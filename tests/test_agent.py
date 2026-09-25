"""Protocol, isolation config and bounded role handoff; no live credentials."""
import support  # noqa: F401  (pack and registry setup)
import ast
import inspect
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
import unittest.mock
from unittest.mock import patch

import agent
from chia.models.opencode import (RateLimitError, AuthenticationError, BillingError,
                                  InvalidRequestError)


def make_agent(web=True, repository='/workspace/original-repository', **kwargs):
    return agent.IsolatedOpenCode(model=agent.MODEL, agent_name='portforge-worker',
                                  web=web, repository=Path(repository),
                                  dangerously_skip_permissions=False, **kwargs)



def setUpModule():
    # Agent nodes read the pack frozen into a run harness; tests use the Parakeet pack.
    global _PORT_PATCH
    _PORT_PATCH = patch.object(agent, "campaign_port", return_value=support.PORT)
    _PORT_PATCH.start()


def tearDownModule():
    _PORT_PATCH.stop()


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'PORTFORGE_REPOSITORY_ROOT': '/workspace/original-repository'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_repository_root_resolves_from_environment(self):
        with patch.object(agent, 'ROOT', Path('/tmp/run/harness')):
            self.assertEqual(agent.repository_root(), Path('/workspace/original-repository'))

    def test_supervisor_protocol_accepts_only_advisory_decisions(self):
        valid = {"action": "continue", "instruction": "Run one tiny independent correctness check."}
        self.assertEqual(agent.parse_decision(json.dumps(valid)), valid)
        for value in (dict(valid, action="reset"), dict(valid, command="shell"),
                      dict(valid, instruction=" "), dict(valid, instruction="x" * 2001), []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                agent.parse_decision(json.dumps(value))
        with self.assertRaises(ValueError):
            agent.parse_decision('{"action":"continue","action":"pause","instruction":"x"}')

    def test_isolation_config_denies_host_writes_exec_and_secrets(self):
        tool = SimpleNamespace(name='parakeet_tools', hostname='127.0.0.1', port=8000)
        config = make_agent(web=True)._build_config([tool])
        permission = config["permission"]
        for key in ("edit", "bash", "task", "question"):
            self.assertEqual(permission[key], "deny", key)
        self.assertEqual(permission["webfetch"], "allow")
        self.assertEqual(permission["websearch"], "allow")
        external = permission["external_directory"]
        self.assertEqual(next(iter(external)), "*")  # broad allow first, denies last
        for denied in ("~/.ssh/**", "~/.config/opencode/**", "~/.local/share/opencode/**",
                       "/workspace/original-repository/**"):
            self.assertEqual(external[denied], "deny", denied)
        self.assertEqual(config["mcp"]["parakeet_tools"],
                         {"type": "remote", "url": "http://127.0.0.1:8000/parakeet_tools/mcp",
                          "enabled": True})
        self.assertEqual(config["share"], "disabled")
        self.assertEqual(config["autoupdate"], False)

    def test_supervisor_variant_denies_web_and_keeps_scoped_tools(self):
        config = make_agent(web=False)._build_config([])
        self.assertEqual(config["permission"]["webfetch"], "deny")
        self.assertEqual(config["permission"]["websearch"], "deny")
        self.assertEqual(config["permission"]["edit"], "deny")
        self.assertNotIn("mcp", config)

    def test_run_command_uses_flag_free_cli_shape_of_current_opencode(self):
        command = make_agent()._build_run_cmd("do the task")
        self.assertEqual(command[:2], ["opencode", "run"])
        self.assertIn("--model", command)
        self.assertEqual(command[command.index("--model") + 1], agent.MODEL)
        self.assertIn("--agent", command)
        # opencode 1.18.x rejects unknown flags; approvals are handled by the
        # permission config, never by a skipped-permissions flag.
        self.assertNotIn("--dangerously-skip-permissions", command)
        self.assertEqual(command[-1], "do the task")

    def test_error_categories_map_typed_provider_failures(self):
        reset = datetime.now(timezone.utc)
        typed = [(RateLimitError("node", reset, "slow down"), "capacity"),
                 (AuthenticationError("node"), "authentication"),
                 (BillingError("node"), "authentication"),
                 (InvalidRequestError("node"), "provider_error")]
        for exc, category in typed:
            with self.subTest(category=category):
                self.assertEqual(agent.error_kind(exc), category)
        for text, category in (("Selected model is at capacity.", "capacity"),
                               ("Authentication failed", "authentication"),
                               ("subprocess timed out", "timeout"), ("other", "provider_error")):
            self.assertEqual(agent.error_kind(text), category)

    def test_rate_limit_failure_is_categorized_without_echoing_context(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(agent, "IsolatedOpenCode") as provider:
                provider.return_value.prompt.side_effect = RateLimitError(
                    "node", datetime.now(timezone.utc), "Secret provider context omitted")
                record = agent.query(root, root, "worker", 1, "task", [], True, 600)
            self.assertFalse(record["success"])
            self.assertEqual(record["error"], "capacity")
            self.assertNotIn("Secret", json.dumps(record))
            self.assertEqual(record["model"], agent.MODEL)

    def test_query_isolates_host_skills_and_restores_environment(self):
        with tempfile.TemporaryDirectory() as root:
            seen = {}

            def prompt(message, tools=None):
                seen.update({k: os.environ.get(k) for k in
                             ("OPENCODE_DISABLE_PROJECT_CONFIG", "OPENCODE_DISABLE_EXTERNAL_SKILLS",
                              "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS")})
                return SimpleNamespace(success=True, result="ok", stderr="")

            with patch.object(agent, "IsolatedOpenCode") as provider, \
                 patch.dict(os.environ, {}, clear=True):
                provider.return_value.prompt.side_effect = prompt
                record = agent.query(root, root, "worker", 1, "task", [], True, 600)
            self.assertTrue(record["success"])
            self.assertEqual(seen["OPENCODE_DISABLE_PROJECT_CONFIG"], "1")
            self.assertEqual(seen["OPENCODE_DISABLE_EXTERNAL_SKILLS"], "1")
            for key in seen:
                self.assertIsNone(os.environ.get(key))

    def test_supervisor_no_tools_empty_workspace_fail_closed_and_durable(self):
        with tempfile.TemporaryDirectory() as root:
            captured = {}
            def query(workspace, run_dir, role, turn, prompt, tools, web, timeout):
                captured.update(files=list(Path(workspace).iterdir()), tools=tools, web=web, timeout=timeout)
                return {"turn": turn, "success": True, "response": "not json", "error": "", "model": agent.MODEL}
            with patch.object(agent, "query", side_effect=query):
                result = agent.supervisor_node._chia_original(root, root, 3, {"pending_jobs": 1})
            self.assertEqual(captured, {"files": [], "tools": [], "web": False,
                                        "timeout": agent.SUPERVISOR_TIMEOUT_SECONDS})
            self.assertEqual(result["decision"]["action"], "pause")
            self.assertFalse(result["success"])
            self.assertEqual(result["error"], "invalid_decision")
            self.assertEqual(len(result["attempt_records"]), 3)
            for relative in result["attempt_records"]:
                self.assertEqual(json.loads((Path(root) / relative).read_text())["response"], "not json")
            self.assertEqual(json.loads((Path(root) / "supervisor-turns/003.json").read_text()), result)

    def test_overlong_reply_is_preserved_and_repaired_without_truncation(self):
        original = json.dumps({"action": "continue", "instruction": "Bounded next step. " * 130})
        repaired = {"action": "continue", "instruction": "Collect the existing pending job; do not resubmit."}
        responses = [original, json.dumps(repaired)]
        with tempfile.TemporaryDirectory() as root:
            def query(workspace, run_dir, role, turn, prompt, tools, web, timeout):
                if role != "supervisor":
                    archived = json.loads((Path(root) / "supervisor-attempt-0-turns/003.json").read_text())
                    self.assertEqual(archived["response"], original)
                    self.assertTrue(archived["provider_success"])
                    self.assertFalse(archived["success"])
                    self.assertIn(json.dumps(original), prompt)
                    self.assertIn('"pending_jobs": 12', prompt)
                    self.assertIn(str(len(json.loads(original)['instruction'])) + ' characters', prompt)
                    self.assertTrue(prompt.endswith(agent.SUPERVISOR_OUTPUT))
                return {"turn": turn, "success": True, "response": responses.pop(0), "error": "", "model": agent.MODEL}
            with patch.object(agent, "query", side_effect=query) as mocked:
                result = agent.supervisor_node._chia_original(root, root, 3, {"pending_jobs": 12})
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual([c.args[2] for c in mocked.call_args_list], ["supervisor", "supervisor-repair-1"])
            self.assertEqual(result["decision"], repaired)
            self.assertTrue(result["success"])
            self.assertEqual(result["response"], json.dumps(repaired))

    def test_invalid_repairs_exhaust_bound_without_synthetic_continue(self):
        replies = ['{"action":"reset","instruction":"x"}',
                   '{"action":"continue","action":"pause","instruction":"x"}',
                   json.dumps({"action": "continue", "instruction": "x" * 2001})]
        with tempfile.TemporaryDirectory() as root, patch.object(agent, "query") as mocked:
            mocked.side_effect = [{"turn": 1, "success": True, "response": text, "error": ""} for text in replies]
            result = agent.supervisor_node._chia_original(root, root, 1, {})
            self.assertEqual(mocked.call_count, 3)
            self.assertFalse(result["success"])
            self.assertEqual(result["decision"]["action"], "pause")
            self.assertEqual(result["error"], "invalid_decision")
            self.assertEqual([json.loads((Path(root) / p).read_text())["response"]
                              for p in result["attempt_records"]], replies)

    def test_valid_pause_is_honored_initially_and_after_repair(self):
        pause = {"action": "pause", "instruction": "Hardware needs external intervention."}
        for repair in (False, True):
            with self.subTest(repair=repair), tempfile.TemporaryDirectory() as root:
                replies = (["not JSON"] if repair else []) + [json.dumps(pause)]
                with patch.object(agent, "query") as mocked:
                    mocked.side_effect = [{"turn": 1, "success": True, "response": text, "error": ""}
                                          for text in replies]
                    result = agent.supervisor_node._chia_original(root, root, 1, {})
                self.assertEqual(mocked.call_count, len(replies))
                self.assertTrue(result["success"])
                self.assertEqual(result["decision"], pause)

    def test_provider_failures_do_not_trigger_format_retries(self):
        for kind in ("capacity", "authentication", "timeout", "provider_error"):
            for repair in (False, True):
                with self.subTest(kind=kind, repair=repair), tempfile.TemporaryDirectory() as root:
                    replies = ([{"turn": 1, "success": True, "response": "bad", "error": ""}] if repair else [])
                    replies += [{"turn": 1, "success": False, "response": "incomplete", "error": kind}]
                    with patch.object(agent, "query", side_effect=replies) as mocked:
                        result = agent.supervisor_node._chia_original(root, root, 1, {})
                    self.assertEqual(mocked.call_count, len(replies))
                    self.assertEqual(result["error"], kind)
                    self.assertFalse(result["success"])
                    self.assertEqual(result["decision"]["action"], "pause")

    def test_repair_provider_logs_are_separate(self):
        with tempfile.TemporaryDirectory() as root, patch.object(agent, "IsolatedOpenCode") as provider:
            provider.return_value.prompt.return_value = SimpleNamespace(success=True, result="{}")
            for role in ("supervisor", "supervisor-repair-1", "supervisor-repair-2"):
                agent.query(root, root, role, 3, "task", [], False, 120)
            log_dirs = [c.kwargs["log_dir"] for c in provider.call_args_list]
            self.assertEqual(len(set(log_dirs)), 3)
            self.assertTrue(all(Path(path).parent.parent == Path(root) / "agent-logs" for path in log_dirs))

    def test_json_protocol_enforced_by_prompt_not_a_cli_schema_flag(self):
        with tempfile.TemporaryDirectory() as root, patch.object(agent, "IsolatedOpenCode") as provider:
            provider.return_value.prompt.return_value = SimpleNamespace(success=True, result="{}")
            for role in ("worker", "supervisor", "supervisor-repair-1"):
                agent.query(root, root, role, 3, "task", [], False, 120)
                self.assertNotIn("--output-schema", provider.call_args.kwargs.get("extra_cli_args", []))
            for call in provider.call_args_list:
                self.assertEqual(call.kwargs["dangerously_skip_permissions"], False)

    def test_custom_phase_instructions_cannot_erase_final_short_protocol(self):
        # Match phase_agent's instruction-assignment replacement, but execute
        # only this function so the test never imports a frozen campaign.
        tree = ast.parse(inspect.getsource(agent.supervisor_node._chia_original))
        node = tree.body[0]
        node.decorator_list = []
        node.body[0] = ast.parse('instructions = "CUSTOM PHASE: qualify all twelve gates.\\n"').body[0]
        ast.fix_missing_locations(tree)
        namespace = dict(vars(agent))
        exec(compile(tree, '<custom-phase-supervisor>', 'exec'), namespace)
        overlong = json.dumps({'action': 'continue', 'instruction': 'x' * 2196})
        concise = json.dumps({'action': 'continue', 'instruction': 'Repair benchmark checkpoint routing and test it.'})
        with tempfile.TemporaryDirectory() as root, patch.dict(namespace, {'query': unittest.mock.Mock()}) as _:
            mocked = namespace['query']
            mocked.side_effect = [{'turn': 1, 'success': True, 'response': text, 'error': ''}
                                  for text in (overlong, concise)]
            result = namespace['supervisor_node'](root, root, 1, {'objective': 'ENTIRE ROADMAP'})
            self.assertTrue(result['success'])
            for call in mocked.call_args_list:
                prompt = call.args[4]
                self.assertTrue(prompt.startswith('CUSTOM PHASE:'))
                self.assertTrue(prompt.endswith(agent.SUPERVISOR_OUTPUT))
                self.assertGreater(prompt.index('FINAL RESPONSE PROTOCOL'), prompt.index('ENTIRE ROADMAP'))
                self.assertIn('200-700 characters', prompt)
                self.assertIn('ONE bounded', prompt)
                self.assertIn('roughly six minutes', prompt)
            self.assertIn('2196 characters', mocked.call_args_list[1].args[4])
            failed = json.loads((Path(root) / result['attempt_records'][0]).read_text())
            self.assertEqual(failed['instruction_characters'], 2196)
            self.assertEqual(failed['response'], overlong)

    def test_three_json_length_failures_remain_fail_closed(self):
        lengths = (2196, 2044, 2035)
        replies = [json.dumps({'action': 'continue', 'instruction': 'x' * n}) for n in lengths]
        with tempfile.TemporaryDirectory() as root, patch.object(agent, "query") as mocked:
            mocked.side_effect = [{"turn": 1, "success": True, "response": text, "error": ""} for text in replies]
            result = agent.supervisor_node._chia_original(root, root, 1, {})
            self.assertEqual(mocked.call_count, 3)
            self.assertFalse(result["success"])
            self.assertEqual(result["decision"]["action"], "pause")
            records = [json.loads((Path(root) / p).read_text()) for p in result["attempt_records"]]
            self.assertEqual([r['instruction_characters'] for r in records], list(lengths))
            self.assertEqual([r['response'] for r in records], replies)

    def test_worker_handoff_has_bounded_timeout_and_state_instruction(self):
        with tempfile.TemporaryDirectory() as root, patch.object(agent, "query") as query:
            query.return_value = {"turn": 2, "success": True, "response": "saved", "error": ""}
            agent.worker_node._chia_original(root, root, ["scoped-tool"], 2, "resume")
            arguments = query.call_args.args
            self.assertIn("STATE.md", arguments[4])
            self.assertTrue(arguments[4].endswith(agent.WORKER_HANDOFF))
            self.assertEqual(arguments[5:], (["scoped-tool"], True, agent.WORKER_TIMEOUT_SECONDS))
            self.assertEqual(agent.WORKER_TIMEOUT_SECONDS, 1800)

    def test_worker_handoff_frames_the_checkpoint_as_a_strong_target(self):
        for phrase in ("strong target, not a hard rule", "thirty minutes",
                       "progress checkpoint to STATE.md"):
            self.assertIn(phrase, agent.WORKER_HANDOFF)
        body = agent.read_prompt("worker", support.PORT)
        self.assertIn("strong guidance", body)
        self.assertIn("BENCHMARK.md", body)
        self.assertIn(support.PORT["repo_id"], body)
        self.assertNotIn("{repo_id}", body)

    def test_phase_worker_retains_compact_batched_handoff_and_scoped_tools(self):
        tree = ast.parse(inspect.getsource(agent.worker_node._chia_original))
        node = tree.body[0]
        node.decorator_list = []
        node.body[0] = ast.parse('instructions = "CUSTOM PHASE: complete frozen gates.\\n"').body[0]
        ast.fix_missing_locations(tree)
        namespace = dict(vars(agent))
        exec(compile(tree, '<custom-phase-worker>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as root, patch.dict(namespace, {'query': unittest.mock.Mock()}):
            query = namespace['query']
            query.return_value = {'turn': 2, 'success': True, 'response': 'saved', 'error': ''}
            namespace['worker_node'](root, root, ['scoped-tool'], 2, 'pending handle example')
            arguments = query.call_args.args
            self.assertTrue(arguments[4].startswith('CUSTOM PHASE:'))
            self.assertTrue(arguments[4].endswith(agent.WORKER_HANDOFF))
            self.assertLess(arguments[4].index('pending handle example'),
                            arguments[4].index('FINAL WORKING PROTOCOL'))
            self.assertEqual(arguments[5:], (['scoped-tool'], True, agent.WORKER_TIMEOUT_SECONDS))


class FakeMCPTool:
    name = 'parakeet_tools'
    hostname = '127.0.0.1'
    port = 8000
    mcp = SimpleNamespace(_tool_manager=SimpleNamespace(list_tools=lambda: [
        SimpleNamespace(name='port_run'), SimpleNamespace(name='port_write')]))


def make_claude(root, web=True, repository='/workspace/original-repository', **kwargs):
    return agent.IsolatedClaude(model='claude-test', work_dir=root, web=web,
                                repository=Path(repository), timeout_seconds=60, retries=1, **kwargs)


def stream(*events):
    return ''.join(json.dumps(event) + '\n' for event in events)


class ClaudeBackendTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'PORTFORGE_REPOSITORY_ROOT': '/workspace/original-repository'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_role_selection_defaults_overrides_and_legacy_model(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(agent.role_selection('worker'), ('opencode', agent.DEFAULT_MODELS['opencode']))
        with patch.dict(os.environ, {'PORTFORGE_AGENT_MODEL': 'zai/legacy'}, clear=True):
            self.assertEqual(agent.role_selection('worker'), ('opencode', 'zai/legacy'))
        with patch.dict(os.environ, {'PORTFORGE_AGENT_MODEL': 'zai/legacy', 'PORTFORGE_WORKER_BACKEND': 'claude',
                                     'PORTFORGE_SUPERVISOR_BACKEND': 'opencode'}, clear=True):
            # A legacy opencode model name never leaks into a Claude role.
            self.assertEqual(agent.role_selection('worker'), ('claude', agent.DEFAULT_MODELS['claude']))
            self.assertEqual(agent.role_selection('supervisor-repair-1'), ('opencode', 'zai/legacy'))
        with patch.dict(os.environ, {'PORTFORGE_SUPERVISOR_BACKEND': 'claude',
                                     'PORTFORGE_SUPERVISOR_MODEL': 'claude-small'}, clear=True):
            self.assertEqual(agent.role_selection('supervisor'), ('claude', 'claude-small'))
        with patch.dict(os.environ, {'PORTFORGE_WORKER_BACKEND': 'codex'}, clear=True), \
             self.assertRaises(ValueError):
            agent.role_selection('worker')

    def test_claude_command_denies_host_writes_exec_and_secrets(self):
        with tempfile.TemporaryDirectory() as root:
            command = make_claude(root)._build_cmd([FakeMCPTool()])
        self.assertEqual(command[:2], ['claude', '--print'])
        self.assertEqual(command[command.index('--model') + 1], 'claude-test')
        for flag in ('--restricted', '--strict-mcp-config', '--no-session-persistence',
                     '--disable-slash-commands'):
            self.assertIn(flag, command)
        self.assertNotIn('--safe-mode', command)  # it would also drop the Chia MCP servers
        self.assertTrue(json.loads(command[command.index('--settings') + 1])['disableAllHooks'])
        self.assertNotIn('--dangerously-skip-permissions', command)
        self.assertEqual(command[command.index('--permission-mode') + 1], 'dontAsk')
        self.assertEqual(command[command.index('--tools') + 1], 'Read,Grep,Glob,WebFetch,WebSearch')
        settings = json.loads(command[command.index('--settings') + 1])['permissions']
        self.assertEqual(settings['defaultMode'], 'dontAsk')
        for denied in ('Bash', 'Edit', 'Write', 'Task', f'Read(/{Path.home()}/.ssh/**)',
                       'Read(//workspace/original-repository/**)'):
            self.assertIn(denied, settings['deny'])
        self.assertEqual(command[command.index('--allowedTools') + 1],
                         'mcp__parakeet_tools__port_run,mcp__parakeet_tools__port_write')
        mcp = json.loads(Path(command[command.index('--mcp-config') + 1]).read_text())
        self.assertEqual(mcp, {'mcpServers': {'parakeet_tools': {
            'type': 'http', 'url': 'http://127.0.0.1:8000/parakeet_tools/mcp'}}})
        self.assertEqual(command[-2:], ['-p', '-'])

    def test_claude_supervisor_variant_has_no_web_or_mcp(self):
        with tempfile.TemporaryDirectory() as root:
            command = make_claude(root, web=False)._build_cmd([])
        self.assertEqual(command[command.index('--tools') + 1], 'Read,Grep,Glob')
        self.assertNotIn('WebFetch', json.loads(command[command.index('--settings') + 1])['permissions']['allow'])
        self.assertNotIn('--mcp-config', command)

    def test_claude_runs_in_workspace_with_timeout_and_returns_final_result(self):
        output = stream({'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'Looking...'}]}},
                        {'type': 'result', 'result': '{"action":"pause","instruction":"done"}'})
        with tempfile.TemporaryDirectory() as root, patch.object(agent.subprocess, 'run') as run:
            run.return_value = SimpleNamespace(stdout=output, stderr='', returncode=0)
            llm = make_claude(root, log_dir=root + '/logs')
            result = llm._run_claude('task', [])
            self.assertEqual(run.call_args.kwargs['cwd'], root)
            self.assertEqual(run.call_args.kwargs['timeout'], 60)
            self.assertEqual(run.call_args.kwargs['input'], 'task')
            self.assertNotIn('CLAUDECODE', run.call_args.kwargs['env'])
            self.assertEqual(run.call_args.kwargs['env']['CLAUDE_CODE_DISABLE_AUTO_MEMORY'], '1')
            self.assertEqual(result.result, '{"action":"pause","instruction":"done"}')
            self.assertIn('Looking...', next(Path(root, 'logs').glob('*.log')).read_text())

    def test_query_uses_claude_and_records_backend(self):
        output = stream({'type': 'result', 'result': 'unit complete'})
        with tempfile.TemporaryDirectory() as root, \
             patch.dict(os.environ, {'PORTFORGE_WORKER_BACKEND': 'claude', 'PORTFORGE_WORKER_MODEL': 'claude-test'}), \
             patch.object(agent.subprocess, 'run', return_value=SimpleNamespace(stdout=output, stderr='', returncode=0)) as run:
            record = agent.query(root, root, 'worker', 2, 'task', [], True, 600)
        self.assertEqual(record, {'turn': 2, 'success': True, 'response': 'unit complete', 'error': '',
                                  'model': 'claude-test', 'backend': 'claude'})
        self.assertEqual(run.call_args.args[0][0], 'claude')

    def test_claude_timeout_and_usage_limit_are_categorized(self):
        with tempfile.TemporaryDirectory() as root, \
             patch.dict(os.environ, {'PORTFORGE_WORKER_BACKEND': 'claude'}):
            with patch.object(agent.subprocess, 'run',
                              side_effect=agent.subprocess.TimeoutExpired('claude', 600, output=b'')):
                self.assertEqual(agent.query(root, root, 'worker', 1, 'task', [], True, 600)['error'], 'timeout')
            limited = stream({'type': 'result', 'is_error': True,
                              'result': "You've hit your limit · resets 5pm (UTC)"})
            with patch.object(agent.subprocess, 'run',
                              return_value=SimpleNamespace(stdout=limited, stderr='', returncode=1)):
                record = agent.query(root, root, 'worker', 1, 'task', [], True, 600)
        self.assertEqual((record['success'], record['error'], record['response']), (False, 'capacity', ''))
        expected = agent.claude_errors.parse_rate_limit_reset("You've hit your limit · resets 5pm (UTC)")
        self.assertAlmostEqual(record['retry_after'], expected.timestamp(), delta=5)
        self.assertEqual(agent.error_kind(agent.claude_errors.AuthenticationError('node')), 'authentication')


if __name__ == "__main__":
    unittest.main()
