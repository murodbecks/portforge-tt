"""Submission and duplicate-job guarantees, without Ray or remote execution."""
import json
import io
import tarfile
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import support  # noqa: F401  (pack and registry setup)
import portforge_loop as loop

support.configure_loop(loop)
from runner import write_json


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.tools = object.__new__(loop.PortTools)
        self.tools.workspace, self.tools.run_dir, self.tools.refs = root / 'work', root / 'run', {}
        self.tools.workspace.mkdir()
        self.tools.run_dir.mkdir()
        support.write_harness(self.tools.run_dir / 'harness')
        self.config = {'tt': {'host': '192.0.2.1'}, 'reference': {'host': '192.0.2.2'}}
        write_json(self.tools.run_dir / 'cluster.json', self.config)
        (self.tools.workspace / 'backend.py').write_text('# fresh implementation')

    def tearDown(self):
        self.temp.cleanup()

    def test_status_exact_old_job_keeps_complete_gate_and_is_read_only(self):
        job = self.tools.run_dir / 'jobs/old-gate'
        request = {'kind': 'evaluate', 'stage': 'bringup', 'model': 'parakeet',
                   'precision': 'bfp8_b', 'code_sha256': 'source', 'cluster_sha256': 'cluster',
                   'suite_definition_sha256': 'definition', 'timeout': 300}
        result = {'returncode': 0, 'stdout': 'x' * 2500, 'stderr': 'y' * 2000,
                  'verdict': {'passed': True, 'accepted': False, 'suite_sha256': 'suite',
                              'checks': [{'name': 'long', 'values': list(range(1200))}]}}
        write_json(job / 'request.json', request)
        write_json(job / 'result.json', result)
        os.utime(job / 'request.json', (1, 1))
        saved = {p.name: p.read_bytes() for p in job.iterdir()}
        for index in range(14):
            write_json(self.tools.run_dir / f'jobs/new-{index}/request.json', {'kind': 'run'})
            write_json(self.tools.run_dir / f'jobs/new-{index}/result.json', {'returncode': 0})
        default = self.tools.status()
        self.assertEqual(len(default), 12)
        self.assertNotIn('old-gate', default)
        with patch.object(loop, 'remote') as remote, patch.object(loop, 'reference_progress') as progress, \
             patch.object(loop, 'write_json') as write:
            exact = self.tools.status(job_id='old-gate')
            compact = self.tools.status(job_id='old-gate', compact=True)
            remote.assert_not_called(); progress.assert_not_called(); write.assert_not_called()
        self.assertEqual(exact, compact)
        self.assertEqual(set(exact), {'old-gate'})
        self.assertEqual(exact['old-gate']['request_identity'], request)
        self.assertEqual(exact['old-gate']['verdict'], result['verdict'])
        self.assertEqual(exact['old-gate']['stdout'], result['stdout'][-1000:])
        self.assertEqual(exact['old-gate']['stderr'], result['stderr'][-1000:])
        self.assertEqual(saved, {p.name: p.read_bytes() for p in job.iterdir()})
        self.assertFalse((self.tools.run_dir / 'baselines').exists())

    def test_status_exact_pending_is_local_and_preserves_request(self):
        request = {'kind': 'reference-study', 'model': 'parakeet', 'cluster_sha256': 'missing'}
        job = self.tools.run_dir / 'jobs/pending'
        write_json(job / 'request.json', request)
        with patch.object(loop, 'remote') as remote, patch.object(loop, 'reference_progress') as progress:
            result = self.tools.status(job_id='pending')['pending']
            remote.assert_not_called(); progress.assert_not_called()
        self.assertEqual(result['status'], 'pending')
        self.assertEqual(result['request_identity'], request)
        self.assertEqual(result['details']['job_id'], 'pending')
        self.assertFalse((job / 'result.json').exists())

    def test_status_exact_historical_dotted_reference_id(self):
        job_id = 'gpu-precision.parakeet-20260921T1020'
        request = {'kind': 'reference-study', 'model': 'parakeet'}
        result = {'returncode': 0, 'modes': {'bf16': {
            'verdict': {'passed': True, 'checks': [{'error': .001}]},
            'measurements': {'cases': [{'seconds': [1., 2.]}]}}}}
        job = self.tools.run_dir / 'jobs' / job_id
        write_json(job / 'request.json', request)
        write_json(job / 'result.json', result)
        with patch.object(loop, 'remote') as remote, patch.object(loop, 'reference_progress') as progress:
            actual = self.tools.status(job_id=job_id)[job_id]
            remote.assert_not_called(); progress.assert_not_called()
        self.assertEqual(actual['modes'], result['modes'])
        self.assertEqual(actual['request_identity'], request)
        self.assertEqual(actual['details']['job_id'], job_id)
        # The TT log validator deliberately retains its separate contract.
        with self.assertRaises(ValueError):
            loop.validate_log_request(job_id, 'stdout', 0, 1)

    def test_status_exact_rejects_invalid_missing_and_symlink_paths(self):
        for job_id in (None, 1, False, [], '.', '..', '../other', '/tmp', 'x/y',
                       'x\\y', 'a' * 81, 'missing', 'trailing\n', '.hidden', '-leading', '_leading'):
            with self.subTest(job_id=job_id), self.assertRaises(ValueError):
                self.tools.status(job_id=job_id)
        jobs = self.tools.run_dir / 'jobs'
        job = jobs / 'own'
        write_json(job / 'request.json', {'kind': 'run'})
        write_json(job / 'result.json', {'returncode': 0})
        for path in (job / 'request.json', job / 'result.json', job, jobs):
            moved = path.with_name(path.name + '-saved')
            path.rename(moved)
            path.symlink_to(moved, target_is_directory=moved.is_dir())
            try:
                with self.subTest(path=str(path)), self.assertRaises(ValueError):
                    self.tools.status(job_id='own')
            finally:
                path.unlink(); moved.rename(path)
        (job / 'result.json').unlink()
        (job / 'result.json').symlink_to(job / 'absent')
        with self.assertRaises(ValueError):
            self.tools.status(job_id='own')

    def timeout_fixture(self):
        health = {'cards': [{'pci': '0000:c1:00.0', 'state': 'quarantined', 'job': 'timed-out'}]}
        job = self.tools.run_dir / 'jobs/timed-out'
        request = {'kind': 'run', 'cluster_sha256': loop.digest(self.config)}
        result = {'returncode': 124, 'failure': 'timeout', 'timeout': True,
                  'pci': '0000:c1:00.0', 'host_memory_limit': False,
                  'job': 'timed-out', 'cluster_sha256': loop.digest(self.config)}
        write_json(job / 'request.json', request)
        write_json(job / 'result.json', result)
        return health, job, request, result

    def test_timeout_recovery_requires_real_probe_and_cannot_repeat(self):
        health, job, request, result = self.timeout_fixture()
        healthy = {'cards': [{**health['cards'][0], 'state': 'healthy'}]}
        with patch.object(loop, 'remote', side_effect=[{'returncode': 0}, healthy]) as remote:
            self.assertEqual(loop.recover_timed_out_card(
                self.tools.run_dir, self.config, health, 'parakeet'), healthy)
            probe = remote.call_args_list[0].args[2]
            self.assertEqual(probe['kind'], 'preflight')
            self.assertEqual(probe['pci'], result['pci'])
            self.assertEqual(probe['model'], 'parakeet')
            self.assertEqual(remote.call_args_list[1].args[2], {'kind': 'health'})
        receipt = json.loads(next((self.tools.run_dir / 'recoveries').glob('*.json')).read_text())
        self.assertEqual(receipt['state'], 'complete')
        self.assertEqual(receipt['failed_job'], job.name)
        with patch.object(loop, 'remote') as remote:
            # Even another terminal timeout on the same card cannot reset the budget.
            health['cards'][0]['job'] = 'second-timeout'
            write_json(job.parent / 'second-timeout/request.json', request)
            write_json(job.parent / 'second-timeout/result.json', {**result, 'job': 'second-timeout'})
            self.assertEqual(loop.recover_timed_out_card(
                self.tools.run_dir, self.config, health, 'parakeet'), health)
            remote.assert_not_called()

    def test_timeout_recovery_refuses_pending_memory_failure_or_wrong_identity(self):
        health, job, request, result = self.timeout_fixture()
        variants = [({}, {'failure': 'host_memory_limit', 'host_memory_limit': True}),
                    ({}, {'timeout': False}), ({}, {'pci': 'different'}),
                    ({}, {'returncode': 0}), ({}, {'job': 'different'}),
                    ({}, {'cluster_sha256': 'old-environment'}),
                    ({'cluster_sha256': 'old-environment'}, {})]
        with patch.object(loop, 'remote') as remote:
            for req, res in variants:
                write_json(job / 'request.json', {**request, **req})
                write_json(job / 'result.json', {**result, **res})
                self.assertEqual(loop.recover_timed_out_card(
                    self.tools.run_dir, self.config, health, 'parakeet'), health)
            write_json(job / 'request.json', request)
            write_json(job / 'result.json', result)
            write_json(job.parent / 'pending/request.json', {'kind': 'evaluate'})
            self.assertEqual(loop.recover_timed_out_card(
                self.tools.run_dir, self.config, health, 'parakeet'), health)
            remote.assert_not_called()

    def test_timeout_recovery_preserves_failed_or_ambiguous_probe(self):
        health, job, request, result = self.timeout_fixture()
        with patch.object(loop, 'remote', side_effect=TimeoutError('transport ambiguous')):
            with self.assertRaises(TimeoutError):
                loop.recover_timed_out_card(self.tools.run_dir, self.config, health, 'parakeet')
        receipt = next((self.tools.run_dir / 'recoveries').glob('*.json'))
        self.assertEqual(json.loads(receipt.read_text())['state'], 'submitted')
        with patch.object(loop, 'remote') as remote:
            self.assertEqual(loop.recover_timed_out_card(
                self.tools.run_dir, self.config, health, 'parakeet'), health)
            remote.assert_not_called()
        # A separate, fresh configuration gets its own single probe budget.
        replacement = {**self.config, 'tt': {'host': '192.0.2.3'}}
        write_json(job / 'request.json', {**request, 'cluster_sha256': loop.digest(replacement)})
        write_json(job / 'result.json', {**result, 'cluster_sha256': loop.digest(replacement)})
        with patch.object(loop, 'remote', side_effect=[{'returncode': 124, 'timeout': True}, health]):
            self.assertEqual(loop.recover_timed_out_card(
                self.tools.run_dir, replacement, health, 'parakeet'), health)

    def test_timeout_recovery_claim_is_exclusive_even_with_stale_existence_check(self):
        health, job, request, result = self.timeout_fixture()
        receipt = self.tools.run_dir / 'recoveries' / (
            loop.digest([loop.digest(self.config), result['pci']]) + '.json')
        write_json(receipt, {'state': 'submitted', 'request': {'job': 'other-probe'}})
        original_exists = Path.exists
        with patch.object(Path, 'exists', lambda p: False if p == receipt else original_exists(p)), \
                patch.object(loop, 'remote') as remote:
            self.assertEqual(loop.recover_timed_out_card(
                self.tools.run_dir, self.config, health, 'parakeet'), health)
            remote.assert_not_called()
        self.assertEqual(json.loads(receipt.read_text())['request']['job'], 'other-probe')

    def test_compact_status_preserves_failure_quality_identity_and_legacy_response(self):
        job = self.tools.run_dir / 'jobs/failure'
        request = {'kind': 'evaluate', 'stage': 'smoke', 'model': 'parakeet', 'precision': 'bf16',
                   'code_sha256': 'source', 'cluster_sha256': 'cluster', 'suite_definition_sha256': 'definition'}
        result = {'returncode': 125, 'failure': 'host_memory_limit', 'timeout': False,
                  'host_memory_limit': True, 'peak_host_rss_bytes': 12345,
                  'host_memory_accounting': 'sampled_sum_rss_shared_pages_counted_per_process',
                  'model': 'parakeet', 'precision': 'bf16', 'pci': '0000:41:00.0',
                  'stdout': 'old ' * 500 + 'last output', 'stderr': 'trace ' * 600 + 'memory failure',
                  'verdict': {'passed': False, 'accepted': False, 'production_certified': False,
                              'suite_sha256': 'actualsuite', 'model_manifest_sha256': 'modelidentity',
                              'precision': {'requested': 'bf16'}, 'failures': ['numerical'],
                              'checks': [{'name': 'one', 'seconds': [1, 2], 'rows': 1}],
                              'quality': {'threshold': 0.1, 'observed': 0.2}}}
        write_json(job / 'request.json', request)
        write_json(job / 'result.json', result)
        saved = (job / 'result.json').read_bytes()
        self.assertEqual(self.tools.status(), {'failure': result})
        compact = self.tools.status(compact=True)['failure']
        self.assertEqual(compact['request_identity'], request)
        for key, value in result.items():
            if key not in ('stdout', 'stderr'):
                self.assertEqual(compact[key], value)
        for stream in ('stdout', 'stderr'):
            self.assertEqual(compact[stream], result[stream][-1000:])
            self.assertEqual(compact['log_tails'][stream], {
                'available_chars': len(result[stream]), 'returned_chars': 1000, 'truncated': True})
        self.assertEqual(compact['details']['tool'], 'port_log')
        self.assertEqual(compact['details']['job_id'], 'failure')
        self.assertEqual((job / 'result.json').read_bytes(), saved)
        self.assertEqual(self.tools.status(), {'failure': result})
        for invalid in ('false', 1, None):
            with self.subTest(compact=invalid), self.assertRaises(ValueError):
                self.tools.status(compact=invalid)

    def test_compact_status_preserves_baseline_registration_without_inventing_gates(self):
        result = {'returncode': 0, 'stdout': 'short', 'stderr': '',
                  'verdict': {'passed': True, 'accepted': False, 'suite_sha256': 'actualsuite'}}
        self.archive('bringup', result)
        compact = self.tools.status(compact=True)['one']
        self.assertEqual(compact['verdict'], result['verdict'])
        self.assertFalse(compact['log_tails']['stdout']['truncated'])
        self.assertEqual(compact['log_tails']['stderr']['returned_chars'], 0)
        baselines = list((self.tools.run_dir / 'baselines').rglob('*.json'))
        self.assertEqual(len(baselines), 1)
        self.assertEqual(json.loads(baselines[0].read_text())['job'], 'one')
        write_json(self.tools.run_dir / 'jobs/shell/request.json', {'kind': 'run'})
        write_json(self.tools.run_dir / 'jobs/shell/result.json', {'returncode': 0})
        shell = self.tools.status(compact=True)['shell']
        self.assertNotIn('verdict', shell)
        self.assertIsNone(shell['request_identity']['precision'])

    def test_compact_status_keeps_old_pending_and_reference_study_details(self):
        run = self.tools.run_dir
        key = loop.digest(self.config)
        write_json(run / 'clusters' / (key + '.json'), self.config)
        for job, kind in (('pending-reference', 'reference-study'), ('pending-tt', 'run')):
            path = run / 'jobs' / job / 'request.json'
            write_json(path, {'kind': kind, 'cluster_sha256': key, 'code_sha256': job + '-source'})
            os.utime(path, (1, 1))
        for index in range(14):
            write_json(run / f'jobs/done-{index}/request.json', {'kind': 'run'})
            write_json(run / f'jobs/done-{index}/result.json', {'returncode': 0})
        study = {'returncode': 0, 'modes': {'fp16': {'verdict': {'passed': False},
                  'measurements': {'cases': [{'seconds': [1.0]}]}}}}
        write_json(run / 'jobs/study/request.json', {'kind': 'reference-study', 'model': 'parakeet'})
        write_json(run / 'jobs/study/result.json', study)
        with patch.object(loop, 'reference_progress', return_value={'phase': 'queued', 'background_nonblocking': True}), \
             patch.object(loop, 'remote', return_value={'phase': 'decode'}):
            legacy, compact = self.tools.status(), self.tools.status(compact=True)
        self.assertEqual(set(compact), set(legacy))
        self.assertEqual(len(compact), 14)
        for job in ('pending-reference', 'pending-tt'):
            self.assertEqual(compact[job]['status'], 'pending')
            self.assertEqual(compact[job]['progress'], legacy[job]['progress'])
            self.assertEqual(compact[job]['request_identity']['code_sha256'], job + '-source')
        self.assertEqual(compact['study']['modes'], study['modes'])
        for job in ('study', 'pending-reference'):
            self.assertEqual(compact[job]['details']['tool'], 'port_status')
            self.assertIs(compact[job]['details']['compact'], False)

    def test_log_registered_and_uses_job_archived_endpoint(self):
        names = []
        self.tools.mcp = SimpleNamespace(add_tool=lambda fn, name: names.append(name))
        self.tools.setup(self.tools.workspace, self.tools.run_dir)
        self.assertIn('port_log', names)
        archived = {'tt': {'host': '192.0.2.99'}, 'reference': {'host': '192.0.2.2'}}
        key = loop.digest(archived)
        write_json(self.tools.run_dir / 'clusters' / (key + '.json'), archived)
        write_json(self.tools.run_dir / 'jobs/own/request.json', {'kind': 'run', 'cluster_sha256': key})
        with patch.object(loop, 'remote', return_value={'text': 'earlier RESULT', 'eof': False}) as remote:
            result = self.tools.log('own', 'stderr', 12, 1024)
        self.assertEqual(result['text'], 'earlier RESULT')
        remote.assert_called_once_with(archived, str(self.tools.run_dir),
            {'kind': 'log', 'job': 'own', 'stream': 'stderr', 'offset_bytes': 12, 'limit_bytes': 1024}, timeout=15)

    def test_log_rejects_unknown_reference_and_bad_bounds_before_remote(self):
        write_json(self.tools.run_dir / 'jobs/reference/request.json', {'kind': 'reference-study'})
        with patch.object(loop, 'remote') as remote:
            for job, kwargs in [('unknown', {}), ('reference', {}), ('../escape', {}),
                                ('unknown', {'stream': 'output'}), ('unknown', {'limit_bytes': 65537}),
                                ('unknown', {'offset_bytes': True})]:
                with self.subTest(job=job, kwargs=kwargs), self.assertRaises(ValueError):
                    self.tools.log(job, **kwargs)
            remote.assert_not_called()

    def test_log_rejects_symlink_job_and_tampered_cluster(self):
        run = self.tools.run_dir
        key = loop.digest(self.config)
        write_json(run / 'jobs/own/request.json', {'kind': 'evaluate', 'cluster_sha256': key})
        write_json(run / 'clusters' / (key + '.json'), {'tt': {'host': '192.0.2.99'}})
        with patch.object(loop, 'remote') as remote:
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                self.tools.log('own')
            (run / 'jobs/link').symlink_to(run / 'jobs/own', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'unsafe job'):
                self.tools.log('link')
            (run / 'jobs/own/request.json').unlink()
            (run / 'jobs/own/request.json').symlink_to(run / 'clusters' / (key + '.json'))
            with self.assertRaisesRegex(ValueError, 'unsafe job'):
                self.tools.log('own')
            remote.assert_not_called()

    def test_run_punctuation_cannot_create_nested_mcp_config_keys(self):
        from types import SimpleNamespace
        from agent import IsolatedOpenCode
        name = loop.tool_name(self.tools.run_dir / 'model-parakeet with spaces')
        self.assertRegex(name, r'^portforge_[a-f0-9]{16}$')
        tool = SimpleNamespace(name=name, hostname='127.0.0.1', port=8002)
        agent = object.__new__(IsolatedOpenCode)
        agent.web, agent.repository, agent.system_message = True, Path('/repo'), ''
        agent.agent_name, agent.config = 'portforge-worker', None
        agent.additional_providers = []
        config = agent._build_config([tool])
        self.assertEqual(list(config['mcp']), [name])
        self.assertEqual(config['mcp'][name]['url'],
                         f'http://127.0.0.1:8002/{name}/mcp')

    def test_tool_identity_is_stable_and_distinguishes_runs(self):
        self.assertEqual(loop.tool_name('/tmp/run/../run'), loop.tool_name('/tmp/run'))
        self.assertNotEqual(loop.tool_name('/tmp/one/model'), loop.tool_name('/tmp/two/model'))

    def make_previous_run(self, name='previous', **manifest):
        root = self.tools.run_dir.parent / name
        source = root / 'workspace'
        (source / 'baselines').mkdir(parents=True)
        (source / '.opencode').mkdir()
        write_json(root / 'manifest.json', {'seed_policy': 'fresh-public-parakeet', 'model': 'parakeet',
                                            'workspace': str(source), **manifest})
        (source / 'STATE.md').write_text('# STATE\nnext: read probe v7 e244a077\n')
        (source / 'RESEARCH.md').write_text('research notes')
        (source / 'TT_NOTES.md').write_text('control notes are kept on resume')
        (source / 'backend.py').write_text('# carried implementation')
        (source / 'baselines/smoke.json').write_text('{"digest": "abc"}')
        (source / '.opencode/state.json').write_text('{}')
        (source / 'TASK.md').write_text('superseded instructions')
        (self.tools.workspace / 'TASK.md').write_text('current instructions')
        return root

    def test_resume_carries_full_workspace_and_state_without_passes(self):
        root = self.make_previous_run(history_runs=['/archive/older'])
        write_json(root / 'jobs/old/request.json', {'kind': 'evaluate', 'stage': 'bringup'})
        write_json(root / 'jobs/old/result.json', {'verdict': {'passed': True}})
        receipt = loop.resume_workspace(self.tools.workspace, root, 'parakeet',
                                        {'worker': 'claude', 'supervisor': 'opencode'}, 'Probe v7 compares x@W to W.T.')
        work = self.tools.workspace
        for name in ('RESEARCH.md', 'TT_NOTES.md', 'backend.py', 'baselines/smoke.json'):
            self.assertEqual((work / name).read_text(), (root / 'workspace' / name).read_text())
        self.assertFalse((work / '.opencode').exists())
        self.assertEqual((work / 'TASK.md').read_text(), 'current instructions')
        state = (work / 'STATE.md').read_text()
        self.assertTrue(state.startswith('# Resumed campaign'))
        for required in ('continues previous', 'opencode/opencode -> claude/opencode',
                         'no pass, pending job or baseline acceptance is inherited',
                         '## Operator note\n\nProbe v7 compares x@W to W.T.',
                         '# STATE\nnext: read probe v7 e244a077'):
            self.assertIn(required, state)
        self.assertEqual(receipt['history_runs'], [str(root.resolve()), '/archive/older'])
        self.assertEqual(receipt['resumed_from'], str(root.resolve()))
        self.assertEqual(self.tools.status(), {})
        with self.assertRaisesRegex(ValueError, 'needs an independent bringup pass'):
            self.tools.submit(model='parakeet')

    def test_resume_rejects_other_model_submitted_or_non_fresh_sources(self):
        root = self.make_previous_run(model='other')
        with self.assertRaisesRegex(ValueError, 'different model'):
            loop.resume_workspace(self.tools.workspace, root, 'parakeet', {'worker': 'claude', 'supervisor': 'claude'})
        root = self.make_previous_run('submitted')
        write_json(root / 'submission.json', {'code_sha256': 'other source'})
        with self.assertRaisesRegex(ValueError, 'differs from its frozen submission'):
            loop.resume_workspace(self.tools.workspace, root, 'parakeet', {'worker': 'claude', 'supervisor': 'claude'})
        root = self.make_previous_run('private', seed_policy='private')
        with self.assertRaisesRegex(ValueError, 'fresh public-source'):
            loop.resume_workspace(self.tools.workspace, root, 'parakeet', {'worker': 'claude', 'supervisor': 'claude'})

    def test_resume_continues_a_submitted_source_that_still_matches(self):
        root = self.make_previous_run('frozen')
        source = loop.snapshot(root / 'workspace')
        write_json(root / 'submission.json', {'files': source, 'code_sha256': loop.code_hash(source)})
        loop.resume_workspace(self.tools.workspace, root, 'parakeet', {'worker': 'claude', 'supervisor': 'claude'})
        self.assertEqual((self.tools.workspace / 'backend.py').read_text(), '# carried implementation')
        self.assertFalse((self.tools.run_dir / 'submission.json').exists())

    def test_historical_full_verdicts_are_operator_only(self):
        root = self.make_previous_run()
        write_json(self.tools.run_dir / 'manifest.json', {'history_runs': [str(root)]})
        write_json(root / 'jobs/final/request.json', {'kind': 'evaluate', 'stage': 'full'})
        write_json(root / 'jobs/final/result.json', {'verdict': {'failures': ['held-out case']}})
        with self.assertRaisesRegex(ValueError, 'operator-only'):
            self.tools.status(job_id='final')
        with self.assertRaisesRegex(ValueError, 'operator-only'):
            self.tools.log('final')
        write_json(self.tools.run_dir / 'jobs/final/request.json', {'kind': 'evaluate', 'stage': 'full'})
        self.assertIn('final', self.tools.status(job_id='final'))  # this run's own receipt stays visible

    def test_history_jobs_are_readable_but_not_listed_reused_or_passing(self):
        root = self.make_previous_run()
        write_json(self.tools.run_dir / 'manifest.json', {'history_runs': [str(root)]})
        archived = {'tt': {'host': '192.0.2.77'}, 'reference': {'host': '192.0.2.2'}}
        key = loop.digest(archived)
        write_json(root / 'clusters' / (key + '.json'), archived)
        write_json(root / 'jobs/e244a077/request.json', {'kind': 'run', 'cluster_sha256': key})
        write_json(root / 'jobs/e244a077/result.json', {'returncode': 0, 'stdout': 'V7 none'})
        self.assertEqual(self.tools.status(), {})
        exact = self.tools.status(job_id='e244a077')['e244a077']
        self.assertEqual((exact['historical_run'], exact['stdout']), (root.name, 'V7 none'))
        with patch.object(loop, 'remote', return_value={'text': 'V7 none'}) as remote:
            self.assertEqual(self.tools.log('e244a077')['text'], 'V7 none')
        self.assertEqual(remote.call_args.args[:2], (archived, str(root)))
        write_json(self.tools.run_dir / 'jobs/current/request.json', {'kind': 'run'})
        self.assertNotIn('historical_run', self.tools.status(job_id='current')['current'])
        with self.assertRaisesRegex(ValueError, 'unknown or unsafe job'):
            self.tools.status(job_id='missing')

    def test_references_are_reused_only_from_the_same_evaluator(self):
        earlier = self.tools.run_dir.parent / 'earlier'
        (earlier / 'harness').mkdir(parents=True)
        self.assertFalse(loop.same_evaluator(earlier))  # no archived evaluator
        (earlier / 'harness/evaluate.py').write_bytes((support.PACK / 'evaluate.py').read_bytes())
        self.assertTrue(loop.same_evaluator(earlier))
        (earlier / 'harness/evaluate.py').write_text('# older oracle')
        self.assertFalse(loop.same_evaluator(earlier))

    def test_capacity_wait_honours_reset_time_and_budget(self):
        now, mono = 1_000_000.0, 500.0
        with patch.object(loop.time, 'time', return_value=now), \
             patch.object(loop.time, 'monotonic', return_value=mono):
            self.assertEqual(loop.capacity_wait({'retry_after': now + 600}, 1, mono + 3600), 630)
            self.assertEqual(loop.capacity_wait({'retry_after': now - 5}, 1, mono + 3600), 30)
            self.assertIsNone(loop.capacity_wait({'retry_after': now + 7200}, 1, mono + 3600))
            self.assertEqual([loop.capacity_wait({}, r, mono + 3600) for r in (1, 2, 5, 9)], [60, 120, 900, 900])
            self.assertIsNone(loop.capacity_wait({}, 31, mono + 36000))

    def test_agent_environment_defaults_older_manifests_to_opencode(self):
        self.assertEqual(loop.agent_environment({}), {'PORTFORGE_WORKER_BACKEND': 'opencode',
                                                      'PORTFORGE_SUPERVISOR_BACKEND': 'opencode'})
        manifest = {'agent_backends': {'worker': 'claude', 'supervisor': 'opencode'},
                    'agent_models': {'worker': 'claude-opus-5-5', 'supervisor': 'zai-coding-plan/glm-5.3'}}
        self.assertEqual(loop.agent_environment(manifest), {
            'PORTFORGE_WORKER_BACKEND': 'claude', 'PORTFORGE_WORKER_MODEL': 'claude-opus-5-5',
            'PORTFORGE_SUPERVISOR_BACKEND': 'opencode', 'PORTFORGE_SUPERVISOR_MODEL': 'zai-coding-plan/glm-5.3'})

    def test_complete_references_requires_both_stages(self):
        root = self.make_previous_run()
        stage = root / 'references/key/parakeet'
        for name in ('smoke/ready.json', 'smoke/reference.tar', 'bringup/ready.json'):
            (stage / name).parent.mkdir(parents=True, exist_ok=True)
            (stage / name).write_text('x')
        self.assertFalse(loop.complete_references(root, 'parakeet'))
        (stage / 'bringup/reference.tar').write_text('x')
        self.assertTrue(loop.complete_references(root, 'parakeet'))

    def test_seed_job_uses_immutable_pass_instead_of_active_workspace(self):
        root = self.tools.run_dir / 'seed'
        write_json(root / 'manifest.json', {'seed_policy': 'fresh-public-parakeet', 'workspace': '/missing'})
        files = {'backend.py': '# fixed passing source', 'STATE.md': 'old pass'}
        write_json(root / 'jobs/passed/source.json', files)
        write_json(root / 'jobs/passed/request.json', {'kind': 'evaluate', 'stage': 'bringup',
                                                     'code_sha256': loop.code_hash(files)})
        write_json(root / 'jobs/passed/result.json', {'returncode': 0, 'verdict': {'passed': True}})
        receipt = loop.seed_workspace(self.tools.workspace, root, 'parakeet', 'passed')
        self.assertEqual(receipt['seed_job'], 'passed')
        self.assertEqual((self.tools.workspace / 'backend.py').read_text(), files['backend.py'])
        self.assertIn('no inherited pending jobs or passes', (self.tools.workspace / 'STATE.md').read_text())
        for changed in ({'returncode': 1, 'verdict': {'passed': True}},
                        {'returncode': 0, 'verdict': {'passed': False}},
                        {'returncode': 0, 'timeout': True, 'verdict': {'passed': True}}):
            write_json(root / 'jobs/passed/result.json', changed)
            with self.assertRaises(ValueError):
                loop.seed_workspace(self.tools.workspace, root, 'parakeet', 'passed')

    def test_same_model_seed_preserves_baseline_dependency_but_not_control_notes_or_passes(self):
        root = self.tools.run_dir / 'seed'
        source = root / 'workspace'
        source.mkdir(parents=True)
        write_json(root / 'manifest.json', {'seed_policy': 'fresh-public-parakeet',
                   'model': 'parakeet', 'workspace': str(source)})
        evidence = ('BASELINE_BF16_3_3B.md', 'BF16_BASELINE_RESULTS.md',
                    'CURRENT_BF16_BASELINE.md', 'CURRENT_BF16_SOURCE_1.md',
                    'CURRENT_BF16_IDENTITY.md', 'CANDIDATE_REVIEW.md', 'BFP8_RESULTS.md')
        controls = ('STATE.md', 'REVIEW_REQUIRED.md', 'HASH_SCOPES.md', 'TT_NOTES.md')
        for name in evidence:
            (source / name).write_text('```python\noriginal_baseline = True\n```\n')
        for name in controls:
            (source / name).write_text('old pass; stale user forbids optimization directive')
        for name in ('TASK.md', 'CONTRACT.md', 'TT_GUIDE.md'):
            (source / name).write_text('superseded instructions')
            (self.tools.workspace / name).write_text('current instructions')
        (source / 'backend.py').write_text('# preserved implementation')
        test = 'from pathlib import Path\nbaseline = Path(__file__).with_name("BASELINE_BF16_3_3B.md").read_text()\n'
        (source / 'test_cross_kv.py').write_text(test)
        write_json(root / 'jobs/old/result.json', {'verdict': {'passed': True}})
        receipt = loop.seed_workspace(self.tools.workspace, root, 'parakeet')
        for name in evidence:
            self.assertEqual((self.tools.workspace / name).read_text(), (source / name).read_text())
        namespace = {'__file__': str(self.tools.workspace / 'test_cross_kv.py')}
        exec(compile(test, namespace['__file__'], 'exec'), namespace)
        self.assertIn('original_baseline = True', namespace['baseline'])
        for name in controls[1:]:
            self.assertFalse((self.tools.workspace / name).exists())
        for name in ('TASK.md', 'CONTRACT.md', 'TT_GUIDE.md'):
            self.assertEqual((self.tools.workspace / name).read_text(), 'current instructions')
        state = (self.tools.workspace / 'STATE.md').read_text()
        for required in ('no inherited pending jobs or passes', 'historical under the archived runtime',
                         'revalidate the source and its same-precision baseline',
                         'advance optimization and standalone deliverables', 'Do not repeat resolved original-runtime'):
            self.assertIn(required, state)
        self.assertNotIn('forbids optimization', state)
        self.assertEqual(receipt['retained_seed_notes'], sorted(evidence))
        self.assertEqual(receipt['omitted_seed_notes'], sorted(controls))
        self.assertEqual(self.tools.status(), {})
        self.assertFalse((self.tools.run_dir / 'baselines').exists())
        self.assertFalse((self.tools.run_dir / 'submission.json').exists())
        with self.assertRaisesRegex(ValueError, 'needs an independent bringup pass'):
            self.tools.submit(model='parakeet')

    def test_cross_model_and_unknown_model_seed_still_drop_baseline_evidence(self):
        root = self.tools.run_dir / 'seed'
        files = {'backend.py': '# source', 'BASELINE_BF16.md': 'old baseline',
                 'BF16_BASELINE_X.md': 'old baseline', 'CURRENT_BF16_SOURCE_1.md': 'old source',
                 'CURRENT_BF16_IDENTITY.md': 'old runtime', 'CANDIDATE_X.md': 'old candidate',
                 'BFP8_X.md': 'old diagnostic', 'RESEARCH.md': 'public research', 'REPORT.md': 'old report'}
        write_json(root / 'manifest.json', {'seed_policy': 'fresh-public-parakeet', 'model': 'parakeet'})
        evidence = ('BASELINE_BF16.md', 'BF16_BASELINE_X.md', 'CURRENT_BF16_SOURCE_1.md',
                    'CURRENT_BF16_IDENTITY.md', 'CANDIDATE_X.md', 'BFP8_X.md')
        for artifact_model, expected in (('parakeet', sorted(list(evidence) + ['RESEARCH.md', 'REPORT.md'])),
                                         ('other-model', ['REPORT.md', 'RESEARCH.md'])):
            frozen = {'files': files, 'code_sha256': loop.code_hash(files), 'model': artifact_model}
            write_json(root / 'submission.json', frozen)
            for name in evidence:
                (self.tools.workspace / name).unlink(missing_ok=True)
            receipt = loop.seed_workspace(self.tools.workspace, root, 'parakeet')
            self.assertEqual(receipt['retained_seed_notes'], expected)
            for name in evidence:
                self.assertEqual((self.tools.workspace / name).exists(), artifact_model == 'parakeet')

    def test_seed_job_tampering_and_traversal_rejected(self):
        root = self.tools.run_dir / 'seed'
        write_json(root / 'manifest.json', {'seed_policy': 'fresh-public-parakeet'})
        write_json(root / 'jobs/pass/source.json', {'backend.py': '# altered'})
        write_json(root / 'jobs/pass/request.json', {'kind': 'evaluate', 'stage': 'bringup', 'code_sha256': 'wrong'})
        write_json(root / 'jobs/pass/result.json', {'returncode': 0, 'verdict': {'passed': True}})
        for job in ('pass', '../escape', '/absolute', '..'):
            with self.subTest(job=job), self.assertRaises(ValueError):
                loop.seed_workspace(self.tools.workspace, root, 'parakeet', job)

    def test_old_pending_reference_remains_visible_after_many_completed_jobs(self):
        run = self.tools.run_dir
        key = loop.digest(self.config)
        write_json(run / 'clusters' / (key + '.json'), self.config)
        old = run / 'jobs/old-reference/request.json'
        write_json(old, {'kind': 'reference-study', 'stage': 'precision-study', 'cluster_sha256': key})
        os.utime(old, (1, 1))
        for index in range(14):
            write_json(run / f'jobs/done-{index}/request.json', {'kind': 'run'})
            write_json(run / f'jobs/done-{index}/result.json', {'returncode': 0})
        with patch.object(loop, 'reference_progress', return_value={'phase': 'queued'}) as progress:
            results = self.tools.status()
        self.assertEqual(results['old-reference']['progress']['phase'], 'queued')
        self.assertEqual(len(results), 13)
        progress.assert_called_once()
        facts = loop.supervisor_facts(self.tools.workspace, run, self.config)
        self.assertTrue(any(j['id'] == 'old-reference' and j['pending'] for j in facts['recent_jobs']))

    def archive(self, stage, result=None, precision='bf16', config=None, model='parakeet'):
        job = self.tools.run_dir / 'jobs/one'
        write_json(job / 'request.json', {'stage': stage, 'code_sha256': loop.code_hash(loop.snapshot(self.tools.workspace)),
                                         'precision': precision, 'cluster_sha256': loop.digest(config or self.config), 'model': model,
                                         'suite_definition_sha256': loop.suite_identity(self.tools.run_dir, model, stage)})
        if result is not None:
            write_json(job / 'result.json', result)

    def test_submission_requires_passing_current_executable_source(self):
        self.delivery_receipt(job='one', model='parakeet')
        self.tools.write('STATE.md', 'Saved progress')
        self.assertEqual(self.tools.submit()['status'], 'frozen_for_full_evaluation')
        submitted = json.loads((self.tools.run_dir / 'submission.json').read_text())
        self.assertEqual(submitted['cluster_sha256'], loop.digest(self.config))
        self.assertEqual(submitted['precision'], 'bf16')
        with self.assertRaises(ValueError):
            self.tools.write('backend.py', '# changed')
        (self.tools.run_dir / 'submission.json').unlink()
        self.tools.write('backend.py', '# changed')
        with self.assertRaises(ValueError):
            self.tools.submit()

    def test_submission_rejects_contradictory_missing_or_mismatched_receipts(self):
        folder, request, result = self.delivery_receipt(model='parakeet')
        changes = [{'returncode': 124}, {'returncode': False}, {'returncode': None},
                   {'failure': 'infrastructure'}, {'timeout': True}, {'host_memory_limit': True},
                   {'job': 'another-job'}, {'cluster_sha256': 'another-runtime'},
                   {'model': 'other-model'}, {'precision': 'fp32'}, {'verdict': None},
                   {'verdict': {'passed': True}}]
        changes += [{'verdict': {**result['verdict'], **change}} for change in
                    ({'passed': 'true'}, {'passed': False}, {'failures': ['numerical']},
                     {'stage': 'smoke'}, {'model_key': 'other-model'}, {'suite_sha256': ''},
                     {'suite_sha256': None}, {'precision': {'requested': 'bfp8_b'}})]
        changes += [{key: None} for key in ('job', 'cluster_sha256', 'model', 'precision')]
        for change in changes:
            with self.subTest(change=change):
                write_json(folder / 'result.json', {**result, **change})
                with self.assertRaisesRegex(ValueError, 'independent bringup pass'):
                    self.tools.submit()
                self.assertFalse((self.tools.run_dir / 'submission.json').exists())
        write_json(folder / 'result.json', result)
        for change in ({'kind': 'run'}, {'kind': 'reference-study'}, {'code_sha256': None},
                       {'code_sha256': 'other-source'}, {'cluster_sha256': 'old-runtime'},
                       {'suite_definition_sha256': 'old-suite'}):
            with self.subTest(request=change):
                write_json(folder / 'request.json', {**request, **change})
                with self.assertRaises(ValueError):
                    self.tools.submit()
                self.assertFalse((self.tools.run_dir / 'submission.json').exists())

    def test_submission_latest_matching_receipt_cannot_fall_back_to_older_pass(self):
        first, _, _ = self.delivery_receipt(job='earlier', model='parakeet')
        later, request, result = self.delivery_receipt(job='later', model='parakeet')
        os.utime(first / 'request.json', (1, 1))
        os.utime(later / 'request.json', (2, 2))
        for receipt in ({**result, 'timeout': True}, {'verdict': {'passed': True}}, None):
            if receipt is None:
                (later / 'result.json').unlink()
            else:
                write_json(later / 'result.json', receipt)
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                self.tools.submit()
            self.assertFalse((self.tools.run_dir / 'submission.json').exists())
        # A genuinely different historical source does not invalidate this pass.
        write_json(later / 'request.json', {**request, 'code_sha256': 'other-source'})
        self.assertEqual(self.tools.submit()['status'], 'frozen_for_full_evaluation')

    def test_submission_accepts_identified_passes_without_changing_precision_interface(self):
        for precision in loop.PORT['precisions']:
            with self.subTest(precision=precision):
                self.delivery_receipt(model='parakeet', precision=precision)
                self.assertEqual(self.tools.submit(precision, 'parakeet')['status'],
                                 'frozen_for_full_evaluation')
                submission = self.tools.run_dir / 'submission.json'
                saved = json.loads(submission.read_text())
                self.assertEqual((saved['model'], saved['precision']), ('parakeet', precision))
                submission.unlink()

    def test_smoke_pass_cannot_authorize_full_submission(self):
        self.archive('smoke', {'verdict': {'passed': True}})
        with self.assertRaises(ValueError):
            self.tools.submit()
        with self.assertRaises(ValueError):
            self.tools.evaluate('full')

    def test_pending_or_passing_identical_stage_reuses_job(self):
        for result in (None, {'verdict': {'passed': True}}):
            self.archive('smoke', result)
            with patch.object(loop.experiment, 'chia_remote') as remote:
                self.assertEqual(self.tools.evaluate('smoke'), {'job': 'one', 'reused': True})
                remote.assert_not_called()

    def test_instruction_protection_handles_equivalent_paths(self):
        for path in ('CONTRACT.md', './CONTRACT.md', './TT_GUIDE.md', './TASK.md'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.tools.write(path, 'weaken contract')

    def test_snapshot_rejects_symlink_inputs(self):
        (self.tools.workspace / 'link.py').symlink_to(self.tools.workspace / 'backend.py')
        with self.assertRaises(ValueError):
            loop.snapshot(self.tools.workspace)

    def test_submission_rejects_changed_precision_or_endpoint(self):
        self.archive('bringup', {'verdict': {'passed': True}}, precision='fp32')
        with self.assertRaises(ValueError):
            self.tools.submit('bf16')
        replacement = {**self.config, 'tt': {'host': '192.0.2.3'}}
        write_json(self.tools.run_dir / 'cluster.json', replacement)
        with self.assertRaises(ValueError):
            self.tools.submit('fp32')

    def test_changed_precision_or_cluster_gets_new_job_with_pinned_config(self):
        self.archive('smoke', {'verdict': {'passed': True}})
        with patch.object(loop.experiment, 'chia_remote', return_value='future') as remote:
            result = self.tools.evaluate('smoke', 'fp32')
            self.assertNotEqual(result['job'], 'one')
            self.assertEqual(remote.call_args.args[-1], self.config)
            request = json.loads((self.tools.run_dir / 'jobs' / result['job'] / 'request.json').read_text())
            self.assertEqual(request['precision'], 'fp32')
            self.assertEqual(request['cluster_sha256'], loop.digest(self.config))
        replacement = {**self.config, 'tt': {'host': '192.0.2.3'}}
        write_json(self.tools.run_dir / 'cluster.json', replacement)
        with patch.object(loop.experiment, 'chia_remote', return_value='future') as remote:
            result = self.tools.evaluate('smoke')
            self.assertNotEqual(result['job'], 'one')
            self.assertEqual(remote.call_args.args[-1], replacement)

    def test_pending_status_uses_original_endpoint_after_config_change(self):
        self.archive('smoke')
        write_json(self.tools.run_dir / 'clusters' / (loop.digest(self.config) + '.json'), self.config)
        write_json(self.tools.run_dir / 'cluster.json', {'tt': {'host': 'replacement'}})
        with patch.object(loop, 'remote', return_value={'phase': 'generate'}) as remote:
            result = self.tools.status()
        self.assertEqual(remote.call_args.args[0], self.config)
        self.assertEqual(result['one']['progress']['phase'], 'generate')

    def test_refresh_unchanged_configuration_does_not_reprobe(self):
        with patch.object(loop, 'load_cluster', return_value=self.config), patch.object(loop, 'prepare') as prepare:
            config, status = loop.refresh_cluster(self.tools.run_dir, 'cluster.yaml', self.config)
        self.assertIs(config, self.config)
        self.assertEqual(status, 'unchanged')
        prepare.assert_not_called()

    def test_refresh_waits_for_pending_jobs_then_prepares_replacement(self):
        self.archive('smoke')
        replacement = {**self.config, 'reference': {'host': '192.0.2.8'}}
        with patch.object(loop, 'load_cluster', return_value=replacement), patch.object(loop, 'prepare') as prepare:
            self.assertEqual(loop.refresh_cluster(self.tools.run_dir, 'cluster.yaml', self.config),
                             (self.config, 'draining'))
            prepare.assert_not_called()
            write_json(self.tools.run_dir / 'jobs/one/result.json', {'returncode': 0})
            self.assertEqual(loop.refresh_cluster(self.tools.run_dir, 'cluster.yaml', self.config),
                             (replacement, 'changed'))
            prepare.assert_called_once_with(self.tools.run_dir, replacement)

    def test_failed_replacement_identity_or_health_keeps_active_cluster(self):
        run = self.tools.run_dir
        (run / 'harness').mkdir(exist_ok=True)
        for name in ('runner.py', 'evaluate.py'):
            (run / 'harness' / name).write_text('# frozen harness')
        replacement = {'tt': {'host': '192.0.2.3', 'work_dir': '/srv/tt', 'python': '/env/python',
                              'metal': '/metal', 'weights_by_model': {'parakeet': '/weights/model.bin'}, 'metal_commit': 'pinned'},
                       'reference': self.config['reference']}
        with patch.object(loop, 'upload'), patch.object(loop, 'ssh', return_value=b'wrong wrong'), \
             patch.object(loop, 'remote') as remote:
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                loop.prepare(run, replacement)
            remote.assert_not_called()
        unhealthy = {'cards': [{'pci': '0000:02:00.0', 'state': 'quarantined'}]}
        with patch.object(loop, 'upload'), \
             patch.object(loop, 'ssh', side_effect=[b'pinned', (loop.model_spec('parakeet')['files']['model.safetensors']['sha256'] + ' model.bin').encode()]), \
             patch.object(loop, 'remote', side_effect=[unhealthy, {'returncode': 1}, unhealthy]):
            with self.assertRaisesRegex(RuntimeError, 'no healthy TT cards'):
                loop.prepare(run, replacement)
        self.assertEqual(json.loads((run / 'cluster.json').read_text()), self.config)
        self.assertFalse((run / 'clusters' / (loop.digest(replacement) + '.json')).exists())

    def test_other_precision_pass_never_deduplicates_or_authorizes_submission(self):
        self.archive('bringup', {'verdict': {'passed': True}}, precision='bf16')
        with self.assertRaises(ValueError):
            self.tools.submit(precision='bfp8_b')
        with patch.object(loop.experiment, 'chia_remote', return_value='future') as remote:
            result = self.tools.evaluate('bringup', precision='bfp8_b')
        self.assertNotEqual(result['job'], 'one')
        self.assertEqual(remote.call_args.args[2]['precision'], 'bfp8_b')

    def test_background_reference_status_never_polls_tt(self):
        job = self.tools.run_dir / 'jobs/study'
        write_json(job / 'request.json', {'kind': 'reference-study', 'cluster_sha256': loop.digest(self.config)})
        write_json(self.tools.run_dir / 'clusters' / (loop.digest(self.config) + '.json'), self.config)
        with patch.object(loop, 'remote') as tt, patch.object(loop, 'reference_progress',
                return_value={'phase': 'generate', 'background_nonblocking': True}) as reference:
            status = self.tools.status()
        tt.assert_not_called()
        reference.assert_called_once_with(self.config, job)
        self.assertTrue(status['study']['progress']['background_nonblocking'])

    def test_supervisor_facts_report_real_source_and_exact_exposed_controls(self):
        self.delivery_receipt(job='one', stage='smoke', model='parakeet')
        facts = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        self.assertTrue(facts['workspace']['has_backend'])
        # Suite sizes come from the pack evaluator; this fixture ships no corpus.
        self.assertEqual(facts['controls']['fixed_suites']['details'], 'CONTRACT.md')
        self.assertEqual(facts['workspace']['current_source_passes'][0]['model'], 'parakeet')
        (self.tools.workspace / 'backend.py').write_text('# changed')
        self.assertEqual(loop.supervisor_facts(self.tools.workspace, self.tools.run_dir,
                                             self.config)['workspace']['current_source_passes'], [])

    def delivery_receipt(self, job='delivery', model='parakeet', precision='bf16', stage='bringup'):
        request = {'kind': 'evaluate', 'model': model, 'precision': precision, 'stage': stage,
                   'code_sha256': loop.code_hash(loop.snapshot(self.tools.workspace)),
                   'cluster_sha256': loop.digest(self.config),
                   'suite_definition_sha256': loop.suite_identity(self.tools.run_dir, model, stage)}
        result = {'job': job, 'returncode': 0, 'cluster_sha256': loop.digest(self.config),
                  'model': model, 'precision': precision,
                  'verdict': {'passed': True, 'stage': stage, 'model_key': model,
                              'precision': {'requested': precision}, 'suite_sha256': 'prepared-suite'}}
        folder = self.tools.run_dir / 'jobs' / job
        write_json(folder / 'request.json', request)
        write_json(folder / 'result.json', result)
        return folder, request, result

    def test_delivery_anchor_does_not_hide_other_models_or_precision_gaps(self):
        self.delivery_receipt(stage='smoke', job='anchor-smoke')
        self.delivery_receipt(job='anchor-bringup')
        (self.tools.workspace / 'STATE.md').write_text('All models and precisions passed!')
        facts = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config, 'parakeet')
        self.assertEqual(facts['selected_model'], 'parakeet')
        scope = facts['delivery_scope']
        self.assertEqual(scope['models'], ['parakeet'])
        self.assertEqual(scope['development_anchor'], facts['selected_model'])
        matrix = scope['short_gates']
        for stage, job in [('smoke', 'anchor-smoke'), ('bringup', 'anchor-bringup')]:
            self.assertEqual(matrix['parakeet']['bf16'][stage]['status'], 'passed')
            self.assertEqual(matrix['parakeet']['bf16'][stage]['job'], job)
        for stage in ('smoke', 'bringup'):
            self.assertEqual(matrix['parakeet']['bfp8_b'][stage]['status'], 'missing')
            self.assertEqual(matrix['parakeet']['bfp8_b'][stage]['job'], None)
        self.assertIn('not IEEE FP8', scope['precision_policy'])
        self.assertIn('ONE production-level reusable', facts['objective'])

    def test_delivery_never_inherits_source_cluster_suite_or_precision_passes(self):
        folder, original_request, original_result = self.delivery_receipt()
        cases = [({'code_sha256': 'old-source'}, {}, {}, 'historical'),
                 ({'cluster_sha256': 'old-runtime'}, {}, {}, 'historical'),
                 ({'suite_definition_sha256': 'old-suite'}, {}, {}, 'historical'),
                 ({}, {'job': 'wrong-job'}, {}, 'unverified'),
                 ({}, {'cluster_sha256': 'old-runtime'}, {}, 'unverified'),
                 ({}, {'model': 'other-model'}, {}, 'unverified'),
                 ({}, {'precision': 'bfp8_b'}, {}, 'unverified'),
                 ({}, {}, {'model_key': 'other-model'}, 'unverified'),
                 ({}, {}, {'stage': 'smoke'}, 'unverified'),
                 ({}, {}, {'precision': {'requested': 'bfp8_b'}}, 'unverified'),
                 ({}, {}, {'suite_sha256': ''}, 'unverified'),
                 ({}, {'returncode': 124}, {}, 'failed'),
                 ({}, {'failure': 'infrastructure'}, {}, 'failed'),
                 ({}, {'timeout': True}, {}, 'failed'),
                 ({}, {'host_memory_limit': True}, {}, 'failed'),
                 ({}, {}, {'passed': False}, 'failed')]
        for req, res, verdict, expected in cases:
            with self.subTest(req=req, res=res, verdict=verdict):
                write_json(folder / 'request.json', {**original_request, **req})
                write_json(folder / 'result.json', {**original_result, **res,
                    'verdict': {**original_result['verdict'], **verdict}})
                facts = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
                matrix = facts['delivery_scope']['short_gates']
                self.assertEqual(matrix['parakeet']['bf16']['bringup']['status'], expected)
                self.assertEqual(matrix['parakeet']['bfp8_b']['bringup']['status'], 'missing')
                self.assertEqual(facts['workspace']['current_source_passes'], [])

    def test_delivery_latest_current_failure_or_pending_supersedes_earlier_pass(self):
        first, _, _ = self.delivery_receipt(job='earlier')
        later, request, result = self.delivery_receipt(job='later')
        os.utime(first / 'request.json', (1, 1))
        os.utime(later / 'request.json', (2, 2))
        for status in ('failed', 'pending'):
            if status == 'failed':
                write_json(later / 'result.json', {**result, 'returncode': 1})
            else:
                (later / 'result.json').unlink()
            facts = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
            cell = facts['delivery_scope']['short_gates']['parakeet']['bf16']['bringup']
            self.assertEqual((cell['job'], cell['status']), ('later', status))
            self.assertEqual(facts['workspace']['current_source_passes'], [])
        write_json(later / 'request.json', {**request, 'code_sha256': 'historical'})
        cell = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir,
            self.config)['delivery_scope']['short_gates']['parakeet']['bf16']['bringup']
        self.assertEqual((cell['job'], cell['status']), ('earlier', 'passed'))
        self.assertEqual(cell['historical_jobs'], ['later'])

    def test_delivery_bfp8_evidence_is_separate_and_reference_study_is_not_a_tt_gate(self):
        self.delivery_receipt(job='tt-bfp8', model='parakeet', precision='bfp8_b')
        folder, request, _ = self.delivery_receipt(job='cuda-study', model='parakeet')
        write_json(folder / 'request.json', {**request, 'kind': 'reference-study'})
        matrix = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir,
                                      self.config)['delivery_scope']['short_gates']
        self.assertEqual(matrix['parakeet']['bfp8_b']['bringup']['status'], 'passed')
        self.assertEqual(matrix['parakeet']['bfp8_b']['bringup']['job'], 'tt-bfp8')
        self.assertEqual(matrix['parakeet']['bf16']['bringup']['status'], 'missing')
        self.assertEqual(matrix['parakeet']['bf16']['bringup']['status'], 'missing')

    def test_delivery_preserves_legacy_other_precision_pass_shape(self):
        self.archive('smoke', {'verdict': {'passed': True}}, precision='fp32')
        facts = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        self.assertEqual(facts['workspace']['current_source_passes'],
                         [{'stage': 'smoke', 'model': 'parakeet', 'precision': 'fp32'}])
        self.assertEqual(facts['delivery_scope']['short_gates']['parakeet']['bf16']['smoke']['status'], 'missing')

    def test_supervisor_distinguishes_backend_from_snapshot_hash(self):
        import hashlib
        self.archive('smoke', {'verdict': {'passed': True}})
        before = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)['workspace']
        expected = hashlib.sha256((self.tools.workspace / 'backend.py').read_bytes()).hexdigest()
        self.assertEqual(before['backend_sha256'], expected)
        self.assertNotEqual(before['backend_sha256'], before['source_sha256'])
        (self.tools.workspace / 'helper.py').write_text('# new inference dependency')
        after = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)['workspace']
        self.assertEqual(after['backend_sha256'], before['backend_sha256'])
        self.assertNotEqual(after['source_sha256'], before['source_sha256'])
        self.assertEqual(after['current_source_passes'], [])
        (self.tools.workspace / 'backend.py').unlink()
        missing = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)['workspace']
        self.assertIsNone(missing['backend_sha256'])

    def test_diagnostic_notes_do_not_hide_implementation_from_supervisor(self):
        for index in range(70):
            (self.tools.workspace / f'A_DIAGNOSTIC_{index:03}.md').write_text('historical note')
        (self.tools.workspace / 'forecast.py').write_text('# standalone entry point')
        facts = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        workspace = facts['workspace']
        self.assertIn('backend.py', workspace['file_names'])
        self.assertIn('forecast.py', workspace['implementation_files'])
        self.assertNotIn('A_DIAGNOSTIC_000.md', workspace['implementation_files'])
        self.assertEqual(workspace['implementation_file_count'], len(workspace['implementation_files']))
        self.assertLessEqual(len(workspace['file_names']), 64)
        self.assertIn('tt-metal-style package', facts['objective'])

    def test_supervisor_can_distinguish_historical_runtime_receipts(self):
        self.archive('bringup', {'verdict': {'passed': True}})
        replacement = json.loads(json.dumps(self.config))
        replacement['tt']['metal'] = '/srv/replacement-runtime'
        replacement['tt']['python'] = '/srv/tokenizer-wrapper/bin/python'
        facts = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, replacement)
        self.assertEqual(facts['environment']['tt_runtime'], '/srv/replacement-runtime')
        self.assertEqual(facts['environment']['tt_python'], '/srv/tokenizer-wrapper/bin/python')
        self.assertEqual(facts['environment']['cluster_sha256'], loop.digest(replacement))
        self.assertEqual(facts['recent_jobs'][0]['cluster_sha256'], loop.digest(self.config))
        self.assertEqual(facts['workspace']['current_source_passes'], [])

    def test_supervisor_handoff_survives_missing_worker_response_without_certifying_claims(self):
        path = self.tools.workspace / 'STATE.md'
        path.write_text('Pending job alpha. Next: collect its receipt; all tests reportedly pass.')
        facts = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        self.assertEqual(facts['worker_state']['text'], path.read_text())
        self.assertTrue(facts['worker_state']['present'])
        self.assertFalse(facts['worker_state']['truncated'])
        self.assertIn('not verified evidence', facts['worker_state']['note'])
        self.assertEqual(facts['workspace']['current_source_passes'], [])
        self.assertEqual(facts['recent_jobs'], [])
        path.write_text('Saved before timeout. Next: finish the checkpoint validation fix.')
        refreshed = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        self.assertEqual(refreshed['worker_state']['text'], path.read_text())

    def test_supervisor_handoff_bounds_large_notes_and_preserves_both_ends(self):
        text = 'CURRENT RESULT\n' + 'long historical detail\n' * 1000 + '\nNEXT: collect job beta'
        (self.tools.workspace / 'STATE.md').write_text(text)
        state = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)['worker_state']
        self.assertTrue(state['truncated'])
        self.assertEqual(state['available_chars'], len(text))
        self.assertLessEqual(len(state['text']), 12000)
        self.assertTrue(state['text'].startswith('CURRENT RESULT\n'))
        self.assertTrue(state['text'].endswith('\nNEXT: collect job beta'))
        self.assertIn('Middle omitted', state['text'])

    def test_supervisor_reports_missing_handoff_without_inventing_state(self):
        state = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)['worker_state']
        self.assertFalse(state['present'])
        self.assertEqual(state['text'], '')
        self.assertEqual(state['available_chars'], 0)
        self.assertFalse(state['truncated'])

    def test_frozen_environment_uses_archive_and_checks_docs_and_registry(self):
        run = self.tools.run_dir
        (run / 'harness/TASK.md').write_text('frozen instruction')
        manifest = {'harness_sha256': loop.harness_identity(run / 'harness'),
                    'repository_root': '/workspace/original-repository'}
        env = loop.frozen_environment(run, manifest)
        self.assertEqual(env['PYTHONPATH'], str(run.resolve() / 'harness'))
        self.assertEqual(env['PORTFORGE_REPOSITORY_ROOT'], '/workspace/original-repository')
        for name in ('TASK.md', 'models.json'):
            path = run / 'harness' / name
            original = path.read_text()
            path.write_text('changed')
            with self.assertRaises(ValueError):
                loop.frozen_environment(run, manifest)
            path.write_text(original)

    def test_campaign_default_model_reaches_run_and_evaluate(self):
        self.tools.model = 'parakeet'
        with patch.object(loop.experiment, 'chia_remote', return_value='future') as remote:
            self.tools.run('python probe.py')
            self.assertEqual(remote.call_args.args[2]['model'], 'parakeet')
            self.tools.evaluate()
            self.assertEqual(remote.call_args.args[2]['model'], 'parakeet')
            self.tools.evaluate(model='parakeet')
            self.assertEqual(remote.call_args.args[2]['model'], 'parakeet')

    def test_explicit_profile_model_selects_checkpoint_and_its_public_suite(self):
        self.tools.model = 'parakeet'
        with patch.object(loop.experiment, 'chia_remote', return_value='future') as remote:
            for model in ('parakeet', 'parakeet'):
                with self.subTest(model=model):
                    job = self.tools.run('python profile.py', public_input_stage='bringup', model=model)
                    request = remote.call_args.args[2]
                    saved = json.loads((self.tools.run_dir / 'jobs' / job['job'] / 'request.json').read_text())
                    self.assertEqual(request['model'], model)
                    self.assertEqual(saved['model'], model)
                    self.assertEqual(saved['suite_definition_sha256'],
                                     loop.suite_identity(self.tools.run_dir, model, 'bringup'))
                    self.assertNotIn('stage', request)
            self.tools.run('python profile.py')
            self.assertEqual(remote.call_args.args[2]['model'], 'parakeet')
        self.assertEqual(self.tools.model, 'parakeet')

    def test_invalid_profile_model_is_rejected_before_submission(self):
        with patch.object(loop.experiment, 'chia_remote') as remote:
            with self.assertRaisesRegex(ValueError, 'Unknown model'):
                self.tools.run('python profile.py', model='unregistered-model')
        remote.assert_not_called()
        self.assertFalse((self.tools.run_dir / 'jobs').exists())

    def test_profile_model_reaches_cached_inputs_and_remote_runner_without_assessment(self):
        request = {'kind': 'run', 'command': 'python profile.py',
                   'public_input_stage': 'smoke', 'model': 'parakeet'}
        with patch.object(loop, 'cached_reference', return_value={'tt_dir': '/prepared/parakeet/smoke'}) as cached, \
             patch.object(loop.reference, 'chia_remote') as gpu, \
             patch.object(loop, 'remote', return_value={'returncode': 0}) as tt, patch.object(loop, 'ssh') as ssh:
            result = loop.experiment._chia_original(str(self.tools.run_dir), 'profile-large', request, self.config)
        cached.assert_called_once_with(str(self.tools.run_dir), 'smoke', self.config, 'parakeet')
        self.assertEqual(tt.call_args.args[2]['model'], 'parakeet')
        self.assertEqual(tt.call_args.args[2]['public_input_stage'], 'smoke')
        self.assertEqual(result['model'], 'parakeet')
        self.assertNotIn('verdict', result)
        gpu.assert_not_called()
        ssh.assert_not_called()

    def test_supervisor_can_see_explicit_profile_models_and_anchor_default(self):
        facts = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config, 'parakeet')
        self.assertEqual(facts['controls']['run']['model'], ['parakeet'])
        self.assertEqual(facts['controls']['run']['default_model'], 'parakeet')

    def test_seed_preserves_code_but_replaces_instructions_and_does_not_copy_passes(self):
        root = self.tools.run_dir / 'seed'
        workspace = root / 'worker'
        workspace.mkdir(parents=True)
        (workspace / 'backend.py').write_text('# independently written previous candidate')
        (workspace / 'TASK.md').write_text('old target')
        (workspace / 'STATE.md').write_text('old job 123 claimed success; user forbids optimization')
        (workspace / 'TT_NOTES.md').write_text('user forbids optimization')
        write_json(root / 'manifest.json', {'workspace': str(workspace),
                   'seed_policy': 'fresh-public-parakeet'})
        write_json(root / 'jobs/123/result.json', {'verdict': {'passed': True}})
        (self.tools.workspace / 'TASK.md').write_text('trusted new target')
        provenance = loop.seed_workspace(self.tools.workspace, root, 'parakeet')
        self.assertEqual((self.tools.workspace / 'backend.py').read_text(),
                         '# independently written previous candidate')
        self.assertEqual((self.tools.workspace / 'TASK.md').read_text(), 'trusted new target')
        self.assertIn('historical', (self.tools.workspace / 'STATE.md').read_text())
        self.assertNotIn('forbids optimization', (self.tools.workspace / 'STATE.md').read_text())
        self.assertFalse((self.tools.workspace / 'TT_NOTES.md').exists())
        self.assertEqual(provenance['omitted_seed_notes'], ['STATE.md', 'TT_NOTES.md'])
        self.assertEqual(provenance['seed_run'], str(root.resolve()))
        self.assertFalse((self.tools.workspace / 'jobs').exists())
        with self.assertRaises(ValueError):
            self.tools.submit(model='parakeet')
        (workspace / 'escape.py').symlink_to('/tmp/outside')
        with self.assertRaises(ValueError):
            loop.seed_workspace(self.tools.workspace, root, 'parakeet')

    def test_seed_submission_hash_is_verified(self):
        root = self.tools.run_dir / 'seed'
        write_json(root / 'manifest.json', {'seed_policy': 'fresh-public-parakeet'})
        files = {'backend.py': '# frozen candidate'}
        write_json(root / 'submission.json', {'files': files, 'code_sha256': loop.code_hash(files)})
        loop.seed_workspace(self.tools.workspace, root, 'parakeet')
        self.assertEqual((self.tools.workspace / 'backend.py').read_text(), files['backend.py'])
        write_json(root / 'submission.json', {'files': files, 'code_sha256': 'tampered'})
        with self.assertRaises(ValueError):
            loop.seed_workspace(self.tools.workspace, root, 'parakeet')

    def test_baselines_are_separate_per_model_precision_and_actual_suite(self):
        for model, precision in (('parakeet', 'bf16'), ('parakeet', 'fp32'), ('parakeet', 'bfp8_b')):
            actual_suite = loop.digest({'model': model, 'precision': precision})
            self.archive('bringup', {'verdict': {'passed': True, 'suite_sha256': actual_suite}},
                         precision=precision, model=model)
            self.tools.status()
            baseline = self.tools.run_dir / 'baselines' / model / precision / loop.digest(self.config) / (actual_suite + '.json')
            saved = json.loads(baseline.read_text())
            self.assertEqual(saved['model'], model)
            self.assertEqual(saved['precision'], precision)
            self.assertEqual(saved['code_sha256'], loop.code_hash(loop.snapshot(self.tools.workspace)))
        self.assertEqual(len(list((self.tools.run_dir / 'baselines').rglob('*.json'))), 3)

    def test_reference_paths_and_cli_are_model_specific(self):
        run = self.tools.run_dir
        write_json(run / 'corpus.json', {})
        config = {'tt': {'work_dir': '/tt'}, 'reference': {'work_dir': '/gpu', 'python': '/env/python'}}
        for stage in ('smoke', 'bringup'):
            with patch.object(loop, 'upload'), patch.object(loop, 'reference_job') as job, \
                 patch.object(loop, 'ssh', return_value=b'archive'):
                result = loop.reference._chia_original(str(run), stage, config, 'parakeet')
            self.assertIn('/parakeet/' + stage, result['reference_dir'])
            self.assertTrue(result['tt_dir'].endswith('/parakeet/' + stage))
            self.assertIn('--model parakeet', job.call_args.args[2])

    def test_ready_deployed_reference_skips_gpu_resource_queue(self):
        run = self.tools.run_dir
        config = {'tt': {'work_dir': '/tt'}, 'reference': {'work_dir': '/gpu', 'python': '/env/python'}}
        key = loop.digest(config['reference'])[:16]
        cache = run / 'references' / key / 'parakeet/smoke'
        expected = {'reference_dir': '/gpu/' + run.name + '/references/' + key + '/parakeet/smoke',
                    'tt_dir': '/tt/' + run.name + '/references/parakeet/smoke', 'model': 'parakeet', 'stage': 'smoke',
                    'endpoint_sha256': loop.digest(config['reference'])}
        write_json(cache / 'ready.json', expected)
        write_json(cache / (loop.digest(config['tt']) + '.deployed.json'), {'tt_dir': expected['tt_dir']})
        (cache / 'reference.tar').write_bytes(b'archive')
        request = {'kind': 'evaluate', 'stage': 'smoke', 'model': 'parakeet'}
        with patch.object(loop.reference, 'chia_remote') as reference, \
             patch.object(loop, 'remote', return_value={'returncode': 1}):
            loop.experiment._chia_original(str(run), 'test', request, config)
            reference.assert_not_called()
        for altered in ({**expected, 'model': 'other-model'}, {**expected, 'stage': 'full'},
                        {**expected, 'endpoint_sha256': 'wrong'}):
            write_json(cache / 'ready.json', altered)
            self.assertIsNone(loop.cached_reference(run, 'smoke', config, 'parakeet'))
        write_json(cache / 'ready.json', expected)
        (cache / 'reference.tar').unlink()
        with patch.object(loop.reference, 'chia_remote', return_value='future') as reference, \
             patch.object(loop, 'get', return_value=expected), \
             patch.object(loop, 'remote', return_value={'returncode': 1}):
            loop.experiment._chia_original(str(run), 'miss', request, config)
            reference.assert_called_once_with(str(run), 'smoke', config, 'parakeet')

    def test_supervisor_tracks_distinct_candidate_outcomes_and_baseline_evidence(self):
        self.archive('bringup', {'verdict': {'passed': True, 'suite_sha256': 'suite'}})
        self.tools.status()
        old = json.loads((self.tools.run_dir / 'jobs/one/request.json').read_text())
        write_json(self.tools.run_dir / 'jobs/two/request.json', {**old, 'code_sha256': 'changed'})
        write_json(self.tools.run_dir / 'jobs/two/result.json', {'verdict': {'passed': False}})
        facts = loop.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        self.assertEqual(facts['evaluated_candidates'], [{'model': 'parakeet', 'precision': 'bf16',
                                                         'distinct_sources': 2, 'passing_sources': 1}])
        self.assertEqual(facts['baselines'][0]['code_sha256'], old['code_sha256'])
        self.assertIn('optimization', facts['objective'])

    def reference_fixture(self, stage='smoke'):
        import evaluate
        pinned = loop.model_spec('parakeet')
        names = [f"libri-{i:04d}" for i in range(15)]
        corpus = {'schema_version': 1, 'task': 'fixture', 'protocol': 'fixture', 'sources': [],
                  'selection': {'smoke': names[0],
                                'bringup': dict(batch0=names[1], batch1=names[4], long=names[-1],
                                                short=names[0], silence='syn-silence'),
                                'full': names[2:] + names[0:2]},
                  'cases': [dict(name=n, transcript=f"T{i}", audio_b64='AAAA', seconds=2.0, source='fixture')
                            for i, n in enumerate(names)]
                           + [dict(name='syn-silence', transcript='', audio_b64='AAAA', seconds=2.0, source='fixture'),
                              dict(name='syn-tone', transcript='', audio_b64='AAAA', seconds=3.0, source='fixture')]}
        with patch.object(evaluate, 'CORPUS_SHA256', loop.digest(corpus)):
            spec, _ = evaluate.make_suite(corpus, stage)
        identity = {k: spec[k] for k in ('model_key', 'model', 'revision', 'model_manifest_sha256')}
        receipt = {'model': 'parakeet', 'repo_id': pinned['repo_id'], 'revision': pinned['revision'],
                   'checkpoint_format': pinned['checkpoint_format'], 'verified_files': pinned['files'],
                   'model_manifest_sha256': spec['model_manifest_sha256'],
                   'suite_sha256': evaluate.suite_hash(spec)}
        meta = {'parameter_dtypes': ['torch.float32'], **identity,
                'precision': {'requested': 'fp32', 'effective': {'mode': 'fp32', 'weights': 'fp32', 'activations': 'fp32'}},
                'cases': [{'name': case['name']} for case in spec['cases']],
                'checkpoint_receipt': receipt, 'suite_sha256': evaluate.suite_hash(spec),
                'timing_protocol': evaluate.TIMING_PROTOCOL}
        entries = {'input/suite.json': json.dumps(spec), 'input/config.json': json.dumps(pinned['architecture']),
                   'input/checkpoint_receipt.json': json.dumps(receipt),
                   'input/inputs.npz': 'fixture-input', 'oracle/oracle.npz': 'fixture-oracle',
                   'oracle/reference.json': json.dumps(meta), 'oracle/labels.json': '{}'}
        return corpus, entries

    def reference_tar(self, path, entries, unsafe=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, 'w') as archive:
            for name, content in entries.items():
                payload = content.encode()
                entry = tarfile.TarInfo(name)
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))
            if unsafe:
                entry = tarfile.TarInfo(unsafe[0])
                entry.type = unsafe[1]
                entry.linkname = '/outside'
                archive.addfile(entry)
        return path

    def test_reference_archive_rejects_wrong_identity_precision_cases_and_unsafe_entries(self):
        import evaluate
        corpus, entries = self.reference_fixture()
        path = self.tools.run_dir / 'reference.tar'
        good = self.reference_tar(path, entries).read_bytes()
        with patch.object(evaluate, 'CORPUS_SHA256', loop.digest(corpus)):
            self.assertEqual(loop.validate_reference_archive(path, corpus, 'smoke', 'parakeet'), good)
            with self.assertRaises(ValueError):
                loop.validate_reference_archive(path, corpus, 'smoke', 'other-model')
            # evaluate.py records the dtype name without the torch prefix.
            plain = dict(entries)
            plain['oracle/reference.json'] = json.dumps(dict(json.loads(entries['oracle/reference.json']),
                                                             parameter_dtypes=['float32']))
            self.assertIsInstance(loop.validate_reference_archive(self.reference_tar(path, plain), corpus,
                                                                  'smoke', 'parakeet'), bytes)
            for name, field, value in [('input/suite.json', 'model', 'wrong'),
                                        ('oracle/reference.json', 'parameter_dtypes', ['bfloat16']),
                                        ('input/suite.json', 'revision', None),
                                        ('input/suite.json', 'weight_sha256', None),
                                        ('input/suite.json', 'corpus_sha256', 'wrong'),
                                        ('input/suite.json', 'gates', {}),
                                        ('input/suite.json', 'cases', []),
                                        ('oracle/reference.json', 'precision', {}),
                                        ('oracle/reference.json', 'cases', [])]:
                changed = dict(entries)
                metadata = json.loads(changed[name])
                metadata[field] = value
                changed[name] = json.dumps(metadata)
                self.reference_tar(path, changed)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    loop.validate_reference_archive(path, corpus, 'smoke', 'parakeet')
            for name, kind in (('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                               ('input/checkpoint_receipt.json', tarfile.SYMTYPE), ('input/checkpoint_receipt.json', tarfile.LNKTYPE)):
                self.reference_tar(path, entries, unsafe=(name, kind))
                with self.assertRaises(ValueError):
                    loop.validate_reference_archive(path, corpus, 'smoke', 'parakeet')
            partial = dict(entries)
            del partial['oracle/oracle.npz']
            self.reference_tar(path, partial)
            with self.assertRaises(ValueError):
                loop.validate_reference_archive(path, corpus, 'smoke', 'parakeet')

    def test_imported_reference_bytes_and_provenance_are_preserved_without_passes(self):
        import evaluate
        run, source = self.tools.run_dir, self.tools.run_dir / 'previous'
        corpus, _ = self.reference_fixture()
        config = {'tt': {'work_dir': '/tt'}, 'reference': {'work_dir': '/gpu'}}
        key = loop.digest(config['reference'])[:16]
        (source / 'harness').mkdir(parents=True)
        (source / 'harness/portforge_loop.py').write_text('# previous verified harness')
        old_hash = loop.digest({'portforge_loop.py': '# previous verified harness'})
        write_json(source / 'manifest.json', {'harness_sha256': old_hash, 'corpus_sha256': loop.digest(corpus)})
        write_json(run / 'corpus.json', corpus)
        originals = {}
        for stage in ('smoke', 'bringup'):
            _, entries = self.reference_fixture(stage)
            folder = source / 'references' / key / 'parakeet' / stage
            originals[stage] = self.reference_tar(folder / 'reference.tar', entries).read_bytes()
            write_json(folder / 'ready.json', {'endpoint_sha256': loop.digest(config['reference'])})
        with patch.object(evaluate, 'CORPUS_SHA256', loop.digest(corpus)), \
             patch.object(loop, 'ssh') as ssh, patch.object(loop, 'upload'):
            loop.import_references(run, source, config, 'parakeet')
            self.assertEqual(ssh.call_count, 4)
            for stage in originals:
                folder = run / 'references' / key / 'parakeet' / stage
                self.assertEqual((folder / 'reference.tar').read_bytes(), originals[stage])
                self.assertEqual(json.loads((folder / 'import.json').read_text())['source_harness_sha256'], old_hash)
                self.assertIsNotNone(loop.cached_reference(run, stage, config, 'parakeet'))
            self.assertFalse((run / 'baseline.json').exists())
            self.assertFalse((run / 'submission.json').exists())
            # A partially completed source fails before either remote is touched.
            (source / 'references' / key / 'parakeet' / 'bringup/ready.json').unlink()
            ssh.reset_mock()
            with self.assertRaises(ValueError):
                loop.import_references(run, source, config, 'parakeet')
            ssh.assert_not_called()
            write_json(source / 'references' / key / 'parakeet' / 'bringup/ready.json', {'endpoint_sha256': 'wrong'})
            with self.assertRaises(ValueError):
                loop.import_references(run, source, config, 'parakeet')
            ssh.assert_not_called()

    def test_profile_run_requests_public_inputs_without_evaluation_dedup(self):
        self.archive('bringup', {'verdict': {'passed': True}})
        with patch.object(loop.experiment, 'chia_remote', return_value='future') as remote:
            first = self.tools.run('python profile.py', public_input_stage='bringup')
            second = self.tools.run('python profile.py', public_input_stage='bringup')
        self.assertNotEqual(first['job'], second['job'])
        self.assertEqual(remote.call_count, 2)
        request = remote.call_args.args[2]
        self.assertEqual(request['kind'], 'run')
        self.assertEqual(request['public_input_stage'], 'bringup')
        self.assertNotIn('stage', request)
        saved = json.loads((self.tools.run_dir / 'jobs' / first['job'] / 'request.json').read_text())
        self.assertEqual(saved['code_sha256'], loop.code_hash(loop.snapshot(self.tools.workspace)))
        self.assertEqual(saved['suite_definition_sha256'], loop.suite_identity(self.tools.run_dir, 'parakeet', 'bringup'))
        with self.assertRaises(ValueError):
            self.tools.run('python profile.py', public_input_stage='full')

    def test_profile_run_resolves_cached_reference_without_assessment_or_gpu_queue(self):
        request = {'kind': 'run', 'command': 'python profile.py', 'public_input_stage': 'smoke'}
        with patch.object(loop, 'cached_reference', return_value={'tt_dir': '/prepared'}) as cached, \
             patch.object(loop.reference, 'chia_remote') as gpu, \
             patch.object(loop, 'remote', return_value={'returncode': 0}) as tt, patch.object(loop, 'ssh') as ssh:
            result = loop.experiment._chia_original(str(self.tools.run_dir), 'profile', request, self.config)
        cached.assert_called_once_with(str(self.tools.run_dir), 'smoke', self.config, 'parakeet')
        gpu.assert_not_called()
        ssh.assert_not_called()
        self.assertNotIn('verdict', result)
        self.assertEqual(tt.call_args.args[2]['public_input_stage'], 'smoke')
        with patch.object(loop, 'cached_reference', return_value=None), \
             patch.object(loop.reference, 'chia_remote', return_value='future') as gpu, \
             patch.object(loop, 'get', return_value={'tt_dir': '/prepared'}), \
             patch.object(loop, 'remote', return_value={'returncode': 0}):
            loop.experiment._chia_original(str(self.tools.run_dir), 'uncached', request, self.config)
        gpu.assert_called_once_with(str(self.tools.run_dir), 'smoke', self.config, 'parakeet')

    def test_full_transport_outlives_configured_job_without_extending_other_calls(self):
        config = json.loads(json.dumps(self.config))
        config['tt']['full_timeout_seconds'] = 64800
        for kind, stage, expected in (('evaluate', 'full', 65200), ('evaluate', 'smoke', 22000),
                                      ('run', None, 22000)):
            request = {'kind': kind, 'stage': stage, 'full_timeout_seconds': 999999}
            with patch.object(loop, 'cached_reference', return_value={'tt_dir': '/prepared'}), \
                 patch.object(loop, 'remote', return_value={'returncode': 1}) as remote:
                loop.experiment._chia_original(str(self.tools.run_dir), kind + str(stage), request, config)
            self.assertEqual(remote.call_args.kwargs['timeout'], expected)
        config['tt']['full_timeout_seconds'] = 86401
        with patch.object(loop, 'cached_reference', return_value={'tt_dir': '/prepared'}), \
             patch.object(loop, 'remote') as remote:
            result = loop.experiment._chia_original(str(self.tools.run_dir), 'invalid-deadline',
                                                   {'kind': 'evaluate', 'stage': 'full'}, config)
        remote.assert_not_called()
        self.assertEqual(result['failure'], 'infrastructure')


if __name__ == '__main__':
    unittest.main()
