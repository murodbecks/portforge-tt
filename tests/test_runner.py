"""Phase deadlines and quarantine classification without accelerator access."""
import support  # noqa: F401  (pack and registry setup)
import json
from pathlib import Path
import tempfile
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import runner


class RunnerTests(unittest.TestCase):
    def test_log_pages_are_bounded_and_require_no_devices_or_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / 'jobs' / 'own-job'
            job.mkdir(parents=True)
            (job / 'stdout.log').write_bytes(b'RESULT\n' + b'x' * 70000)
            (job / 'stderr.log').write_bytes('é'.encode())
            with patch.object(runner, 'configure', side_effect=AssertionError('no runtime setup')), \
                 patch.object(runner, 'cards', side_effect=AssertionError('no device work')):
                first = runner.main(root, {'kind': 'log', 'job': 'own-job'})
                self.assertEqual(first['next_offset_bytes'], 65536)
                self.assertTrue(first['text'].startswith('RESULT\n'))
                self.assertFalse(first['eof'])
                second = runner.main(root, {'kind': 'log', 'job': 'own-job', 'offset_bytes': 65536})
                self.assertEqual(first['text'] + second['text'], (job / 'stdout.log').read_text())
                self.assertEqual(second['next_offset_bytes'], second['size_bytes'])
                self.assertTrue(second['eof'])
                for offset in (second['size_bytes'], second['size_bytes'] + 10):
                    end = runner.main(root, {'kind': 'log', 'job': 'own-job', 'offset_bytes': offset})
                    self.assertEqual(end['text'], '')
                    self.assertEqual(end['next_offset_bytes'], offset)
                    self.assertTrue(end['eof'])
                split = runner.main(root, {'kind': 'log', 'job': 'own-job', 'stream': 'stderr', 'limit_bytes': 1})
                self.assertEqual(split['text'], '\ufffd')
                self.assertEqual(split['next_offset_bytes'], 1)

    def test_log_rejects_invalid_paths_streams_and_bounds(self):
        base = {'kind': 'log', 'job': 'own-job'}
        invalid = [('job', '../oracle'), ('job', '/absolute'), ('job', ''),
                   ('stream', 'oracle'), ('stream', '../stdout'), ('offset_bytes', -1),
                   ('offset_bytes', True), ('offset_bytes', 1.5), ('offset_bytes', 2**63),
                   ('limit_bytes', 0), ('limit_bytes', 65537), ('limit_bytes', False)]
        with tempfile.TemporaryDirectory() as directory:
            for key, value in invalid:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    runner.main(directory, {**base, key: value})
            with self.assertRaises(FileNotFoundError):
                runner.main(directory, base)

    def test_log_rejects_symlink_components_and_nonregular_files(self):
        for component in ('jobs', 'job', 'stdout.log'):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                actual = root / 'other'
                actual.mkdir()
                (actual / 'stdout.log').write_text('not authorized')
                if component == 'jobs':
                    (root / 'jobs').symlink_to(actual, target_is_directory=True)
                elif component == 'job':
                    (root / 'jobs').mkdir()
                    (root / 'jobs/job').symlink_to(actual, target_is_directory=True)
                else:
                    (root / 'jobs/job').mkdir(parents=True)
                    (root / 'jobs/job/stdout.log').symlink_to(actual / 'stdout.log')
                with self.assertRaises(OSError):
                    runner.main(root, {'kind': 'log', 'job': 'job'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'jobs/job/stdout.log').mkdir(parents=True)
            with self.assertRaises(ValueError):
                runner.main(root, {'kind': 'log', 'job': 'job'})

    def setUp(self):
        names = ('METAL', 'PYTHON', 'HEALTH', 'LEASES', 'MOUNTS', 'CARD_ALLOWLIST', 'WEIGHTS_BY_MODEL', 'HOST_MEMORY_LIMIT_BYTES')
        original = {name: getattr(runner, name) for name in names}
        self.addCleanup(lambda: [setattr(runner, name, value) for name, value in original.items()])

    def config(self, root):
        return {'metal': '/deployment/tt-metal', 'python': '/deployment/python/bin/python',
                'weights_by_model': {'parakeet': '/models/parakeet/model.safetensors'}, 'health_dir': str(root / 'health'),
                'lease_dir': str(root / 'shared-leases'), 'runtime_mounts': ['/deployment/python'],
                'cards': []}

    def test_runtime_configuration_has_no_host_or_user_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            runner.configure(config)
            args = runner.sandbox(root, root / 'job', '0000:02:00.0', '7', 'parakeet')
            self.assertEqual(runner.WEIGHTS_BY_MODEL['parakeet'], Path('/models/parakeet/model.safetensors'))
            index = args.index('/deployment/python')
            self.assertEqual(args[index - 1], '--ro-bind')
            self.assertNotIn('/models/parakeet', args)
            self.assertIn('/models/parakeet/model.safetensors', args)
            self.assertIn('/models.py', args)
            self.assertIn('/models.json', args)

    def test_sandbox_hides_worktree_git_pointer_as_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metal = root / 'metal'
            metal.mkdir()
            (metal / '.git').write_text('gitdir: /private/repository/worktrees/metal\n')
            config = self.config(root)
            config['metal'] = str(metal)
            runner.configure(config)
            args = runner.sandbox(root, root / 'job', '0000:02:00.0', '7', 'parakeet')
            for alias in (str(metal), '/opt/tt-metal'):
                index = args.index(alias + '/.git')
                self.assertEqual(args[index - 2:index], ['--ro-bind', '/dev/null'])
            self.assertNotIn('/private/repository/worktrees/metal', args)
            self.assertFalse(any(part.startswith(('/home/', '/Users/')) for part in args))
            self.assertNotIn('miniconda', ' '.join(args))
            for key, value in (('metal', 'relative/path'), ('weights_by_model', {'parakeet': '/a/../b'}),
                               ('runtime_mounts', ['relative']), ('cards', ['not-pci'])):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.configure({**config, key: value})

    def test_sysfs_discovery_and_optional_pci_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.configure(self.config(root))
            devices, sysfs = root / 'devices', root / 'sysfs'
            devices.mkdir()
            for index, pci in enumerate(('0000:02:00.0', '0000:03:00.0')):
                (devices / str(index)).touch()
                node = sysfs / ('tenstorrent!' + str(index))
                node.mkdir(parents=True)
                (node / 'device').symlink_to(root / 'pci' / pci)
            self.assertEqual(runner.cards(devices, sysfs), {'0000:02:00.0': '0', '0000:03:00.0': '1'})
            runner.configure({**self.config(root), 'cards': ['0000:03:00.0']})
            self.assertEqual(runner.cards(devices, sysfs), {'0000:03:00.0': '1'})

    def test_sharded_mount_exposes_only_selected_manifest_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'weights-parakeet'
            checkpoint.mkdir()
            spec = runner.model_spec('parakeet')
            for filename in spec['files']:
                (checkpoint / filename).write_text('test bytes')
            cached = root / 'official-cache' / 'blob'
            cached.parent.mkdir()
            cached.write_text('published weights')
            (checkpoint / spec['weight_files'][0]).unlink()
            (checkpoint / spec['weight_files'][0]).symlink_to(cached)
            (checkpoint / 'earlier-private-backend.py').write_text('private')
            runner.configure({**self.config(root), 'weights_by_model': {'parakeet': str(checkpoint)}})
            args = runner.sandbox(root, root / 'job', '0000:02:00.0', '7', 'parakeet')
            self.assertEqual(runner.checkpoint('parakeet'), (checkpoint, '/weights'))
            self.assertNotIn(str(checkpoint), args)
            self.assertNotIn(str(checkpoint / 'earlier-private-backend.py'), args)
            self.assertNotIn('/models/parakeet/model.safetensors', args)
            self.assertNotIn(str(cached.parent), args)
            for filename in spec['files']:
                self.assertIn(str((checkpoint / filename).resolve()), args)

    def test_model_mapping_rejects_unknown_models_and_unconfigured_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                runner.configure({**self.config(root), 'weights_by_model': {'unknown-model': '/weights/checkpoint.bin'}})
            runner.configure(self.config(root))
            with self.assertRaises(ValueError):
                runner.checkpoint('unknown-model')
            self.assertEqual(runner.checkpoint('parakeet')[1], '/weights/model.safetensors')

    def test_evaluation_uses_model_specific_inputs_and_excludes_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'harness').mkdir()
            config = {**self.config(root), 'weights_by_model': {'parakeet': '/models/large/model.safetensors'}}
            (root / 'harness/runtime.json').write_text(json.dumps(config))
            inputs = root / 'references/parakeet/smoke/input'
            inputs.mkdir(parents=True)
            (inputs / 'suite.json').write_text(json.dumps({'model_key': 'parakeet'}))
            (inputs / 'unexpected-oracle.json').write_text('must not leak')
            captured = []
            def execute(args, job, *unused):
                captured.append(args)
                for filename in ('stdout.log', 'stderr.log'):
                    (job / filename).write_text('')
                return {'returncode': 0, 'timeout': False}
            with patch.object(runner, 'cards', return_value={'0000:02:00.0': '7'}), \
                 patch.object(runner, 'health', return_value={'state': 'healthy'}), \
                 patch.object(runner, 'execute', side_effect=execute):
                for kind in ('evaluate', 'run'):
                    with self.subTest(kind=kind):
                        (inputs / 'suite.json').write_text(json.dumps({'model_key': 'parakeet'}))
                        request = {'kind': kind, 'job': 'larger-' + kind, 'model': 'parakeet'}
                        request.update({'stage': 'smoke'} if kind == 'evaluate' else
                                       {'public_input_stage': 'smoke', 'command': 'python profile.py'})
                        result = runner.main(root, request)
                        self.assertEqual(result['model'], 'parakeet')
                        self.assertIn('/models/large/model.safetensors', captured[-1])
                        self.assertNotIn('/models/parakeet/model.safetensors', captured[-1])
                        self.assertEqual({p.name for p in (root / 'jobs' / request['job'] / 'input').iterdir()},
                                         {'suite.json'})
                        (inputs / 'suite.json').write_text(json.dumps({'model_key': 'other'}))
                        with self.assertRaises(ValueError):
                            runner.main(root, {**request, 'job': 'wrong-model-' + kind})

    def test_candidate_command_passes_validated_precision_and_weight_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'harness').mkdir()
            (root / 'harness/runtime.json').write_text(json.dumps(self.config(root)))
            (root / 'references/parakeet/smoke/input').mkdir(parents=True)
            captured = []
            def execute(args, job, *unused):
                captured.append(args)
                for name in ('stdout.log', 'stderr.log'):
                    (job / name).write_text('')
                return {'returncode': 0, 'timeout': False}
            with patch.object(runner, 'cards', return_value={'0000:02:00.0': '7'}), \
                 patch.object(runner, 'health', return_value={'state': 'healthy'}), \
                 patch.object(runner, 'execute', side_effect=execute):
                for precision in (None, 'fp32', 'fp16', 'bfp8_b'):
                    request = {'model': 'parakeet', 'kind': 'evaluate', 'stage': 'smoke', 'job': 'job-' + str(precision)}
                    if precision:
                        request['precision'] = precision
                    runner.main(root, request)
                    self.assertEqual(captured[-1][-4:], ['--weights', '/weights/model.safetensors',
                                                       '--precision', precision or 'bf16'])
                with self.assertRaises(ValueError):
                    runner.main(root, {'model': 'parakeet', 'kind': 'evaluate', 'stage': 'smoke', 'job': 'invalid', 'precision': 'fake'})

    def run_phase(self, phase, stage, terminal=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'output').mkdir()
            (root / 'output/progress.json').write_text(json.dumps({'sequence': 1, 'phase': phase,
                                                                 'timeout_seconds': 999999}))
            clock = [0]
            process = SimpleNamespace(pid=999999, returncode=terminal)
            process.poll = lambda: process.returncode
            def wait(timeout):
                process.returncode = -15
            process.wait = wait
            def sleep(seconds):
                clock[0] += seconds
            with patch.object(runner.subprocess, 'Popen', return_value=process), \
                 patch.object(runner.time, 'monotonic', side_effect=lambda: clock[0]), \
                 patch.object(runner.time, 'sleep', side_effect=sleep), \
                 patch.object(runner.os, 'killpg'):
                return runner.execute(['unused'], root, 1000, True, stage)

    def test_import_runtime_receives_initialization_budget(self):
        for phase in ('import_runtime', 'verify_checkpoint'):
            result = self.run_phase(phase, 'smoke')
            self.assertEqual(result['seconds'], 601)
            self.assertTrue(result['timeout'])
            self.assertEqual(result['returncode'], 124)

    def test_nondefault_model_preflight_and_hashing_timeout_do_not_require_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'harness').mkdir()
            config = self.config(root)
            config['weights_by_model'] = {'parakeet': '/models/large/model.safetensors'}
            (root / 'harness/runtime.json').write_text(json.dumps(config))
            (root / 'references/parakeet/smoke/input').mkdir(parents=True)
            current = {'returncode': 0, 'timeout': False}
            def execute(args, job, *unused):
                self.assertIn('/models/large/model.safetensors', args)
                for filename in ('stdout.log', 'stderr.log'):
                    (job / filename).write_text('')
                return dict(current)
            with patch.object(runner, 'cards', return_value={'0000:02:00.0': '7'}), \
                 patch.object(runner, 'health', return_value={'state': 'healthy'}), \
                 patch.object(runner, 'mark') as mark, patch.object(runner, 'execute', side_effect=execute):
                result = runner.main(root, {'kind': 'preflight', 'pci': '0000:02:00.0', 'job': 'preflight', 'model': 'parakeet'})
                self.assertEqual(result['returncode'], 0)
                mark.assert_called_once_with('0000:02:00.0', 'healthy', 'preflight')
                mark.reset_mock()
                current = {'returncode': 124, 'timeout': True, 'progress': {'phase': 'verify_checkpoint'}}
                result = runner.main(root, {'kind': 'evaluate', 'stage': 'smoke', 'job': 'slow-hash', 'model': 'parakeet'})
                self.assertEqual(result['failure'], 'timeout')
                mark.assert_not_called()

    def test_full_generation_uses_trusted_stage_budget_not_candidate_deadline(self):
        self.assertEqual(self.run_phase('generate', 'smoke')['seconds'], 181)
        self.assertEqual(self.run_phase('generate', 'full')['seconds'], 301)

    def test_external_timeout_is_reported_as_timeout(self):
        result = self.run_phase('generate', 'full', terminal=124)
        self.assertTrue(result['timeout'])
        self.assertEqual(runner.failure_kind(result['returncode'], '', result['timeout']), 'timeout')

    def test_candidate_errors_do_not_automatically_quarantine_device(self):
        self.assertEqual(runner.failure_kind(1, 'ValueError: incompatible shape'), 'candidate_error')
        self.assertEqual(runner.failure_kind(1, 'Read 0xffffffff over PCIe'), 'device_initialization')
        self.assertIsNone(runner.failure_kind(0, 'firmware initialized'))

    def test_normal_device_startup_does_not_reclassify_python_exception(self):
        startup = 'UMD: Established firmware bundle version: 19.4.2\n{"phase":"open_device"}\n'
        for exception in ('IsADirectoryError: /weights', 'ValueError: invalid token ids',
                          'FileNotFoundError: checkpoint missing'):
            text = startup + 'Traceback (most recent call last):\n  backend.create_backend()\n' + exception
            self.assertEqual(runner.failure_kind(1, text), 'candidate_error')
        self.assertEqual(runner.failure_kind(1, startup + 'ValueError: invalid shape'), 'candidate_error')

    def test_real_device_faults_and_timeouts_still_require_recovery(self):
        for text in ('Traceback (most recent call last):\n  ttnn.open_device()\nRuntimeError: initialization failed',
                     'firmware initialization failed', 'Physical core 1 timed out', 'Read 0xffffffff over PCIe'):
            self.assertEqual(runner.failure_kind(1, text), 'device_initialization')
        self.assertEqual(runner.failure_kind(124, 'firmware initialized'), 'timeout')

    @staticmethod
    def proc_row(start=1, parent=1, group=10, rss=0, state='S'):
        return dict(starttime=start, ppid=parent, pgrp=group, rss=rss, state=state)

    @staticmethod
    def stat_line(pid, parent=1, group=10, start=1, rss=2):
        fields = ['S', str(parent), str(group)] + ['0'] * 21
        fields[19], fields[21] = str(start), str(rss)
        return f'{pid} (name with ) spaces) ' + ' '.join(fields)

    def test_proc_rss_parses_comm_parentheses_and_excludes_disappeared_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '123').mkdir()
            (root / '123/stat').write_text(self.stat_line(123, parent=10, group=123, start=987, rss=7))
            (root / '124').mkdir()  # Exited before stat could be read.
            (root / 'self').mkdir()
            with patch.object(runner.os, 'sysconf', return_value=4096):
                self.assertEqual(runner.proc_snapshot(root),
                                 {123: self.proc_row(start=987, parent=10, group=123, rss=7 * 4096)})
            self.assertIsNone(runner.proc_snapshot(root / 'absent'))

    def test_owned_tree_tracks_detached_and_reparented_children_not_reused_pids(self):
        known = {10: 100}
        rows = {10: self.proc_row(start=100),
                11: self.proc_row(start=101, parent=10, group=11, rss=20),
                12: self.proc_row(start=102, parent=11, group=11, rss=30),
                13: self.proc_row(start=103, parent=1, group=10, rss=40),
                90: self.proc_row(start=900, parent=1, group=90, rss=99999)}
        self.assertEqual(set(runner.owned_processes(rows, 10, known)), {10, 11, 12, 13})
        later = {11: self.proc_row(start=101, parent=1, group=11, rss=20),
                 12: self.proc_row(start=999, parent=1, group=12, rss=99999),
                 14: self.proc_row(start=104, parent=11, group=11, rss=30),
                 90: rows[90]}
        self.assertEqual(set(runner.owned_processes(later, 10, known)), {11, 14})
        # A reused leader cannot re-authorize its new process group.
        self.assertEqual(runner.owned_processes({10: self.proc_row(start=9999)}, 10, known), {})

    def test_memory_budget_is_trusted_bounded_runtime_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            runner.configure(config)
            self.assertEqual(runner.HOST_MEMORY_LIMIT_BYTES, 64 * 1024**3)
            runner.configure({**config, 'host_memory_limit_bytes': 32 * 1024**3})
            self.assertEqual(runner.HOST_MEMORY_LIMIT_BYTES, 32 * 1024**3)
            for limit in (0, -1, True, '64000000000', 129 * 1024**3):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    runner.configure({**config, 'host_memory_limit_bytes': limit})

    def test_full_job_deadline_is_bounded_trusted_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            runner.configure(config)
            self.assertEqual(runner.FULL_TIMEOUT_SECONDS, 21600)
            runner.configure({**config, 'full_timeout_seconds': 64800})
            self.assertEqual(runner.FULL_TIMEOUT_SECONDS, 64800)
            for value in (0, 3599, 86401, True, 64800.0, '64800'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    runner.configure({**config, 'full_timeout_seconds': value})
            runner.configure(config)
            self.assertEqual(runner.FULL_TIMEOUT_SECONDS, 21600)

    def test_full_deadline_changes_only_full_evaluation_and_ignores_candidate_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'harness').mkdir()
            config = {**self.config(root), 'full_timeout_seconds': 64800}
            (root / 'harness/runtime.json').write_text(json.dumps(config))
            for stage in ('smoke', 'bringup', 'full'):
                (root / 'references' / 'parakeet' / stage / 'input').mkdir(parents=True)
            captured = []
            def execute(args, job, total, monitor, stage='smoke'):
                captured.append((int(args[2]), total, monitor, stage))
                for name in ('stdout.log', 'stderr.log'):
                    (job / name).write_text('')
                return {'returncode': 0, 'timeout': False}
            with patch.object(runner, 'cards', return_value={'0000:02:00.0': '7'}), \
                 patch.object(runner, 'health', return_value={'state': 'healthy'}), \
                 patch.object(runner, 'execute', side_effect=execute):
                for stage, deadline in (('smoke', 900), ('bringup', 1200), ('full', 64800)):
                    runner.main(root, {'model': 'parakeet', 'kind': 'evaluate', 'stage': stage, 'job': stage,
                                      'timeout': 999999, 'full_timeout_seconds': 999999})
                    self.assertEqual(captured[-1], (deadline, deadline + 2, True, stage))
                runner.main(root, {'model': 'parakeet', 'kind': 'run', 'job': 'diagnostic', 'command': 'true', 'timeout': 999999})
                self.assertEqual(captured[-1][0], 300)

    def test_memory_crossing_stops_owned_tree_and_retains_peak_without_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = SimpleNamespace(pid=10, returncode=None)
            process.poll = lambda: process.returncode
            low = {10: self.proc_row(start=100, rss=40)}
            exact = {**low, 11: self.proc_row(start=101, parent=10, group=11, rss=60)}
            high = {**low, 11: self.proc_row(start=101, parent=10, group=11, rss=61),
                    90: self.proc_row(start=900, group=90, rss=999999)}
            def stop(child, known):
                self.assertIs(child, process)
                self.assertEqual(known, {10: 100, 11: 101})
                process.returncode = -9
            with patch.object(runner.subprocess, 'Popen', return_value=process), \
                 patch.object(runner, 'proc_snapshot', side_effect=[low, exact, high]), \
                 patch.object(runner, 'HOST_MEMORY_LIMIT_BYTES', 100), \
                 patch.object(runner, 'terminate_job', side_effect=stop) as terminate, \
                 patch.object(runner.time, 'sleep') as sleep:
                result = runner.execute(['unused'], root, 60, False)
            terminate.assert_called_once()
            sleep.assert_called_once_with(1)  # Equality does not exceed the budget.
            self.assertEqual(result['peak_host_rss_bytes'], 101)
            self.assertEqual(result['host_memory_limit_bytes'], 100)
            self.assertTrue(result['host_memory_limit'])
            self.assertFalse(result['timeout'])
            self.assertEqual(result['returncode'], 125)
            self.assertEqual(runner.failure_kind(-9, 'firmware error', True, True), 'host_memory_limit')

    def test_cleanup_kills_verified_detached_survivor_after_leader_exits(self):
        process = SimpleNamespace(pid=10, returncode=None)
        process.poll = lambda: process.returncode
        def wait(timeout):
            process.returncode = -15
        process.wait = wait
        row = self.proc_row(start=101, parent=1, group=11, rss=20)
        snapshot = {10: self.proc_row(start=100), 11: row,
                    12: self.proc_row(start=999, group=12, rss=50)}
        with patch.object(runner, 'proc_snapshot', return_value=snapshot), \
             patch.object(Path, 'read_text', side_effect=lambda: self.stat_line(11, start=101)), \
             patch.object(runner.os, 'killpg') as group_signal, \
             patch.object(runner.os, 'kill') as pid_signal:
            runner.terminate_job(process, {10: 100, 11: 101, 12: 102})
        group_signal.assert_called_once_with(10, runner.signal.SIGTERM)
        self.assertEqual([c.args for c in pid_signal.call_args_list],
                         [(11, runner.signal.SIGTERM), (11, runner.signal.SIGKILL)])

    def test_unreadable_proc_does_not_prevent_group_cleanup(self):
        process = SimpleNamespace(pid=10, returncode=None)
        process.poll = lambda: process.returncode
        process.wait = lambda timeout: setattr(process, 'returncode', -15)
        with patch.object(runner, 'proc_snapshot', side_effect=PermissionError('proc unreadable')), \
             patch.object(runner.os, 'killpg') as signal_group:
            runner.terminate_job(process, {})
        signal_group.assert_called_once_with(10, runner.signal.SIGTERM)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'real /proc RSS accounting requires Linux')
    def test_linux_memory_watchdog_stops_detached_allocating_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'detached-child'
            child = ('import os,time; from pathlib import Path; '
                     f'Path({str(marker)!r}).write_text(str(os.getpid())); '
                     'allocation=bytearray(96*1024**2); time.sleep(60)')
            parent = ('import subprocess,sys,time; '
                      f'subprocess.Popen([sys.executable,"-c",{child!r}],start_new_session=True); '
                      'time.sleep(60)')
            with patch.object(runner, 'HOST_MEMORY_LIMIT_BYTES', 64 * 1024**2):
                result = runner.execute([sys.executable, '-c', parent], root, 15, False)
            self.assertTrue(result['host_memory_limit'])
            self.assertFalse(result['timeout'])
            self.assertGreater(result['peak_host_rss_bytes'], result['host_memory_limit_bytes'])
            self.assertTrue(marker.exists())
            child_pid = int(marker.read_text())
            # A reparented zombie may await init reaping, but must not be running.
            deadline = time.monotonic() + 2
            while True:
                snapshot = runner.proc_snapshot()
                stopped = child_pid not in snapshot or snapshot[child_pid]['state'] == 'Z'
                if stopped or time.monotonic() >= deadline:
                    break
                time.sleep(.01)
            self.assertTrue(stopped)

    def test_sampling_failure_still_terminates_launched_job(self):
        with tempfile.TemporaryDirectory() as directory:
            process = SimpleNamespace(pid=10)
            with patch.object(runner.subprocess, 'Popen', return_value=process), \
                 patch.object(runner, 'proc_snapshot', side_effect=PermissionError('proc unreadable')), \
                 patch.object(runner, 'terminate_job') as terminate:
                with self.assertRaises(PermissionError):
                    runner.execute(['unused'], Path(directory), 60, False)
                terminate.assert_called_once_with(process, {})

    def test_memory_interruption_quarantines_card_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'harness').mkdir()
            (root / 'harness/runtime.json').write_text(json.dumps(self.config(root)))
            def execute(args, job, *unused):
                for filename in ('stdout.log', 'stderr.log'):
                    (job / filename).write_text('')
                return {'returncode': 125, 'timeout': False, 'host_memory_limit': True,
                        'peak_host_rss_bytes': 65 * 1024**3, 'host_memory_limit_bytes': 64 * 1024**3}
            with patch.object(runner, 'cards', return_value={'0000:02:00.0': '7'}), \
                 patch.object(runner, 'health', return_value={'state': 'healthy'}), \
                 patch.object(runner, 'execute', side_effect=execute), patch.object(runner, 'mark') as mark:
                result = runner.main(root, {'model': 'parakeet', 'kind': 'run', 'job': 'memory-limit', 'command': 'python candidate.py'})
            self.assertEqual(result['failure'], 'host_memory_limit')
            mark.assert_called_once_with('0000:02:00.0', 'quarantined', 'memory-limit')
            self.assertEqual(json.loads((root / 'jobs/memory-limit/result.json').read_text()), result)
            self.assertFalse(result['timeout'])

    def test_paths_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ('../outside', '/absolute', 'a\\b', ''):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    runner.safe_file(root, name)

    def test_real_watchdog_terminates_sleeping_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'child-terminated'
            child = ('import signal,time,sys; from pathlib import Path; '
                     f'signal.signal(signal.SIGTERM, lambda *args: (Path({str(marker)!r}).write_text("terminated"), sys.exit(0))); '
                     'time.sleep(60)')
            parent = ('import subprocess,sys,time; '
                      f'subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(60)')
            result = runner.execute([sys.executable, '-c', parent], root, 1, False)
            end = time.monotonic() + 2
            while not marker.exists() and time.monotonic() < end:
                time.sleep(.01)
            self.assertTrue(result['timeout'])
            self.assertEqual(result['returncode'], 124)
            self.assertTrue(marker.exists(), 'grandchild did not receive process-group termination')

    def test_profile_run_mounts_only_readonly_public_inputs_with_normal_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'harness').mkdir()
            (root / 'harness/runtime.json').write_text(json.dumps(self.config(root)))
            public = root / 'references/parakeet/smoke/input'
            public.mkdir(parents=True)
            for name, content in (('suite.json', '{"model_key":"parakeet","stage":"smoke"}'),
                                  ('config.json', '{}'), ('inputs.npz', 'public tokens'),
                                  ('oracle.npz', 'forbidden accidental file')):
                (public / name).write_text(content)
            (public.parent / 'oracle').mkdir()
            (public.parent / 'oracle/private.json').write_text('private oracle')
            captured = []
            def execute(args, job, total, monitor, *unused):
                for name in ('stdout.log', 'stderr.log'):
                    (job / name).write_text('')
                captured.append((args, {p.name for p in (job / 'input').iterdir()}, total, monitor))
                return {'returncode': 0, 'timeout': False}
            with patch.object(runner, 'cards', return_value={'0000:02:00.0': '7'}), \
                 patch.object(runner, 'health', return_value={'state': 'healthy'}), \
                 patch.object(runner, 'execute', side_effect=execute):
                result = runner.main(root, {'model': 'parakeet', 'kind': 'run', 'job': 'profile', 'command': 'python profile.py',
                                            'public_input_stage': 'smoke', 'timeout': 45})
                args, names, total, monitor = captured[-1]
                self.assertEqual(names, {'suite.json', 'config.json', 'inputs.npz'})
                source = str((root / 'jobs/profile/input').resolve())
                position = args.index(source)
                self.assertEqual(args[position - 1:position + 2], ['--ro-bind', source, '/input'])
                self.assertNotIn(str(public.parent / 'oracle'), args)
                self.assertEqual(total, 47)
                self.assertFalse(monitor)
                self.assertNotIn('verdict', result)
                runner.main(root, {'model': 'parakeet', 'kind': 'run', 'job': 'normal', 'command': 'true'})
                self.assertEqual(captured[-1][1], set())
                with self.assertRaises(ValueError):
                    runner.main(root, {'model': 'parakeet', 'kind': 'run', 'job': 'invalid', 'command': 'true', 'public_input_stage': 'full'})
                (public / 'inputs.npz').unlink()
                (public / 'inputs.npz').symlink_to(public.parent / 'oracle/private.json')
                with self.assertRaises(ValueError):
                    runner.main(root, {'model': 'parakeet', 'kind': 'run', 'job': 'link', 'command': 'true', 'public_input_stage': 'smoke'})


if __name__ == '__main__':
    unittest.main()
