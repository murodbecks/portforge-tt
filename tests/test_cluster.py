"""Native CHIA inventory, explicit SSH transport and durable job identity."""
import support  # noqa: F401  (pack and registry setup)
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cluster
from runner import write_json


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = {'cluster_name': 'test', 'provider': {'head_ip': '127.0.0.1'},
            'auth': {'ssh_user': 'tt-user', 'ssh_private_key': '~/.ssh/tt-test',
                     'overrides': {'192.0.2.2': {'ssh_user': 'gpu-user', 'ssh_private_key': '~/.ssh/gpu-test'}}},
            'available_node_types': {
                'tt': {'compatible_ips': ['192.0.2.1'], 'num_workers': 1, 'resources': {'tt': 4}},
                'reference': {'compatible_ips': ['192.0.2.2'], 'num_workers': 1, 'resources': {'reference': 1}}},
            'porting': {'tt': {'work_dir': '/srv/tt jobs', 'python': '/env/tt/bin/python'},
                        'reference': {'work_dir': '/srv/ref', 'python': '/env/gpu/bin/python',
                                      'executor': 'direct', 'hf_home': '/models/cache'}}}

    def tearDown(self):
        self.temp.cleanup()

    def load(self, raw=None):
        path = self.root / 'cluster.yaml'
        path.write_text(json.dumps(raw or self.raw))
        return cluster.load(path)

    def test_native_schema_resolves_per_host_auth_and_resources(self):
        config = self.load()
        self.assertEqual(config['tt']['host'], '192.0.2.1')
        self.assertEqual(config['tt']['user'], 'tt-user')
        self.assertEqual(config['reference']['user'], 'gpu-user')
        self.assertEqual(config['reference']['key'], str(Path.home() / '.ssh/gpu-test'))
        self.assertEqual(config['tt']['resources']['tt'], 4)

    def test_inventory_rejects_ambiguous_hosts_unsafe_paths_and_executor(self):
        invalid = []
        multiple = copy.deepcopy(self.raw)
        multiple['available_node_types']['tt']['compatible_ips'].append('192.0.2.3')
        invalid.append(multiple)
        relative = copy.deepcopy(self.raw)
        relative['porting']['tt']['python'] = 'relative/python'
        invalid.append(relative)
        executor = copy.deepcopy(self.raw)
        executor['porting']['reference']['executor'] = 'unapproved'
        invalid.append(executor)
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.load(raw)

    def test_ssh_ignores_aliases_and_requires_explicit_identity_and_host_verification(self):
        endpoint = {**self.load()['reference'], 'known_hosts': '~/.ssh/test-known', 'host_key_alias': 'gpu-instance'}
        with patch.object(cluster.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, b'ok', b'')) as run:
            self.assertEqual(cluster.ssh(endpoint, 'echo hello', b'data', 30), b'ok')
        args = run.call_args.args[0]
        self.assertEqual(args[:3], ['ssh', '-F', '/dev/null'])
        for option in ('BatchMode=yes', 'IdentitiesOnly=yes', 'StrictHostKeyChecking=yes', 'HostKeyAlias=gpu-instance'):
            self.assertIn(option, args)
        self.assertEqual(args[-3:], ['--', '192.0.2.2', 'echo hello'])
        self.assertEqual(args[args.index('-i') + 1], endpoint['key'])
        self.assertEqual(args[args.index('-l') + 1], 'gpu-user')
        self.assertEqual(run.call_args.kwargs['input'], b'data')

    def test_model_checkpoint_inventory_is_explicit_and_validated(self):
        raw = copy.deepcopy(self.raw)
        raw['porting']['tt']['weights_by_model'] = {'parakeet': '/models/parakeet'}
        self.assertEqual(self.load(raw)['tt']['weights_by_model']['parakeet'], '/models/parakeet')
        for value in ({'unknown-model': '/models/ambiguous'}, {'parakeet': '../relative'},
                      {'parakeet': '/models/../other'}, {'parakeet': 1}, []):
            raw['porting']['tt']['weights_by_model'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(raw)

    def test_direct_and_slurm_launch_once_then_reuse_record(self):
        for executor in ('direct', 'slurm'):
            endpoint = {**self.load()['reference'], 'executor': executor, 'slurm': {'partition': 'gpu'}}
            record = self.root / (executor + '.json')
            with patch.object(cluster, 'upload') as upload, patch.object(cluster, 'ssh', side_effect=[b'12345\n', b'0\n']) as ssh:
                cluster.reference_job(endpoint, '/srv/ref/job', 'python reference.py', True, record)
                upload.assert_called_once()
                script = upload.call_args.args[2]['job.sh']
                self.assertIn('flock -n 9', script)
                self.assertIn('PORTFORGE_REFERENCE_EXECUTION=' + executor, script)
                self.assertEqual('.gpu-reference.lock' in script, executor == 'direct')
                launch = ssh.call_args_list[0].args[1]
                self.assertIn('sbatch' if executor == 'slurm' else 'nohup timeout', launch)
            with patch.object(cluster, 'upload') as upload, patch.object(cluster, 'ssh', return_value=b'0\n') as ssh:
                cluster.reference_job(endpoint, '/srv/ref/job', 'python reference.py', True, record)
                upload.assert_not_called()
                self.assertEqual(ssh.call_count, 1)
                self.assertNotIn('sbatch', ssh.call_args.args[1])
                self.assertNotIn('nohup', ssh.call_args.args[1])

    def test_cpu_reference_preparation_does_not_hold_gpu_lease(self):
        endpoint = self.load()['reference']
        with patch.object(cluster, 'upload') as upload, patch.object(cluster, 'ssh', side_effect=[b'99\n', b'0\n']):
            cluster.reference_job(endpoint, '/srv/ref/preparation', 'prepare', False, self.root / 'cpu-job.json')
        self.assertNotIn('.gpu-reference.lock', upload.call_args.args[2]['job.sh'])

    def test_remote_runner_quotes_paths_and_uses_configured_endpoint(self):
        config = self.load()
        with patch.object(cluster, 'ssh', return_value=b'{"cards": []}') as ssh:
            self.assertEqual(cluster.remote(config, self.root / 'run', {'kind': 'health'}), {'cards': []})
        endpoint, command, body, timeout = ssh.call_args.args
        self.assertEqual(endpoint, config['tt'])
        self.assertIn("'/srv/tt jobs/run/harness/runner.py'", command)
        self.assertEqual(json.loads(body), {'kind': 'health'})

    def test_polling_failure_preserves_submitted_job_record(self):
        endpoint, record = self.load()['reference'], self.root / 'job.json'
        saved = {'directory': '/srv/ref/job', 'endpoint_sha256': cluster.identity(endpoint),
                 'submission': 'submitted', 'job': '12345'}
        write_json(record, saved)
        with patch.object(cluster, 'ssh', side_effect=RuntimeError('SSH disconnected')) as ssh, \
             patch.object(cluster.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'transport unavailable'):
                cluster.reference_job(endpoint, '/srv/ref/job', 'run', True, record)
            self.assertEqual(ssh.call_count, 6)
        self.assertEqual(json.loads(record.read_text()), saved)

    def test_ambiguous_submission_records_intent_and_never_submits_again(self):
        endpoint, record = self.load()['reference'], self.root / 'job.json'
        with patch.object(cluster, 'upload'), patch.object(cluster, 'ssh', side_effect=subprocess.TimeoutExpired('ssh', 90)):
            with self.assertRaises(subprocess.TimeoutExpired):
                cluster.reference_job(endpoint, '/srv/ref/job', 'run', True, record)
        self.assertEqual(json.loads(record.read_text())['submission'], 'intent')
        with patch.object(cluster, 'upload') as upload, patch.object(cluster, 'ssh') as ssh:
            with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
                cluster.reference_job(endpoint, '/srv/ref/job', 'run', True, record)
            upload.assert_not_called()
            ssh.assert_not_called()

    def test_saved_job_refuses_changed_machine_or_directory(self):
        endpoint, record = self.load()['reference'], self.root / 'job.json'
        write_json(record, {'directory': '/srv/ref/job', 'endpoint_sha256': cluster.identity(endpoint),
                            'submission': 'submitted', 'job': '12345'})
        with patch.object(cluster, 'ssh') as ssh:
            for changed, directory in (({**endpoint, 'host': '192.0.2.8'}, '/srv/ref/job'),
                                       (endpoint, '/different')):
                with self.assertRaises(ValueError):
                    cluster.reference_job(changed, directory, 'run', True, record)
            ssh.assert_not_called()


if __name__ == '__main__':
    unittest.main()
