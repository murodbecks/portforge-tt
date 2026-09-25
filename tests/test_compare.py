import support  # noqa: F401  (pack and registry setup)
import copy
import unittest

from compare import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.job = dict(request=dict(code_sha256='base', cluster_sha256='cluster', precision='bf16', model='parakeet', stage='bringup'),
                        result=dict(returncode=0, failure=None, timeout=False, pci='card',
                                    cluster_sha256='cluster', precision='bf16', model='parakeet',
                                    verdict=dict(passed=True, stage='bringup', model_key='parakeet', suite_sha256='suite',
                                                 model_manifest_sha256='model', timing_protocol='sync', precision={'requested': 'bf16'},
                                                 case_count=1, checks=[dict(name='short', rows=1, forecast_points=64,
                                                                            quantile_max_row_nrmse=.01, timing_samples=3, seconds=[1., 2., 3.])])) )
        self.candidate = copy.deepcopy(self.job)
        self.candidate['request']['code_sha256'] = 'new'
        self.candidate['result']['verdict']['checks'][0]['seconds'] = [.5, 1., 1.5]

    def test_paired_case_observation_preserves_samples(self):
        report = compare(self.job, self.candidate)
        self.assertEqual(report['cases'][0]['observed_latency_ratio'], 2)
        self.assertEqual(report['minimum_samples_per_case'], 3)
        self.assertTrue(report['cases'][0]['both_within_oracle_gate'])

    def test_unchanged_code_is_repeatability(self):
        self.assertEqual(compare(self.job, self.job)['comparison'], 'repeatability')

    def test_different_card_config_model_or_precision_rejected(self):
        for key in ('pci', 'cluster_sha256', 'model', 'precision'):
            with self.subTest(key=key):
                other = copy.deepcopy(self.candidate)
                other['result'][key] = 'different'
                with self.assertRaises(ValueError):
                    compare(self.job, other)

    def test_failed_quality_or_execution_rejected(self):
        for field, value in (('returncode', 1), ('timeout', True), ('failure', 'device')):
            other = copy.deepcopy(self.candidate)
            other['result'][field] = value
            with self.assertRaises(ValueError):
                compare(self.job, other)
        self.candidate['result']['verdict']['passed'] = False
        with self.assertRaises(ValueError):
            compare(self.job, self.candidate)

    def test_policy_or_suite_change_rejected(self):
        for field in ('precision', 'suite_sha256', 'timing_protocol', 'model_manifest_sha256'):
            other = copy.deepcopy(self.candidate)
            other['result']['verdict'][field] = 'different'
            with self.assertRaises(ValueError):
                compare(self.job, other)

    def test_shorter_output_cannot_inflate_speedup(self):
        self.candidate['result']['verdict']['checks'][0]['forecast_points'] = 16
        with self.assertRaises(ValueError):
            compare(self.job, self.candidate)

    def test_nan_zero_and_missing_samples_rejected(self):
        for values in ([float('nan'), 1., 2.], [0., 1., 2.], []):
            self.candidate['result']['verdict']['checks'][0]['seconds'] = values
            with self.assertRaises(ValueError):
                compare(self.job, self.candidate)

    def test_duplicate_cases_rejected(self):
        checks = self.candidate['result']['verdict']['checks']
        checks.append(copy.deepcopy(checks[0]))
        with self.assertRaises(ValueError):
            compare(self.job, self.candidate)

    def test_outputs_outside_the_oracle_gate_cannot_be_compared(self):
        self.candidate['result']['verdict']['checks'][0]['quantile_max_row_nrmse'] = .2
        with self.assertRaises(ValueError):
            compare(self.job, self.candidate)

    def test_missing_nrmse_evidence_rejected(self):
        del self.candidate['result']['verdict']['checks'][0]['quantile_max_row_nrmse']
        with self.assertRaises(ValueError):
            compare(self.job, self.candidate)

    def test_identically_mislabeled_verdicts_rejected(self):
        for field, value in (('model_key', 'other'), ('precision', {'requested': 'fp32'})):
            left, right = copy.deepcopy(self.job), copy.deepcopy(self.candidate)
            for job in (left, right):
                job['result']['verdict'][field] = value
            with self.assertRaises(ValueError):
                compare(left, right)

    def test_candidate_workload_types_checked(self):
        self.candidate['result']['verdict']['checks'][0]['rows'] = True
        with self.assertRaises(ValueError):
            compare(self.job, self.candidate)


if __name__ == '__main__':
    unittest.main()
