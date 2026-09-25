"""Compare two saved TT evaluation jobs, refusing mismatched benchmark conditions."""
import argparse
import json
import math
from pathlib import Path
import statistics


def load_job(path):
    path = Path(path)
    return {name: json.loads((path / (name + '.json')).read_text())
            for name in ('request', 'result')}


def compare(baseline, candidate):
    """Return descriptive per-case ratios, not statistical or production claims.

    A precision-policy change, different device, failed quality gate, missing
    sample or changed output length cannot become an implementation speedup.
    Source hashes and all sample counts remain visible in the report.
    """
    left, right = baseline['result'], candidate['result']
    lv, rv = left.get('verdict', {}), right.get('verdict', {})
    for result, verdict in ((left, lv), (right, rv)):
        if result.get('returncode') != 0 or result.get('timeout') or result.get('failure') or verdict.get('passed') is not True:
            raise ValueError('both jobs must pass correctness without execution failure')
    for key in ('pci', 'cluster_sha256', 'precision', 'model'):
        if not left.get(key) or left[key] != right.get(key):
            raise ValueError('different or missing ' + key)
    for key in ('stage', 'suite_sha256', 'model_manifest_sha256', 'timing_protocol', 'precision'):
        if not lv.get(key) or lv[key] != rv.get(key):
            raise ValueError('different or missing verdict ' + key)
    for job in (baseline, candidate):
        req, result = job['request'], job['result']
        verdict = result['verdict']
        for key in ('cluster_sha256', 'precision', 'model'):
            if req.get(key) != result[key]:
                raise ValueError('request/result mismatch: ' + key)
        if not req.get('code_sha256'):
            raise ValueError('missing evaluated source identity')
        if (verdict.get('model_key') != result['model']
                or verdict['precision'].get('requested') != result['precision']
                or verdict['stage'] != req.get('stage')):
            raise ValueError('request/result/verdict identity mismatch')
    def indexed(verdict):
        cases = verdict['checks']
        by_name = {c['name']: c for c in cases}
        if not cases or len(by_name) != len(cases) or len(cases) != verdict['case_count']:
            raise ValueError('incomplete or duplicate cases')
        return by_name
    a, b = indexed(lv), indexed(rv)
    if set(a) != set(b):
        raise ValueError('different case sets')
    def samples(case):
        values = case.get('seconds', [])
        if (not values or len(values) != case.get('timing_samples')
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values)):
            raise ValueError('invalid or missing timing samples')
        return dict(samples=len(values), median=statistics.median(values),
                    minimum=min(values), maximum=max(values), seconds=values)
    rows = []
    for name, first in a.items():
        second = b[name]
        for key in ('rows', 'forecast_points'):
            if (type(first.get(key)) is not int or type(second.get(key)) is not int
                    or first[key] <= 0 or first[key] != second[key]):
                raise ValueError('different or missing workload ' + key + ': ' + name)
        x, y = samples(first), samples(second)
        quantile_error = [c.get('quantile_max_row_nrmse') for c in (first, second)]
        if any(e is None or e > 0.04 for e in quantile_error):
            raise ValueError('outputs are not both within the oracle NRMSE gate: ' + name)
        rows.append(dict(name=name, rows=first['rows'], forecast_points=first['forecast_points'],
                         both_within_oracle_gate=True, baseline=x, candidate=y,
                         observed_latency_ratio=x['median'] / y['median']))
    same_source = baseline['request']['code_sha256'] == candidate['request']['code_sha256']
    return dict(comparison='repeatability' if same_source else 'same_precision_candidate',
                baseline_source=baseline['request']['code_sha256'], candidate_source=candidate['request']['code_sha256'],
                pci=left['pci'], cluster_sha256=left['cluster_sha256'],
                suite_sha256=lv['suite_sha256'], precision=lv['precision'],
                timing_protocol=lv['timing_protocol'], cases=rows,
                minimum_samples_per_case=min(min(c['baseline']['samples'], c['candidate']['samples']) for c in rows),
                limitations=['Descriptive observations; no statistical confidence or serving SLA.',
                             'Same card/config does not prove identical thermal or contention conditions.',
                             'Requires both outputs to be within the same FP32-oracle NRMSE gate; quality tradeoffs are separate.',
                             'No aggregate ratio across heterogeneous cases; memory is reported separately.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(load_job(args.baseline), load_job(args.candidate)), indent=2, allow_nan=False))
