"""Validate the balanced-capacity data and optionally compare a fresh rerun."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAVED = 'balanced_reproduction'


def identity(row):
    return row['kind'], row['instance'], row.get('budget', 0), row['repetition']


def validate(folder):
    directory = ROOT/'code/results'/folder
    result = json.loads((directory/'runs.json').read_text())
    assert not result['pilot'] and not result['failures']
    assert len(result['rows']) == 438 and len(result['references']) == 12
    count = Counter(row['kind'] for row in result['rows'])
    assert count == {'capacity':378,'structured':60}
    assert len(set(map(identity,result['rows']))) == 438
    assert all(n==3 for n in Counter((row['kind'],row['instance'],row.get('budget',0)) for row in result['rows']).values())
    data = np.load(directory/'instances.npz')
    matrices = [data[name] for name in data.files if name.endswith('_G')]
    assert len(matrices)==42
    for G in matrices:
        assert np.linalg.matrix_rank(G)==G.shape[1]
        np.testing.assert_allclose(np.linalg.svd(G,compute_uv=False),.75,atol=1e-12)
    for seed in result['protocol']['seeds']:
        for scale in result['protocol']['capacity_factors']:
            base=f'q32_p8_s{seed}_c{scale}'
            for p in [2,4]:
                other=f'q32_p{p}_s{seed}_c{scale}'
                np.testing.assert_allclose(data[other+'_G'],data[base+'_G'][:,:p],atol=1e-14)
                for field in ['A','b','c']:
                    np.testing.assert_array_equal(data[other+'_'+field],data[base+'_'+field])
    for row in result['rows']:
        assert row['maximum_forward_residual'] < 2e-7
        assert row['maximum_dual_residual'] < 1e-7
        assert row['maximum_distance_gap'] < 1e-6
        assert row['adaptive_distance']+3e-7 >= row['original_distance']
        assert row['adaptive_total_seconds'] >= row['common_seconds']
        assert row['original_total_seconds'] >= row['common_seconds']
        for stage in ['forward_seconds','anchor_seconds','upper_seconds','bookkeeping_seconds']:
            assert row[stage]>=0 and np.isfinite(row[stage])
        assert abs(row['adaptive_total_seconds']-row['common_seconds']-row['adaptive_seconds'])<1e-8
        assert abs(row['common_seconds']-sum(row[s] for s in ['forward_seconds','anchor_seconds','upper_seconds','bookkeeping_seconds']))<1e-7
        if row['kind']=='capacity':
            assert row['queries']<=row['budget']
            assert row['plain_upper']+1e-8>=row['upper']>=row['variance']-1e-7
            assert row['source_margin']>=-1e-12
            if 'reference_anchor_upper' in row:
                assert row['upper']+3e-6>=row['reference_anchor_upper']
                assert row['adaptive_gap']<=row['reference_penalty_upper']+3e-6
        elif row['affine_control']:
            assert row['adaptive_gap']<1e-8 and row['original_gap']<1e-8
        else:
            assert 0<row['adaptive_gap']<=row['penalty_reference']+3e-6
            assert abs(row['R_reference']-(1-1/(2*row['p'])))<1e-12
            assert row['upper']>=row['R_reference']-1e-7
    for ref in result['references'].values():
        assert ref['tested_active_subsets']==386
        assert ref['probe_error']<2e-6 and ref['P_upper']-ref['P_lower']<3e-5
        assert ref['R_lower']<=ref['R_upper']+1e-8
        assert ref['supplied_center_adaptive_gap']<=ref['penalty_upper']+3e-6
    return result


def compare(first, second):
    a,b={identity(row):row for row in first['rows']},{identity(row):row for row in second['rows']}
    assert a.keys()==b.keys()
    maximum=0.
    for key,row in a.items():
        for field,value in row.items():
            if field.endswith('_seconds') or field in ['repetition','reference_seconds']:
                continue
            other=b[key][field]
            if isinstance(value,(float,int)) and not isinstance(value,bool):
                error=abs(value-other)
                assert error<=3e-6*max(1.,abs(value)),(key,field,value,other)
                maximum=max(maximum,error)
            elif isinstance(value,list):
                np.testing.assert_allclose(value,other,atol=3e-6,rtol=3e-6)
            else:
                assert value==other,(key,field)
    return dict(rows=438,maximum_absolute_difference=maximum,timings_compared=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh',help='Fresh simple result directory name')
    parser.add_argument('--compare-initial',action='store_true')
    args=parser.parse_args()
    saved=validate(SAVED)
    report=dict(passed=True,timed_runs=438,independent_global_references=12,
                balanced_capacity_instances=42,structured_conditions=20,
                positive_structured_conditions=10,affine_zero_penalty_conditions=10,
                warnings_retained=saved['warning_count'])
    if args.fresh:
        if Path(args.fresh).name!=args.fresh or any(c in args.fresh for c in '/\\:'):
            raise ValueError('Use a simple result directory name')
        report['fresh_reproduction']=compare(saved,validate(args.fresh))
    if args.compare_initial:
        initial=json.loads((ROOT/'code/results/balanced_initial/runs.json').read_text())
        report['independent_reexecution']=compare(saved,initial)
    spec=importlib.util.spec_from_file_location('balanced_report',ROOT/'code/experiments/report_balanced_base.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected,summary=module.render(saved)
    if (ROOT/'sections/balanced_base_results.tex').exists():
        assert expected==(ROOT/'sections/balanced_base_results.tex').read_text(encoding='utf-8')
    assert summary==json.loads((ROOT/'verification/balanced_table_values.json').read_text())
    jgo_reporter = ROOT/'code/experiments/report_jogo.py'
    if jgo_reporter.exists():
        spec=importlib.util.spec_from_file_location('jgo_report',jgo_reporter)
        jgo=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(jgo)
        text,jgo_summary=jgo.render(saved)
        assert all(jgo_summary[key]==value for key,value in summary.items())
        assert jgo_summary==json.loads((ROOT/'verification/jogo_table_values.json').read_text())
        if (ROOT/'sections/jogo_results.tex').exists():
            assert text==(ROOT/'sections/jogo_results.tex').read_text(encoding='utf-8')
            assert jgo.parts(saved)[1]==(ROOT/'sections/folded_controls.tex').read_text(encoding='utf-8')
        report['jgo_presentation_uses_complete_data']=True
    (ROOT/'verification/balanced_result_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
