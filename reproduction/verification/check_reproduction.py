"""Compare fresh computations with supplied results, allowing floating-point tolerances."""
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(folder,name):
    with (ROOT/'code/results'/folder/name).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def compare(saved,fresh,name,keys,fields,tolerance):
    first=read(saved,name);second=read(fresh,name)
    key=lambda row:tuple(row[k] for k in keys)
    a={key(row):row for row in first};b={key(row):row for row in second}
    assert a.keys()==b.keys(),(saved,fresh,'Different configurations')
    maximum=0.
    for identity in a:
        for field in fields:
            x=float(a[identity][field]);y=float(b[identity][field])
            error=abs(x-y)
            assert error<=tolerance*max(1.,abs(x)),(saved,identity,field,x,y)
            maximum=max(maximum,error)
    return dict(cases=len(a),maximum_absolute_difference=maximum,relative_or_absolute_tolerance=tolerance)


def compare_capacity(fresh):
    saved=json.loads((ROOT/'code/results/scalar_capacity/runs.json').read_text())
    actual=json.loads((ROOT/'code/results'/fresh/'runs.json').read_text())
    key=lambda row:(row['dimension'],row['seed'],row['capacity_factor'])
    first={key(r):r for r in saved['cases']};second={key(r):r for r in actual['cases']}
    assert first.keys()==second.keys() and len(first)==120
    assert not saved['failures'] and not actual['failures']
    maximum=0.
    fields=['mixed_lower','mixed_upper','single_risk','ratio','actual_penalty',
            'supplied_center_lower','adaptive_lower','delta_lower','margin']
    for identity,row in first.items():
        other=second[identity]
        for field in fields:
            difference=abs(row[field]-other[field]);maximum=max(maximum,difference)
            assert difference<=2e-6*max(1.,abs(row[field])),(identity,field)
        for a,b in zip(row['finite_query'],other['finite_query']):
            assert a['queries']==b['queries'] and a['adaptive_positive']==b['adaptive_positive'] and a['positive']==b['positive']
            assert abs(a['adaptive_lower_value']-b['adaptive_lower_value'])<=2e-6
            assert abs(a['global_upper']-b['global_upper'])<=2e-6
    return dict(cases=len(first),maximum_absolute_difference=maximum,finite_query_flags_match=True)


def compare_multidimensional(fresh):
    saved=json.loads((ROOT/'code/results/anisotropic_controls/runs.json').read_text())
    actual=json.loads((ROOT/'code/results'/fresh/'runs.json').read_text())
    key=lambda r:(r['q'],r['p'],r['seed'],r['eta'])
    first={key(r):r for r in saved['cases']};second={key(r):r for r in actual['cases']}
    assert first.keys()==second.keys() and len(first)==135
    assert not saved['failures'] and not actual['failures']
    assert saved['protocol_sha256']==actual['protocol_sha256']
    maximum=0.
    for identity,row in first.items():
        other=second[identity]
        assert row['latent_rank']==other['latent_rank']==row['p']
        for a,b in zip(row['measurements'],other['measurements']):
            assert a['queries']==b['queries']
            assert a['original_positive']==b['original_positive'] and a['adaptive_positive']==b['adaptive_positive']
            for field in ['variance_lower','global_upper','covering_error','original_lower','adaptive_lower']:
                difference=abs(a[field]-b[field]);maximum=max(maximum,difference)
                assert difference<=2e-6*max(1.,abs(a[field])),(identity,field,difference)
    return dict(cases=135,maximum_absolute_difference=maximum,finite_query_detections_match=True,timings_compared=False)


def compare_complexity(fresh):
    saved=ROOT/'code/results/computational_separation'
    actual=ROOT/'code/results'/fresh
    first=json.loads((saved/'runs.json').read_text())
    second=json.loads((actual/'runs.json').read_text())
    assert [r['name'] for r in first]==[r['name'] for r in second] and len(first)==15
    maximum=0.
    for a,b in zip(first,second):
        for field in ['clauses','satisfying_assignments','source_nullity','patterns']:
            assert a[field]==b[field],(a['name'],field)
        for field in ['proper_lower','proper_upper','mixed_risk','central_risk']:
            difference=abs(a[field]-b[field]);maximum=max(maximum,difference)
            assert difference<2e-7,(a['name'],field,difference)
    assert json.loads((saved/'exact_controls.json').read_text())==json.loads((actual/'exact_controls.json').read_text())
    return dict(instances=15,maximum_absolute_difference=maximum,exact_controls_match=True,timings_compared=False)


def compare_curvature(fresh):
    saved=ROOT/'code/results/smooth_curvature'
    actual=ROOT/'code/results'/fresh
    first=json.loads((saved/'runs.json').read_text())
    second=json.loads((actual/'runs.json').read_text())
    assert first.keys()==second.keys()
    maximum=0.
    for group in first:
        assert len(first[group])==len(second[group])
        for a,b in zip(first[group],second[group]):
            assert a.keys()==b.keys()
            for field in a:
                if field=='sample_minimax_disagreement':
                    assert a[field]<1e-10 and b[field]<1e-10
                elif isinstance(a[field],float):
                    difference=abs(a[field]-b[field]);maximum=max(maximum,difference)
                    assert difference<1e-9*max(1.,abs(a[field])),(group,field,difference)
                else:
                    assert a[field]==b[field],(group,field)
    assert json.loads((actual/'summary.json').read_text())['total_cases']==37
    return dict(cases=37,maximum_absolute_difference=maximum,exact_controls_match=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',default='reproduction')
    parser.add_argument('--curvature-only',action='store_true',help='Check only the added local-curvature study.')
    args=parser.parse_args()
    if not args.prefix or Path(args.prefix).name!=args.prefix or any(c in args.prefix for c in '/\\:'):
        raise ValueError('Use a simple prefix')
    p=args.prefix
    if args.curvature_only:
        report=dict(passed=True,prefix=p,curvature=compare_curvature(p+'_curvature'),
                    scope='Only the added curvature study; previous studies were not rerun.')
        (ROOT/'verification/curvature_reproduction_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
        return
    report=dict(
        planar=compare('planar_recovery',p+'_recovery','runs.csv',
            ['seed','target_type','budget'],['lower','upper','reference_risk'],2e-6),
        two_branch=compare('mixture_recovery',p+'_mixture','phase.csv',
            ['length_ratio','angle_degrees'],['mixed_formula','single_formula','mixed_computed','single_computed'],1e-7),
        projection_realizations=compare('mixture_recovery',p+'_mixture','embeddings.csv',
            ['dimension','seed','length_ratio','cosine'],['ratio','mixed_error','single_error'],1e-7),
        mixture=compare('mixture_recovery',p+'_mixture','cases.csv',
            ['name'],['single_lower','single_upper','mixed_lower','mixed_upper'],2e-6),
        fiber=compare('fiber_study',p+'_fiber','runs.csv',
            ['name'],['delta_lower','delta_upper','margin','squared_gap_lower'],2e-6),
        facet_classification=compare('classification_study',p+'_classification','facets.csv',
            ['dimension','seed','family','beta','radius'],['mixed_normalized','proper_normalized','ratio'],5e-5),
        information_reversal=compare('classification_study',p+'_classification','information.csv',
            ['upper','source_tolerance'],['before_mixed','before_proper','after_mixed','after_proper'],1e-7),
        global_polygons=compare('polygon_study',p+'_polygon','cases.csv',
            ['name'],['mixed','proper','ratio','projection_error','conic_error'],1e-7),
        global_paths=compare('projection_path_study',p+'_paths','cases.csv',
            ['edges'],['mixed_lower','mixed_upper','single_risk','ratio_lower','ratio_upper'],2e-6),
        capacity=compare_capacity(p+'_capacity'),
        multidimensional=compare_multidimensional(p+'_multidimensional'),
        computational_separation=compare_complexity(p+'_complexity'),
        curvature=compare_curvature(p+'_curvature'))
    for folder in [p+'_'+name for name in ['recovery','mixture','fiber','classification','polygon','paths','capacity']]:
        metadata=json.loads((ROOT/'code/results'/folder/'metadata.json').read_text())
        assert metadata['protocol_sha256']
    for row in read(p+'_recovery','runs.csv'):
        assert row['converged']=='True' and float(row['gap'])<=1e-7
    for row in read(p+'_mixture','cases.csv'):
        assert row['single_converged']=='True' and row['mixed_converged']=='True'
        assert float(row['single_gap'])<=2e-6
    report.update(passed=True,prefix=p,timings_compared=False,
        scope='Local fresh-environment reproduction, not external replication. Runtimes and solver-warning counts may vary.')
    (ROOT/'verification/reproduction_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
