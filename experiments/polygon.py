"""Fixed global polygon examples: exact rational certificates and numerical checks."""
import argparse,csv,hashlib,json,os,sys,time,warnings
from fractions import Fraction
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.polygon_recovery import rational_instances,exact_case,project_polygon
from experiments.classification import enclosing


def serialize(value):
    if isinstance(value,Fraction):return str(value)
    if isinstance(value,dict):return {k:serialize(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serialize(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='polygon')
    args=parser.parse_args()
    if not args.output or Path(args.output).name!=args.output or any(c in args.output for c in '/\\:'):
        raise ValueError('Use a simple output name')
    output=ROOT/'results'/args.output
    if output.exists():raise FileExistsError('Preserve previous computations')
    output.mkdir(parents=True);rows=[];details={};started=time.perf_counter()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        for instance in rational_instances():
            begin=time.perf_counter();case=exact_case(instance)
            vertices=np.array(instance['vertices'],dtype=float)
            offset=np.array(instance['offset'],dtype=float);direction=np.array(instance['direction'],dtype=float)
            projection_error=0.
            for index,interval in enumerate(case['partition']):
                t=np.linspace(float(interval['start']),float(interval['end']),7)
                actual=project_polygon(offset+t[:,None]*direction,vertices)
                expected=vertices[index]+np.linspace(0,1,7)[:,None]*(vertices[index+1]-vertices[index])
                projection_error=max(projection_error,float(np.max(np.linalg.norm(actual-expected,axis=1))))
            mixed,_=enclosing(vertices)
            proper=min(enclosing(vertices,[a,b])[0] for a,b in zip(vertices[:-1],vertices[1:]))
            rows.append(dict(name=case['name'],mixed=float(case['mixed']),proper=float(case['proper']),
                ratio=float(case['ratio']),mixed_exact=str(case['mixed']),proper_exact=str(case['proper']),
                projection_error=projection_error,conic_error=max(abs(mixed-float(case['mixed'])),abs(proper-float(case['proper']))),
                seconds=time.perf_counter()-begin))
            details[case['name']]=serialize(dict(inputs=instance,certificate=case))
    analysis=dict(cases=len(rows),passed=all(r['ratio']>64/27 and r['ratio']<4 and r['projection_error']<1e-10 and r['conic_error']<1e-7 for r in rows),
                  maximum_projection_error=max(r['projection_error'] for r in rows),
                  maximum_conic_error=max(r['conic_error'] for r in rows),maximum_ratio=max(r['ratio'] for r in rows))
    assert analysis['passed'] and analysis['maximum_ratio']>3
    with (output/'cases.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    protocol=ROOT/'configs/polygon_protocol.json'
    metadata=dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        exact_arithmetic='Python fractions.Fraction for image partition and radii',
        seconds=time.perf_counter()-started,details=details,
        solver_warning_count=len(seen),solver_warnings=[str(w.message) for w in seen])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (output/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':main()
