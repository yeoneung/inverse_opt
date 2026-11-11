"""Check exact path realization, global sharpness and segment-count bounds."""
import argparse,csv,hashlib,importlib.metadata,json,platform,sys,time,warnings
from pathlib import Path
from fractions import Fraction as Q
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.projection_paths import exact_chain_check,float_lift,proper_radius,mixed_radius,qp_check

def run(output):
    directory=ROOT/'results'/output
    directory.mkdir(parents=True,exist_ok=False)
    protocol=ROOT/'configs/paths_protocol.json'
    cfg=json.loads(protocol.read_text())
    rng=np.random.default_rng(cfg['seed']);exact=[]
    for q in cfg['base_dimensions']:
        for n in cfg['arbitrary_chain_edges']:
            points=rng.integers(-5,6,size=(n+1,q)).tolist()
            exact.append(exact_chain_check(points,Q(1,5*n*n)))
    exact.append(exact_chain_check([[0,0],[1,0],[0,0],[-1,0],[0,0]],Q(1,100)))
    cases=[];start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for n in cfg['circle_edges']:
            angle=np.arange(n+1)*2*np.pi/n
            points=np.column_stack((np.cos(angle),np.sin(angle)));points[-1]=points[0]
            delta=n**-3;vertices,M,intervals=float_lift(points,delta)
            P,center=proper_radius(vertices);Rlo,Rhi,mixed=mixed_radius(vertices)
            eps2=delta*delta*(n*n+n**4)
            base_proper=1+3*np.cos(np.pi/n)**2
            theoretical=base_proper/(1+eps2/4)
            assert abs(proper_radius(points)[0]-base_proper)<1e-10
            assert 1-1e-9<=Rlo<=Rhi<=1+eps2/4+1e-8
            assert base_proper-1e-9<=P<=base_proper+eps2+1e-8
            longest=max(np.linalg.norm(b-a) for a,b in zip(vertices[:-1],vertices[1:]))
            instance_upper=(1+np.sqrt(max(0.,1-longest**2/(4*Rhi))))**2
            count_upper=(1+np.sqrt(1-1/(2*n*n)))**2
            err,normal=qp_check(vertices,M,intervals,sorted(set([0,n//3,n//2,n-1])))
            assert P/Rhi+1e-8>=theoretical and P/Rlo<=4+1e-7
            assert P/Rlo<=instance_upper+1e-7<=count_upper+1e-7
            assert err<1e-5
            cases.append(dict(edges=n,delta=delta,M=M,mixed_lower=Rlo,mixed_upper=Rhi,
                              single_risk=P,ratio_lower=P/Rhi,ratio_upper=P/Rlo,
                              theoretical_ratio_lower=theoretical,instance_ratio_upper=instance_upper,
                              segment_count_ratio_upper=count_upper,qp_projection_error=err,
                              normal_condition_violation=normal,minimum_plateau=float(np.min(intervals[1:,0]-intervals[:-1,1]))))
    report=dict(exact_checks=exact,cases=cases,seconds=time.perf_counter()-start,
                warnings=[str(w.message) for w in caught],
                conclusion='C_line=4 follows from the algebraic normal-cone construction; the numerical cases are independent diagnostics.')
    (directory/'runs.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for c in cases:print(c['edges'],c['ratio_lower'],c['ratio_upper'],c['theoretical_ratio_lower'],c['qp_projection_error'])
    print('Exact arbitrary-chain checks:',len(exact),'seconds',report['seconds'],'warnings',len(caught))
    with (directory/'cases.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    metadata=dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        cases=len(cases),exact_arbitrary_chains=len(exact),seconds=report['seconds'],
        python=platform.python_version(),cvxpy=importlib.metadata.version('cvxpy'),
        solver_warning_count=len(caught),solver_warnings=report['warnings'])
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    analysis=dict(passed=True,cases=len(cases),exact_arbitrary_chains=len(exact),
        maximum_projection_error=max(c['qp_projection_error'] for c in cases),
        maximum_ratio_lower=max(c['ratio_lower'] for c in cases))
    (directory/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not args.output or Path(args.output).name!=args.output or any(c in args.output for c in '/\\:'):
        raise ValueError('Use a simple fresh output name')
    run(args.output)


if __name__=='__main__':main()
