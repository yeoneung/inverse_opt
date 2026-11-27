"""Render the integrated paper from supplied final data; preserve recorded runtimes."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon
ROOT=Path(__file__).resolve().parents[1]
REPORT_ROOT=ROOT.parent/'verification'/'rendered_outputs'
for folder in ['sections','figures']:
    (REPORT_ROOT/folder).mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
from src.polygon_recovery import rational_instances,exact_case

def figures():
    plt.rcParams.update({'font.family':'serif','font.size':7.5,'axes.titlesize':8,
        'legend.fontsize':6.2,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':1.,'patch.linewidth':1.,'xtick.major.width':1.,'xtick.minor.width':1.,
        'ytick.major.width':1.,'ytick.minor.width':1.})
    width=119/25.4;blue='#2166ac';orange='#d66000'
    fig,axes=plt.subplots(1,2,figsize=(width,2.55))
    r=.3;x0=np.ones(2)/2;n=-np.ones(2)/np.sqrt(2);e=np.array([1.,-1.])/np.sqrt(2)
    d=e/np.sqrt(3)+np.sqrt(2/3)*n
    endpoints=np.array([x0-r*e/np.sqrt(3),x0,x0+r*d]);mixed=x0+r*n/np.sqrt(6);single=x0+r*d/4
    ax=axes[0];ax.add_patch(Polygon([[0,0],[1,0],[0,1]],facecolor='#eeeeee',edgecolor='gray',linewidth=1.))
    ax.plot(endpoints[:,0],endpoints[:,1],color='black',lw=2,label='Response image')
    ax.add_patch(Circle(mixed,r/np.sqrt(2),fill=False,ls='--',color=blue,lw=1))
    ax.scatter(*mixed,c=blue,marker='*',s=50,zorder=4,label='Mixed center')
    ax.scatter(*single,c=orange,s=24,zorder=4,label='Single-model center')
    ax.set(xlim=(-.06,1.04),ylim=(-.06,1.04),xlabel='Allocation 1',ylabel='Allocation 2',title='Shared capacity')
    ax.set_aspect('equal')
    instance=rational_instances()[1];case=exact_case(instance);vertices=np.array(instance['vertices'],dtype=float)
    ax=axes[1];ax.add_patch(Polygon(vertices,facecolor='#eeeeee',edgecolor='gray',linewidth=1.))
    ax.plot(vertices[:,0],vertices[:,1],color='black',lw=2)
    mixed=np.array(case['mixed_center'],dtype=float);single=np.array(case['single_center'],dtype=float)
    ax.add_patch(Circle(mixed,np.sqrt(float(case['mixed'])),fill=False,ls='--',color=blue,lw=1))
    ax.scatter(*mixed,c=blue,marker='*',s=50,zorder=4);ax.scatter(*single,c=orange,s=24,zorder=4)
    ax.set(xlim=(-.48,1.65),ylim=(-.22,1.86),xlabel='Action 1',ylabel='Action 2',title='Global latent line')
    ax.set_aspect('equal');fig.subplots_adjust(left=.11,right=.99,bottom=.24,top=.89,wspace=.34)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),
               ncol=3,columnspacing=1.,handletextpad=.5,frameon=False)
    for suffix in ['pdf','png']:fig.savefig(REPORT_ROOT/'figures'/('projection_geometry.'+suffix),dpi=220)
    plt.close(fig)
    chain=json.loads((ROOT/'results/projection_path_study/runs.json').read_text())['cases']
    groups=json.loads((ROOT/'results/scalar_capacity/summary.json').read_text())['summary']
    fig,axes=plt.subplots(1,2,figsize=(width,2.25))
    count=[c['edges'] for c in chain]
    axes[0].plot(count,[c['ratio_lower'] for c in chain],'-o',markersize=2.5,label='Computed price')
    axes[0].plot(count,[c['theoretical_ratio_lower'] for c in chain],':',label='Lower bound')
    axes[0].axhline(4,color='black',ls='--',lw=1.,label='Sharp supremum')
    axes[0].set(xlabel='Response segments',ylabel='Squared-risk ratio',title='Global line images')
    axes[0].legend(fontsize=5.7,frameon=False,loc='lower right')
    q=[r['dimension'] for r in groups]
    axes[1].semilogy(q,[100*r['median_supplied_center_fraction'] for r in groups],'-o',markersize=2.5,label='Original margin')
    axes[1].semilogy(q,[100*r['median_adaptive_fraction'] for r in groups],'-s',markersize=2.5,label='Adaptive radius')
    axes[1].set(xlabel='Action dimension',ylabel='Actual penalty captured (%)',title='Coupled-capacity calibration')
    axes[1].set_xticks(q);axes[1].legend(fontsize=5.7,frameon=False)
    fig.tight_layout(pad=.6,w_pad=.7)
    active=ROOT.parent/'figures'
    active.mkdir(exist_ok=True)
    for suffix in ['pdf','png']:
        fig.savefig(REPORT_ROOT/'figures'/('global_calibration.'+suffix),dpi=220)
        fig.savefig(active/('Fig2.'+suffix),dpi=220)
    plt.close(fig)

def main():
    data=json.loads((ROOT/'results/scalar_capacity/runs.json').read_text())
    groups=json.loads((ROOT/'results/scalar_capacity/summary.json').read_text())['summary']
    global_data=json.loads((ROOT/'results/projection_path_study/runs.json').read_text())
    calibration=[];cost=[]
    for r in groups:
        calibration.append(str(r['dimension'])+' & '+str(r['cases'])+' & '+str(r['positive_penalties'])+' & '+
            f"{100*r['median_supplied_center_fraction']:.2f}"+' & '+f"{100*r['median_adaptive_fraction']:.2f}"+' & '+
            f"{r['finite_query_detections']['129']}/{r['adaptive_finite_query_detections']['129']}"+r'\\')
        cost.append(str(r['dimension'])+' & '+f"{r['median_certificate_ms']:.1f}"+' & '+
            f"{r['median_adaptive_ms']:.1f}"+' & '+f"{r['median_full_recovery_ms']:.1f}"+' & '+str(r['maximum_adaptive_solves'])+r'\\')
    rows=data['cases'];positive=sum(r['positive_penalties'] for r in groups)
    old_hits=sum(r['finite_query_detections']['129'] for r in groups)
    hits=sum(r['adaptive_finite_query_detections']['129'] for r in groups)
    hits_33=sum(r['adaptive_finite_query_detections']['33'] for r in groups)
    false=sum(b['adaptive_positive'] and c['actual_penalty']<=1e-7 for c in rows for b in c['finite_query'])
    forward=max(c['forward_path_error'] for c in rows);mantissa,exponent=f'{forward:.2e}'.split('e')
    forward_tex=mantissa+r'\times10^{'+str(int(exponent))+'}'
    text=r'''\section{Numerical Calculations}\label{sec:numerics}
\subsection{Analytic Controls and Global Paths}
The analytic checks use 96 facet cases, 15 information cases, 77 two-branch
configurations and 48 fixed-fiber cases. Forward QPs and enclosing balls
check the formulas, including zero-penalty controls. Coupled controls use
an enclosing-center method for convex semi-infinite programs
\cite{paruchuri2023}; matrices and settings are supplied with the code.

For Theorem~\ref{thm:path-lift}, thirteen integer-vertex chains in base
dimensions $1,2,4,7$, including a repeated-vertex chain, satisfy the normal
identities in rational arithmetic. Regular-polygon lifts have
4, 8, 12, 16, 24, 32, 48 and 64 edges, with $\delta=m^{-3}$.
Inequalities \eqref{eq:path-normal-identity} tighten $M$ without changing the images.
Figure~\ref{fig:global-calibration} shows convergence toward four; the largest
computed price lies between 3.992641013 and 3.992641014. Independent full-hull
QP projections differ from the prescribed paths by less than
$2.0\times10^{-11}$. Rational planar examples exceed the two-branch bound.

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{global_calibration.pdf}
\caption{Left: prices of lifted line images in action dimension four and the
constructive lower bound. Right: median fractions of actual penalty captured
on the coupled-capacity cases with positive penalties; the vertical scale
is logarithmic.}
\label{fig:global-calibration}
\end{figure}

\subsection{Coupled Capacity Problems and Calibration}
Use $q=4,8,16,32$, $t\in[0,4]$, a normalized standard Gaussian $d$, and
$y^{\rm src}=0.3\mathbf1_{\{d>0\}}$. All source responses
$\proj_{[0,0.3]^q}(y^{\rm src}+td)$ equal the observed $y^{\rm src}$.
The target is
\[
 \Phi(t)=\proj_{\mathcal X}(c+td),\qquad
 \mathcal X=\{x:0\le x\le\bar x,\ A_{\rm cap}x\le b_{\rm cap}\}.
\]
Set $c=y^{\rm src}+0.25\zeta^{\rm ctx}$ with standard Gaussian
$\zeta^{\rm ctx}$. Draw $\max(2,\lfloor q/2\rfloor)$ dense capacity rows
uniformly on $[0.2,1]^q$ and normalize them. Before common scaling,
$\bar x$ has independent uniform entries in $[0.7,1.3]$, and
$b_{\rm cap}=A_{\rm cap}u^{\rm cap}$ with $u^{\rm cap}$ uniform on $[0.2,0.55]^q$.
Ten seeds and scales $0.6,1,1.6$ give 120 cases, with complete settings in the code.

Forward QPs recover candidate response segments; their KKT inequalities
cover $[0,4]$. Endpoint enclosing balls and segmentwise minimax solves give
reference risks. The maximum forward disagreement is $1.86\times10^{-6}$.
Table~\ref{tab:capacity-calibration} compares \eqref{eq:fiber-witness} and
\eqref{eq:adaptive-witness}, using contact-weight means and retaining the
variance/upper-risk correction. Fractions are lower/actual penalty, with
medians over the 113 penalties exceeding $10^{-7}$; a numerical detection
requires a lower value exceeding $10^{-6}$.

\begin{table}[htbp]\centering\small
\caption{Coupled-capacity calibration. Fractions are percentages of the
actual penalty captured, conditional on a positive reference penalty.
The last column gives original/adaptive detections at 129 forward queries.}
\label{tab:capacity-calibration}
\begin{tabular}{rrrrrr}\toprule
$q$ & Cases & Positive & Original (\%) & Adaptive (\%) & Detections\\\midrule
4 & 30 & 23 & 50.71 & 79.62 & 3/6\\
8 & 30 & 30 & 1.54 & 59.85 & 2/18\\
16 & 30 & 30 & 0.24 & 45.95 & 0/22\\
32 & 30 & 30 & 0.02 & 41.96 & 0/25\\
\bottomrule\end{tabular}
\end{table}

\subsection{Finite Queries and Computational Cost}
Using $9,33,129$ uniform forward queries, the certificate computes contact
means and \eqref{eq:query-upper} without a response partition or reference
center. At 129 queries adaptation increases detections from 5 to 71 with
zero false positives at the stated thresholds. Nine queries detect none;
33 detect five with adaptation. Loose global upper values limit detection.

Table~\ref{tab:capacity-cost} reports median times. Checks exclude anchor
and query costs; each radius uses a primal and dual SOCP. Reference times
include path recovery and both centers. Adaptation costs more and is slower
than complete scalar-path recovery here.

\begin{table}[htbp]\centering\small
\caption{Median milliseconds. Checks use a supplied anchor; reference includes
complete path recovery. Radii counts candidate distances (maximum per dimension).}
\label{tab:capacity-cost}
\begin{tabular}{rrrrr}\toprule
$q$ & Original & Adaptive & Reference & Radii\\\midrule
4 & 4.8 & 18.7 & 7.0 & 4\\
8 & 4.8 & 19.6 & 11.6 & 5\\
16 & 5.0 & 25.2 & 16.4 & 6\\
32 & 5.2 & 31.7 & 27.8 & 7\\
\bottomrule\end{tabular}
\end{table}

The solvers also cover homogeneous conic fitting with interval or ball priors;
timings above concern intervals. Conic and KKT diagnostics are floating-point,
not interval certification. These tests address neither statistical
generalization nor closed-loop performance.
\input{sections/computational_experiment}
'''
    replacements={'@@RATIOLO@@':f"{global_data['cases'][-1]['ratio_lower']:.9f}",
        '@@RATIOHI@@':f"{global_data['cases'][-1]['ratio_upper']:.9f}",
        '@@FORWARD@@':forward_tex,'@@POSITIVE@@':str(positive),'@@OLDHITS@@':str(old_hits),
        '@@HITS@@':str(hits),'@@HITS33@@':str(hits_33),'@@FALSE@@':str(false),
        '@@CALIBRATION@@':'\n'.join(calibration),'@@COST@@':'\n'.join(cost)}
    for key,value in replacements.items():text=text.replace(key,value)
    (REPORT_ROOT/'sections/numerics_submission.tex').write_text(text,encoding='utf-8')
    figures()
    print(json.dumps(dict(cases=len(rows),positive=positive,original_hits=old_hits,adaptive_hits=hits,figures=2,tables=2),indent=2))


if __name__=='__main__':main()
