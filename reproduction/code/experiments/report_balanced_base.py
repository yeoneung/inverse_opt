"""Render balanced-capacity tables and the computational figure from retained raw runs."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT/'code/results/balanced_reproduction/runs.json'


def tex_sci(value):
    mantissa, exponent = f'{value:.2e}'.split('e')
    return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'


def condition_medians(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['kind'], row['instance'], row.get('budget', 0))].append(row)
    medians = []
    for group in groups.values():
        row = group[0].copy()
        for key in row:
            if isinstance(row[key], (int, float)) and not isinstance(row[key], bool):
                row[key] = float(np.median([r[key] for r in group]))
        medians.append(row)
    return medians


def render(report):
    rows = condition_medians(report['rows'])
    refs = report['references']
    lines = [r'\subsection{Accuracy against the Global References}',
        r'Table~\ref{tab:balanced-reference} compares the query certificates with the independent',
        r'global risks at 513 queries. The reference intervals also identify positive penalties.',
        r'Capture percentages include zero detections on positive-reference cases and divide each lower penalty',
        r'by the midpoint of its reference penalty interval. The times compare complete',
        r'implementations, including all query and anchor costs. The exhaustive reference',
        r'is a transparent small-instance baseline, not an optimized global solver.',
        r'\begin{table}[htbp]\centering\small',
        r'\caption{Global-reference comparison on six instances per row. Original and',
        r'adaptive columns count positive detections. Capture is the adaptive median',
        r'percentage. Total and reference are median times in milliseconds.}',
        r'\label{tab:balanced-reference}',
        r'\begin{tabular}{rrrrrrr}\toprule',
        r'$p$ & Regions & Original & Adaptive & Capture (\%) & Total & Reference\\\midrule']
    summary = {'reference': [], 'scaling': [], 'structured': []}
    for p in [2, 3]:
        s = [row for row in rows if row['kind']=='capacity' and row['q']==4 and row['p']==p and row['budget']==513]
        reg = [refs[row['instance']]['regions'] for row in s]
        original = sum(row['original_gap'] > 1e-6 for row in s)
        adaptive = sum(row['adaptive_gap'] > 1e-6 for row in s)
        fraction = [100*row['adaptive_gap']/((refs[row['instance']]['penalty_lower']+refs[row['instance']]['penalty_upper'])/2) for row in s if refs[row['instance']]['penalty_lower']>1e-6]
        total = 1000*np.median([row['adaptive_total_seconds'] for row in s])
        reference = 1000*np.median([refs[row['instance']]['seconds'] for row in s])
        lines.append(f'{p} & {min(reg)}--{max(reg)} & {original}/6 & {adaptive}/6 & {np.median(fraction):.2f} & {total:.1f} & {reference:.1f}'+r'\\')
        summary['reference'].append(dict(p=p, original_detections=original, adaptive_detections=adaptive,
            captured_percent=float(np.median(fraction)), total_ms=float(total), reference_ms=float(reference)))
    lines += [r'\bottomrule\end{tabular}\end{table}']
    widths = [v['P_upper']-v['P_lower'] for v in refs.values()]
    probe = max(v['probe_error'] for v in refs.values())
    lines += [f'The largest single-model reference interval has width {tex_sci(max(widths))},',
        f'and the largest independent forward-probe error is {tex_sci(probe)}.',
        r'Original and adaptive detection counts are reported separately.',
        r'A failed detection does not establish that the penalty vanishes.']
    reference_rows=[row for row in rows if row['kind']=='capacity' and row['q']==4 and row['budget']==513]
    cover_count=sum(row['adaptive_cover_gap']>1e-6 for row in reference_rows)
    positive_counts=[sum(refs[row['instance']]['penalty_lower']>1e-6 for row in reference_rows if row['p']==p) for p in [2,3]]
    lines += [f'The reference lower values establish positive penalties in {positive_counts[0]}/6 cases',
        f'at $p=2$ and {positive_counts[1]}/6 at $p=3$. The covering-only upper bound detects',
        f'{cover_count}/12 cases at the same 513-query budget.']
    supplied = []
    for p in [2,3]:
        group = [v for k,v in refs.items() if f'_p{p}_' in k]
        fractions = [100*v['supplied_center_adaptive_gap']/((v['penalty_lower']+v['penalty_upper'])/2) for v in group if v['penalty_lower']>1e-6]
        supplied.append(float(np.median(fractions)))
    lines += [f'When the independently computed center is supplied solely for calibration,',
        f'the adaptive median captured fractions are {supplied[0]:.2f}\\% and {supplied[1]:.2f}\\%',
        r'for $p=2$ and $p=3$, respectively. This supplied-center calculation is excluded',
        r'from the query procedure and its detection counts.']
    lines += [r'\subsection{Latent Dimension and Complete Computational Cost}',
        r'Table~\ref{tab:balanced-scaling} reports the larger coupled-capacity problems.',
        r'There are six instances in each row. Detection counts use the adaptive',
        r'certificate and the upper bound with affine-region checks. The covering-only',
        r'ablation uses exactly the same queried responses.',
        r'\begin{table}[htbp]\centering\small',
        r'\caption{Detections as the query budget increases and complete median times',
        r'at 513 queries. Both time columns include forward setup, every query,',
        r'all anchor updates, the upper-risk calculation and the indicated SOCP checks.}',
        r'\label{tab:balanced-scaling}',
        r'\begin{tabular}{rrrrrrr}\toprule',
        r'$q$ & $p$ & 33 & 129 & 513 & Original (ms) & Adaptive (ms)\\\midrule']
    for q,p in [(8,2),(16,4),(32,2),(32,4),(32,8)]:
        counts = []
        for budget in [33,129,513]:
            group = [row for row in rows if row['kind']=='capacity' and (row['q'],row['p'],row['budget'])==(q,p,budget)]
            counts.append(sum(row['adaptive_gap']>1e-6 for row in group))
        original = 1000*np.median([row['original_total_seconds'] for row in group])
        adaptive = 1000*np.median([row['adaptive_total_seconds'] for row in group])
        lines.append(f'{q} & {p} & {counts[0]}/6 & {counts[1]}/6 & {counts[2]}/6 & {original:.1f} & {adaptive:.1f}'+r'\\')
        summary['scaling'].append(dict(q=q,p=p,detections=counts,original_ms=float(original),adaptive_ms=float(adaptive)))
    large_rows=[row for row in rows if row['kind']=='capacity' and row['q']>4 and row['budget']==513]
    large_cover=sum(row['adaptive_cover_gap']>1e-6 for row in large_rows)
    lines += [r'\bottomrule\end{tabular}\end{table}',
        f'At 513 queries the covering-only ablation detects {large_cover}/{len(large_rows)} larger instances.',
        r'Figure~\ref{fig:balanced-scaling} locates the obstruction to detection.',
        r'The positive lower value requires the squared separation to exceed $U-\nu$.',
        r'At $p=2$, affine-region checks eventually make this upper-risk excess small.',
        r'At larger latent dimensions it can remain larger than the separation term.',
        r'The unsuccessful rows therefore demonstrate the limitation of this query',
        r'budget rather than a successful dimension-independent algorithm.',
        r'\begin{figure}[htbp]\centering',
        r'\includegraphics[width=\textwidth]{Fig3.pdf}',
        r'\caption{Left: median upper-risk excess $U-\nu$ (solid) and squared adaptive',
        r'separation (dashed) against query budget for the three $q=32$ rows of',
        r'Table~\ref{tab:balanced-scaling}. Detection requires the dashed value to exceed',
        r'the solid one in an individual instance. Action dimension is fixed at 32.',
        r'Right: complete timing stages at',
        r'513 queries. Medians of stages need not sum to the median total.}',
        r'\label{fig:balanced-scaling}\end{figure}',
        r'The adaptive SOCP stage improves some bounds at an additional cost.',
        r'The full totals prevent that improvement from being mistaken for a',
        r'reduction in the cost of obtaining witnesses and a valid global upper risk.']
    lines += [r'\subsection{Full-Dimensional Controls with Exact Risks}\label{sec:balanced-controls}',
        r'The folded-ball family in \eqref{eq:folded-family}, with radius one, gives',
        r'global reference values $\mathcal P=1$ and $\mathcal R=1-1/(2p)$.',
        r'We rotate its $2p$ action coordinates by a dense orthogonal matrix.',
        r'The target inequalities then couple the displayed action coordinates,',
        r'while every singular value of the latent map is $\sqrt2$.',
        r'Two rotations at each $p=2,4,8,16,32$ give ten positive controls.',
        r'Removing the target inequalities gives ten affine zero-penalty controls.',
        r'This is a structured family with a known upper-risk formula, not a',
        r'general substitute for covering a multidimensional parameter class.',
        r'The procedure queries $0,\pm e_j$ and computes their sampled anchor.',
        r'In the unrotated coordinates that anchor is nonnegative. For any such',
        r'anchor $a$ and any $\|u\|\le1$,',
        r'\[',
        r' \|\Phi(u)-a\|^2\le\|a\|^2+\max\{0,1-2\min_i a_i\}.',
        r'\]',
        r'Indeed, $a^\top\Phi(u)\ge(\min_i a_i)\|u\|_1',
        r'\ge(\min_i a_i)\|u\|$ and the remaining quadratic is maximized',
        r'at an endpoint of $[0,1]$. This supplies the global upper value;',
        r'the analytical minimax center is not supplied to the algorithm.',
        r'Table~\ref{tab:balanced-controls} reports the positive controls.',
        r'\begin{table}[htbp]\centering\small',
        r'\caption{Rotated folded-ball controls. Lower is the adaptive penalty bound.',
        r'Times include queries, the sampled anchor, the structural upper value',
        r'and separation. Each row aggregates two rotations and three timing repeats.}',
        r'\label{tab:balanced-controls}',
        r'\begin{tabular}{rrrrrr}\toprule',
        r'$p$ & Queries & True penalty & Lower & Capture (\%) & Total (ms)\\\midrule']
    for p in [2,4,8,16,32]:
        group = [row for row in rows if row['kind']=='structured' and row['p']==p and not row['affine_control']]
        lower = float(np.median([row['adaptive_gap'] for row in group]))
        total = float(1000*np.median([row['adaptive_total_seconds'] for row in group]))
        fraction = 100*lower/(1/(2*p))
        lines.append(f'{p} & {2*p+1} & {1/(2*p):.6f} & {lower:.6f} & {fraction:.2f} & {total:.1f}'+r'\\')
        summary['structured'].append(dict(p=p,true_penalty=1/(2*p),lower=lower,captured_percent=fraction,total_ms=total))
    maximum_forward = max(row['maximum_forward_residual'] for row in report['rows'])
    maximum_dual = max(row['maximum_dual_residual'] for row in report['rows'])
    maximum_gap = max(row['maximum_distance_gap'] for row in report['rows'])
    lines += [r'\bottomrule\end{tabular}\end{table}',
        r'Every positive control is detected and every affine control returns zero.',
        r'The captured fraction follows the $1/(2p)$ face-margin limitation in',
        r'Appendix~\ref{sec:allocation-certificate}; radius adaptation does not remove it.',
        r'The contrast with the generic capacity problems isolates two different',
        r'obstacles: a shrinking geometric lower bound and a loose global upper risk.',
        f'Across all {len(report["rows"])} timed runs, the maximum forward optimality residual is {tex_sci(maximum_forward)},',
        f'the maximum separation-dual feasibility residual is {tex_sci(maximum_dual)},',
        f'and the maximum separation primal--dual difference including the backoff is {tex_sci(maximum_gap)}.',
        r'Some conic solves report an inaccurate-solution status; all warnings are',
        r'retained, and acceptance uses the recorded residual and gap checks.',
        r'These floating-point diagnostics do not replace interval validation.']
    return '\n'.join(lines)+'\n', summary


def figure(report):
    rows = condition_medians(report['rows'])
    plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(6.0,2.9),layout='constrained')
    colors = ['#2365a8','#b85916','#317650']
    pairs = [(32,2),(32,4),(32,8)]
    for (q,p),color,marker in zip(pairs,colors,['o','^','D']):
        excess, separation = [], []
        for budget in [33,129,513]:
            group = [row for row in rows if row['kind']=='capacity' and (row['q'],row['p'],row['budget'])==(q,p,budget)]
            excess.append(np.median([row['upper']-row['variance'] for row in group]))
            separation.append(np.median([row['adaptive_distance']**2 for row in group]))
        axes[0].loglog([33,129,513],excess,marker+'-',color=color,label=f'p={p}')
        axes[0].loglog([33,129,513],separation,marker+'--',color=color)
    axes[0].set_xticks([33,129,513],['33','129','513'])
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_xlabel('Forward-query budget')
    axes[0].set_ylabel('Squared error')
    axes[0].legend(fontsize=8,loc='lower left')
    axes[0].grid(alpha=.18)
    stages = ['forward_seconds','anchor_seconds','upper_seconds']
    labels = ['Queries','Anchors','Upper bound','SOCP + other']
    base = np.zeros(3)
    for stage,label,color,hatch in zip(stages+['remaining'],labels,['#5680a8','#dbb372','#72a889','#aaa5b8'],['','///','..','xx']):
        heights=[]
        for q,p in pairs:
            group = [row for row in rows if row['kind']=='capacity' and (row['q'],row['p'],row['budget'])==(q,p,513)]
            if stage=='remaining':
                values=[row['adaptive_seconds']+row['bookkeeping_seconds'] for row in group]
            else:
                values=[row[stage] for row in group]
            heights.append(1000*np.median(values))
        axes[1].bar(range(3),heights,bottom=base,color=color,label=label,width=.65,
                    hatch=hatch,edgecolor='#333333',linewidth=.4)
        base+=heights
    axes[1].set_xticks(range(3),['p=2','p=4','p=8'])
    axes[1].set_ylabel('Wall-clock time (ms)')
    axes[1].legend(fontsize=7.3,loc='lower left',bbox_to_anchor=(0.,1.02),
                   ncol=2,frameon=False,borderaxespad=0.)
    fig.savefig(ROOT/'figures/Fig3.pdf')
    fig.savefig(ROOT/'figures/Fig3.png',dpi=300)
    plt.close(fig)


if __name__=='__main__':
    report=json.loads(RESULT.read_text())
    text,summary=render(report)
    for directory in ['sections','figures','verification']:
        (ROOT/directory).mkdir(exist_ok=True)
    (ROOT/'sections/balanced_base_results.tex').write_text(text,encoding='utf-8')
    (ROOT/'verification/balanced_table_values.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    figure(report)
    print('Rendered balanced-capacity tables and Fig3 from all retained timed runs.')
