"""Render the multidimensional study from its retained paired measurements."""
from pathlib import Path
import json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    report=json.loads((ROOT/'results/anisotropic_controls/runs.json').read_text())
    assert not report['failures'] and len(report['cases'])==135
    rows=[]
    for p in [2,4,8]:
        cells=[]
        for eta in [1e-4,1e-3,1e-2,.1,1.]:
            cases=[c for c in report['cases'] if c['p']==p and c['eta']==eta]
            measurements=[next(m for m in c['measurements'] if m['queries']==129) for c in cases]
            assert len(cases)==9
            cells.append(str(sum(m['adaptive_positive'] for m in measurements))+'/9')
        measured=[next(m for m in c['measurements'] if m['queries']==129) for c in report['cases'] if c['p']==p]
        check=1000*np.median([m['adaptive_seconds'] for m in measured])
        total=1000*np.median([m['adaptive_total_seconds'] for m in measured])
        rows.append(str(p)+' & '+' & '.join(cells)+f' & {check:.1f} & {total:.1f}'+r'\\')
    totals={n:sum(m['adaptive_positive'] for c in report['cases'] for m in c['measurements'] if m['queries']==n) for n in [9,33,129]}
    assert totals[9]==totals[33]==0
    positive=sum(c['reference_penalty_lower']>1e-7 for c in report['cases'])
    last=[m for c in report['cases'] for m in c['measurements'] if m['queries']==129]
    forward=1000*np.median([m['forward_seconds'] for m in last])
    anchor=1000*np.median([m['anchor_seconds'] for m in last])
    upper_time=1000*np.median([m['upper_bound_seconds'] for m in last])
    text=r'''\subsection{Multidimensional Inverse Classes and Separated Costs}\label{sec:multidimensional}
We add independent latent directions to the coupled-capacity instances,
retaining their scalar central images for validation. For $p=2,4,8$ and
$q=8,16,32$, choose $W_{\rm n}\in\R^{q\times(p-1)}$ with orthonormal columns
orthogonal to the unit direction $d$. For nuisance radius
$\epsilon_{\rm n}\in\{10^{-4},10^{-3},10^{-2},10^{-1},1\}$, let
\[
 V=[-1,1]^p,\quad
 G=[2d,\epsilon_{\rm n}W_{\rm n}/\sqrt{p-1}],\quad
 \Phi(v)=\proj_{\mathcal X}(c+2d+Gv).
\]
The matrix $G$ has rank $p$. A source box $[0,.3]^q$ with observed upper
corner and latent $.3\mathbf1_q+|G|\mathbf1_p+Gv$, where $|G|$ is entrywise
absolute value, makes every parameter exactly source-consistent. Three
seeds per action dimension give nine base contexts and 135 paired conditions.

For $n_{\rm q}=9,33,129$, uniform queries on the central segment
$v=(s,0,\ldots,0)$, $s\in[-1,1]$, give contact witnesses and an anchor $x$.
Every response is within $2/(n_{\rm q}-1)+\epsilon_{\rm n}$ of a queried
response by nonexpansiveness. Hence a valid global upper value is
\[
 U=\left(\max_j\norm{x-\Phi(v_j)}+
       \frac{2}{n_{\rm q}-1}+\epsilon_{\rm n}\right)^2.
\]
The box support function, a sum of coordinatewise maxima, gives the
primal/dual convex separation problems; no response partition is supplied.
Query, anchor and upper-value costs are separated from the original and
adaptive checks. The query and anchor measurements are shared across
matched nuisance conditions.

For validation only, recover the scalar central image $Y_0$ and put
$P_0:=\mathcal P(Y_0)$, $R_0:=\mathcal R(Y_0)$. Since $Y_0\subseteq Y$ and
every point of $Y$ is within $\epsilon_{\rm n}$ of $Y_0$, the same radius
argument as in \eqref{eq:budget-persistence} gives
\begin{equation*}
 \begin{split}
 (\sqrt{P_0}-\epsilon_{\rm n})_+^2-(\sqrt{R_0}+\epsilon_{\rm n})^2
 &\le\mathcal P(Y)-\mathcal R(Y),\\
 \mathcal P(Y)-\mathcal R(Y)
 &\le(\sqrt{P_0}+\epsilon_{\rm n})^2-R_0.
 \end{split}
\end{equation*}
The lower envelope is positive in @@POSITIVE@@ of the 135 conditions. These reference
envelopes and independent forward probes check the reported values; they
are not inputs to either certificate.

Table~\ref{tab:multidimensional} reports 129-query detections and costs.
The original check detects none of the 135 conditions, whereas adaptation
detects @@HITS@@. Both detect none at 9 or 33 queries, despite the positive reference
envelopes: an insufficiently tight upper value can erase the useful distance
bound. At 129 queries the median forward and anchor costs are @@FORWARD@@
and @@ANCHOR@@ ms respectively; the median cost of forming $U$ is @@UPPER@@ ms.

\begin{table}[htbp]\centering\small
\caption{Multidimensional boxes at 129 queries. Detections are out of nine
base contexts for each nuisance radius. Check and total times are median
milliseconds over 45 paired conditions per parameter dimension; total
includes queries, the anchor, the upper value and the adaptive check.}
\label{tab:multidimensional}
\begin{tabular}{rrrrrrrr}\toprule
$p$ & $10^{-4}$ & $10^{-3}$ & $10^{-2}$ & $10^{-1}$ & $1$ & Check & Total\\\midrule
@@ROWS@@
\bottomrule\end{tabular}
\end{table}

The benchmark retains a distinguished scalar direction, with nuisance
radii spanning four orders of magnitude. It tests certificate validity,
sensitivity and preparation cost;
it does not compare against full multidimensional minimax recovery or
establish performance for isotropic uncertainty. The preparation cost here
exceeds the convex checking cost, so a fast residual solve alone would not
establish an efficient end-to-end prediction method. A zero lower value
does not establish zero penalty, especially where the reference envelope
is inconclusive at the larger nuisance radii.
'''
    text=text.replace('@@POSITIVE@@',str(positive)).replace('@@HITS@@',str(totals[129]))
    text=text.replace('@@FORWARD@@',f'{forward:.1f}').replace('@@ANCHOR@@',f'{anchor:.1f}').replace('@@ROWS@@','\n'.join(rows))
    text=text.replace('@@UPPER@@',f'{upper_time:.3f}')
    (ROOT.parent/'sections/multidimensional_submission.tex').write_text(text,encoding='utf-8')
    print(json.dumps(dict(conditions=135,adaptive_detections=totals,rows=rows),indent=2))


if __name__=='__main__':
    main()
