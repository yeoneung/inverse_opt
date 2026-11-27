"""Render the computational controls from the supplied numerical and exact data."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
REPORT_ROOT=ROOT.parent/'verification'/'rendered_outputs'
for folder in ['sections','figures']:
    (REPORT_ROOT/folder).mkdir(parents=True,exist_ok=True)


def main():
    folder=ROOT/'results/computational_separation'
    summary=json.loads((folder/'summary.json').read_text())
    controls=json.loads((folder/'exact_controls.json').read_text())
    assert [row['proper_risk'] for row in controls]==['13/2','125/36','649/72']
    text=r'''\subsection{Computational Separation Controls}\label{sec:complexity-experiment}
We check three analytic controls and twelve fixed-seed formulas, four each
at three, five and seven Boolean variables. Every generated variable occurs,
and $\ker A_{\rm B}$ has dimension two or three. Six generated formulas have
planted assignments; six use unrestricted positive clauses with repetitions. Independent Boolean
enumeration labels SATCOUNT instances satisfiable and UNSATCOUNT unsatisfiable;
the projections in \eqref{eq:sign-projections} reproduce every label.
The largest primal--dual gap is $1.93\times10^{-10}$, with source and box
residuals below $8.00\times10^{-12}$ and $2.06\times10^{-11}$.
The smallest unsatisfiable excess is $0.22222222>1/6$.

Rational box-face enumeration gives Table~\ref{tab:complexity-controls}
exactly. Its last row forces each variable coordinate to equal minus one
third of the reference coordinate.
\begin{table}[htbp]\centering\small
\caption{Exact computational controls. $k_{\rm B}$ includes the reference
coordinate. The mixed risk is explicit in every case.}
\label{tab:complexity-controls}
\begin{tabular}{lrrrl}\toprule
Formula & $k_{\rm B}$ & $\mathcal R$ & $\mathcal P$ & Satisfiable\\\midrule
One clause, three variables & 4 & $4$ & $13/2$ & Yes\\
One triply repeated variable & 2 & $2$ & $125/36$ & No\\
All four distinct triples & 5 & $5$ & $649/72$ & No\\
\bottomrule\end{tabular}
\end{table}
'''
    text=text.replace('UNSATCOUNT',str(summary['unsatisfiable'])).replace('SATCOUNT',str(summary['satisfiable']))
    # Check the diagnostics against stored values; the displayed upper rounding
    # remains unchanged if the renderer is rerun on the supplied data.
    assert summary['instances']==15 and summary['random_source_nullities']==[2,3]
    assert summary['max_risk_primal_dual_gap']<1.93e-10
    assert summary['max_equality_residual']<8.00e-12
    assert summary['max_box_violation']<2.06e-11
    (REPORT_ROOT/'sections/computational_experiment.tex').write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
