"""Render the added curvature study from stored data into a separate check folder."""
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
style_spec = importlib.util.spec_from_file_location('publication_style', Path(__file__).with_name('publication_math_style.py'))
style = importlib.util.module_from_spec(style_spec)
style_spec.loader.exec_module(style)


def render():
    folder = ROOT / 'results/smooth_curvature'
    runs = json.loads((folder / 'runs.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['total_cases'] == 37 and summary['rational_energy_identities']
    disks = {row['radius']: row for row in runs['disk_controls'] if row['height'] == 2.0}
    rows = '\n'.join(f"{row['radius']:.4f} & {row['scaled_penalty']:.8f} & "
                     f"{disks[row['radius']]['scaled_penalty']:.8f}\\\\"
                     for row in runs['smooth_curve'])
    text = r'''\subsection{Local Curvature and HJB Controls}\label{sec:curvature-experiment}
We check the local formulas on an asymmetric smooth response and two
families of inverse quadratic decisions with curved targets.
For $\gamma(t)=(t+0.4t^2,0.7t^2+0.3t^3)$, normal acceleration gives
coefficient $0.49$ in \eqref{eq:curvature-law}.
Bounds valid over the whole interval certify the finite-radius condition:
take $M_\gamma\le\norm{(0.8,1.4)}+1.8r$ and bound speed below and above by
$1-r(\norm{(0.8,1.4)}+1.8r)$ and
$1+r(\norm{(0.8,1.4)}+1.8r)$.
We solve the balanced-center equation in \eqref{eq:curve-exact} and compare
it with a separate bounded minimization over 4,001 sampled responses.
Their largest risk disagreement is below $10^{-10}$.

For the unit disk and preferred points $(t,h_{\rm d})$, $h_{\rm d}>1$,
the potential $U_X(z,1)=(\norm z-1)^2/2$ is smooth near $(0,h_{\rm d})$.
Here $\gamma(t)=(t,h_{\rm d})/\sqrt{h_{\rm d}^2+t^2}$, with
velocity $(1/h_{\rm d},0)$ and acceleration $(0,-1/h_{\rm d}^2)$.
The exact risks and penalty are
\[
 \mathcal R=\frac{r^2}{h_{\rm d}^2+r^2},\qquad
 \mathcal P=2\left(1-\frac{h_{\rm d}}{\sqrt{h_{\rm d}^2+r^2}}\right),
 \qquad
 \mathcal P-\mathcal R=
 \left(1-\frac{h_{\rm d}}{\sqrt{h_{\rm d}^2+r^2}}\right)^2.
\]
Thus \eqref{eq:hj-curvature} predicts coefficient $1/(4h_{\rm d}^4)$.
Twelve controls use heights $1.25,2,3$ and the four radii below.
Table~\ref{tab:curvature-scaling} shows the scaled penalties for the
asymmetric curve and height two; both approach their stated limits.

\begin{table}[htbp]\centering\small
\caption{Scaled penalties $(\mathcal P-\mathcal R)/r^4$ for local smooth
responses. The last row gives the limits predicted by the curvature law.}
\label{tab:curvature-scaling}
\begin{tabular}{rrr}\toprule
$r$ & Asymmetric curve & Disk at height two\\\midrule
ROWS
$r\downarrow0$ & $0.49$ & $1/64=0.015625$\\
\bottomrule\end{tabular}
\end{table}

Nine parabolic controls check \eqref{eq:parabolic-prices}
and \eqref{eq:parabolic-energy} in rational arithmetic, using
$\kappa=1/2,1,2$, endpoint $b_r=1/5,1/10,1/20$ and
$r=b_r+\kappa^2b_r^3/2$. Here the parameter radius is chosen from the
rational endpoint. A separate paired comparison fixes $r=0.2,0.1,0.05$
and varies only $\kappa=0,1/2,1,2$, giving twelve cases with unchanged
compatible preferred points. The flat target has zero penalty and each
curved target has positive penalty. These calculations check a local
mechanism for scalar ambiguity; extension to multidimensional classes and feedback
regularity through constraint switches remains open.
'''
    return style.apply(text.replace('ROWS', rows), 'smooth_curvature_experiment.tex')


def main():
    output = ROOT.parent / 'verification/rendered_outputs/sections'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'smooth_curvature_experiment.tex').write_text(render(), encoding='utf-8')
    print('Rendered curvature controls from 37 supplied cases.')


if __name__ == '__main__':
    main()
