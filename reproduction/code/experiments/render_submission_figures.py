"""Render four readable vector figures from the retained data only.

Run this script from any working directory.  The default output is the
reproduction root's figures directory; --output selects another directory.
No optimization problem or experiment is rerun.  The JSON check report records
the exact plotted arrays, source hashes, panel mappings, and export checks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "reproduction_figure_fonts")
)

import matplotlib

matplotlib.use("Agg")
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(CODE))
from src.polygon_recovery import exact_case, rational_instances

WIDTH_IN = 119 / 25.4
BLUE, ORANGE, GREEN = "#2166ac", "#b35806", "#287550"
SOURCES = {
    "projection_path": CODE / "results/projection_path_study/runs.json",
    "scalar_capacity": CODE / "results/scalar_capacity/summary.json",
    "balanced_capacity": CODE / "results/balanced_reproduction/runs.json",
    "global_checks": CODE / "global_study/independent_result_checks.json",
    "global_summary": CODE / "global_study/validated_summary.json",
    "battery_reference": CODE / "global_study/battery_tight_reference.json",
    "battery_grid": CODE / "global_study/battery_grid_validation/runs.json",
    "rational_geometry": CODE / "src/polygon_recovery.py",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def plain(values):
    """Make saved values JSON-serializable without rounding them."""
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, np.generic):
        return values.item()
    if isinstance(values, dict):
        return {str(key): plain(value) for key, value in values.items()}
    if isinstance(values, (list, tuple)):
        return [plain(value) for value in values]
    return values


def configure():
    # Fixed physical dimensions, no tight-bbox resizing, and no superscript
    # log ticks keep all text readable at the final 119-mm manuscript width.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "figure.constrained_layout.h_pad": 0.055,
        "figure.constrained_layout.w_pad": 0.055,
        "figure.constrained_layout.hspace": 0.1,
        "figure.constrained_layout.wspace": 0.1,
    })


def panel(ax, name):
    ax.text(0, 1.035, f"({name})", transform=ax.transAxes,
            fontsize=10, ha="left", va="bottom")


def assert_line(line, x, y):
    assert np.array_equal(np.asarray(line.get_xdata()), np.asarray(x))
    assert np.array_equal(np.asarray(line.get_ydata()), np.asarray(y))


def log_ticks(ax, axis, locations, labels):
    selected = ax.xaxis if axis == "x" else ax.yaxis
    selected.set_major_locator(ticker.FixedLocator(locations))
    selected.set_major_formatter(ticker.FixedFormatter(labels))
    selected.set_minor_formatter(ticker.NullFormatter())


def condition_medians(rows):
    """Exactly the aggregation in the retained balanced-study renderer."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["kind"], row["instance"], row.get("budget", 0))].append(row)
    medians = []
    for group in grouped.values():
        row = group[0].copy()
        for key in row:
            if isinstance(row[key], (int, float)) and not isinstance(row[key], bool):
                row[key] = float(np.median([r[key] for r in group]))
        medians.append(row)
    return medians


def figure1():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.95))
    r = 0.3
    x0 = np.ones(2) / 2
    n = -np.ones(2) / np.sqrt(2)
    e = np.array([1.0, -1.0]) / np.sqrt(2)
    d = e / np.sqrt(3) + np.sqrt(2 / 3) * n
    endpoints = np.array([x0 - r * e / np.sqrt(3), x0, x0 + r * d])
    mixed = x0 + r * n / np.sqrt(6)
    single = x0 + r * d / 4
    ax = axes[0]
    triangle = np.array([[0, 0], [1, 0], [0, 1]])
    ax.add_patch(Polygon(triangle, facecolor="#eeeeee", edgecolor="#888888"))
    line, = ax.plot(endpoints[:, 0], endpoints[:, 1], color="black", lw=2,
                    label="Response image")
    assert_line(line, endpoints[:, 0], endpoints[:, 1])
    ax.add_patch(Circle(mixed, r / np.sqrt(2), fill=False, ls="--", color=BLUE))
    ax.scatter(*mixed, color=BLUE, marker="*", s=55, zorder=4, label="Mixed center")
    ax.scatter(*single, color=ORANGE, marker="o", s=25, zorder=4,
               label="Single-model center")
    ax.set(xlim=(-0.06, 1.04), ylim=(-0.06, 1.04),
           xlabel="Allocation 1", ylabel="Allocation 2")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_aspect("equal")
    geometry = {"shared_capacity": {
        "feasible_vertices": triangle, "response_vertices": endpoints,
        "mixed_center": mixed, "single_model_center": single,
        "enclosing_radius": r / np.sqrt(2),
    }}

    instance = rational_instances()[1]
    case = exact_case(instance)
    vertices = np.array(instance["vertices"], dtype=float)
    mixed = np.array(case["mixed_center"], dtype=float)
    single = np.array(case["single_center"], dtype=float)
    radius = np.sqrt(float(case["mixed"]))
    ax = axes[1]
    ax.add_patch(Polygon(vertices, facecolor="#eeeeee", edgecolor="#888888"))
    line, = ax.plot(vertices[:, 0], vertices[:, 1], color="black", lw=2)
    assert_line(line, vertices[:, 0], vertices[:, 1])
    ax.add_patch(Circle(mixed, radius, fill=False, ls="--", color=BLUE))
    ax.scatter(*mixed, color=BLUE, marker="*", s=55, zorder=4)
    ax.scatter(*single, color=ORANGE, marker="o", s=25, zorder=4)
    ax.set(xlim=(-0.48, 1.65), ylim=(-0.22, 1.86),
           xlabel="Action 1", ylabel="Action 2")
    ax.set_xticks([0, 0.5, 1, 1.5])
    ax.set_yticks([0, 0.5, 1, 1.5])
    ax.set_aspect("equal")
    geometry["global_latent_line"] = {
        "feasible_vertices": vertices, "response_vertices": vertices,
        "mixed_center": mixed, "single_model_center": single,
        "enclosing_radius": radius,
        "squared_mixed_risk_exact": str(case["mixed"]),
        "squared_single_model_risk_exact": str(case["proper"]),
    }
    for ax, name in zip(axes, "ab"):
        panel(ax, name)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.31, top=0.91, wspace=0.48)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045),
               ncol=2, columnspacing=1.2, handletextpad=0.5, frameon=False)
    return fig, {
        "panel_mapping": {"a": "shared-capacity geometry (left)",
                          "b": "global latent-line geometry (right)"},
        "sources": ["rational_geometry"], "data": geometry,
        "value_alignment": "Identical geometry and exact rational instance as the original renderer; plotted line arrays checked exactly.",
    }


def figure2():
    chain = read_json(SOURCES["projection_path"])["cases"]
    groups = read_json(SOURCES["scalar_capacity"])["summary"]
    count = np.array([case["edges"] for case in chain])
    computed = np.array([case["ratio_lower"] for case in chain])
    lower = np.array([case["theoretical_ratio_lower"] for case in chain])
    dimensions = np.array([row["dimension"] for row in groups])
    original = np.array([100 * row["median_supplied_center_fraction"] for row in groups])
    adaptive = np.array([100 * row["median_adaptive_fraction"] for row in groups])
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH_IN, 4.7), layout="constrained")
    line, = axes[0].plot(count, computed, "-o", color=BLUE, ms=4, label="Computed price")
    assert_line(line, count, computed)
    line, = axes[0].plot(count, lower, ":", color=ORANGE, lw=1.8, label="Lower bound")
    assert_line(line, count, lower)
    axes[0].axhline(4, color="black", ls="--", lw=1.2, label="Sharp supremum")
    axes[0].set(xlabel="Response segments", ylabel="Squared-risk ratio")
    axes[0].legend(loc="lower right", frameon=False)
    line, = axes[1].semilogy(dimensions, original, "-o", color=BLUE, ms=4,
                            label="Original margin")
    assert_line(line, dimensions, original)
    line, = axes[1].semilogy(dimensions, adaptive, "--s", color=ORANGE, ms=4,
                            label="Adaptive radius")
    assert_line(line, dimensions, adaptive)
    axes[1].set(xlabel="Action dimension", ylabel="Actual penalty\ncaptured (%)")
    axes[1].set_xticks(dimensions)
    log_ticks(axes[1], "y", [0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    axes[1].legend(loc="lower left", frameon=False)
    for ax, name in zip(axes, "ab"):
        panel(ax, name)
        ax.grid(alpha=0.16)
    return fig, {
        "panel_mapping": {"a": "global-line price curve (top)",
                          "b": "scalar coupled-capacity calibration (bottom)"},
        "sources": ["projection_path", "scalar_capacity"],
        "data": {"segments": count, "computed_price": computed, "lower_bound": lower,
                 "sharp_supremum": 4, "action_dimensions": dimensions,
                 "original_captured_percent": original, "adaptive_captured_percent": adaptive},
        "value_alignment": "Saved values used without rounding; all four plotted line arrays checked exactly.",
    }


def figure3():
    report = read_json(SOURCES["balanced_capacity"])
    rows = condition_medians(report["rows"])
    pairs = [(32, 2), (32, 4), (32, 8)]
    budgets = [33, 129, 513]
    colors = [BLUE, ORANGE, GREEN]
    markers = ["o", "^", "D"]
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH_IN, 5.05), layout="constrained",
                             gridspec_kw={"height_ratios": [1.15, 1]})
    curves = []
    for (q, p), color, marker in zip(pairs, colors, markers):
        excess, separation = [], []
        for budget in budgets:
            group = [row for row in rows if row["kind"] == "capacity" and
                     (row["q"], row["p"], row["budget"]) == (q, p, budget)]
            assert len(group) == 6
            excess.append(float(np.median([row["upper"] - row["variance"] for row in group])))
            separation.append(float(np.median([row["adaptive_distance"] ** 2 for row in group])))
        line, = axes[0].loglog(budgets, excess, marker + "-", color=color, ms=4)
        assert_line(line, budgets, excess)
        line, = axes[0].loglog(budgets, separation, marker + "--", color=color, ms=4)
        assert_line(line, budgets, separation)
        curves.append({"q": q, "p": p, "upper_excess": excess,
                       "squared_adaptive_separation": separation})
    log_ticks(axes[0], "x", budgets, [str(value) for value in budgets])
    log_ticks(axes[0], "y", [1e-8, 1e-6, 1e-4, 1e-2, 1],
              ["1e-8", "1e-6", "1e-4", "0.01", "1"])
    axes[0].set(xlabel="Forward-query budget", ylabel="Squared error")
    axes[0].grid(alpha=0.16)
    dimension_handles = [Line2D([0], [0], color=color, marker=marker, label=f"p = {p}")
                         for (_, p), color, marker in zip(pairs, colors, markers)]
    first_legend = axes[0].legend(handles=dimension_handles, loc="lower left", frameon=False,
                                   ncol=3, columnspacing=0.75, handlelength=1.25)
    axes[0].add_artist(first_legend)
    style_handles = [Line2D([0], [0], color="black", ls="-", label="Upper excess"),
                     Line2D([0], [0], color="black", ls="--", label="Squared separation")]
    axes[0].legend(handles=style_handles, loc="lower right", bbox_to_anchor=(1, 1.01),
                   frameon=False, borderaxespad=0)

    stage_keys = ["forward_seconds", "anchor_seconds", "upper_seconds", "remaining"]
    stage_labels = ["Queries", "Anchors", "Upper bound", "SOCP + other"]
    stage_colors = ["#5680a8", "#dbb372", "#72a889", "#aaa5b8"]
    base = np.zeros(3)
    timing = {}
    for stage, label, color, hatch in zip(stage_keys, stage_labels, stage_colors, ["", "///", "..", "xx"]):
        heights = []
        for q, p in pairs:
            group = [row for row in rows if row["kind"] == "capacity" and
                     (row["q"], row["p"], row["budget"]) == (q, p, 513)]
            values = ([row["adaptive_seconds"] + row["bookkeeping_seconds"] for row in group]
                      if stage == "remaining" else [row[stage] for row in group])
            heights.append(float(1000 * np.median(values)))
        bars = axes[1].bar(range(3), heights, bottom=base, width=0.6, color=color,
                           label=label, hatch=hatch, edgecolor="#333333", linewidth=0.5)
        assert np.array_equal(np.array([bar.get_height() for bar in bars]), heights)
        base += heights
        timing[stage] = heights
    axes[1].set_xticks(range(3), ["p = 2", "p = 4", "p = 8"])
    axes[1].set(xlabel="Latent dimension (action dimension 32)", ylabel="Wall-clock time (ms)")
    axes[1].legend(loc="upper left", frameon=False, ncol=2, columnspacing=0.9,
                   handlelength=1.5)
    axes[1].set_ylim(0, max(base) * 1.32)
    for ax, name in zip(axes, "ab"):
        panel(ax, name)
    return fig, {
        "panel_mapping": {"a": "upper excess and squared separation (top)",
                          "b": "complete median timing stages at 513 queries (bottom)"},
        "sources": ["balanced_capacity"],
        "data": {"raw_timed_rows": len(report["rows"]), "condition_median_rows": len(rows),
                 "budgets": budgets, "curves": curves, "stage_milliseconds": timing,
                 "stage_sums_not_median_total": base},
        "value_alignment": "Exactly the original condition-median aggregation, six instances per curve point; all curves and bar heights checked exactly.",
    }


def global_groups():
    checked = read_json(SOURCES["global_checks"])
    assert checked["passed"] and checked["compact_p8_complete"]
    groups = {
        "capacity_q16_p4": [row for row in checked["rows"] if row["label"].startswith("p4validation_q16_")],
        "capacity_q32_p4": [row for row in checked["rows"] if row["label"].startswith("p4validation_q32_")],
        "capacity_q12_p8": [row for row in checked["rows"] if row["label"].startswith("compact_capacity_extended/")],
        "battery_q8_p4": [row for row in checked["rows"] if row["label"].startswith("battery_grid_")],
    }
    summary = read_json(SOURCES["global_summary"])
    for key, rows in groups.items():
        assert len(rows) == summary[key]["cases"]
        assert sum(row["detected"] for row in rows) == summary[key]["positive"]
        assert float(np.median([row["pipeline_and_repair_seconds"] for row in rows])) == summary[key]["median_total_seconds_including_final_validation"]
    return groups, summary


def figure4():
    groups, summary = global_groups()
    keys = list(groups)[:3]
    fractions = [summary[key]["positive"] / summary[key]["cases"] for key in keys]
    medians = [summary[key]["median_total_seconds_including_final_validation"] for key in keys]
    labels = ["q=16\np=4", "q=32\np=4", "q=12\np=8"]
    fig = plt.figure(figsize=(WIDTH_IN, 4.95), layout="constrained")
    grid_layout = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    axes = [fig.add_subplot(grid_layout[0, 0]), fig.add_subplot(grid_layout[0, 1]),
            fig.add_subplot(grid_layout[1, :])]
    bars = axes[0].bar(range(3), fractions, color=[BLUE, BLUE, ORANGE],
                        hatch=["", "//", ".."], edgecolor="#333333", linewidth=0.5)
    assert np.array_equal([bar.get_height() for bar in bars], fractions)
    for index, key in enumerate(keys):
        axes[0].text(index, fractions[index] + 0.035,
                     f"{summary[key]['positive']}/{summary[key]['cases']}", ha="center", fontsize=9)
    axes[0].set_xticks(range(3), labels)
    axes[0].set(ylim=(0, 1.2), ylabel="Positive-certificate\nfraction")
    axes[0].set_yticks([0, 0.5, 1])
    bars = axes[1].bar(range(3), medians, color=[BLUE, BLUE, ORANGE],
                        hatch=["", "//", ".."], edgecolor="#333333", linewidth=0.5)
    assert np.array_equal([bar.get_height() for bar in bars], medians)
    axes[1].set_xticks(range(3), labels)
    axes[1].set(yscale="log", ylim=(1, 130), ylabel="Complete median\nwall time (s)")
    log_ticks(axes[1], "y", [1, 10, 100], ["1", "10", "100"])

    grid = read_json(SOURCES["battery_grid"])["rows"]
    checks = {row["label"].rsplit("/", 1)[-1]: row for row in groups["battery_q8_p4"]}
    powers, reserves = [1, 2, 4], [4, 8, 12]
    values = np.array([
        [checks[next(row["label"] for row in grid if row["target_power_kw"] == power
                     and row["reserve_amplitude_kwh"] == reserve)]["validated_certificate"]
         for reserve in reserves] for power in powers
    ])
    image = axes[2].imshow(values, origin="lower", aspect="auto", cmap="Blues", vmin=0)
    assert np.array_equal(np.asarray(image.get_array()), values)
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, f"{values[i, j]:.2g}", ha="center", va="center",
                         color="white" if values[i, j] > 0.0007 else "black", fontsize=9)
    axes[2].set_xticks(range(3), [str(value) for value in reserves])
    axes[2].set_yticks(range(3), [str(value) for value in powers])
    axes[2].set(xlabel="Reserve-band half-width (kWh)", ylabel="Power limit (kW)")
    for ax, name in zip(axes, "abc"):
        panel(ax, name)
    return fig, {
        "panel_mapping": {"a": "global-exchange detections (top left)",
                          "b": "complete median times including validation (top right)",
                          "c": "all nine battery lower bounds (bottom)"},
        "sources": ["global_checks", "global_summary", "battery_grid"],
        "data": {"capacity_cohorts": keys, "detection_numerators": [summary[key]["positive"] for key in keys],
                 "detection_denominators": [summary[key]["cases"] for key in keys],
                 "detection_fractions": fractions, "median_complete_seconds": medians,
                 "heatmap_rows_power_kw": powers, "heatmap_columns_reserve_kwh": reserves,
                 "heatmap_lower_bounds": values, "heatmap_display_precision": ".2g"},
        "value_alignment": "All cohort counts and medians independently reconstructed from retained checks and matched exactly to saved summary; every bar height and all nine heatmap values checked exactly.",
    }


def figure5():
    tight = read_json(SOURCES["battery_reference"])["rows"][0]
    assert tight["label"] == "battery_corner_q8_p4_cosine"
    hours = np.arange(9) * 3
    fig = plt.figure(figsize=(WIDTH_IN, 4.85), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]
    schedules = {}
    for key, label, color, style, marker in [
            ("mixed_physical_schedule", "Mixed center", BLUE, "-", "o"),
            ("proper_physical_schedule", "One objective", ORANGE, "--", "s")]:
        schedule = tight[key]
        power = np.array(schedule["power_kw"])
        state = np.r_[16, schedule["soc_kwh"]]
        stairs = axes[0].stairs(power, hours, label=label, color=color, linestyle=style, lw=1.6)
        assert np.array_equal(stairs.get_data().values, power)
        assert np.array_equal(stairs.get_data().edges, hours)
        line, = axes[1].plot(hours, state, color=color, linestyle=style, marker=marker,
                            ms=3, label=label, lw=1.6)
        assert_line(line, hours, state)
        schedules[key] = {"power_kw": power, "soc_kwh_including_initial": state}
    axes[0].set(xlabel="Hour", ylabel="Battery power (kW)")
    axes[1].axhline(4, color="#777777", ls=":", lw=1.1)
    axes[1].axhline(32, color="#777777", ls=":", lw=1.1)
    axes[1].set(xlabel="Hour", ylabel="Stored energy (kWh)", ylim=(2, 34))
    for ax in axes[:2]:
        ax.set_xticks([0, 12, 24])
    axes[1].set_yticks([4, 16, 32])
    risk_intervals = []
    for y, key, color, marker in [(0, "mixed", BLUE, "o"), (1, "proper", ORANGE, "s")]:
        lo, hi = tight[f"{key}_risk_lower"], tight[f"{key}_risk_upper"]
        midpoint = (lo + hi) / 2
        artist = axes[2].errorbar(midpoint, y, xerr=np.array([[midpoint - lo], [hi - midpoint]]),
                                  fmt=marker, capsize=6, color=color, ms=5, lw=1.4)
        assert_line(artist.lines[0], [midpoint], [y])
        risk_intervals.append({"kind": key, "lower": lo, "upper": hi, "midpoint": midpoint})
    axes[2].set_yticks([0, 1], ["Mixed center", "One objective"])
    axes[2].set(xlabel="Worst squared error in Q metric", ylim=(-0.65, 1.65))
    axes[2].xaxis.set_major_locator(ticker.MaxNLocator(4))
    axes[2].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    axes[2].grid(axis="x", alpha=0.16)
    # These two previously displayed summary values are retained as a compact
    # annotation; the exact intervals are recorded below and are not rounded.
    axes[2].text(0.5, 0.5, "Risk ratio: 1.01345\nCertified excess captured: 43.18%",
                 transform=axes[2].transAxes, fontsize=9, ha="center", va="center")
    for ax, name in zip(axes, "abc"):
        panel(ax, name)
    handles, labels = axes[0].get_legend_handles_labels()
    # A figure legend between the two rows would compete with panel labels;
    # reserve an independent strip above the physical schedules instead.
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False,
               columnspacing=1.1, handlelength=2.3)
    return fig, {
        "panel_mapping": {"a": "24-hour power schedules (top left)",
                          "b": "stored-energy trajectories (top right)",
                          "c": "independent global risk intervals (bottom)"},
        "sources": ["battery_reference"],
        "data": {"reference_case": tight["label"], "hours": hours, "schedules": schedules,
                 "state_bounds_kwh": [4, 32], "global_risk_intervals": risk_intervals,
                 "squared_risk_ratio_interval": tight["squared_risk_ratio_interval"],
                 "captured_fraction_interval": tight["captured_fraction_interval"],
                 "displayed_ratio": "1.01345", "displayed_captured_percent": "43.18"},
        "value_alignment": "All eight power values, nine state points per schedule and exact risk endpoints copied from the same retained reference; all plotted arrays checked exactly.",
    }


def inspect_pdf(path):
    """Optional PDF-level checks; rendering itself needs only Matplotlib/NumPy."""
    try:
        import pymupdf
    except ImportError:
        return {"inspection": "PyMuPDF unavailable; use the recorded Matplotlib font sizes."}
    document = pymupdf.open(path)
    page = document[0]
    spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
             for line in block["lines"] for span in line["spans"] if span["text"].strip()]
    minimum = min(span["size"] for span in spans)
    fonts = []
    for font in page.get_fonts(full=True):
        extracted = document.extract_font(font[0])
        fonts.append({"base_font": font[3], "subtype": font[2],
                      "embedded_bytes": len(extracted[3])})
    assert minimum >= 8.5 - 0.01, (path.name, minimum)
    assert all(font["embedded_bytes"] > 0 for font in fonts)
    assert abs(page.rect.width - WIDTH_IN * 72) < 0.01
    assert all(0 <= span["bbox"][0] < span["bbox"][2] <= page.rect.width + 0.05
               and 0 <= span["bbox"][1] < span["bbox"][3] <= page.rect.height + 0.05
               for span in spans), path.name
    result = {"width_points": page.rect.width, "height_points": page.rect.height,
              "minimum_text_span_points": minimum,
              "font_sizes_points": sorted({round(span["size"], 4) for span in spans}),
              "fonts": fonts, "all_fonts_embedded": True, "all_text_inside_media_box": True}
    document.close()
    return result


def save(fig, info, name, directory):
    fig.canvas.draw()
    labels = [item for item in fig.findobj(Text) if item.get_text().strip() and item.get_visible()]
    minimum = min(item.get_fontsize() for item in labels)
    assert minimum >= 8.5, (name, minimum)
    assert all(not ax.get_title() for ax in fig.axes)
    before = {ext: digest(directory / f"{name}.{ext}") if (directory / f"{name}.{ext}").exists()
              else None for ext in ["pdf", "png"]}
    for extension in ["pdf", "png"]:
        fig.savefig(directory / f"{name}.{extension}", dpi=300,
                    metadata={"Creator": "Matplotlib", "CreationDate": None,
                              "ModDate": None} if extension == "pdf" else None)
    info.update({"physical_width_mm": 119, "physical_height_inches": fig.get_figheight(),
                 "minimum_matplotlib_font_points": minimum, "internal_panel_titles": [],
                 "grayscale_distinction": "Line styles and markers, hatched timing bars, or directly labelled heatmap cells supplement color.",
                 "previous_file_sha256": before,
                 "output_sha256": {ext: digest(directory / f"{name}.{ext}") for ext in ["pdf", "png"]},
                 "pdf_export": inspect_pdf(directory / f"{name}.pdf")})
    plt.close(fig)
    return plain(info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument("--checks", type=Path,
                        default=ROOT / "verification/length_revision_20261128/figure_checks.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.checks.parent.mkdir(parents=True, exist_ok=True)
    previous_checks = read_json(args.checks) if args.checks.exists() else {}
    configure()
    checks = {"method": "Existing data and deterministic geometry; no experiments rerun.",
              "renderer": "code/experiments/render_submission_figures.py",
              "source_files": {key: {"relative_path": path.relative_to(ROOT).as_posix(),
                                     "sha256": digest(path)} for key, path in SOURCES.items()},
              "figures": {}}
    # The global overview duplicates the capacity and battery tables; retain
    # its rendering function for optional reuse, outside the publication set.
    for index, render in enumerate([figure1, figure2, figure3, figure5], 1):
        fig, info = render()
        name = f"Fig{index}"
        checks["figures"][name] = save(fig, info, name, args.output)
        # Preserve the first pre-revision hashes when adjusting a layout.
        previous = previous_checks.get("figures", {}).get(name, {})
        if previous.get("previous_file_sha256"):
            checks["figures"][name]["previous_file_sha256"] = previous["previous_file_sha256"]
    checks["source_files_unchanged"] = all(digest(SOURCES[key]) == value["sha256"]
                                           for key, value in checks["source_files"].items())
    assert checks["source_files_unchanged"]
    # A visual review remains valid only while every reviewed PNG is identical.
    if previous_checks.get("visual_inspection") and all(
            checks["figures"][name]["output_sha256"]["png"] ==
            previous_checks.get("figures", {}).get(name, {}).get("output_sha256", {}).get("png")
            for name in checks["figures"]):
        checks["visual_inspection"] = previous_checks["visual_inspection"]
    args.checks.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print("Rendered Fig1--Fig4 at 119 mm with unchanged data; saved figure checks.")


if __name__ == "__main__":
    main()
