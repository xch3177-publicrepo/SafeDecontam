#!/usr/bin/env python3
"""Render only recorded operating points from a complete current run."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    manifest = json.loads((args.run_dir / "run_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise SystemExit("Run is incomplete; no final figure produced.")
    source = args.run_dir / "results.json"
    r = json.loads(source.read_text())["primary"]
    series = [("core", "All-feature logistic", "#2670B6"),
              ("mlp_margin", "MLP margin", "#168466"),
              ("no_constraints", "No-constraint logistic", "#B57705"),
              ("fusion", "Similarity fusion", "#784EAA")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    points = []
    for key, label, color in series:
        for budget, marker, size in [("eps5", "o", 28), ("eps10", "^", 36)]:
            x = r["variants"][key][budget]["test"]
            ax.scatter(x["CR"], x["CRR"], color=color, marker=marker, s=size,
                       edgecolor="white", linewidth=.5, zorder=4)
            points.append({"scorer": key, "budget": budget, "recall": x["CR"],
                           "clean_retention": x["CRR"],
                           "source_key": "primary.variants." + key + "." + budget + ".test"})
    min_y = min(p["clean_retention"] for p in points)
    lower = min(97.5, min_y - .8)
    ax.set_xlim(-3, 102)
    ax.set_ylim(lower, 100.15)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([98.0, 98.5, 99.0, 99.5, 100.0])
    ax.set_xlabel("Contamination recall (%)")
    ax.set_ylabel("Clean retention (%)")
    ax.grid(color="#D9DDE2", linewidth=.55)
    ax.set_axisbelow(True)
    for name in ["top", "right"]:
        ax.spines[name].set_visible(False)
    handles = [Line2D([], [], marker="s", linestyle="none", markersize=4.7,
                      color=color, label=label) for _, label, color in series]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.0,
              handlelength=.7, handletextpad=.4, labelspacing=.23, borderpad=.05)
    ax.text(.985, .985, "● 5%     ▲ 10% budget", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.0)
    fig.tight_layout(pad=.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=400)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps({
        "source_results_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "Eight saved calibrated operating points; no curves or interpolation", "points": points,
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
