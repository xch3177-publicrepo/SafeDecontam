#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据研究的论文图(IEEE 单栏,PDF)。调色板:已验证的分类色序(blue/aqua/yellow/violet),
序数热图用单蓝色渐变,超预算单元用 critical 红描边 + 文本标注(不只靠颜色)。"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle, FancyBboxPatch

C = dict(blue="#2a78d6", aqua="#1baf7a", yellow="#eda100", violet="#4a3aa7",
         red="#e34948", critical="#d03b3b", ink="#0b0b0b", ink2="#52514e",
         muted="#898781", grid="#e1e0d9", axis="#c3c2b7", surface="#ffffff")
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "figure.dpi": 200})
_HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(_HERE, "results")
FIG = os.path.join(_HERE, "figs")
import os; os.makedirs(FIG, exist_ok=True)

res = json.load(open(f"{R}/v2_results.json"))
npz = np.load(f"{R}/scores_v2.npz")
test = pd.read_csv(f"{R}/frame_test_v2.csv")

def style_ax(ax):
    ax.grid(True, color=C["grid"], linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)

# ---------- Fig: 系统流程(重绘原 Fig1 的简版) ----------
def fig_flow():
    fig, ax = plt.subplots(figsize=(7.0, 1.5))
    ax.axis("off")
    boxes = [("Benchmark item\n+ public answer\n+ candidate + dates", C["grid"]),
             ("Train-frozen\nsimilarity layer", "#e8f1fc"),
             ("Observable constraint\nparser (evidence)", "#e8f1fc"),
             ("Auditable score\n(linear / MLP)", "#e8f1fc"),
             ("Family-stratified\nexact calibration", "#eaf7f1"),
             ("remove / retain\n/ abstain", "#fdf3e0")]
    xw = 1.0 / len(boxes)
    for i, (t, fc) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((i * xw + 0.006, 0.18), xw - 0.03, 0.62,
                     boxstyle="round,pad=0.008", fc=fc, ec=C["axis"], lw=0.8,
                     transform=ax.transAxes))
        ax.text(i * xw + xw / 2 - 0.009, 0.49, t, ha="center", va="center",
                fontsize=6.8, color=C["ink"], transform=ax.transAxes)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(i * xw + xw - 0.021, 0.49), xytext=(i * xw + xw - 0.003, 0.49),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="<-", color=C["ink2"], lw=0.9))
    ax.text(0.5, 0.02, "training fits representations + scorer;  calibration selects τ per family stratum;  the locked test only reports",
            ha="center", fontsize=6.5, color=C["ink2"], transform=ax.transAxes)
    fig.savefig(f"{FIG}/fig_flow.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig: 真实数据 threshold sweep ----------
def sweep(variant):
    m = test.family.isin(["arithmetic", "mcq"]).values
    z = npz[f"{variant}_test"][m]
    y = (test[m].decision_label == "remove").values
    taus = np.unique(z)
    cr, crr = [], []
    for t in np.concatenate([taus, [np.inf]]):
        rm = z >= t
        cr.append(rm[y].mean()); crr.append(1 - rm[~y].mean())
    return np.array(cr), np.array(crr)

def fig_sweep():
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    series = [("core", "SafeDecontam core (linear)", C["blue"]),
              ("mlp", "MLP verifier", C["aqua"]),
              ("no_constraints", "No constraints", C["yellow"]),
              ("fusion", "Similarity fusion", C["violet"])]
    for v, lab, col in series:
        cr, crr = sweep(v)
        o = np.argsort(cr)
        ax.plot(100 * cr[o], 100 * crr[o], color=col, lw=1.3, label=lab)
    ax.set_xlabel("Contamination recall (%)"); ax.set_ylabel("Clean retention (%)")
    ax.set_xlim(0, 101); ax.set_ylim(84, 100.6)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=6.4, loc="lower left")
    # 操作点标记
    for v, col in [("core", C["blue"]), ("mlp", C["aqua"])]:
        t5 = res["variants"][v]["eps5"]["test"]
        ax.scatter([t5["CR"]], [t5["CRR"]], s=16, color=col, zorder=5, edgecolor="white", lw=0.6)
    ax.annotate("5% operating points", xy=(res["variants"]["mlp"]["eps5"]["test"]["CR"],
                res["variants"]["mlp"]["eps5"]["test"]["CRR"]), xytext=(-30, -12),
                textcoords="offset points", fontsize=6.4, color=C["ink2"])
    fig.savefig(f"{FIG}/fig_sweep_real.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig: 全局 vs 家族本地 ----------
def fig_local():
    loc = res["family_local"]
    groups = ["Global\n(arith+MCQ)", "Arithmetic", "MCQ", "MATH"]
    core_v = [res["variants"]["core"]["eps5"]["test"]["CR"], loc["core"]["arithmetic"]["CR"],
              loc["core"]["mcq"]["CR"], loc["core"]["math"]["CR"]]
    prob_v = [res["variants"]["mlp"]["eps5"]["test"]["CR"], loc["mlp"]["arithmetic"]["CR"],
              loc["mlp"]["mcq"]["CR"], loc["mlp"]["math"]["CR"]]
    marg_v = [res["variants"]["mlp_margin"]["eps5"]["test"]["CR"], loc["mlp_margin"]["arithmetic"]["CR"],
              loc["mlp_margin"]["mcq"]["CR"], loc["mlp_margin"]["math"]["CR"]]
    x = np.arange(4); w = 0.26
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bars = [ax.bar(x - w, core_v, w*0.92, color=C["blue"], label="Linear core"),
            ax.bar(x,     prob_v, w*0.92, color=C["yellow"], label="MLP (probability)"),
            ax.bar(x + w, marg_v, w*0.92, color=C["aqua"], label="MLP (margin)")]
    for bs in bars:
        for r in bs:
            v = r.get_height()
            ax.text(r.get_x() + r.get_width()/2, v + 1.6, f"{v:.0f}", ha="center",
                    fontsize=5.6, color=C["ink2"])
    ax.text(3, 12, "infeasible", ha="center", fontsize=5.4, color=C["critical"], rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=6.2)
    ax.set_ylabel("Locked-test CR @ 5% budget (%)"); ax.set_ylim(0, 112)
    style_ax(ax); ax.legend(frameon=False, fontsize=6.0, loc="upper center",
                            bbox_to_anchor=(0.5, 1.18), ncols=3, columnspacing=0.8, handletextpad=0.4)
    fig.savefig(f"{FIG}/fig_local_global.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig: 迁移热图 ----------
def fig_transfer():
    fams = ["arithmetic", "mcq", "math"]
    M = res["transfer"]["matrix"]
    reg = np.array([[M[f"{s}->{t}"]["regret"] for t in fams] for s in fams])
    gcdr = np.array([[M[f"{s}->{t}"]["GCDR"] for t in fams] for s in fams])
    fig, ax = plt.subplots(figsize=(3.0, 2.55))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
    im = ax.imshow(reg, cmap=cmap, vmin=0, vmax=65)
    for i in range(3):
        for j in range(3):
            over = gcdr[i, j] > 5.0
            txt = f"{reg[i,j]:.1f}" + ("\nGCDR " + f"{gcdr[i,j]:.1f}%" if over else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.6,
                    color=("white" if reg[i, j] > 32 else C["ink"]),
                    fontweight=("bold" if over else "normal"))
            if over:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       ec=C["critical"], lw=1.6))
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["arith", "MCQ", "MATH"], fontsize=6.8)
    ax.set_yticklabels(["arith", "MCQ", "MATH"], fontsize=6.8)
    ax.set_xlabel("Target family"); ax.set_ylabel("Calibration family")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Lost recall (points)", fontsize=6.6); cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    fig.savefig(f"{FIG}/fig_transfer_real.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig: 种子稳定性 ----------
def fig_seeds():
    core = [x["CR"] for x in res["sensitivity"]["core"]]
    prob = [x["CR"] for x in res["sensitivity"]["mlp"]]
    marg = [x["CR"] for x in res["sensitivity"]["mlp_margin"]]
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    for i, (vals, col) in enumerate([(core, C["blue"]), (prob, C["yellow"]), (marg, C["aqua"])]):
        xj = i + rng.uniform(-0.07, 0.07, len(vals))
        ax.scatter(xj, vals, s=16, color=col, alpha=0.9, edgecolor="white", lw=0.5, zorder=5)
        med = np.median(vals)
        ax.hlines(med, i - 0.2, i + 0.2, color=col, lw=1.4)
        ax.text(i + 0.23, med, f"{med:.0f}", fontsize=6.2, color=C["ink2"], va="center")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Linear core", "MLP (prob.)", "MLP (margin)"], fontsize=6.6)
    ax.set_ylabel("Locked-test CR @ 5% (%)"); ax.set_xlim(-0.5, 2.6); ax.set_ylim(-6, 106)
    ax.annotate("5/10 seeds\ninfeasible", xy=(1.06, 1.5), xytext=(1.3, 20), fontsize=5.6,
                color=C["critical"], arrowprops=dict(arrowstyle="-", color=C["critical"], lw=0.6))
    style_ax(ax)
    fig.savefig(f"{FIG}/fig_seed_strip.pdf", bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig_flow(); fig_sweep(); fig_local(); fig_transfer(); fig_seeds()
    print("figures ->", FIG)
