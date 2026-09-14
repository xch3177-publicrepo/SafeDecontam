#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 全量实验:三个真实家族。
主校准 = {arithmetic, mcq} 两家族全局(H=2,Bonferroni α/2);math 家族(253 组)按精确
上界的样本量可行性,采用家族本地校准(H=1)并参与 3×3 阈值迁移。适配器的精确一致指示
特征只依据校准分区决定(核对 cal CCR),锁定测试各变体只评一次。"""
import json, os, os, time, hashlib
import numpy as np, pandas as pd
import safedecontam as sd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("SAFEDECONTAM_DATA", "/home/claude/work/论文数据集")
SEED = 42579
SEEDS = [SEED] + [int(hashlib.sha256(f"SafeDecontam-real-v2-{i}".encode()).hexdigest()[:8], 16) % 100000
                  for i in range(1, 10)]

def load_pairs():
    dfs = []
    for f in ["pairs_arith_v2.csv", "pairs_mcq_v2.csv", "pairs_math_v2.csv"]:
        dfs.append(pd.read_csv(f"{DATA}/{f}"))
    p = pd.concat(dfs, ignore_index=True)
    p["benchmark_answer"] = p["benchmark_answer"].fillna("")
    bad = p.benchmark_text.isna() | p.candidate_text.isna() | (p.candidate_text.astype(str).str.strip() == "")
    if bad.sum(): print("剔除空文本:", int(bad.sum()))
    return p[~bad].reset_index(drop=True)

def split_z(prep, part, z, fams):
    fr = prep["frames"][part]
    m = fr.family.isin(fams).values
    return fr[m].reset_index(drop=True), z[m]

def calibrated(prep, zc, zt, fams, eps, alpha=0.05, ci=False):
    calf, zcf = split_z(prep, "cal", zc, fams)
    tef, ztf = split_z(prep, "test", zt, fams)
    tau = sd.select_threshold(calf, zcf, eps, alpha)
    r = dict(tau=float(tau), eps=eps, cal=sd.eval_at(calf, zcf, tau), test=sd.eval_at(tef, ztf, tau))
    if ci: r["test_CR_ci95"] = sd.bootstrap_cr_ci(tef, ztf, tau)
    for fam in fams:
        tf2, zf2 = split_z(prep, "test", zt, [fam])
        r[f"test_{fam}"] = sd.eval_at(tf2, zf2, tau)
    return r

def breakdown(prep, zt, tau, fams):
    fr, z = split_z(prep, "test", zt, fams)
    fr = fr.assign(rm=z >= tau)
    def flat(g): return {"|".join(map(str, k)): int(v) for k, v in g.items()}
    miss = fr[(fr.decision_label == "remove") & (~fr.rm)]
    fp = fr[(fr.decision_label == "retain") & (fr.rm)]
    return dict(missed_by=flat(miss.groupby(["family","relation","candidate_source"]).size().to_dict()),
                false_removed_by=flat(fp.groupby(["family","relation","candidate_source"]).size().to_dict()),
                n_miss=int(len(miss)), n_fp=int(len(fp)))

def main():
    pairs = load_pairs()
    print("配对总数:", len(pairs), pairs.family.value_counts().to_dict())
    t0 = time.time(); prep = sd.prepare(pairs, seed=SEED)
    print(f"prepare {time.time()-t0:.0f}s; splits={prep['splits']}")
    res = {"seed": SEED, "splits": {k: list(v) for k, v in prep["splits"].items()}, "variants": {}}

    # ---- 适配器 A/B(只看校准分区)----
    ab = {}
    for v in ["core", "core_noind"]:
        zc, zt = sd.variant_scores(prep, v)
        calf, zcf = split_z(prep, "cal", zc, ["arithmetic", "mcq"])
        tau = sd.select_threshold(calf, zcf, 0.05)
        ab[v] = dict(tau=float(tau), cal=sd.eval_at(calf, zcf, tau), z=(zc, zt))
        print(f"[适配器A/B|仅校准] {v:11s} cal CR={ab[v]['cal']['CR']:.2f} 损组={ab[v]['cal']['damaged_groups']}")
    core_variant = "core" if (ab["core"]["cal"]["CR"], -ab["core"]["cal"]["damaged_groups"]) >= \
                            (ab["core_noind"]["cal"]["CR"], -ab["core_noind"]["cal"]["damaged_groups"]) else "core_noind"
    res["adapter_choice"] = dict(chosen=core_variant,
                                 cal_core=ab["core"]["cal"], cal_core_noind=ab["core_noind"]["cal"])
    print("锁定适配器:", core_variant)

    # ---- 主表:全部变体(全局 H=2)----
    variants = [core_variant, "mlp", "no_constraints", "no_chronology",
                "ngram4", "shingle3", "bm25n", "lsa", "fusion"]
    zs = {}
    for v in variants:
        zc, zt = ab[v]["z"] if v in ab else sd.variant_scores(prep, v)
        zs[v] = (zc, zt)
        res["variants"][v] = {}
        for eps in [0.05, 0.10]:
            res["variants"][v][f"eps{int(eps*100)}"] = calibrated(
                prep, zc, zt, ["arithmetic", "mcq"], eps, ci=(v in (core_variant, "mlp")))
        r5 = res["variants"][v]["eps5"]
        print(f"{v:14s} tau={r5['tau']:.4f} | test CR={r5['test']['CR']:6.2f} "
              f"CDR={r5['test']['CDR']:.3f} GCDR={r5['test']['GCDR']:.3f} U95={r5['test']['U95_worst']:.2f}")

    # ---- math 家族(本地校准 H=1)----
    res["math_local"] = {}
    for v in [core_variant, "mlp", "no_constraints", "fusion", "ngram4"]:
        zc, zt = zs.get(v) or sd.variant_scores(prep, v)
        res["math_local"][v] = calibrated(prep, zc, zt, ["math"], 0.05, ci=(v == core_variant))
    rm = res["math_local"][core_variant]
    print(f"math 本地: core CR={rm['test']['CR']:.2f} GCDR={rm['test']['GCDR']:.3f} tau={rm['tau']:.4f}")

    # ---- 逐关系误差 ----
    for v in [core_variant, "no_constraints", "mlp"]:
        res["variants"][v]["eps5"]["breakdown"] = breakdown(
            prep, zs[v][1], res["variants"][v]["eps5"]["tau"], ["arithmetic", "mcq"])
    res["math_local"][core_variant]["breakdown"] = breakdown(
        prep, zs[core_variant][1], res["math_local"][core_variant]["tau"], ["math"])

    # ---- 3×3 阈值迁移(固定全局模型,家族本地阈值)----
    fams = ["arithmetic", "mcq", "math"]
    zc, zt = zs[core_variant]
    taus, local_cr = {}, {}
    for f in fams:
        calf, zcf = split_z(prep, "cal", zc, [f])
        taus[f] = sd.select_threshold(calf, zcf, 0.05)
        tf, ztf = split_z(prep, "test", zt, [f])
        local_cr[f] = sd.eval_at(tf, ztf, taus[f])["CR"]
    transfer = {}
    for src in fams:
        for tgt in fams:
            tf, ztf = split_z(prep, "test", zt, [tgt])
            e = sd.eval_at(tf, ztf, taus[src])
            transfer[f"{src}->{tgt}"] = dict(CR=e["CR"], CDR=e["CDR"], GCDR=e["GCDR"],
                                             regret=max(0.0, local_cr[tgt] - e["CR"]))
    res["transfer"] = dict(taus={k: float(v) for k, v in taus.items()}, local_CR=local_cr, matrix=transfer)
    worst = max(transfer.values(), key=lambda d: d["regret"])
    print("迁移最大 regret:", round(worst["regret"], 2))

    # ---- GSM1k-50 新鲜干净题保留率(部署式检查,不入校准)----
    import csv as _csv
    g50 = list(_csv.DictReader(open(os.environ.get("SAFEDECONTAM_DATA", "/home/claude/work/论文数据集") + "/new_2024_2026/gsm1k_public_50.csv", encoding="utf-8")))
    te_ar = prep["frames"]["test"].query("family=='arithmetic'").drop_duplicates("group_id")
    btexts = te_ar.benchmark_text.tolist()
    rows = []
    for d in g50:
        for b in btexts:
            rows.append(dict(benchmark_text=b, candidate_text=d["question"], benchmark_answer="",
                             candidate_date="2024-05-01", benchmark_release="2021-10-27",
                             family="arithmetic", group_id="g50", decision_label="retain", relation="fresh"))
    gdf = pd.DataFrame(rows)
    Xs = prep["sim"].batch(list(gdf.benchmark_text), list(gdf.candidate_text))
    Xc = sd.constraint_frame(gdf).reset_index(drop=True)
    Xg = pd.concat([Xs, Xc], axis=1)
    for c in ["a_num", "a_rel", "a_tgt", "a_ent"]: Xg[c + "_x"] = (Xg[c] >= 0.999).astype(float)
    cols = sd.SIM_COLS + sd.CON_COLS + (sd.EXACT_COLS if core_variant == "core" else []) + sd.CHRONO
    zg, _, _ = None, None, None
    zfit, clf, scaler = sd.fit_score(prep, cols)  # 同一模型系数
    zg = clf.predict_proba(scaler.transform(Xg[cols]))[:, 1].reshape(len(g50), len(btexts)).max(1)
    tau_main = res["variants"][core_variant]["eps5"]["tau"]
    res["gsm1k_fresh_retention"] = dict(n=50, deleted=int((zg >= tau_main).sum()),
                                        retained_pct=100 * float((zg < tau_main).mean()))
    print("GSM1k-50 保留率:", res["gsm1k_fresh_retention"]["retained_pct"], "%")

    # ---- 10 种子敏感性 ----
    sens = {v: [] for v in [core_variant, "mlp", "no_constraints", "fusion", "ngram4"]}
    sens_math = []
    for si, s in enumerate(SEEDS):
        pr = prep if s == SEED else sd.prepare(pairs, seed=s)
        for v in sens:
            zc2, zt2 = (zs[v] if s == SEED else sd.variant_scores(pr, v))
            r = calibrated(pr, zc2, zt2, ["arithmetic", "mcq"], 0.05)
            sens[v].append(dict(seed=s, CR=r["test"]["CR"], GCDR=r["test"]["GCDR"],
                                CDR=r["test"]["CDR"], damaged=r["test"]["damaged_groups"]))
        zc2, zt2 = (zs[core_variant] if s == SEED else sd.variant_scores(pr, core_variant))
        rm2 = calibrated(pr, zc2, zt2, ["math"], 0.05)
        sens_math.append(dict(seed=s, CR=rm2["test"]["CR"], GCDR=rm2["test"]["GCDR"]))
        print(f"seed {s} 完成 ({si+1}/10)")
    res["sensitivity"] = sens; res["sensitivity_math_core"] = sens_math

    # ---- 保存 ----
    np.savez(os.path.join(OUT, "scores_v2.npz"),
             **{f"{v}_{p}": z for v, (zc2, zt2) in zs.items() for p, z in [("cal", zc2), ("test", zt2)]})
    prep["frames"]["cal"].to_csv(os.path.join(OUT, "frame_cal_v2.csv"), index=False)
    prep["frames"]["test"].to_csv(os.path.join(OUT, "frame_test_v2.csv"), index=False)
    with open(os.path.join(OUT, "v2_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("\n保存完成 results/v2_results.json")

if __name__ == "__main__":
    main()
