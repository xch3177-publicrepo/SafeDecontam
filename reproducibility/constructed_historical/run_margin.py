#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复现:MLP margin 校准实验(饱和修复)。10 种子 + math 本地 + 家族本地,写回 v2_results.json。"""
import json, os, os, hashlib
import numpy as np, pandas as pd
import safedecontam as sd
from run_v2 import load_pairs, calibrated

SEED = 42579
SEEDS = [SEED] + [int(hashlib.sha256(f"SafeDecontam-real-v2-{i}".encode()).hexdigest()[:8], 16) % 100000
                  for i in range(1, 10)]

def main():
    pairs = load_pairs()
    res = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + "/v2_results.json"))
    sens, math_sens, primary = [], [], {}
    for s in SEEDS:
        prep = sd.prepare(pairs, seed=s)
        zc, zt = sd.variant_scores(prep, "mlp_margin")
        r = calibrated(prep, zc, zt, ["arithmetic", "mcq"], 0.05, ci=(s == SEED))
        rm = calibrated(prep, zc, zt, ["math"], 0.05)
        sens.append(dict(seed=s, CR=r["test"]["CR"], GCDR=r["test"]["GCDR"]))
        math_sens.append(dict(seed=s, CR=rm["test"]["CR"], GCDR=rm["test"]["GCDR"]))
        if s == SEED:
            r10 = calibrated(prep, zc, zt, ["arithmetic", "mcq"], 0.10)
            primary = dict(eps5=r, eps10={"tau": r10["tau"], "test": r10["test"]}, math_local=rm)
            np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + "/scores_margin.npz", cal=zc, test=zt)
            fl = {}
            for fam in ["arithmetic", "mcq", "math"]:
                rr = calibrated(prep, zc, zt, [fam], 0.05)
                fl[fam] = dict(tau=rr["tau"], CR=rr["test"]["CR"], CDR=rr["test"]["CDR"],
                               GCDR=rr["test"]["GCDR"], U95=rr["test"]["U95_worst"])
            res.setdefault("family_local", {})["mlp_margin"] = fl
        print(f"seed {s}: global {r['test']['CR']:.2f} | math {rm['test']['CR']:.2f}")
    res["variants"]["mlp_margin"] = {"eps5": primary["eps5"], "eps10": primary["eps10"]}
    res["math_local"]["mlp_margin"] = primary["math_local"]
    res["sensitivity"]["mlp_margin"] = sens
    res["sensitivity_math_margin"] = math_sens
    cr = np.array([x["CR"] for x in sens])
    print(f"10 种子: {cr.mean():.2f}±{cr.std():.2f} [{cr.min():.2f},{cr.max():.2f}]")
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + "/v2_results.json", "w"), indent=1, default=str)

if __name__ == "__main__":
    main()
