#!/usr/bin/env python3
"""Replay fixed web-pilot pairs and a frozen-model cross-lingual probe.

Use the same unmodified historical scorer implementation and seed as the
constructed primary run. No model or parameter is selected on either probe.
The input pair CSVs are fixed archived constructions, not new annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Directory containing the three constructed CSVs, r2_wild_pairs.csv, and pairs_xling.csv")
    ap.add_argument("--source-dir", type=Path, required=True,
                    help="Directory containing the unchanged historical safedecontam.py")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--primary-state", type=Path,
                    help="Optional local trusted primary-run cache; omit to rebuild from the fixed training split")
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use an empty output directory; existing evidence will not be overwritten.")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_dir.resolve()))
    sd = importlib.import_module("safedecontam")
    started = datetime.now(timezone.utc).isoformat()
    seed = 42579
    cols = sd.SIM_COLS + sd.CON_COLS + sd.EXACT_COLS + sd.CHRONO
    data_paths = [args.input_dir / f for f in ["pairs_arith_v2.csv", "pairs_mcq_v2.csv", "pairs_math_v2.csv",
                                             "r2_wild_pairs.csv", "pairs_xling.csv"]]
    if args.primary_state:
        # Only load a cache generated locally by the trusted primary-run code.
        cache = joblib.load(args.primary_state)
        prep = cache["prep"]
        if "mlp" in cache and "mlp_scaler" in cache:
            clf_m, sc_m = cache["mlp"], cache["mlp_scaler"]
            clf_l, sc_l = cache["core"], cache["core_scaler"]
            zm = {p: cache["scores"]["mlp_margin_" + p] for p in ["cal", "test"]}
            zl = {p: cache["scores"]["core_" + p] for p in ["cal", "test"]}
        elif "mlp_margin" in cache and "core" in cache:
            margin = cache["mlp_margin"]
            linear = cache["core"]
            zm, clf_m, sc_m = margin["scores"], margin["model"], margin["scaler"]
            zl, clf_l, sc_l = linear["scores"], linear["model"], linear["scaler"]
        else:
            zm, clf_m, sc_m = sd.fit_score_margin(prep, cols)
            zl, clf_l, sc_l = sd.fit_score(prep, cols, model="logreg")
    else:
        pairs = pd.concat([pd.read_csv(p) for p in data_paths[:3]], ignore_index=True)
        pairs["benchmark_answer"] = pairs["benchmark_answer"].fillna("")
        bad = pairs.benchmark_text.isna() | pairs.candidate_text.isna() | (pairs.candidate_text.astype(str).str.strip() == "")
        pairs = pairs[~bad].reset_index(drop=True)
        prep = sd.prepare(pairs, seed=seed)
        zm, clf_m, sc_m = sd.fit_score_margin(prep, cols)
        zl, clf_l, sc_l = sd.fit_score(prep, cols, model="logreg")

    def features(frame):
        xs = prep["sim"].batch(frame.benchmark_text.tolist(), frame.candidate_text.tolist())
        xc = sd.constraint_frame(frame).reset_index(drop=True)
        x = pd.concat([xs, xc], axis=1)
        for c in ["a_num", "a_rel", "a_tgt", "a_ent"]:
            x[c + "_x"] = (x[c] >= .999).astype(float)
        return x

    def scores(x):
        return {"mlp_margin": sd.mlp_logits(clf_m, sc_m.transform(x[cols])),
                "core": clf_l.predict_proba(sc_l.transform(x[cols]))[:, 1]}

    np.savez_compressed(args.output / "frozen_models.npz",
        mlp_weight_0=clf_m.coefs_[0], mlp_weight_1=clf_m.coefs_[1],
        mlp_bias_0=clf_m.intercepts_[0], mlp_bias_1=clf_m.intercepts_[1],
        mlp_scaler_mean=sc_m.mean_, mlp_scaler_scale=sc_m.scale_,
        core_coef=clf_l.coef_, core_intercept=clf_l.intercept_,
        core_scaler_mean=sc_l.mean_, core_scaler_scale=sc_l.scale_)
    dump(args.output / "frozen_models.json", {
        "feature_columns": cols, "mlp_params": clf_m.get_params(), "core_params": clf_l.get_params(),
        "model_origin": "primary constructed experiment seed 42579; no auxiliary refitting or selection",
    })

    cal = prep["frames"]["cal"]
    mcq_mask = cal.family.eq("mcq").values
    mcq_cal = cal[mcq_mask].reset_index(drop=True)
    mcq_z = {"mlp_margin": zm["cal"][mcq_mask], "core": zl["cal"][mcq_mask]}
    mcq_tau = {name: sd.select_threshold(mcq_cal, score, .05, alpha=.05)
               for name, score in mcq_z.items()}
    dump(args.output / "primary_mcq_calibration.json", {
        "seed": seed, "feature_columns": cols, "thresholds": {n: float(t) for n, t in mcq_tau.items()},
        "groups": int(mcq_cal.group_id.nunique()),
        "row_ids": mcq_cal.group_id.tolist(), "decision_labels": mcq_cal.decision_label.tolist(),
        "scores": {n: s.tolist() for n, s in mcq_z.items()},
        "clean_group_maxima": {n: sd.group_max_scores(mcq_cal, s) for n, s in mcq_z.items()},
    })

    pilot = pd.read_csv(data_paths[3]).fillna("")
    # Historical removal decisions are excluded from the new inference input.
    pilot = pilot.drop(columns=[c for c in pilot if c.startswith("remove_")])
    xp = features(pilot)
    sp = scores(xp)
    pilot_result = {"condition": "fixed primary model; MCQ-local calibration; no pilot refitting",
                    "n": len(pilot), "thresholds": {n: float(t) for n, t in mcq_tau.items()}, "by_silver": {}}
    for name, z in sp.items():
        pilot["score_" + name] = z
        pilot["removed_" + name] = z >= mcq_tau[name]
    for category, sub in pilot.groupby("silver"):
        pilot_result["by_silver"][category] = {"n": len(sub), **{
            name: {"removed": int(sub["removed_" + name].sum()),
                   "removal_pct": 100 * float(sub["removed_" + name].mean())} for name in sp}}
    pilot.to_csv(args.output / "web_pilot_predictions.csv", index=False)
    xp.to_csv(args.output / "web_pilot_features.csv", index=False)
    dump(args.output / "web_pilot_results.json", pilot_result)
    print(json.dumps(pilot_result, indent=2), flush=True)

    xl = pd.read_csv(data_paths[4]).fillna("")
    tr, ca, te = sd.grouped_split(xl.group_id, seed)
    xl["part"] = ["train" if g in tr else ("cal" if g in ca else "test") for g in xl.group_id]
    xl[["group_id", "family", "relation", "part"]].to_csv(args.output / "xling_split_manifest.csv", index=False)
    frames = {p: xl[xl.part == p].reset_index(drop=True) for p in ["cal", "test"]}
    sx, xx = {}, {}
    for part, frame in frames.items():
        xx[part] = features(frame)
        sx[part] = scores(xx[part])
        xx[part].to_csv(args.output / f"xling_features_{part}.csv", index=False)
    xresult = {"condition": "frozen English three-family model; cross-lingual train partition unused; local calibration only",
               "n_groups": {p: int(f.group_id.nunique()) for p, f in frames.items()},
               "n_pairs": {p: len(f) for p, f in frames.items()}, "variants": {}}
    for name in sp:
        zc, zt = sx["cal"][name], sx["test"][name]
        tau = sd.select_threshold(frames["cal"], zc, .05, alpha=.05)
        fr = frames["test"].assign(removed=zt >= tau)
        by_relation = {r: {"n": len(g), "removed": int(g.removed.sum())} for r, g in fr.groupby("relation")}
        xresult["variants"][name] = {
            "tau": float(tau), "cal": sd.eval_at(frames["cal"], zc, tau),
            "test": sd.eval_at(frames["test"], zt, tau), "by_relation": by_relation,
            "calibration_clean_group_maxima": sd.group_max_scores(frames["cal"], zc),
        }
        for part in frames:
            frames[part]["score_" + name] = sx[part][name]
            frames[part]["removed_" + name] = sx[part][name] >= tau
    for part, frame in frames.items():
        frame.to_csv(args.output / f"xling_predictions_{part}.csv", index=False)
    np.savez_compressed(args.output / "scores.npz",
        **{"web_pilot_" + name: score for name, score in sp.items()},
        **{"xling_" + part + "_" + name: score for part, values in sx.items() for name, score in values.items()})
    dump(args.output / "xling_results.json", xresult)
    print(json.dumps(xresult, indent=2), flush=True)
    dump(args.output / "run_manifest.json", {
        "status": "new independent rerun, not recovered historical output",
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "pandas": pd.__version__,
        "environment": {k: os.environ.get(k) for k in ["PYTHONHASHSEED", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]},
        "command_arguments": sys.argv[1:],
        "inputs": {p.name: sha256(p) for p in data_paths},
        "source_hashes": {"safedecontam.py": sha256(sd.__file__), "run_auxiliary_probes.py": sha256(__file__)},
        "primary_state_sha256": sha256(args.primary_state) if args.primary_state else None,
        "feature_columns": cols,
        "primary_training": "fixed three-family training split, seed 42579; no probe rows used for training",
        "calibration": "Unmodified historical threshold selector; exploratory legacy protocol, not the raw-search clean-only guarantee.",
        "human_labels": "None used; web-pilot reference is silver/control construction, xling reference is paired construction.",
    })


if __name__ == "__main__":
    main()
