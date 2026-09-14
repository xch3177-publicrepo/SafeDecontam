#!/usr/bin/env python3
"""Run the fixed historical constructed protocol in a new, fully recorded run.

This wrapper preserves the supplied scoring implementation and hyperparameters.
It never searches seeds or tunes to manuscript values. A new output directory is
required. Set PYTHONHASHSEED=0 and single-thread BLAS before starting Python.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time


def safe(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    return x


def write_json(p, x):
    p.write_text(json.dumps(safe(x), indent=2, allow_nan=False) + "\n")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--source-dir", type=Path, default=Path(__file__).parent / "historical_source")
    ap.add_argument("--private-runtime-dir", type=Path, help="Optional local joblib export for related frozen-scorer probes; exclude from release.")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists; preserving it. Choose a fresh directory.")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Start Python with PYTHONHASHSEED=0 to fix set-iteration order.")
    for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        if os.environ.get(k) != "1":
            raise SystemExit("Start Python with " + k + "=1.")
    sys.path.insert(0, str(args.source_dir.resolve()))
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import safedecontam as sd
    from sklearn.metrics import roc_auc_score
    args.output.mkdir(parents=True)
    out = args.output
    (out / "source").mkdir()
    for name in ["safedecontam.py", "run_v2.py"]:
        shutil.copy2(args.source_dir / name, out / "source" / name)
    shutil.copy2(Path(__file__), out / "source" / Path(__file__).name)
    seeds = [42579] + [int(hashlib.sha256(f"SafeDecontam-real-v2-{i}".encode()).hexdigest()[:8], 16) % 100000 for i in range(1, 10)]
    inputs = [args.input_dir / f"pairs_{f}_v2.csv" for f in ["arith", "mcq", "math"]]
    start = time.time()
    manifest = {
        "identity": "Current independent complete constructed run; distinct from all historical results",
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_list": seeds, "seeds_selected_from": "Unchanged historical SHA256-derived seed list",
        "platform": platform.platform(), "python": platform.python_version(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "environment": {k: os.environ[k] for k in ["PYTHONHASHSEED", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
        "inputs": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in inputs},
        "source_sha256": {p.name: sha(p) for p in (out / "source").glob("*.py")},
        "hyperparameters": {"MLP": {"hidden_layer_sizes": [16], "activation": "relu", "solver": "lbfgs", "alpha": .001, "max_iter": 4000, "random_state": 0},
                            "logistic": {"C": 1.0, "class_weight": "balanced", "max_iter": 2000},
                            "main_families": ["arithmetic", "mcq"], "alpha": .05, "budgets": [.05, .10]},
        "limitations": ["Exploratory configuration inherited from a pipeline that inspected calibration feedback.", "New numerical output; not a reconstruction of the missing93.70% run.", "No settings or seeds tuned to match either93.70% or90.21%.", "Nonfinite abstention thresholds represented as null in JSON and interpreted as positive infinity."],
    }
    write_json(out / "run_manifest.json", manifest)
    dfs = []
    for p in inputs:
        df = pd.read_csv(p)
        df["source_csv"] = p.name
        df["source_csv_row"] = np.arange(len(df))
        dfs.append(df)
    pairs = pd.concat(dfs, ignore_index=True)
    pairs["benchmark_answer"] = pairs["benchmark_answer"].fillna("")
    bad = pairs.benchmark_text.isna() | pairs.candidate_text.isna() | (pairs.candidate_text.astype(str).str.strip() == "")
    manifest["removed_empty_rows"] = int(bad.sum())
    pairs = pairs[~bad].reset_index(drop=True)
    manifest["usable_pairs"] = len(pairs)
    write_json(out / "run_manifest.json", manifest)
    main_fams = ["arithmetic", "mcq"]

    def subset(prep, part, z, fams):
        fr = prep["frames"][part]
        m = fr.family.isin(fams).values
        return fr[m].reset_index(drop=True), z[m]

    def evaluate(prep, zc, zt, fams, eps, with_ci=False):
        cf, c = subset(prep, "cal", zc, fams)
        tf, t = subset(prep, "test", zt, fams)
        tau = sd.select_threshold(cf, c, eps, .05)
        r = {"tau": float(tau), "eps": eps, "families": fams,
             "cal": sd.eval_at(cf, c, tau), "test": sd.eval_at(tf, t, tau)}
        if with_ci:
            r["test_CR_ci95"] = sd.bootstrap_cr_ci(tf, t, tau)
        r["AUROC"] = float(roc_auc_score(tf.decision_label.eq("remove"), t))
        for fam in fams:
            f, z = subset(prep, "test", zt, [fam])
            r["test_" + fam] = sd.eval_at(f, z, tau)
        return r

    def errors(prep, zt, tau):
        fr, z = subset(prep, "test", zt, main_fams)
        fr = fr.assign(rm=z >= tau)
        miss = fr[fr.decision_label.eq("remove") & ~fr.rm]
        fp = fr[fr.decision_label.eq("retain") & fr.rm]
        cols = ["family", "relation", "candidate_source"]
        fmt = lambda df: {"|".join(map(str, k)): int(v) for k, v in df.groupby(cols).size().items()}
        return {"n_miss": len(miss), "n_fp": len(fp), "missed_by": fmt(miss), "false_removed_by": fmt(fp)}

    def save_model(dest, clf, scaler, cols):
        if isinstance(clf.coefs_, list) if hasattr(clf, "coefs_") else False:
            arrays = {"coef_" + str(i): v for i, v in enumerate(clf.coefs_)}
            arrays.update({"intercept_" + str(i): v for i, v in enumerate(clf.intercepts_)})
        else:
            arrays = {"coef": clf.coef_, "intercept": clf.intercept_}
        arrays.update({"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "scaler_var": scaler.var_})
        np.savez_compressed(dest.with_suffix(".npz"), **arrays)
        write_json(dest.with_suffix(".json"), {"feature_columns": cols, "estimator": type(clf).__name__, "params": clf.get_params(), "classes": clf.classes_.tolist(), "n_iter": int(clf.n_iter_) if np.ndim(clf.n_iter_) == 0 else clf.n_iter_.tolist()})

    result = {"identity": manifest["identity"], "seed": seeds[0], "seed_list": seeds, "primary": {}, "sensitivity": {"mlp": [], "mlp_margin": []}, "sensitivity_math_margin": []}
    for seed in seeds:
        seed_start = time.time()
        dest = out / "seeds" / str(seed)
        dest.mkdir(parents=True)
        prep = sd.prepare(pairs, seed=seed)
        frame_files = {}
        matrices = {}
        for part in ["train", "cal", "test"]:
            frame = prep["frames"][part]
            # Exact row order is recoverable from canonical input CSV and row index.
            meta_cols = ["source_csv", "source_csv_row", "family", "group_id", "part", "decision_label", "relation", "candidate_source"]
            frame[meta_cols].to_csv(dest / ("frame_" + part + ".csv"), index=False)
            matrices[part] = prep["X"][part].to_numpy()
            frame_files[part] = {"rows": len(frame), "sha256": sha(dest / ("frame_" + part + ".csv"))}
        np.savez_compressed(dest / "features.npz", **matrices)
        write_json(dest / "features.json", {"columns": prep["X"]["train"].columns.tolist(), "frames": frame_files, "group_splits": prep["splits"]})
        cols = sd.SIM_COLS + sd.CON_COLS + sd.EXACT_COLS + sd.CHRONO
        prob, mlp, mlp_scaler = sd.fit_score(prep, cols, model="mlp")
        margin = {p: sd.mlp_logits(mlp, mlp_scaler.transform(prep["X"][p][cols])) for p in ["cal", "test"]}
        save_model(dest / "mlp_model", mlp, mlp_scaler, cols)
        scores = {"mlp_cal": prob["cal"], "mlp_test": prob["test"], "mlp_margin_cal": margin["cal"], "mlp_margin_test": margin["test"]}
        seed_res = {"seed": seed, "splits": prep["splits"], "variants": {}}
        for name, zs in [("mlp", prob), ("mlp_margin", margin)]:
            rr = evaluate(prep, zs["cal"], zs["test"], main_fams, .05, with_ci=seed == seeds[0] and name == "mlp_margin")
            seed_res["variants"][name] = {"eps5": rr}
            result["sensitivity"][name].append({"seed": seed, "CR": rr["test"]["CR"], "CDR": rr["test"]["CDR"], "GCDR": rr["test"]["GCDR"], "damaged_groups": rr["test"]["damaged_groups"], "tau": rr["tau"]})
        rm = evaluate(prep, margin["cal"], margin["test"], ["math"], .05)
        seed_res["math_margin_local"] = rm
        result["sensitivity_math_margin"].append({"seed": seed, "CR": rm["test"]["CR"], "GCDR": rm["test"]["GCDR"]})
        if seed == seeds[0]:
            for name in ["ngram4", "shingle3", "bm25n", "fusion", "no_constraints", "core", "mlp", "mlp_margin"]:
                if name in ["mlp", "mlp_margin"]:
                    zc, zt = scores[name + "_cal"], scores[name + "_test"]
                elif name in ["core", "no_constraints"]:
                    c = cols if name == "core" else sd.SIM_COLS + sd.CHRONO
                    zs, clf, scaler = sd.fit_score(prep, c, model="logreg")
                    zc, zt = zs["cal"], zs["test"]
                    save_model(dest / (name + "_model"), clf, scaler, c)
                    if name == "core":
                        core, core_scaler = clf, scaler
                else:
                    zc, zt = sd.variant_scores(prep, name)
                scores[name + "_cal"], scores[name + "_test"] = zc, zt
                seed_res["variants"].setdefault(name, {})
                for eps in [.05, .10]:
                    key = "eps" + str(round(100 * eps))
                    if key not in seed_res["variants"][name]:
                        seed_res["variants"][name][key] = evaluate(prep, zc, zt, main_fams, eps)
                seed_res["variants"][name]["eps5"]["breakdown"] = errors(prep, zt, seed_res["variants"][name]["eps5"]["tau"])
            fams = ["arithmetic", "mcq", "math"]
            local = {f: evaluate(prep, margin["cal"], margin["test"], [f], .05) for f in fams}
            seed_res["margin_family_local"] = local
            transfer = {}
            for src in fams:
                for tgt in fams:
                    tf, z = subset(prep, "test", margin["test"], [tgt])
                    metric = sd.eval_at(tf, z, local[src]["tau"])
                    metric["regret"] = max(0., local[tgt]["test"]["CR"] - metric["CR"])
                    transfer[src + "->" + tgt] = metric
            seed_res["margin_transfer"] = {"taus": {f: local[f]["tau"] for f in fams}, "matrix": transfer, "scorer": "MLP margin"}
            result["primary"] = seed_res
            if args.private_runtime_dir:
                import joblib
                args.private_runtime_dir.mkdir(parents=True, exist_ok=False)
                joblib.dump({"prep": prep, "mlp": mlp, "mlp_scaler": mlp_scaler, "core": core, "core_scaler": core_scaler,
                             "scores": scores, "feature_columns": cols, "margin_family_local": local,
                             "core_mcq_local": evaluate(prep, scores["core_cal"], scores["core_test"], ["mcq"], .05)},
                            args.private_runtime_dir / "primary_runtime.joblib", compress=3)
                print("PRIMARY_RUNTIME_READY", args.private_runtime_dir / "primary_runtime.joblib", flush=True)
                write_json(args.private_runtime_dir / "primary_results.json", seed_res)
        np.savez_compressed(dest / "scores.npz", **scores)
        seed_res["elapsed_s"] = time.time() - seed_start
        write_json(dest / "results.json", seed_res)
        write_json(out / "results.json", result)
        print("SEED", seed, "prob", seed_res["variants"]["mlp"]["eps5"]["test"]["CR"], "margin", seed_res["variants"]["mlp_margin"]["eps5"]["test"]["CR"], "seconds", round(seed_res["elapsed_s"], 2), flush=True)
    result["sensitivity_summary"] = {}
    for name, rows in result["sensitivity"].items():
        v = np.array([r["CR"] for r in rows])
        result["sensitivity_summary"][name] = {"mean": float(v.mean()), "population_sd": float(v.std()), "min": float(v.min()), "max": float(v.max()), "zero_recall_splits": int((v == 0).sum())}
    v = np.array([r["CR"] for r in result["sensitivity_math_margin"]])
    result["sensitivity_math_margin_summary"] = {"mean": float(v.mean()), "population_sd": float(v.std()), "min": float(v.min()), "max": float(v.max())}
    write_json(out / "results.json", result)
    manifest["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["elapsed_s"] = time.time() - start
    manifest["status"] = "complete"
    write_json(out / "run_manifest.json", manifest)
    checksums = [sha(p) + "  " + str(p.relative_to(out)) for p in sorted(out.rglob("*")) if p.is_file()]
    (out / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    print("COMPLETE", manifest["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
