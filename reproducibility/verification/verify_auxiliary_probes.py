#!/usr/bin/env python3
"""Verify frozen-model inference and auxiliary counts from numerical artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    root = args.directory
    model = np.load(root / "frozen_models.npz")
    cols = json.loads((root / "frozen_models.json").read_text())["feature_columns"]

    def check_scores(frame, feature_file):
        x = pd.read_csv(root / feature_file, float_precision="round_trip")[cols].to_numpy()
        xm = (x - model["mlp_scaler_mean"]) / model["mlp_scaler_scale"]
        margin = (np.maximum(0, xm @ model["mlp_weight_0"] + model["mlp_bias_0"])
                  @ model["mlp_weight_1"] + model["mlp_bias_1"]).ravel()
        xl = (x - model["core_scaler_mean"]) / model["core_scaler_scale"]
        linear = expit(xl @ model["core_coef"].T + model["core_intercept"]).ravel()
        for name, score in [("mlp_margin", margin), ("core", linear)]:
            np.testing.assert_allclose(score, frame["score_" + name], rtol=1e-12, atol=1e-10)

    def threshold_from_saved(group_ids, labels, scores):
        frame = pd.DataFrame({"group": group_ids, "label": labels, "score": scores})
        clean = frame[frame.label == "retain"].groupby("group").score.max().to_numpy()
        positive = frame[frame.label == "remove"].score.to_numpy()
        n = len(clean)
        best = None
        for threshold in np.unique(np.r_[scores, np.inf]):
            k = int((clean >= threshold).sum())
            upper = beta.ppf(.95, k + 1, n - k) if k < n else 1.
            if upper > .05:
                continue
            key = (np.mean(positive >= threshold), -k, threshold if np.isfinite(threshold) else 1e18)
            if best is None or key > best[0]:
                best = key, threshold
        return best[1] if best else np.inf

    pilot = pd.read_csv(root / "web_pilot_predictions.csv", float_precision="round_trip").fillna("")
    check_scores(pilot, "web_pilot_features.csv")
    pr = json.loads((root / "web_pilot_results.json").read_text())
    cal = json.loads((root / "primary_mcq_calibration.json").read_text())
    for name in ["mlp_margin", "core"]:
        threshold = threshold_from_saved(cal["row_ids"], cal["decision_labels"], cal["scores"][name])
        assert threshold == pr["thresholds"][name]
        np.testing.assert_array_equal(pilot["score_" + name] >= threshold, pilot["removed_" + name])
        for category, row in pr["by_silver"].items():
            sub = pilot[pilot.silver == category]
            assert len(sub) == row["n"]
            assert int(sub["removed_" + name].sum()) == row[name]["removed"]

    xr = json.loads((root / "xling_results.json").read_text())
    xc = pd.read_csv(root / "xling_predictions_cal.csv", float_precision="round_trip").fillna("")
    xt = pd.read_csv(root / "xling_predictions_test.csv", float_precision="round_trip").fillna("")
    if (root / "scores.npz").exists():
        saved = np.load(root / "scores.npz")
        for name in ["mlp_margin", "core"]:
            np.testing.assert_array_equal(saved["web_pilot_" + name], pilot["score_" + name])
            np.testing.assert_array_equal(saved["xling_cal_" + name], xc["score_" + name])
            np.testing.assert_array_equal(saved["xling_test_" + name], xt["score_" + name])
    check_scores(xc, "xling_features_cal.csv")
    check_scores(xt, "xling_features_test.csv")
    for name, result in xr["variants"].items():
        threshold = threshold_from_saved(xc.group_id, xc.decision_label, xc["score_" + name])
        assert np.isclose(threshold, result["tau"], rtol=0, atol=1e-10)
        for part, frame in [("cal", xc), ("test", xt)]:
            removed = frame["score_" + name] >= result["tau"]
            np.testing.assert_array_equal(removed, frame["removed_" + name])
            positive = frame.decision_label == "remove"
            assert int(removed[positive].sum()) == result[part]["removed_contam"]
            assert int(removed[~positive].sum()) == result[part]["removed_clean_pairs"]
            assert positive.sum() == result[part]["total_contam"]
        for relation, row in result["by_relation"].items():
            sub = xt[xt.relation == relation]
            assert len(sub) == row["n"] and int(sub["removed_" + name].sum()) == row["removed"]
    print(json.dumps({"passed": True, "checks": ["manual MLP and logistic numerical inference",
          "MCQ and xling threshold recomputation", "saved decisions", "all subgroup counts"],
          "web_pilot": pr, "xling_mlp_margin": xr["variants"]["mlp_margin"]}, indent=2))


if __name__ == "__main__":
    main()
