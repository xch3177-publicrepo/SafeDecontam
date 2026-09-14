#!/usr/bin/env python3
"""Verify saved external-audit scores independently of the training script."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score


def verify(directory):
    result = json.loads((directory / "results.json").read_text())
    rows = [json.loads(line) for line in (directory / "all_split_scores.jsonl").read_text().splitlines()]
    assert len(rows) == result["item_count"]
    assert len({r["source_id"] for r in rows}) == len(rows)
    groups = {s: [r for r in rows if r["split"] == s] for s in result["split_counts"]}
    assert {s: len(v) for s, v in groups.items()} == result["split_counts"]
    models = list(rows[0]["scores"])
    def arr(split, model):
        return (np.array([r["silver_label"] for r in groups[split]]),
                np.array([r["scores"][model] for r in groups[split]]))
    validation = {}
    for model in models:
        y, score = arr("validation", model)
        threshold = np.quantile(score[y == 0], .95, method="higher")
        metrics = dict(cr=np.mean(score[y == 1] >= threshold), cdr=np.mean(score[y == 0] >= threshold),
                       auroc=roc_auc_score(y, score), threshold=threshold)
        for key, value in metrics.items():
            assert np.isclose(value, result["validation_model_selection"][model][key], rtol=0, atol=1e-12), (model,key)
        validation[model] = metrics
    selected = max(models, key=lambda model: (validation[model]["cr"], -validation[model]["cdr"]))
    assert selected == result["selected_scorer"]
    y, score = arr("calibration", selected)
    clean = np.sort(score[y == 0])
    n = len(clean)
    feasible = [k for k in range(n) if beta.ppf(.95, k + 1, n - k) <= .05]
    assert feasible
    k = max(feasible)
    threshold = np.nextafter(clean[n - k - 1], np.inf)
    cal = result["calibration"]
    assert n == cal["clean_groups"] and k == cal["allowed_exceedances"]
    assert threshold == cal["threshold"]
    assert np.isclose(beta.ppf(.95, k + 1, n - k), cal["upper_bound"], rtol=0, atol=1e-12)
    y, score = arr("test", selected)
    removed = score >= threshold
    actual = dict(cr=np.mean(removed[y == 1]), cdr=np.mean(removed[y == 0]),
                  positive_n=int((y == 1).sum()), retain_n=int((y == 0).sum()),
                  false_negative_n=int((~removed[y == 1]).sum()), false_positive_n=int(removed[y == 0].sum()),
                  auroc_diagnostic=roc_auc_score(y, score))
    for key, value in actual.items():
        assert np.isclose(value, result["locked_test"][key], rtol=0, atol=1e-12), key
    frozen = [json.loads(line) for line in (directory / "locked_test_predictions.jsonl").read_text().splitlines()]
    assert [r["source_id"] for r in frozen] == [r["source_id"] for r in groups["test"]]
    assert np.array_equal(np.array([r["score"] for r in frozen]), score)
    assert np.array_equal(np.array([r["removed"] for r in frozen]), removed)
    features = np.load(directory / "raw_item_features.npz")
    for split, rr in groups.items():
        assert features[f"features_{split}"].shape == (len(rr), 27)
        assert np.array_equal(features[f"labels_{split}"], [r["silver_label"] for r in rr])
    return dict(dataset=result["dataset"], passed=True, item_count=len(rows), selected_scorer=selected,
                calibration_clean_groups=n, allowed_exceedances=k, threshold=threshold, **actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([verify(p) for p in args.directory], indent=2))
