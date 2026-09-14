#!/usr/bin/env python3
"""Auditable web-evidence experiment for SafeDecontam.

The positive evidence comes from the public release accompanying Li et al.
(Findings of EMNLP 2024). Retain controls are within-split cross-item pairings:
the web document is real, while the negative relation is constructed.  This
script deliberately keeps validation, calibration, and test roles separate.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "external/Contamination_Detector/reports/mmlu_report.json"
OUT = ROOT / "output/experiments/web_evidence"
SEED = 20260715


def clean_text(s: str) -> str:
    s = html.unescape(re.sub(r"</?b>", "", s or ""))
    return re.sub(r"\s+", " ", s).strip()


def stable_fraction(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16) / 16**16


def split_for(query: str) -> str:
    x = stable_fraction(query)
    if x < 0.40:
        return "train"
    if x < 0.60:
        return "validation"
    if x < 0.80:
        return "calibration"
    return "test"


def select_public_records() -> list[dict]:
    raw = json.loads(REPORT.read_text())["matches"]
    records = []
    seen_query, seen_url = set(), set()
    for group in raw:
        candidates = [r for r in group if r.get("score", 0) >= 0.7]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: (r.get("score", 0), r.get("score_label", 0)))
        query = clean_text(best["query"])
        url = best.get("url", "").strip()
        doc = clean_text(" ".join([best.get("name", ""), best.get("snippet", ""), best.get("match_string", "")]))
        if not query or not url or not doc or query in seen_query or url in seen_url:
            continue
        seen_query.add(query)
        seen_url.add(url)
        records.append({
            "source_id": hashlib.sha256(query.encode()).hexdigest()[:16],
            "query": query,
            "url": url,
            "document": doc,
            "silver_type": "answer-bearing" if best.get("score_label", 0) >= 0.7 else "question-only",
            "released_score": float(best.get("score", 0)),
            "released_label_score": float(best.get("score_label", 0)),
            "split": split_for(query),
        })
    return records


def deranged_controls(records: list[dict]) -> list[dict]:
    """Pair each query with another real document inside the same locked split."""
    rng = np.random.default_rng(SEED)
    pairs = []
    for split in ("train", "validation", "calibration", "test"):
        part = [r for r in records if r["split"] == split]
        order = np.arange(len(part))
        rng.shuffle(order)
        # A cyclic shift guarantees no fixed point while retaining a random order.
        other = np.roll(order, 1)
        for pos_idx, neg_idx in zip(order, other):
            pos, negdoc = part[pos_idx], part[neg_idx]
            pairs.append({**pos, "label": 1, "relation": "released-silver-match"})
            pairs.append({
                **pos,
                "url": negdoc["url"],
                "document": negdoc["document"],
                "label": 0,
                "relation": "within-split-cross-item-control",
                "silver_type": "retain-control",
            })
    return pairs


def token_set(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def char_ngrams(s: str, n: int = 4) -> set[str]:
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i:i+n] for i in range(max(0, len(s) - n + 1))}


def row_cos(a, b) -> np.ndarray:
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def handcrafted(pairs: list[dict]) -> np.ndarray:
    out = []
    for r in pairs:
        q, d = r["query"], r["document"]
        tq, td = token_set(q), token_set(d)
        cq, cd = char_ngrams(q), char_ngrams(d)
        nums_q, nums_d = set(re.findall(r"\d+(?:\.\d+)?", q)), set(re.findall(r"\d+(?:\.\d+)?", d))
        out.append([
            len(tq & td) / max(1, len(tq | td)),
            len(cq & cd) / max(1, len(cq)),
            len(nums_q & nums_d) / max(1, len(nums_q)),
            min(len(q), len(d)) / max(1, max(len(q), len(d))),
            float(q.lower() in d.lower()),
            float(bool(nums_q) and nums_q <= nums_d),
        ])
    return np.asarray(out, dtype=float)


def exact_calibration_threshold(clean_scores: np.ndarray, epsilon=.05, alpha=.05):
    """One-sided CP/order-statistic threshold; scorer is already fixed."""
    q = np.sort(np.asarray(clean_scores))
    n = len(q)
    feasible = []
    for k in range(n):
        upper = beta.ppf(1 - alpha, k + 1, n - k)
        if upper <= epsilon:
            feasible.append((k, float(upper)))
    if not feasible:
        return None
    k, upper = feasible[-1]
    boundary = q[n - k - 1]
    return float(np.nextafter(boundary, np.inf)), k, upper


def metrics(rows, scores, threshold):
    y = np.asarray([r["label"] for r in rows])
    pred = scores >= threshold
    pos, neg = y == 1, y == 0
    # Exactly one positive and one retain pair per source group in this audit.
    return {
        "cr": float(pred[pos].mean()),
        "cdr": float(pred[neg].mean()),
        "gcdr": float(pred[neg].mean()),
        "positive_n": int(pos.sum()),
        "retain_n": int(neg.sum()),
        "false_negative_n": int((~pred[pos]).sum()),
        "false_positive_n": int(pred[neg].sum()),
    }


def bootstrap_ci(rows, scores, threshold, reps=2000):
    rng = np.random.default_rng(SEED + 1)
    by_group = {}
    for i, r in enumerate(rows):
        by_group.setdefault(r["source_id"], []).append(i)
    groups = list(by_group)
    vals = []
    for _ in range(reps):
        sampled = rng.choice(groups, len(groups), replace=True)
        idx = [i for g in sampled for i in by_group[g]]
        m = metrics([rows[i] for i in idx], scores[idx], threshold)
        vals.append([m["cr"], m["cdr"], m["gcdr"]])
    a = np.asarray(vals)
    return {k: [float(x) for x in np.quantile(a[:, j], [.025, .975])]
            for j, k in enumerate(("cr", "cdr", "gcdr"))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = select_public_records()
    pairs = deranged_controls(records)
    (OUT / "dataset.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs))

    train = [r for r in pairs if r["split"] == "train"]
    all_text_train = [r["query"] for r in train] + [r["document"] for r in train]
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True).fit(all_text_train)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000, sublinear_tf=True).fit(all_text_train)

    features = {}
    for split in ("train", "validation", "calibration", "test"):
        rows = [r for r in pairs if r["split"] == split]
        q_w, d_w = word.transform([r["query"] for r in rows]), word.transform([r["document"] for r in rows])
        q_c, d_c = char.transform([r["query"] for r in rows]), char.transform([r["document"] for r in rows])
        features[split] = np.column_stack([row_cos(q_w, d_w), row_cos(q_c, d_c), handcrafted(rows)])

    scaler = StandardScaler().fit(features["train"])
    X = {k: scaler.transform(v) for k, v in features.items()}
    rows = {k: [r for r in pairs if r["split"] == k] for k in features}
    y = {k: np.asarray([r["label"] for r in rows[k]]) for k in rows}
    models = {
        "word_tfidf": None,
        "char_tfidf": None,
        "logistic": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        "mlp16": MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-3, solver="lbfgs", max_iter=1000, random_state=SEED),
    }
    models["logistic"].fit(X["train"], y["train"])
    models["mlp16"].fit(X["train"], y["train"])

    def score(name, split):
        if name == "word_tfidf": return features[split][:, 0]
        if name == "char_tfidf": return features[split][:, 1]
        return models[name].predict_proba(X[split])[:, 1]

    # Model selection is validation-only. Threshold here is diagnostic, not certified.
    validation = {}
    for name in models:
        s = score(name, "validation")
        clean = s[y["validation"] == 0]
        t = float(np.quantile(clean, .95, method="higher"))
        m = metrics(rows["validation"], s, t)
        validation[name] = {**m, "threshold": t, "auroc": float(roc_auc_score(y["validation"], s))}
    selected = max(models, key=lambda n: (validation[n]["cr"], -validation[n]["cdr"]))

    cal_scores = score(selected, "calibration")
    calibrated = exact_calibration_threshold(cal_scores[y["calibration"] == 0])
    if calibrated is None:
        raise RuntimeError("Calibration abstained: insufficient clean groups for 5%/95% target")
    threshold, allowance, upper = calibrated
    test_scores = score(selected, "test")
    test_result = metrics(rows["test"], test_scores, threshold)
    test_result["cluster_bootstrap_95_ci"] = bootstrap_ci(rows["test"], test_scores, threshold)
    test_result["auroc_diagnostic"] = float(roc_auc_score(y["test"], test_scores))

    result = {
        "data_source": "Li et al. 2024 public MMLU Bing-search release",
        "limitations": [
            "Positive labels are released search-overlap silver labels, not independent human labels.",
            "Documents are search titles/snippets/matched spans, not archived full pages.",
            "Retain relations are within-split cross-item controls over real web evidence.",
        ],
        "unique_released_records": len(records),
        "pair_count": len(pairs),
        "split_group_counts": dict(Counter(r["split"] for r in records)),
        "silver_type_counts": dict(Counter(r["silver_type"] for r in records)),
        "validation_model_selection": validation,
        "selected_scorer": selected,
        "calibration": {
            "epsilon": .05, "alpha": .05, "clean_groups": int((y["calibration"] == 0).sum()),
            "allowed_exceedances": allowance, "upper_bound": upper, "threshold": threshold,
        },
        "locked_test": test_result,
        "versions": {
            "numpy": np.__version__,
        },
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
