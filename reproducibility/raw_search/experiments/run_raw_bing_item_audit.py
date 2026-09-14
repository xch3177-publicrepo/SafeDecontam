#!/usr/bin/env python3
"""Item-level audit using the released raw Bing responses and silver labels.

Unlike the pair benchmark, this evaluates one real search-result bundle per
MMLU item. Labels are the authors' released clean/input/input+label categories.
They remain silver labels because they were derived from the same search data.
"""

from __future__ import annotations

import glob
import json
import os
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from run_web_evidence_experiment import (
    OUT as PAIR_OUT, ROOT, bootstrap_ci, clean_text,
    exact_calibration_threshold, handcrafted, row_cos, stable_fraction,
)
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_ID = "BAAI/bge-small-en-v1.5"


RAW_ROOT = ROOT / "data/raw/contamination_detector/bing_search/bing_search"
ANN_ROOT = ROOT / "external/Contamination_Detector/reports"
OUT_ROOT = ROOT / "output/experiments"
SEED = 20260715


def split_for(item_id, split_salt=""):
    key = item_id if not split_salt else f"{item_id}::{split_salt}"
    x = stable_fraction(key)
    return "train" if x < .4 else "validation" if x < .6 else "calibration" if x < .8 else "test"


def load_rows(dataset="mmlu", split_salt=""):
    raw_dir = RAW_ROOT / dataset
    ann_path = ANN_ROOT / f"{dataset}_annotations.json"
    ann = json.loads(ann_path.read_text())
    rows = []
    for path in glob.glob(str(raw_dir / "*.json")):
        item_id = os.path.basename(path)[:-5]
        if item_id not in ann:
            continue
        raw = json.load(open(path))
        pages = raw.get("webPages", {}).get("value", [])
        if not pages:
            continue
        page_records = []
        for p in pages:
            url = p.get("url", "").strip()
            text = clean_text(" ".join([p.get("name", ""), p.get("snippet", "")]))
            if url and text:
                page_records.append({"url": url, "text": text})
        query = clean_text(raw.get("queryContext", {}).get("originalQuery", ""))
        category, released_score = ann[item_id]
        if query and page_records:
            rows.append({
                "source_id": item_id,
                "query": query,
                "pages": page_records,
                "category": category,
                "released_max_score": released_score,
                "label": int(category != "clean"),
                "split": split_for(item_id, split_salt),
            })
    # Prevent a repeated URL from crossing partitions by removing it from all
    # bundles where its split ownership is ambiguous.
    owners = {}
    for r in rows:
        for p in r["pages"]:
            owners.setdefault(p["url"], set()).add(r["split"])
    conflict = {u for u, s in owners.items() if len(s) > 1}
    for r in rows:
        r["pages"] = [p for p in r["pages"] if p["url"] not in conflict]
        r["urls"] = sorted({p["url"] for p in r["pages"]})
        r["document"] = " ".join(p["text"] for p in r["pages"])
    return [r for r in rows if r["pages"]]


def metric(rows, scores, threshold):
    y = np.asarray([r["label"] for r in rows])
    pred = scores >= threshold
    return {
        "cr": float(pred[y == 1].mean()),
        "cdr": float(pred[y == 0].mean()),
        "gcdr": float(pred[y == 0].mean()),
        "positive_n": int((y == 1).sum()),
        "retain_n": int((y == 0).sum()),
        "false_negative_n": int((~pred[y == 1]).sum()),
        "false_positive_n": int(pred[y == 0].sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mmlu",
                        choices=("mmlu", "ARC", "commonsense_qa", "hellaswag", "winogrande", "ceval"))
    parser.add_argument("--split-salt", default="")
    parser.add_argument("--output", default=None)
    parser.add_argument("--disable-dense", action="store_true",
                        help="Run a lightweight lexical+constraint external audit without BGE.")
    args = parser.parse_args()
    out = Path(args.output) if args.output else (
        OUT_ROOT / ("raw_bing_item_audit" if args.dataset == "mmlu" and not args.split_salt
                    else f"raw_bing_item_audit_{args.dataset}{'_' + args.split_salt if args.split_salt else ''}"))
    out.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows(args.dataset, args.split_salt)
    (out / "dataset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))
    rows = {s: [r for r in all_rows if r["split"] == s] for s in ("train", "validation", "calibration", "test")}
    train_text = ([r["query"] for r in rows["train"]] +
                  [p["text"] for r in rows["train"] for p in r["pages"]])
    word = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=30000, sublinear_tf=True).fit(train_text)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=40000, sublinear_tf=True).fit(train_text)
    # The dense adapter is frozen before validation/calibration. No labels are
    # used to fit the public embedding model. External audits may explicitly
    # disable it to test the lightweight lexical+constraint adapter.
    dense_model = None
    if not args.disable_dense:
        from sentence_transformers import SentenceTransformer
        dense_model = SentenceTransformer(MODEL_ID)
    rawx = {}
    y = {}
    for s, rr in rows.items():
        flat = [(i, r, p) for i, r in enumerate(rr) for p in r["pages"]]
        page_queries = [x[1]["query"] for x in flat]
        page_docs = [x[2]["text"] for x in flat]
        qw, dw = word.transform(page_queries), word.transform(page_docs)
        qc, dc = char.transform(page_queries), char.transform(page_docs)
        if dense_model is None:
            dense_cos = np.zeros(len(flat))
        else:
            unique_eq = dense_model.encode(
                ["Represent this sentence for searching relevant passages: " + r["query"] for r in rr],
                batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            ed = dense_model.encode(page_docs, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            dense_cos = np.asarray([np.dot(unique_eq[i], ed[j]) for j, (i, _, _) in enumerate(flat)])
        rels = [{"query": q, "document": d} for q, d in zip(page_queries, page_docs)]
        page_x = np.column_stack([row_cos(qw,dw), row_cos(qc,dc), dense_cos, handcrafted(rels)])
        # Multiple-instance representation: one benchmark item is removed if
        # any retrieved page supplies strong evidence. Max is load-bearing;
        # top-2 mean and mean encode evidence concentration and background.
        agg = []
        for i in range(len(rr)):
            z = page_x[[j for j, (owner, _, _) in enumerate(flat) if owner == i]]
            ordered = np.sort(z, axis=0)
            agg.append(np.concatenate([z.max(axis=0), ordered[-min(2, len(z)):].mean(axis=0), z.mean(axis=0)]))
        rawx[s] = np.asarray(agg)
        y[s] = np.asarray([r["label"] for r in rr])
    scaler = StandardScaler().fit(rawx["train"])
    X = {s: scaler.transform(x) for s,x in rawx.items()}
    models = {
        "word_tfidf": None,
        "char_tfidf": None,
        "logistic": LogisticRegression(C=1, max_iter=2000, random_state=SEED).fit(X["train"],y["train"]),
        "mlp16": MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-3, solver="lbfgs", max_iter=1000, random_state=SEED).fit(X["train"],y["train"]),
    }
    if dense_model is not None:
        models["bge_small_en_v1_5"] = None
    def score(name,s):
        if name=="word_tfidf": return rawx[s][:,0]
        if name=="char_tfidf": return rawx[s][:,1]
        if name=="bge_small_en_v1_5": return rawx[s][:,2]
        return models[name].predict_proba(X[s])[:,1]
    validation={}
    for name in models:
        sv=score(name,"validation")
        t=float(np.quantile(sv[y["validation"]==0],.95,method="higher"))
        validation[name]={**metric(rows["validation"],sv,t),"threshold":t,"auroc":float(roc_auc_score(y["validation"],sv))}
    selected=max(models,key=lambda n:(validation[n]["cr"],-validation[n]["cdr"]))
    sc=score(selected,"calibration")
    cal=exact_calibration_threshold(sc[y["calibration"]==0])
    result={
        "data_source":f"Raw Bing responses + released {args.dataset} silver annotations from Li et al. 2024",
        "dataset": args.dataset,
        "split_salt": args.split_salt,
        "limitations":["Labels are search-overlap silver annotations, not independent human judgments.","Evidence is title/snippet bundles, not archived full pages."],
        "item_count":len(all_rows),
        "split_counts":{s:len(v) for s,v in rows.items()},
        "category_counts":dict(Counter(r["category"] for r in all_rows)),
        "validation_model_selection":validation,
        "selected_scorer":selected,
        "dense_model": MODEL_ID if dense_model is not None else None,
        "adapter_variant": "full" if dense_model is not None else "lexical+constraint (dense disabled)",
    }
    if cal is None:
        result["calibration"]={"abstained":True,"clean_groups":int((y["calibration"]==0).sum())}
    else:
        threshold,k,upper=cal
        st=score(selected,"test")
        mt=metric(rows["test"],st,threshold)
        # bootstrap_ci expects the same metric semantics, which hold for one row/group.
        mt["cluster_bootstrap_95_ci"]=bootstrap_ci(rows["test"],st,threshold)
        mt["auroc_diagnostic"]=float(roc_auc_score(y["test"],st))
        mt["recall_by_category"] = {}
        for category in ("input contamination", "input-and-label contamination"):
            idx = np.asarray([r["category"] == category for r in rows["test"]])
            mt["recall_by_category"][category] = {
                "n": int(idx.sum()),
                "removed": int((st[idx] >= threshold).sum()),
                "recall": float((st[idx] >= threshold).mean()),
            }
        result["calibration"]={"abstained":False,"epsilon":.05,"alpha":.05,"clean_groups":int((y["calibration"]==0).sum()),"allowed_exceedances":k,"upper_bound":upper,"threshold":threshold}
        result["locked_test"]=mt
        prediction_rows = []
        for row, score_value in zip(rows["test"], st):
            prediction_rows.append({
                "source_id": row["source_id"],
                "silver_category": row["category"],
                "silver_label": row["label"],
                "score": float(score_value),
                "threshold": float(threshold),
                "removed": bool(score_value >= threshold),
            })
        (out / "locked_test_predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in prediction_rows)
        )
        sensitivity = {}
        for eps in (.025, .05, .10):
            c = exact_calibration_threshold(sc[y["calibration"]==0], epsilon=eps, alpha=.05)
            if c is None:
                sensitivity[str(eps)] = {"abstained": True}
            else:
                t2, k2, u2 = c
                sensitivity[str(eps)] = {
                    "abstained": False, "threshold": t2,
                    "allowed_exceedances": k2, "upper_bound": u2,
                    **metric(rows["test"], st, t2),
                }
        result["budget_sensitivity"] = sensitivity
    (out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
