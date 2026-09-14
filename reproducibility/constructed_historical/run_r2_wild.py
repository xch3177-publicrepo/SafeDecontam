import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R2 野外审计试点:用 Li et al. (EMNLP-F 2024) 发布的 MMLU↔真实网页匹配作为 d 侧,
以冻结的主种子模型 + MCQ 家族本地阈值做 remove/retain 判定,对照其银标签
(input-and-label / input / clean)报告一致性。银标签≠金标;结果按试点披露。
运行:python3 run_r2_wild.py(需先有 v2 配对表;约 1 分钟)"""
import csv, json, re
import numpy as np, pandas as pd
import safedecontam as sd
from run_v2 import load_pairs

DATA = os.environ.get("SAFEDECONTAM_DATA", "/home/claude/work/论文数据集")
REP = f"{DATA}/derived_real/contamination_reports_realweb"
LETTERS = "ABCD"
SUBJECTS = ["abstract_algebra", "high_school_us_history", "sociology"]

def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()

def read_items():
    items = {}
    for s in SUBJECTS:
        with open(f"{DATA}/benchmarks/mmlu_subjects_sample/{s}_test.csv", newline="", encoding="utf-8") as f:
            for i, r in enumerate(csv.reader(f)):
                if len(r) >= 6:
                    ans = r[1 + LETTERS.index(r[5].strip())] if r[5].strip() in LETTERS else r[1]
                    items[norm(r[0])] = dict(subject=s, idx=i, q=r[0].strip(),
                                             opts=[x.strip() for x in r[1:5]], ans=ans.strip())
    return items

def main():
    items = read_items()
    ann = json.load(open(f"{REP}/mmlu_annotations.json"))
    matches = json.load(open(f"{REP}/mmlu_report.json"))["matches"]
    # query = 题干 + 正确答案文本;用"题干为前缀"连接
    prefix_index = sorted(items.items(), key=lambda kv: -len(kv[0]))
    def find_item(query_norm):
        for qn, it in prefix_index:
            if query_norm.startswith(qn): return it
        return None
    rows = []
    for mlist in matches:
        if not mlist: continue
        it = find_item(norm(mlist[0]["query"]))
        if not it: continue
        ev = mlist
        d_parts = []
        for m in ev[:3]:
            snip = re.sub(r"</?b>", "", m.get("snippet") or "")
            d_parts.append(" ".join(x for x in [m.get("name", ""), snip, m.get("match_string", "")] if x))
        d_text = " ".join(d_parts)
        b_text = f"{it['q']} " + " ".join(f"{L}. {o}" for L, o in zip(LETTERS, it["opts"]))
        key = f"{it['subject']} ?"
        rows.append(dict(benchmark_text=b_text, benchmark_answer=it["ans"], candidate_text=d_text,
                         candidate_date="", benchmark_release="2020-09-07", family="mcq",
                         group_id=f"wild_{it['subject']}_{it['idx']:04d}", decision_label="retain",
                         relation="wild", subject=it["subject"],
                         max_score=max(m["score"] for m in ev),
                         answer_flag=max(float(m.get("score_label") or 0) for m in ev)))
    # 交叉负例:b 配同学科"另一道题"的真实网页证据(真实网页、题材相同、非本题)-> 期望 retain
    import hashlib
    pos = pd.DataFrame(rows)
    neg_rows = []
    for subj, sub in pos.groupby("subject"):
        recs = sub.to_dict("records")
        if len(recs) < 2: continue
        for i, r in enumerate(recs):
            j = (i + 1 + int(hashlib.sha256(r["group_id"].encode()).hexdigest(), 16) % (len(recs) - 1)) % len(recs)
            if j == i: j = (i + 1) % len(recs)
            n = dict(r); n["candidate_text"] = recs[j]["candidate_text"]
            n["group_id"] = r["group_id"] + "_x"; n["silver"] = "cross-item (clean)"
            neg_rows.append(n)
    pos["silver"] = np.where(pos.answer_flag > 0, "web page bears answer", "web page bears question")
    wild = pd.concat([pos, pd.DataFrame(neg_rows)], ignore_index=True)
    print("野外可评条目:", len(wild), wild.silver.value_counts().to_dict())
    print("按学科:", wild.subject.value_counts().to_dict())

    # 冻结主种子模型与家族本地阈值
    pairs = load_pairs()
    prep = sd.prepare(pairs, seed=42579)
    res = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + "/v2_results.json"))
    tau_marg = res["family_local"]["mlp_margin"]["mcq"]["tau"]
    tau_core = res["transfer"]["taus"]["mcq"]
    Xs = prep["sim"].batch(list(wild.benchmark_text), list(wild.candidate_text))
    Xc = sd.constraint_frame(wild).reset_index(drop=True)
    X = pd.concat([Xs, Xc], axis=1)
    for c in ["a_num", "a_rel", "a_tgt", "a_ent"]: X[c + "_x"] = (X[c] >= 0.999).astype(float)
    cols = sd.SIM_COLS + sd.CON_COLS + sd.EXACT_COLS + sd.CHRONO
    zc_m, clf_m, sc_m = sd.fit_score_margin(prep, cols)
    zm = sd.mlp_logits(clf_m, sc_m.transform(X[cols]))
    zc_l, clf_l, sc_l = sd.fit_score(prep, cols, model="logreg")
    zl = clf_l.predict_proba(sc_l.transform(X[cols]))[:, 1]
    wild["remove_margin"] = zm >= tau_marg
    wild["remove_core"] = zl >= tau_core

    out = {"n": int(len(wild)), "thresholds": {"mlp_margin_mcq_local": float(tau_marg), "core_mcq_local": float(tau_core)},
           "by_silver": {}}
    for lab, sub in wild.groupby("silver"):
        out["by_silver"][lab] = dict(n=int(len(sub)),
                                     removal_margin=100 * float(sub.remove_margin.mean()),
                                     removal_core=100 * float(sub.remove_core.mean()))
        print(f"{lab:18s} n={len(sub):3d}  margin移除率={100*sub.remove_margin.mean():6.2f}%  core移除率={100*sub.remove_core.mean():6.2f}%")
    wild.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + "/r2_wild_pairs.csv", index=False)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + "/r2_wild.json", "w"), indent=1)
    print("saved results/r2_wild.{json,csv} | 注意:银标签由证据字段按 Li et al. 阈值规则重建,论文中如实披露")

if __name__ == "__main__":
    main()
