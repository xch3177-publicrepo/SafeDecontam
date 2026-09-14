#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四家族:跨语言 MCQ 探针(xling)。
构造:b = 原版 MMLU(3 学科用官方原文,其余用 MMLU-SR 双变体重建,已知 7 处编码退化+1 处词级出入,披露);
C1_en / C1_zh = LMSYS 全量英文改写 / 中文翻译改写完整提示(含 Answer 行,即真实污染工件原样),
仅使用"条数+答案序列"双重验证对齐的学科交集;H1 = 同学科另一题;H3 = LogiQA;R = 跨学科随机题。
协议:模型完全冻结(主种子三家族训练所得);xling 仅做家族本地校准(cal 选 τ,单层 α=0.05)+ 锁定测试。
这模拟部署场景:新家族到来,不重训,只用本地干净校准组定阈值。"""
import csv, hashlib, json, os, re, collections
import numpy as np, pandas as pd
import safedecontam as sd
from run_v2 import load_pairs, calibrated, split_z

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SAFEDECONTAM_DATA", "/home/claude/work/论文数据集")
LETTERS = "ABCD"
PER_SUBJ = 40  # 每学科最多采样源题数(确定性)

def parse_lmsys(path):
    by = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        t = json.loads(line)["text"]
        m = re.match(r"The following are multiple choice questions \(with answers\) about\s+(.+?)\.\n", t)
        s = m.group(1).strip().replace(" ", "_") if m else "?"
        am = re.search(r"Answer:\s*([A-D])\s*$", t.strip())
        body = re.sub(r"^The following are.*?\n\n", "", t, flags=re.S).strip()
        by[s].append(dict(body=body, ans=am.group(1) if am else "?"))
    return by

def aligned_subjects(by, recon):
    ok = set()
    for s, items in by.items():
        if s in recon and len(items) == len(recon[s]) and all(x["ans"] == r["ans"] for x, r in zip(items, recon[s])):
            ok.add(s)
    return ok

def pick(lst, key, salt):
    return lst[int(hashlib.sha256(f"{key}|{salt}".encode()).hexdigest(), 16) % len(lst)]

def build_pairs():
    _cand = [os.path.join(DATA, "new_2024_2026", "mmlu_full_reconstructed.json"),
             os.path.join(HERE, "results", "mmlu_full_reconstructed.json")]
    recon = json.load(open(next(p for p in _cand if os.path.exists(p))))
    # 3 个学科替换为官方原文
    for s in ["abstract_algebra", "high_school_us_history", "sociology"]:
        with open(f"{DATA}/benchmarks/mmlu_subjects_sample/{s}_test.csv", newline="", encoding="utf-8") as f:
            recon[s] = [dict(q=r[0].strip(), opts=[x.strip() for x in r[1:5]], ans=r[5].strip())
                        for r in csv.reader(f) if len(r) >= 6]
    en = parse_lmsys(f"{DATA}/derived_real/rephrased_lmsys/mmlu_english_full_prompt.jsonl")
    zh = parse_lmsys(f"{DATA}/derived_real/rephrased_lmsys/mmlu_chinese_full_prompt.jsonl")
    subs = sorted(aligned_subjects(en, recon) & aligned_subjects(zh, recon))
    print("双语对齐学科交集:", len(subs), subs[:6])
    # LogiQA 与跨学科随机池
    blocks = open(f"{DATA}/benchmarks/logiqa/Test.txt", encoding="utf-8").read().strip().split("\n\n")
    logiqa = [" ".join([x.strip() for x in b.strip().split("\n") if x.strip()][1:7]) for b in blocks]
    logiqa = [x for x in logiqa if len(x) > 60]
    rows = []
    all_items = [(s, i) for s in subs for i in range(len(recon[s]))]
    for s in subs:
        items = recon[s]
        idxs = sorted(range(len(items)),
                      key=lambda i: hashlib.sha256(f"{s}|{i}".encode()).hexdigest())[:PER_SUBJ]
        for i in idxs:
            it = items[i]
            gid = f"xling_{s}_{i:04d}"
            opts = " ".join(f"{L}. {o}" for L, o in zip(LETTERS, it["opts"]))
            b_text = f"{it['q']} {opts}"
            ans_txt = it["opts"][LETTERS.index(it["ans"])] if it["ans"] in LETTERS else ""
            def add(rel, lab, cand, date, src):
                rows.append(dict(group_id=gid, family="xling", relation=rel, decision_label=lab,
                                 benchmark_text=b_text, benchmark_answer=ans_txt, candidate_text=cand,
                                 candidate_date=date, candidate_source=src, benchmark_release="2020-09-07"))
            add("C0_exact", "remove", f"{it['q']} {opts} Answer: {it['ans']}. {ans_txt}", "2023-11-08", "mmlu_copy")
            add("C1_en_rephrase", "remove", en[s][i]["body"], "2023-11-08", "lmsys_en_full")
            add("C1_zh_rephrase", "remove", zh[s][i]["body"], "2023-11-08", "lmsys_zh_full")
            j = pick([k for k in range(len(items)) if k != i], gid, "h1")
            oth = items[j]
            add("H1_same_topic", "retain",
                oth["q"] + " " + " ".join(f"{L}. {o}" for L, o in zip(LETTERS, oth["opts"])),
                "2020-09-07", "mmlu_same_subject")
            add("H3_predating", "retain", pick(logiqa, gid, "h3"), "2020-04-16", "logiqa")
            so, io_ = pick([x for x in all_items if x[0] != s], gid, "r")
            ro = recon[so][io_]
            add("R_random", "retain", ro["q"] + " " + " ".join(f"{L}. {o}" for L, o in zip(LETTERS, ro["opts"])),
                "2020-09-07", "mmlu_other_subject")
    df = pd.DataFrame(rows)
    df.to_csv(f"{DATA}/pairs_xling.csv", index=False)
    print(f"xling 家族: {len(df)} 对 / {df.group_id.nunique()} 组 -> pairs_xling.csv")
    return df

def main():
    xl = build_pairs()
    pairs = load_pairs()                       # 三家族(用于冻结模型)
    prep = sd.prepare(pairs, seed=42579)       # 冻结:相似度层 + 训练分区
    # xling 的组切分(同种子;train 分区弃用——模型不重训)
    tr, ca, te = sd.grouped_split(xl.group_id, 42579)
    xl["part"] = ["train" if g in tr else ("cal" if g in ca else "test") for g in xl.group_id]
    frames = {p: xl[xl.part == p].reset_index(drop=True) for p in ["cal", "test"]}
    X = {}
    for p, df in frames.items():
        Xs = prep["sim"].batch(list(df.benchmark_text), list(df.candidate_text))
        Xc = sd.constraint_frame(df).reset_index(drop=True)
        Xp = pd.concat([Xs, Xc], axis=1)
        for c in ["a_num", "a_rel", "a_tgt", "a_ent"]: Xp[c + "_x"] = (Xp[c] >= 0.999).astype(float)
        X[p] = Xp
    cols = sd.SIM_COLS + sd.CON_COLS + sd.EXACT_COLS + sd.CHRONO
    out = {"n_groups": {p: int(frames[p].group_id.nunique()) for p in frames},
           "n_pairs": {p: int(len(frames[p])) for p in frames}, "variants": {}}
    scorers = {}
    z_m, clf_m, sc_m = sd.fit_score_margin(prep, cols)
    scorers["mlp_margin"] = lambda df, Xp: sd.mlp_logits(clf_m, sc_m.transform(Xp[cols]))
    z_l, clf_l, sc_l = sd.fit_score(prep, cols, model="logreg")
    scorers["core"] = lambda df, Xp: clf_l.predict_proba(sc_l.transform(Xp[cols]))[:, 1]
    for name, fn in scorers.items():
        zc, zt = fn(frames["cal"], X["cal"]), fn(frames["test"], X["test"])
        tau = sd.select_threshold(frames["cal"], zc, 0.05, alpha=0.05)
        ev = sd.eval_at(frames["test"], zt, tau)
        # 逐关系
        fr = frames["test"].assign(rm=zt >= tau)
        rel = {r: dict(n=int(len(g)), removed=int(g.rm.sum()))
               for r, g in fr.groupby("relation")}
        out["variants"][name] = dict(tau=float(tau), test=ev, cal=sd.eval_at(frames["cal"], zc, tau),
                                     by_relation=rel)
        print(f"xling {name:10s} tau={tau if np.isfinite(tau) else float('inf'):.4f} "
              f"CR={ev['CR']:6.2f} CDR={ev['CDR']:.3f} GCDR={ev['GCDR']:.3f} U95={ev['U95_worst']:.2f}")
        for r, v in sorted(rel.items()):
            print(f"    {r:ymax18s}".replace("ymax", "") if False else f"    {r:18s} {v['removed']}/{v['n']}")
    res = json.load(open(os.path.join(HERE, "results", "v2_results.json")))
    res["xling_probe"] = out
    json.dump(res, open(os.path.join(HERE, "results", "v2_results.json"), "w"), indent=1, default=str)
    print("saved -> v2_results.json[xling_probe]")

if __name__ == "__main__":
    main()
