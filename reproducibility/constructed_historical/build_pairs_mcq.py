import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构造 MCQ 家族真实配对表(MMLU 3 学科 × LMSYS 官方改写 + 真实负类)。
关系菜单(与算术家族不同处已在论文中说明):
  C0 exact          = 原题干+选项+答案                               remove
  C1 transformed    = LMSYS 改写题干(跳过空条目)+ 原选项(无答案)   remove
  C2 answer-bearing = 无题干;选项 + "The correct answer is X: ..."   remove
  H1 same-topic     = 同学科另一道真实 MMLU 题                        retain
  H1b cross-bench   = LogiQA 逻辑 MCQ(2020-04,亦为 H3 predating)   retain
  R random          = SVAMP 应用题文本                                retain
"""
import csv, hashlib, json, os

DATA = os.environ.get("SAFEDECONTAM_DATA", "/home/claude/work/论文数据集")
MMLU_RELEASE = "2020-09-07"   # arXiv:2009.03300 v1
DATES = {"lmsys": "2023-11-08", "logiqa": "2020-04-16", "svamp": "2021-03-15", "copy": "2023-11-08"}
LETTERS = "ABCD"

def read_mmlu(subject):
    rows = []
    with open(f"{DATA}/benchmarks/mmlu_subjects_sample/{subject}_test.csv", newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 6:
                rows.append(dict(q=r[0].strip(), opts=[x.strip() for x in r[1:5]], ans=r[5].strip()))
    return rows

def read_rephrase(subject):
    out = []
    with open(f"{DATA}/derived_real/rephrased_lmsys/{subject}_test_rephrase_english_filtered_question.jsonl") as f:
        for line in f:
            out.append(json.loads(line).get("text", "").strip())
    return out

def read_logiqa():
    blocks = open(f"{DATA}/benchmarks/logiqa/Test.txt", encoding="utf-8").read().strip().split("\n\n")
    out = []
    for b in blocks:
        ls = [x.strip() for x in b.strip().split("\n") if x.strip()]
        if len(ls) >= 7:
            out.append(ls[1] + " " + ls[2] + " " + " ".join(ls[3:7]))
    return out

def read_svamp():
    with open(f"{DATA}/hard_negatives/svamp/SVAMP.json") as f:
        return [(d.get("Body", "") + " " + d.get("Question", "")).strip() for d in json.load(f)]

def pick(lst, key, salt):
    return lst[int(hashlib.sha256(f"{key}|{salt}".encode()).hexdigest(), 16) % len(lst)]

def main():
    logiqa, svamp = read_logiqa(), read_svamp()
    rows = []
    for subject in ["abstract_algebra", "high_school_us_history", "sociology"]:
        items, reph = read_mmlu(subject), read_rephrase(subject)
        assert len(items) == len(reph), (subject, len(items), len(reph))
        n_skip = sum(1 for t in reph if not t)
        print(f"{subject}: {len(items)} 题, 改写空条目 {n_skip}")
        for i, it in enumerate(items):
            gid = f"mmlu_{subject}_{i:04d}"
            opts_txt = " ".join(f"{L}. {o}" for L, o in zip(LETTERS, it["opts"]))
            ans_idx = LETTERS.index(it["ans"]) if it["ans"] in LETTERS else 0
            ans_txt = it["opts"][ans_idx]
            b_text = f"{it['q']} {opts_txt}"
            def add(rel, label, cand, date, src):
                rows.append(dict(group_id=gid, family="mcq", relation=rel, decision_label=label,
                                 benchmark_text=b_text, benchmark_answer=ans_txt,
                                 candidate_text=cand, candidate_date=date, candidate_source=src,
                                 benchmark_release=MMLU_RELEASE))
            add("C0_exact", "remove", f"{it['q']} {opts_txt} Answer: {it['ans']}. {ans_txt}",
                DATES["copy"], "mmlu_copy")
            if reph[i]:
                add("C1_transformed", "remove", f"{reph[i]} {opts_txt}", DATES["lmsys"], "lmsys_rephrase")
            add("C2_answer_bearing", "remove",
                f"Options were: {opts_txt}. The correct answer is {it['ans']}: {ans_txt}.",
                DATES["copy"], "mmlu_answer_frag")
            j = (i + 7) % len(items)
            if j == i: j = (i + 1) % len(items)
            other = items[j]
            add("H1_same_topic", "retain",
                other["q"] + " " + " ".join(f"{L}. {o}" for L, o in zip(LETTERS, other["opts"])),
                MMLU_RELEASE, "mmlu_same_subject_other_item")
            add("H3_predating", "retain", pick(logiqa, gid, "h3"), DATES["logiqa"], "logiqa")
            add("R_random", "retain", pick(svamp, gid, "r"), DATES["svamp"], "svamp")
    out = f"{DATA}/pairs_mcq_real.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    n_rm = sum(1 for r in rows if r["decision_label"] == "remove")
    gs = len({r["group_id"] for r in rows})
    print(f"共 {len(rows)} 对 / {gs} 组(remove {n_rm} / retain {len(rows)-n_rm})-> {out}")

if __name__ == "__main__":
    main()
