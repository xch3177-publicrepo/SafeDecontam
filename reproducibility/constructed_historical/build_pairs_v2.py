#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 配对表:在 v1 基础上加入 2024–2025 最新真实变体,并新增 MATH 家族。
新增来源:
  GSM-Symbolic (Apple, 2024-12, CC-BY-NC-ND):answer≠原答案 -> H2_symbolic/p1/p2(retain)
  MMLU-SR (2024-06):符号替换变体,答案保持 -> C1_sr(remove)
  MATH-Perturb (ICML 2025) + MATH 原题(shingle 唯一匹配回链):新家族 math
匹配规程(论文披露):MATH-P-Simple 逐题与全部 12,500 道 MATH 原题(同 type、Level 5)
按 3-词 shingle 召回匹配;仅保留 top1>=0.35 且 top1>=1.5*top2 的唯一匹配;
Hard 共享 problem_id,经由 Simple 的匹配取得原题。
"""
import csv, hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
csv.field_size_limit(10_000_000)

DATA = os.environ.get("SAFEDECONTAM_DATA", "/home/claude/work/论文数据集")
REPOS = os.environ.get("SAFEDECONTAM_REPOS", "/home/claude/work/repos")
NEW = os.path.join(DATA, "new_2024_2026")
def _find(*cands):
    for c in cands:
        if os.path.exists(c): return c
    raise FileNotFoundError(cands)

LETTERS = "ABCD"

def toks(t): return re.findall(r"[a-z0-9$%.,'/\\{}^_+=()-]+", t.lower())
def shingles(t, n=3):
    tt = toks(t); return set(tuple(tt[i:i+n]) for i in range(len(tt)-n+1))
def pick(lst, key, salt):
    return lst[int(hashlib.sha256(f"{key}|{salt}".encode()).hexdigest(), 16) % len(lst)]
def norm_num(s):
    s = str(s).replace(",", "").strip()
    try: return round(float(s), 6)
    except Exception: return s.strip()

# ---------- 算术家族 v2:v1 + GSM-Symbolic ----------
def build_arith():
    rows = list(csv.DictReader(open(f"{DATA}/pairs_gsm8k_real.csv", encoding="utf-8")))
    for r in rows: r["benchmark_release"] = "2021-10-27"
    by_q = {}
    for fn, src in [("GSM_symbolic.jsonl", "gsm_symbolic"), ("GSM_p1.jsonl", "gsm_symbolic_p1"),
                    ("GSM_p2.jsonl", "gsm_symbolic_p2")]:
        for l in open(_find(f"{REPOS}/ml-gsm-symbolic/generated_data/{fn}", f"{NEW}/{fn}")):
            d = json.loads(l)
            by_q.setdefault(d["original_question"].strip(), {}).setdefault(src, []).append(d)
    b_index = {}
    for r in rows:
        b_index.setdefault(r["benchmark_text"].strip(), (r["group_id"], r["benchmark_answer"]))
    added, skipped_same = 0, 0
    for oq, srcs in by_q.items():
        hit = b_index.get(oq)
        if not hit: continue
        gid, b_ans = hit
        for src, insts in srcs.items():
            k = 2 if src == "gsm_symbolic" else 1
            insts_sorted = sorted(insts, key=lambda d: d["instance"])
            chosen, ci = [], 0
            while len(chosen) < k and ci < len(insts_sorted):
                d = insts_sorted[ci]; ci += 1
                if norm_num(d["answer"]) == norm_num(d["original_answer"]):
                    skipped_same += 1; continue
                chosen.append(d)
            for d in chosen:
                rows.append(dict(group_id=gid, family="arithmetic", relation=f"H2_{src}",
                                 decision_label="retain", benchmark_text=oq, benchmark_answer=b_ans,
                                 candidate_text=d["question"].strip(), candidate_date="2024-12-05",
                                 candidate_source=src, benchmark_release="2021-10-27"))
                added += 1
    print(f"arithmetic: +{added} 条 GSM-Symbolic(跳过答案巧合相同 {skipped_same});总 {len(rows)}")
    return rows

# ---------- MCQ 家族 v2:v1 + MMLU-SR ----------
def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.reader(f) if len(r) >= 6]

def build_mcq():
    rows = list(csv.DictReader(open(f"{DATA}/pairs_mcq_real.csv", encoding="utf-8")))
    added = 0
    for subject in ["abstract_algebra", "high_school_us_history", "sociology"]:
        orig = read_csv_rows(f"{DATA}/benchmarks/mmlu_subjects_sample/{subject}_test.csv")
        sr = read_csv_rows(_find(f"{REPOS}/MMLU-SR/dataset/question_and_answer_test/question_and_answer_{subject}_test.csv", f"{NEW}/question_and_answer_{subject}_test.csv"))
        assert len(orig) == len(sr), (subject, len(orig), len(sr))
        for i, (o, s) in enumerate(zip(orig, sr)):
            gid = f"mmlu_{subject}_{i:04d}"
            ans_idx = LETTERS.index(o[5].strip()) if o[5].strip() in LETTERS else 0
            b_text = f"{o[0].strip()} " + " ".join(f"{L}. {x.strip()}" for L, x in zip(LETTERS, o[1:5]))
            cand = f"{s[0].strip()} " + " ".join(f"{L}. {x.strip()}" for L, x in zip(LETTERS, s[1:5]))
            rows.append(dict(group_id=gid, family="mcq", relation="C1_sr", decision_label="remove",
                             benchmark_text=b_text, benchmark_answer=o[1 + ans_idx].strip(),
                             candidate_text=cand, candidate_date="2024-06-20",
                             candidate_source="mmlu_sr_qa", benchmark_release="2020-09-07"))
            added += 1
    print(f"mcq: +{added} 条 MMLU-SR;总 {len(rows)}")
    return rows

# ---------- MATH 家族(新):MATH-Perturb + 原题匹配 ----------
def boxed_answer(sol):
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", sol)
    return m[-1].strip() if m else ""

def build_math():
    orig = []
    for fn, split in [("math_test_mirror.jsonl", "test"), ("math_train_mirror.jsonl", "train")]:
        for l in open(_find(f"{REPOS}/{fn}", f"{NEW}/{fn}")):
            d = json.loads(l)
            if d["level"] == "Level 5":
                orig.append(dict(problem=d["problem"].strip(), type=d["type"], split=split,
                                 solution=d["solution"], answer=boxed_answer(d["solution"])))
    print(f"MATH L5 原题池: {len(orig)}")
    sh = [shingles(o["problem"]) for o in orig]
    simple = [json.loads(l) for l in open(_find(f"{REPOS}/MATH-Perturb/math_perturb/math_perturb_simple.jsonl", f"{NEW}/math_perturb_simple.jsonl"))]
    hard = {d["problem_id"]: d for d in (json.loads(l) for l in open(_find(f"{REPOS}/MATH-Perturb/math_perturb/math_perturb_hard.jsonl", f"{NEW}/math_perturb_hard.jsonl")))}
    matches, amb = {}, 0
    for s in simple:
        ss = shingles(s["problem"])
        if not ss: continue
        scores = []
        for j, o in enumerate(orig):
            if o["type"] != s["type"] or o["split"] != s["original_split"]: continue
            inter = len(ss & sh[j])
            scores.append((inter / max(1, len(ss)), j))
        scores.sort(reverse=True)
        if not scores or scores[0][0] < 0.35: amb += 1; continue
        if len(scores) > 1 and scores[0][0] < 1.5 * scores[1][0]: amb += 1; continue
        matches[s["problem_id"]] = (scores[0][1], s)
    print(f"MATH-P-Simple 唯一匹配: {len(matches)}/279(弃 {amb})")
    rows = []
    type_pool = {}
    for j, o in enumerate(orig): type_pool.setdefault(o["type"], []).append(j)
    other_pool = [o["problem"] for o in orig]
    for pid, (j, s) in sorted(matches.items()):
        o = orig[j]; gid = f"math_{pid}"
        def add(rel, label, cand, date, src):
            rows.append(dict(group_id=gid, family="math", relation=rel, decision_label=label,
                             benchmark_text=o["problem"], benchmark_answer=o["answer"],
                             candidate_text=cand, candidate_date=date, candidate_source=src,
                             benchmark_release="2021-03-05"))
        add("C0_exact", "remove", f"{o['problem']}\nSolution: {o['solution']}", "2025-02-10", "math_copy")
        tail = o["solution"].split(". ")[-2:]
        add("C2_answer_bearing", "remove",
            "Final steps of a worked solution: " + ". ".join(tail), "2025-02-10", "math_solution_frag")
        add("H2_perturb_simple", "retain", s["problem"].strip(), "2025-02-10", "math_perturb_simple")
        if pid in hard:
            add("H2_perturb_hard", "retain", hard[pid]["problem"].strip(), "2025-02-10", "math_perturb_hard")
        pool = [x for x in type_pool[o["type"]] if x != j]
        add("H1_same_topic", "retain", orig[pick(pool, gid, "h1")]["problem"] if pool else pick(other_pool, gid, "h1"),
            "2021-03-05", "math_same_type")
        add("R_random", "retain", pick(other_pool, gid, "r"), "2021-03-05", "math_random")
    gs = len({r["group_id"] for r in rows})
    print(f"math 家族: {len(rows)} 对 / {gs} 组")
    return rows

def write(rows, path):
    cols = ["group_id","family","relation","decision_label","benchmark_text","benchmark_answer",
            "candidate_text","candidate_date","candidate_source","benchmark_release"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print("->", path)

if __name__ == "__main__":
    write(build_arith(), f"{DATA}/pairs_arith_v2.csv")
    write(build_mcq(), f"{DATA}/pairs_mcq_v2.csv")
    write(build_math(), f"{DATA}/pairs_math_v2.csv")
