#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四家族脚手架:跨语言 MCQ(等 HF 数据到位后运行)。
前置:先在 real_data_set/ 下运行
  python3 mac本地补齐/download_hf_datasets.py         # cais/mmlu 全量
  python3 new_2024_2026/download_hf_datasets_v2.py    # Global-MMLU / MMLU-Redux-2.0 / GSM8K-Platinum 等
设计(与论文关系分类对齐;构造后请抽查 20 行再入实验):
  b 侧      = cais/mmlu 全量 test 条目(有 Global-MMLU sample_id 回链的子集)
  C1_xling  = Global-MMLU 对应语种翻译(题+选项;答案标签不变)          -> remove
  C0_redux  = MMLU-Redux-2.0 中逐字沿用的条目(error_type 保留)         -> remove(再标注真实副本)
  H1        = 同学科另一条真实条目                                       -> retain
  H2_redux  = MMLU-Redux-2.0 标为标注错误且给出 correct_answer 的条目
              (同题面、决定性答案不同)                                  -> retain(边界情形,论文中单列讨论)
  R         = 跨学科随机条目                                             -> retain
运行:cd code/ && python3 build_family4.py [数据目录] [语种,默认 zh]
产出:{DATA}/pairs_mcq_xling.csv;之后我把它接入 run_v2 的家族列表重跑。
"""
import os, sys, hashlib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "..", "..", "real_data_set"))
LANG = sys.argv[2] if len(sys.argv) > 2 else "zh"
MMLU_RELEASE = "2020-09-07"

def need(path):
    if not os.path.exists(path):
        raise SystemExit(f"缺少 {path} —— 请先运行 HF 下载脚本(见文件头)")
    return path

def pick_idx(n, key, salt):
    return int(hashlib.sha256(f"{key}|{salt}".encode()).hexdigest(), 16) % n

def main():
    mmlu = pd.read_parquet(need(f"{DATA}/benchmarks/mmlu_full/mmlu_test.parquet"))
    gm = pd.read_parquet(need(f"{DATA}/derived_real/global_mmlu/Global-MMLU_test.parquet"))
    print("mmlu 列:", list(mmlu.columns)[:8]); print("global-mmlu 列:", list(gm.columns)[:10])
    # Global-MMLU: sample_id 形如 "subject/idx";按实际 schema 调整以下三行
    gm = gm[gm.get("lang", gm.get("language", pd.Series(dtype=str))) == LANG] if ("lang" in gm or "language" in gm) else gm
    rows = []
    made = 0
    for _, g in gm.iterrows():
        sid = str(g.get("sample_id", ""))
        q_en = str(g.get("question_en", "") or "")
        # 连接策略 1:sample_id;策略 2:英文题面文本匹配 —— 视 schema 打印结果选择
        # TODO(数据到位后确认字段名;以下按常见 schema 书写)
        q = str(g.get("question", "")); opts = [str(g.get(k, "")) for k in ["option_a","option_b","option_c","option_d"]]
        ans = str(g.get("answer", ""))
        if not q or not any(opts): continue
        gid = f"xling_{hashlib.sha256(sid.encode()).hexdigest()[:10]}"
        b_text = (q_en + " ") if q_en else ""
        rows.append(dict(group_id=gid, family="mcq_xling", relation="C1_xling", decision_label="remove",
                         benchmark_text=b_text or q, benchmark_answer=ans,
                         candidate_text=q + " " + " ".join(f"{L}. {o}" for L, o in zip("ABCD", opts)),
                         candidate_date="2024-12-05", candidate_source=f"global_mmlu_{LANG}",
                         benchmark_release=MMLU_RELEASE))
        made += 1
        if made >= 2000: break
    out = f"{DATA}/pairs_mcq_xling.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"草稿配对 {made} 条 -> {out}")
    print("注意:这是脚手架——先跑一次看两个 parquet 的真实字段名,把 TODO 处连接逻辑确认后再扩 H1/H2/R,")
    print("然后把结果发我(或直接连文件夹),我来补全负类构造并接入主实验重跑。")

if __name__ == "__main__":
    main()
