#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dense 检索基线(在你的 Mac 上运行;云端访问不了 Hugging Face 模型)。
冻结的 sentence-transformers all-MiniLM-L6-v2 余弦相似度作为分数 z,
与论文完全相同的分组切分(种子 42579)与家族分层精确校准规则。

用法:
  pip3 install -r requirements.txt sentence-transformers
  cd code/ && python3 dense_baseline_mac.py [数据目录]     # 默认 ../../real_data_set
输出:全局(arith+MCQ)与三个家族本地的 CR/CDR/GCDR@5%,追加写入 results/dense_baseline.json
论文用途:填补 §Threats 承认缺失的 frozen dense retriever 基线;
预期(与模拟器一致的假说):排序不错,但在严格组预算下无法区分 H1/H2 —— 以实际跑出的数字为准。
"""
import json, os, sys
import numpy as np, pandas as pd
import safedecontam as sd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "..", "..", "real_data_set"))
os.environ["SAFEDECONTAM_DATA"] = DATA

def main():
    from sentence_transformers import SentenceTransformer
    from run_v2 import load_pairs, calibrated
    pairs = load_pairs()
    prep = sd.prepare(pairs, seed=42579)   # 只用它的切分与 frames;dense 分数独立计算
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def z_of(part):
        fr = prep["frames"][part]
        eb = model.encode(list(fr.benchmark_text), batch_size=64, normalize_embeddings=True,
                          show_progress_bar=True)
        ed = model.encode(list(fr.candidate_text), batch_size=64, normalize_embeddings=True,
                          show_progress_bar=True)
        return (eb * ed).sum(1)
    zc, zt = z_of("cal"), z_of("test")
    out = {"model": "all-MiniLM-L6-v2 (frozen)", "seed": 42579}
    r = calibrated(prep, zc, zt, ["arithmetic", "mcq"], 0.05)
    out["global_eps5"] = {k: r[k] for k in ["tau", "cal", "test"]}
    print(f"dense 全局@5%: CR={r['test']['CR']:.2f} CDR={r['test']['CDR']:.3f} GCDR={r['test']['GCDR']:.3f}")
    out["family_local"] = {}
    for fam in ["arithmetic", "mcq", "math"]:
        rr = calibrated(prep, zc, zt, [fam], 0.05)
        out["family_local"][fam] = {"tau": rr["tau"], "test": rr["test"]}
        print(f"dense {fam} 本地@5%: CR={rr['test']['CR']:.2f} GCDR={rr['test']['GCDR']:.3f}")
    p = os.path.join(HERE, "results", "dense_baseline.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1, default=str)
    print("saved", p, "\n把这些数字发我,我可以直接把 dense 基线行加进论文 Table 与文字。")

if __name__ == "__main__":
    main()
