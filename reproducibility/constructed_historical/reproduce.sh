#!/usr/bin/env bash
# SafeDecontam 真实数据研究 · 一键复现
# 用法:cd code/ && bash reproduce.sh [数据目录]   (默认 ../../real_data_set)
set -euo pipefail
cd "$(dirname "$0")"
export SAFEDECONTAM_DATA="${1:-$(cd ../../real_data_set && pwd)}"
echo "数据目录: $SAFEDECONTAM_DATA"
python3 -c "import pandas, numpy, scipy, sklearn, matplotlib" || { echo "先执行: pip3 install -r requirements.txt"; exit 1; }
mkdir -p results
echo "== 1/5 构造 MCQ v1 配对 ==";              python3 build_pairs_mcq.py
echo "== 2/5 构造三家族 v2 配对 ==";            python3 build_pairs_v2.py
echo "== 3/5 主实验(约 8 分钟)==";             python3 run_v2.py
echo "== 4/5 margin 校准实验(约 7 分钟)==";    python3 run_margin.py
echo "== 5/5 野外审计试点 + 论文图 ==";          python3 run_r2_wild.py && python3 make_figures.py
echo "== 6/6(可选)跨语言探针家族 ==";       python3 run_xling.py || echo "(xling 可选步骤失败,不影响主结果)"
echo "完成:results/v2_results.json、results/r2_wild.json 与 figs/ 即论文全部数字与图。"
