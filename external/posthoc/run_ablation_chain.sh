#!/usr/bin/env bash
# A2 zinciri (analysis_log Ek 14). Proje kökünden: bash external/posthoc/run_ablation_chain.sh
set -u
R=~/mimic-echo/runs/rev_abl; mkdir -p "$R"
for seed in 42 43 44; do
  for cfg in K A0 A1; do
    out="$R/${cfg}_s${seed}"
    [ -f "$out/binary_results.json" ] && { echo "var: $out"; continue; }
    case $cfg in K) extra="";; A0) extra="--aux-weight 0";; A1) extra="--aux-drop LAVi";; esac
    echo "=== $(date '+%F %T') $cfg seed $seed"
    python3 grading/train_binary.py --out-dir "$out" --seed $seed --stop-after 9 $extra > "$R/${cfg}_s${seed}.log" 2>&1 \
      || echo "HATA: $cfg seed $seed (bkz. $R/${cfg}_s${seed}.log)"
  done
done
python3 external/posthoc/infer_ablation.py "$R"/*_s4? > "$R/infer_external.log" 2>&1 || echo "HATA: dış çıkarım"
echo "=== $(date '+%F %T') zincir bitti"
