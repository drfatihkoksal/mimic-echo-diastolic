#!/usr/bin/env bash
# A4 zinciri (analysis_log Ek 16). Proje kökünden: bash external/posthoc/run_a4_chain.sh
set -eu
O=~/mimic-echo/rev_a4; mkdir -p "$O"
echo "=== $(date '+%F %T') prep112";  python3 external/posthoc/a4_pipeline.py prep112 --workers 18
echo "=== $(date '+%F %T') görünüm";  python3 grading/view_classify.py --npz-dir "$O/npz112" --out "$O/_views.csv" --fp16
echo "=== $(date '+%F %T') hires";    python3 external/posthoc/a4_pipeline.py hires --workers 18
echo "=== $(date '+%F %T') infer";    python3 external/posthoc/a4_pipeline.py infer
echo "=== $(date '+%F %T') A4 zinciri bitti"
