#!/usr/bin/env bash
# OOF bitince ensemble'ı başlat (tek GPU → sıralı çalışmalı).
# $1: beklenecek OOF sarmalayıcı PID'i (boş bırakılırsa hemen başlar).
#
# Bu makinede GPU uzun yük altında GSP kilitlenmesiyle PCIe'den düşüyor. Eski sürüm
# bu durumda 6 denemenin hepsini dakikalar içinde CUDA-init hatasıyla yakıyordu; artık
# her denemeden ÖNCE kartın geri gelmesi bekleniyor. ensemble_eval.py de her epoch
# durumu atomik yazıyor → çöküşte en fazla ~1 epoch kaybedilir.
set -u
cd "$(dirname "$0")"
LOG=~/mimic-echo/ensemble.log
CHAIN=~/mimic-echo/ens_chain.log
OOF_PID=${1:-}
MAX_TRY=${MAX_TRY:-20}
GPU_WAIT_MIN=${GPU_WAIT_MIN:-30}

gpu_ok() { nvidia-smi -L >/dev/null 2>&1; }

wait_for_gpu() {
  local waited=0
  while ! gpu_ok; do
    if [ $waited -ge $((GPU_WAIT_MIN * 60)) ]; then
      echo "[zincir] !!! GPU $GPU_WAIT_MIN dk'dır yok — sürücü reset/reboot GEREKİYOR. Bırakılıyor." | tee -a "$CHAIN"
      return 1
    fi
    if [ $waited -eq 0 ]; then
      echo "[zincir] !!! GPU görünmüyor (GSP kilitlenmesi olabilir). Geri gelmesi bekleniyor..." | tee -a "$CHAIN"
      echo "    Düzeltmek için:  sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia" | tee -a "$CHAIN"
    fi
    sleep 60; waited=$((waited + 60))
  done
  [ $waited -gt 0 ] && echo "[zincir] === GPU geri geldi ($((waited / 60)) dk sonra) — devam ===" | tee -a "$CHAIN"
  return 0
}

if [ -n "$OOF_PID" ]; then
  echo "[zincir] OOF (pid $OOF_PID) bekleniyor — $(date '+%F %T')" | tee -a "$CHAIN"
  while kill -0 "$OOF_PID" 2>/dev/null; do sleep 60; done
  echo "[zincir] OOF süreci bitti — $(date '+%F %T')" | tee -a "$CHAIN"
fi

if [ -f ~/mimic-echo/runs/oof_predictions.csv ]; then
  echo "[zincir] OOF çıktısı hazır ($(($(wc -l < ~/mimic-echo/runs/oof_predictions.csv) - 1)) study)" | tee -a "$CHAIN"
else
  # ensemble OOF'tan bağımsızdır (kilitli test üzerinde çalışır) → yine de devam edilir.
  echo "[zincir] UYARI: oof_predictions.csv yok — OOF başarısız görünüyor." | tee -a "$CHAIN"
  echo "[zincir] ensemble OOF'tan bağımsız olduğu için yine de başlatılıyor." | tee -a "$CHAIN"
fi

echo "[zincir] ensemble başlıyor — $(date '+%F %T')" | tee -a "$CHAIN"
for try in $(seq 1 $MAX_TRY); do
  wait_for_gpu || exit 1
  echo "=== ensemble deneme $try/$MAX_TRY — $(date '+%F %T') ===" | tee -a "$LOG"
  python3 ensemble_eval.py >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[zincir] ENSEMBLE TAMAMLANDI (deneme $try) — $(date '+%F %T')" | tee -a "$CHAIN"
    exit 0
  fi
  done_folds=$(ls ~/mimic-echo/runs/ensemble/ens_fold*_test.csv 2>/dev/null | wc -l)
  echo "[zincir] çıkış kodu $rc — $done_folds/5 fold bitmiş; epoch-durumu korunuyor, devam edilecek" | tee -a "$CHAIN"
  sleep 30
done

echo "[zincir] ensemble $MAX_TRY denemede bitmedi — $(date '+%F %T')" | tee -a "$CHAIN"
exit 1
