#!/usr/bin/env bash
# 5-kat OOF'u çökme-dayanıklı çalıştır.
#
# Bu makinede GPU uzun yük altında GSP firmware kilitlenmesiyle düşüyor ve PCIe'den
# tamamen kayboluyor (nvidia-smi "No devices were found"). İki ders:
#   1) GPU yokken yeniden denemek anlamsız — CUDA init'te anında patlar ve denemeleri yakar.
#      Bu yüzden her denemeden ÖNCE GPU sağlığını kontrol edip geri gelmesini bekliyoruz.
#   2) kfold_oof.py artık HER EPOCH durumu diske yazıyor → çöküşte en fazla ~1 epoch kaybolur,
#      yeniden başlayınca kaldığı epoch'tan devam eder.
set -u
cd "$(dirname "$0")"
LOG=~/mimic-echo/oof.log
MAX_TRY=${MAX_TRY:-20}
GPU_WAIT_MIN=${GPU_WAIT_MIN:-30}    # GPU'nun geri gelmesi için kaç dakika beklensin

gpu_ok() { nvidia-smi -L >/dev/null 2>&1; }

wait_for_gpu() {
  local waited=0
  while ! gpu_ok; do
    if [ $waited -ge $((GPU_WAIT_MIN * 60)) ]; then
      echo "!!! GPU $GPU_WAIT_MIN dk'dır yok — sürücü reset/reboot GEREKİYOR. Bırakılıyor." | tee -a "$LOG"
      return 1
    fi
    if [ $waited -eq 0 ]; then
      echo "!!! GPU görünmüyor (GSP kilitlenmesi olabilir). Geri gelmesi bekleniyor..." | tee -a "$LOG"
      echo "    Düzeltmek için:  sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia" | tee -a "$LOG"
      echo "    ya da:           sudo reboot" | tee -a "$LOG"
    fi
    sleep 60; waited=$((waited + 60))
  done
  [ $waited -gt 0 ] && echo "=== GPU geri geldi ($((waited / 60)) dk sonra) — devam ===" | tee -a "$LOG"
  return 0
}

for try in $(seq 1 $MAX_TRY); do
  wait_for_gpu || exit 1
  echo "=== deneme $try/$MAX_TRY — $(date '+%F %T') ===" | tee -a "$LOG"
  python3 kfold_oof.py --folds 5 >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== OOF TAMAMLANDI (deneme $try) — $(date '+%F %T') ===" | tee -a "$LOG"
    exit 0
  fi
  done_folds=$(ls ~/mimic-echo/runs/oof_fold*.csv 2>/dev/null | wc -l)
  echo "=== çıkış kodu $rc — $done_folds fold bitmiş; epoch-durumu korunuyor, devam edilecek ===" | tee -a "$LOG"
  sleep 30
done

echo "=== $MAX_TRY denemede bitmedi, vazgeçildi — $(date '+%F %T') ===" | tee -a "$LOG"
exit 1
