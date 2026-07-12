#!/usr/bin/env bash
# OOF bitince prognostik analizleri çalıştır: 1-yıllık mortalite + NT-proBNP.
# $1: beklenecek OOF sarmalayıcı PID'i (boş bırakılırsa hemen başlar).
#
# Bunlar SALT CPU (Cox / korelasyon) → ensemble GPU'yu kullanırken paralel koşabilirler,
# kuyrukta ensemble'ı beklemeleri gerekmez.
#
# KRİTİK: OOF çıktısı yoksa test-seti tahminlerine DÜŞÜLMEZ, çıkılır. Test seti (438 study,
# ~28 ölüm / ~74 NT-proBNP eşleşmesi) bu iki analiz için güçsüz — EF-düzeltilmiş Cox orada
# HR 0.97 (%95 GA 0.16–6.02) veriyordu. Sessizce güçsüz bir sonuç üretmektense hiç üretmemek
# daha iyi: "sonuçsuz" ile "güçsüz" karıştırılırsa bir analiz turu çöpe gider.
set -u
cd "$(dirname "$0")"
OOF=~/mimic-echo/runs/oof_predictions.csv
LOG=~/mimic-echo/analysis.log
OOF_PID=${1:-}

if [ -n "$OOF_PID" ]; then
  echo "[analiz] OOF (pid $OOF_PID) bekleniyor — $(date '+%F %T')" | tee -a "$LOG"
  while kill -0 "$OOF_PID" 2>/dev/null; do sleep 60; done
  echo "[analiz] OOF süreci bitti — $(date '+%F %T')" | tee -a "$LOG"
fi

if [ ! -f "$OOF" ]; then
  echo "[analiz] !!! $OOF YOK — OOF tamamlanmamış." | tee -a "$LOG"
  echo "[analiz] !!! Test-seti tahminleriyle koşmuyorum (güçsüz: ~28 ölüm / ~74 lab)." | tee -a "$LOG"
  echo "[analiz] !!! OOF'u bitirip bu scripti PID'siz yeniden çalıştır." | tee -a "$LOG"
  exit 1
fi
echo "[analiz] OOF hazır: $(($(wc -l < "$OOF") - 1)) study — analizler başlıyor $(date '+%F %T')" | tee -a "$LOG"

rc=0

echo "" | tee -a "$LOG"
echo "=================== 1-YILLIK MORTALİTE (OOF) ===================" | tee -a "$LOG"
python3 mortality_analysis.py --preds "$OOF" \
        --out ~/mimic-echo/runs/mortality_cohort_oof.csv >> "$LOG" 2>&1 \
  && echo "[analiz] mortalite TAMAM" | tee -a "$LOG" \
  || { echo "[analiz] mortalite BAŞARISIZ (çıkış $?)" | tee -a "$LOG"; rc=1; }

echo "" | tee -a "$LOG"
echo "=================== NT-proBNP (OOF) ===================" | tee -a "$LOG"
python3 ntprobnp_analysis.py --preds "$OOF" \
        --out ~/mimic-echo/runs/ntprobnp_cohort_oof.csv >> "$LOG" 2>&1 \
  && echo "[analiz] NT-proBNP TAMAM" | tee -a "$LOG" \
  || { echo "[analiz] NT-proBNP BAŞARISIZ (çıkış $?)" | tee -a "$LOG"; rc=1; }

echo "" | tee -a "$LOG"
if [ $rc -eq 0 ]; then
  echo "[analiz] === HER İKİ ANALİZ TAMAMLANDI — $(date '+%F %T') ===" | tee -a "$LOG"
else
  echo "[analiz] === EN AZ BİR ANALİZ BAŞARISIZ — $(date '+%F %T') ===" | tee -a "$LOG"
fi
exit $rc
