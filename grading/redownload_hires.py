#!/usr/bin/env python3
"""
Faz B sonrası — TUTULAN (A4C/A2C/PLAX) klipleri GCS'ten YÜKSEK ÇÖZÜNÜRLÜKTE yeniden indir+işle.

Neden: Faz A NPZ'leri 112×112 gri + 32 frame + en-boy EZİLMİŞ (squash) — nihai eğitim için kayıplı.
Bu adım sadece view-filtreli kliplere (~34.8k) GCS orijinalini bir kez daha çeker ve KAYIPSIZ-geometrili
üretir. Bir kez indir, bir daha indirme (tüm frame'ler saklanır; eğitimde alt-örnekle).

Format (kullanıcı kararı):
  - TÜM frame'ler (native, değişken uzunluk T).
  - 224×224, EN-BOY KORUNARAK: US-sektör bbox'a kırp -> kareye PAD (siyah) -> resize (squash YOK).
  - Gri = YBR Y-kanalı (gerçek luma; düşük-res'teki R-kanalından temiz). Tek kanal (yüklerken 3'e kopyala).
  - Etiketler manifest_study'den + view/view_prob (Faz B) NPZ'ye gömülür.

Girdi:  ~/mimic-echo/redownload_kept.csv (dicom_id,study_id,view,prob,gcs_uri)
Çıktı:  ~/mimic-echo/npz_hires/<dicom_id>.npz   (frames=(T,224,224) uint8 + meta)
Resumable: çıktı NPZ varsa atlanır.

Kullanım:
  # pilot (görsel doğrulama, 4 klip):
  python3 redownload_hires.py --kept ~/mimic-echo/redownload_kept.csv \
    --manifest ~/mimic-echo/manifest_study.csv --out-dir ~/mimic-echo/npz_hires --limit 4 --preview
  # tam koşu:
  nohup python3 redownload_hires.py --kept ~/mimic-echo/redownload_kept.csv \
    --manifest ~/mimic-echo/manifest_study.csv --out-dir ~/mimic-echo/npz_hires \
    --procs 12 --io-threads 12 > ~/mimic-echo/redownload_hires.log 2>&1 &
"""
from __future__ import annotations
import argparse, csv, io, os, subprocess, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import pydicom
from pydicom.pixels import iter_pixels
import cv2
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shrink_pipeline import classify

BUCKET = "mimic-iv-echo-1.0.physionet.org"
GCS_PREFIX = f"gs://{BUCKET}/"
URL = f"https://storage.googleapis.com/{BUCKET}/"
BILLING = os.environ.get("GCP_BILLING_PROJECT", "")   # export GCP_BILLING_PROJECT=<projeniz>

SESSION = requests.Session()
SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=64, pool_connections=64))

_tok = {'val': None, 'ts': 0.0}
_tok_lock = threading.Lock()
def get_token(force=False):
    with _tok_lock:
        if force or _tok['val'] is None or (time.time() - _tok['ts']) > 2400:
            _tok['val'] = subprocess.run(['gcloud', 'auth', 'print-access-token'],
                                         capture_output=True, text=True).stdout.strip()
            _tok['ts'] = time.time()
        return _tok['val']

def http_get(obj, retries=5):
    for a in range(retries):
        try:
            h = {'Authorization': f'Bearer {get_token()}', 'x-goog-user-project': BILLING}
            r = SESSION.get(URL + obj, headers=h, timeout=180)
            if r.status_code in (200, 206):
                return r.content
            if r.status_code == 401:
                get_token(force=True)
        except Exception:
            pass
        time.sleep(1.5 * (a + 1))
    return None

# --------------------------------------------------------------------------- #
def process_clip_hires(ds, bbox, size=224):
    """DICOM -> (T, size, size) uint8. TÜM frame, Y-luma, bbox kırp, kareye PAD, resize.
    Bellek: iter_pixels ile frame-frame decode (tepe ~tek native kare)."""
    x0 = x1 = y0 = y1 = None
    if bbox:
        x0, x1, y0, y1 = bbox
    out = []
    for fr in iter_pixels(ds, raw=True):                 # native YBR, lazy (tüm frame)
        g = fr[..., 0] if fr.ndim == 3 else fr           # Y-kanalı = luma
        if bbox and x1 > x0 and y1 > y0:
            g = g[y0:y1, x0:x1]                           # US sektör kırp
        h, w = g.shape
        s = max(h, w)                                     # kareye pad (en-boy koru)
        if h != w:
            sq = np.zeros((s, s), np.uint8)
            yo = (s - h) // 2; xo = (s - w) // 2
            sq[yo:yo + h, xo:xo + w] = g
            g = sq
        out.append(cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA))
    return np.stack(out) if out else None

# --- istatistik ---
_stat = defaultdict(int)
_stat_lock = threading.Lock()
def bump(k, v=1):
    with _stat_lock: _stat[k] += v

def process_one(rec, labels, out_dir, size):
    did, sid, view, prob, obj = rec
    outp = os.path.join(out_dir, f"{did}.npz")
    if os.path.exists(outp):
        bump('exists'); return
    data = http_get(obj)
    if data is None:
        bump('dl_fail'); return
    try:
        ds = pydicom.dcmread(io.BytesIO(data))
        cat, bbox, meta = classify(ds)
        clip = process_clip_hires(ds, bbox, size)
    except Exception:
        bump('decode_fail'); return
    if clip is None:
        bump('empty'); return
    lab = labels.get(sid, {})
    np.savez_compressed(
        outp, frames=clip, view=view, view_prob=str(prob),
        study_id=sid, subject_id=lab.get('subject_id', ''),
        grade_2025=lab.get('grade_2025', ''), grade_2016=lab.get('grade_2016', ''),
        confidence=lab.get('confidence', ''), age=lab.get('age', ''),
        gender=lab.get('gender', ''), ef=lab.get('EF', ''),
        n_orig_frames=meta['n_frames'], cine_rate=str(meta['cine_rate']),
        rows=meta['rows'], cols=meta['cols'], bbox=str(bbox), photometric=meta['photometric'])
    bump('written'); bump('frames_total', clip.shape[0])

def work_chunk(payload):
    recs, labels, out_dir, size, io_threads = payload
    _stat.clear()
    with ThreadPoolExecutor(max_workers=io_threads) as tp:
        list(tp.map(lambda r: process_one(r, labels, out_dir, size), recs))
    return dict(_stat)

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kept', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--io-threads', type=int, default=12)
    ap.add_argument('--chunk', type=int, default=150)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    labels = {}
    with open(args.manifest, newline='') as f:
        for r in csv.DictReader(f): labels[r['study_id']] = r

    recs = []
    with open(args.kept, newline='') as f:
        for r in csv.DictReader(f):
            obj = r['gcs_uri'].replace(GCS_PREFIX, '')
            recs.append((r['dicom_id'], r['study_id'], r['view'], r['prob'], obj))
    if args.limit: recs = recs[:args.limit]
    total = len(recs)
    print(f"[{time.strftime('%H:%M:%S')}] Tutulan klip: {total} | size={args.size} tüm-frame "
          f"| procs={args.procs} io={args.io_threads}", flush=True)

    chunks = [recs[i:i + args.chunk] for i in range(0, total, args.chunk)]
    payloads = [(c, labels, args.out_dir, args.size, args.io_threads) for c in chunks]
    agg = defaultdict(int); t0 = time.time(); done_clips = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = {ex.submit(work_chunk, p): i for i, p in enumerate(payloads)}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                st = fut.result()
                for kk, vv in st.items(): agg[kk] += vv
            except Exception as e:
                print(f"[HATA] chunk: {e}", flush=True); continue
            done_clips = agg.get('written', 0) + agg.get('exists', 0)
            if k % 10 == 0 or k == len(chunks):
                el = time.time() - t0; rate = done_clips / el if el else 0
                eta = (total - done_clips) / rate / 60 if rate else 0
                print(f"[{time.strftime('%H:%M:%S')}] chunk {k}/{len(chunks)} | yazılan={agg.get('written',0)} "
                      f"| {rate:.0f} klip/s | ETA {eta:.0f} dk | {dict(sorted(agg.items()))}", flush=True)
    el = time.time() - t0
    fr = agg.get('frames_total', 0); wr = max(agg.get('written', 1), 1)
    print(f"\n[{time.strftime('%H:%M:%S')}] BİTTİ. {el/60:.1f} dk | {dict(sorted(agg.items()))}")
    print(f"Ortalama frame/klip: {fr/wr:.1f}")

    if args.preview:
        prev_dir = os.path.join(args.out_dir, '_preview'); os.makedirs(prev_dir, exist_ok=True)
        import glob
        for fp in sorted(glob.glob(os.path.join(args.out_dir, '*.npz')))[:args.limit or 4]:
            z = np.load(fp, allow_pickle=True); f = z['frames']
            mid = f[len(f) // 2]
            cv2.imwrite(os.path.join(prev_dir, os.path.basename(fp).replace('.npz', f"_{z['view']}.png")), mid)
        print(f"Önizleme: {prev_dir}")

if __name__ == '__main__':
    main()
