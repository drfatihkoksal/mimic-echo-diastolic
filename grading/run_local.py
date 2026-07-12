#!/usr/bin/env python3
"""
Yerel tam koşu — MIMIC-IV-ECHO B-mode cine -> NPZ (VM'siz, bu makinede).

Bu ortamda GCS bant genişliği yüksek (27-588 MB/s), bu yüzden VM yerine yerelde koşuyoruz.
Egress'i yarıya indirmek için ÖNCE header range-read ile B-mode cine'leri seçer, SADECE onları
tam indirir (Doppler'i hiç indirmez -> ~431 GB yerine header+B-mode ~460 GB).

Teknik:
  - Auth: mevcut gcloud kişisel-hesap access-token (`gcloud auth print-access-token`), ~40dk'da yenilenir.
  - GCS erişim: doğrudan HTTPS + `x-goog-user-project` (requester-pays), paralel (ThreadPool).
  - Resumable: tamamlanan study'ler `_done.txt`'e yazılır; tekrar koşunca atlanır.
  - İşleme: shrink_pipeline.classify/process_clip yeniden kullanılır (pilotta doğrulandı).

Kullanım:
  nohup python3 run_local.py --manifest-dicom ~/mimic-echo/manifest_dicom.csv \
    --manifest ~/mimic-echo/manifest_study.csv --out-dir ~/mimic-echo/npz \
    > ~/mimic-echo/run_local.log 2>&1 &
"""
from __future__ import os
import annotations
import argparse, csv, io, os, subprocess, sys, threading, time, glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import pydicom
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shrink_pipeline import classify, process_clip   # pilotta doğrulanmış işleme

BUCKET = "mimic-iv-echo-1.0.physionet.org"
GCS_PREFIX = f"gs://{BUCKET}/"
URL = f"https://storage.googleapis.com/{BUCKET}/"
# requester-pays için KENDİ GCP projenizin kimliği — ortam değişkeninden okunur:
#   export GCP_BILLING_PROJECT=<projeniz>
BILLING = os.environ.get("GCP_BILLING_PROJECT", "")
if not BILLING:
    raise SystemExit("GCP_BILLING_PROJECT tanımlı değil — requester-pays indirme için gerekli.")
HEADER_BYTES = 131071          # header için ilk 128KB (yeterli, test edildi)

# --- Session (büyük bağlantı havuzu) ---
SESSION = requests.Session()
SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=64, pool_connections=64))

# --- Token yönetimi (thread-safe, otomatik yenileme) ---
_tok = {'val': None, 'ts': 0.0}
_tok_lock = threading.Lock()
def get_token(force=False):
    with _tok_lock:
        if force or _tok['val'] is None or (time.time() - _tok['ts']) > 2400:  # 40dk
            _tok['val'] = subprocess.run(['gcloud', 'auth', 'print-access-token'],
                                         capture_output=True, text=True).stdout.strip()
            _tok['ts'] = time.time()
        return _tok['val']

def http_get(obj, rng=None, retries=5):
    """GCS nesnesini (opsiyonel Range) indir. Döner bytes | None."""
    for a in range(retries):
        try:
            h = {'Authorization': f'Bearer {get_token()}', 'x-goog-user-project': BILLING}
            if rng: h['Range'] = f'bytes={rng}'
            r = SESSION.get(URL + obj, headers=h, timeout=120)
            if r.status_code in (200, 206):
                return r.content
            if r.status_code == 401:
                get_token(force=True)
        except Exception:
            pass
        time.sleep(1.5 * (a + 1))
    return None

# --- Progres/istatistik (thread-safe) ---
_stat = defaultdict(int)
_stat_lock = threading.Lock()
def bump(k, v=1):
    with _stat_lock: _stat[k] += v

# --------------------------------------------------------------------------- #
def read_header(obj):
    """Range-read header -> (kategori, bbox, meta) | (None,..)."""
    data = http_get(obj, rng=f'0-{HEADER_BYTES}')
    if data is None: return (None, None, None)
    try:
        ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True, force=True)
        return classify(ds)
    except Exception:
        return (None, None, None)

def download_process(obj, bbox, meta, label, out_dir, size, n_frames, io_pool):
    """B-mode cine'yi tam indir -> shrink -> NPZ."""
    di = os.path.basename(obj).replace('.dcm', '')
    outp = os.path.join(out_dir, f"{di}.npz")
    if os.path.exists(outp):
        bump('npz_exists'); return
    data = http_get(obj)
    if data is None: bump('dl_fail'); return
    try:
        ds = pydicom.dcmread(io.BytesIO(data))
        clip = process_clip(ds, bbox, meta['photometric'], size, n_frames)
    except Exception:
        bump('decode_fail'); return
    np.savez_compressed(
        outp, frames=clip,
        study_id=label.get('study_id', ''), subject_id=label.get('subject_id', ''),
        grade_2025=label.get('grade_2025', ''), grade_2016=label.get('grade_2016', ''),
        confidence=label.get('confidence', ''), age=label.get('age', ''),
        gender=label.get('gender', ''), ef=label.get('EF', ''),
        n_orig_frames=meta['n_frames'], cine_rate=str(meta['cine_rate']))
    bump('npz_written')

def work_study(payload):
    """Bir process içinde çalışır: bir study'nin header+download+shrink işi.
    Döner (sid, n_cine, stats). I/O thread'lerle; decode process-paralel (GIL süreçler arası yok)."""
    sid, uris, label, out_dir, size, n_frames, io_threads = payload
    _stat.clear()
    objs = [u.replace(GCS_PREFIX, '') for u in uris]
    with ThreadPoolExecutor(max_workers=io_threads) as tp:
        heads = list(tp.map(read_header, objs))
        bmode = [(o, b, m) for o, (c, b, m) in zip(objs, heads) if c == 'bmode_cine']
        for c, _, _ in heads:
            bump(c or 'unreadable')
        list(tp.map(lambda x: download_process(x[0], x[1], x[2], label, out_dir,
                                               size, n_frames, None), bmode))
    return sid, len(bmode), dict(_stat)

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest-dicom', required=True)
    ap.add_argument('--manifest', required=True, help='manifest_study.csv (etiketler)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--size', type=int, default=112)
    ap.add_argument('--n-frames', type=int, default=32)
    ap.add_argument('--procs', type=int, default=16, help='paralel study process sayısı')
    ap.add_argument('--io-threads', type=int, default=12, help='process başına I/O thread')
    ap.add_argument('--limit', type=int, default=0, help='pilot için ör. 3')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    done_path = os.path.join(args.out_dir, '_done.txt')
    done = set()
    if os.path.exists(done_path):
        done = set(open(done_path).read().split())

    labels = {}
    with open(args.manifest, newline='') as f:
        for r in csv.DictReader(f): labels[r['study_id']] = r
    by_study = defaultdict(list)
    with open(args.manifest_dicom, newline='') as f:
        for r in csv.DictReader(f): by_study[r['study_id']].append(r['gcs_uri'])

    studies = [s for s in by_study if s not in done]
    if args.limit: studies = studies[:args.limit]
    total = len(studies)
    print(f"[{time.strftime('%H:%M:%S')}] Toplam study: {len(by_study)} | tamamlanmış: {len(done)} "
          f"| işlenecek: {total} | procs={args.procs} io-threads={args.io_threads}", flush=True)

    payloads = [(sid, by_study[sid], labels.get(sid, {}), args.out_dir,
                 args.size, args.n_frames, args.io_threads) for sid in studies]
    t0 = time.time()
    agg = defaultdict(int); npz = 0; i = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex, open(done_path, 'a') as dfh:
        futs = {ex.submit(work_study, p): p[0] for p in payloads}
        for fut in as_completed(futs):
            i += 1
            try:
                sid, n, st = fut.result()
                for k, v in st.items(): agg[k] += v
                npz = agg.get('npz_written', 0)
                dfh.write(sid + '\n'); dfh.flush()
            except Exception as e:
                print(f"[HATA] s{futs[fut]}: {e}", flush=True)
                continue
            if i % 25 == 0 or i == total:
                el = time.time() - t0; rate = i / el; eta = (total - i) / rate / 60 if rate else 0
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{total} | NPZ={npz} "
                      f"| {rate:.2f} study/s | ETA {eta:.0f} dk | {dict(sorted(agg.items()))}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] BİTTİ. NPZ yazılan: {npz} "
          f"| istatistik: {dict(sorted(agg.items()))}", flush=True)

if __name__ == '__main__':
    main()
