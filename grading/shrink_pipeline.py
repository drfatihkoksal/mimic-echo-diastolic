#!/usr/bin/env python3
"""
Küçültme pipeline'ı (Faz A) — MIMIC-IV-ECHO DICOM -> B-mode cine NPZ.

Bölüm 6.1/6.2/11 bulgularına dayanır. Her study için:
  header oku -> METADATA FİLTRE (B-mode cine) -> JPEG decode -> US-sektör bbox kırp ->
  gri'ye çevir -> mekânsal downsample (112x112) -> sabit frame örnekle (32) -> NPZ.

Metadata filtre (piksel analizi GEREKMEZ):
  B-mode cine  <=>  UltrasoundColorDataPresent != 1  &  RegionDataType tissue(1)  &  NumberOfFrames > 1
  (Color Doppler=%49 ücretsiz elenir; still=1 frame elenir.)

Kullanım:
  # PİLOT (lokalde indirilmiş study'ler):
  python3 shrink_pipeline.py --local-dir <dir> --manifest ~/mimic-echo/manifest_study.csv \
      --out-dir <out> --limit 20 --preview
  # VM (GCS'ten indirerek stream):
  python3 shrink_pipeline.py --gcs --manifest-dicom ~/mimic-echo/manifest_dicom.csv \
      --out-dir <out> --billing-project <gcp-projeniz>

Faz B (PanEcho view sınıflandırma) AYRI adım — bu script view-agnostic B-mode cine üretir;
view etiketi sonra eklenir (Bölüm 6.3).
"""
from __future__ import annotations
import argparse, csv, glob, json, os, subprocess, sys, tempfile
from collections import defaultdict
import numpy as np
import pydicom
import cv2

# --------------------------------------------------------------------------- #
# 1. Metadata filtre                                                          #
# --------------------------------------------------------------------------- #
def classify(ds):
    """Döner (kategori, bbox|None, meta). kategori: 'bmode_cine' | 'color_doppler'
    | 'spectral' | 'bmode_still' | 'other'."""
    color = getattr(ds, 'UltrasoundColorDataPresent', 0)
    nf = int(getattr(ds, 'NumberOfFrames', 1) or 1)
    rdt, bbox = None, None
    seq = getattr(ds, 'SequenceOfUltrasoundRegions', None)
    if seq:
        r = seq[0]
        rdt = getattr(r, 'RegionDataType', None)
        try:
            bbox = (int(r.RegionLocationMinX0), int(r.RegionLocationMaxX1),
                    int(r.RegionLocationMinY0), int(r.RegionLocationMaxY1))
        except Exception:
            bbox = None
    if rdt in (3, 4):          cat = 'spectral'
    elif color == 1 or rdt == 2: cat = 'color_doppler'
    elif nf > 1:               cat = 'bmode_cine'
    else:                      cat = 'bmode_still'
    meta = {'n_frames': nf, 'cine_rate': getattr(ds, 'CineRate', None),
            'frame_time': getattr(ds, 'FrameTime', None),
            'rows': int(getattr(ds, 'Rows', 0)), 'cols': int(getattr(ds, 'Columns', 0)),
            'photometric': str(getattr(ds, 'PhotometricInterpretation', ''))}
    return cat, bbox, meta

# --------------------------------------------------------------------------- #
# 2. Decode + kırp + gri + downsample + frame örnekle                          #
# --------------------------------------------------------------------------- #
def to_gray(frame, photometric):
    """(H,W,3) -> (H,W) gri. YBR ise Y(kanal0)=luma; RGB ise luma-ağırlık."""
    if frame.ndim == 2:
        return frame
    if photometric.upper().startswith('YBR'):
        return frame[..., 0]                       # Y = luma (B-mode için doğru)
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

def sample_indices(n_total, n_want):
    if n_total >= n_want:
        return np.linspace(0, n_total - 1, n_want).round().astype(int)
    return np.array(list(range(n_total)) * ((n_want // n_total) + 1))[:n_want]

def process_clip(ds, bbox, photometric, size=112, n_frames=32):
    """DICOM -> (n_frames, size, size) uint8 gri. Kırp+gri+resize+örnekle.

    BELLEK: eski `ds.pixel_array` TÜM kareleri (ör. 77 kare 708x1016x3 ≈ 166 MB, decode
    tepe ≈ 1.5 GB) çözüyordu; tam-koşuda procs×io_threads eşzamanlı decode ile OOM sebebi.
    Artık SADECE örneklenen benzersiz kareler iter_pixels ile tek tek çözülür -> tepe bellek
    klip başına ~tek kare (~2-27 MB), çıktı eskiyle BAYT-BAYT AYNI (raw=False = YBR->RGB,
    legacy pixel_array ile birebir; doğrulandı)."""
    from pydicom.pixels import iter_pixels
    T = int(getattr(ds, 'NumberOfFrames', 1) or 1)
    idx = sample_indices(T, n_frames)               # uzunluk n_frames, tekrarlı olabilir
    uniq = sorted({int(i) for i in idx})            # her kare bir kez decode edilsin
    x0 = x1 = y0 = y1 = None
    if bbox:
        x0, x1, y0, y1 = bbox
    small = {}                                      # kare_idx -> (size,size) uint8
    for fi, fr in zip(uniq, iter_pixels(ds, indices=uniq)):
        g = to_gray(fr, photometric)
        if bbox and x1 > x0 and y1 > y0:
            g = g[y0:y1, x0:x1]                      # US sektör kırp (chrome at)
        small[fi] = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    out = np.empty((n_frames, size, size), dtype=np.uint8)
    for i, fi in enumerate(idx):
        out[i] = small[int(fi)]
    return out

# --------------------------------------------------------------------------- #
# 3. Etiket yükle (manifest_study.csv)                                         #
# --------------------------------------------------------------------------- #
def load_labels(manifest_study):
    lab = {}
    with open(manifest_study, newline='') as f:
        for r in csv.DictReader(f):
            lab[r['study_id']] = r
    return lab

# --------------------------------------------------------------------------- #
# 4. Study işle                                                                #
# --------------------------------------------------------------------------- #
def process_study(study_id, dcm_paths, label, out_dir, size, n_frames, preview_dir=None):
    """Bir study'nin DICOM'larını işle. Döner istatistik dict."""
    stat = defaultdict(int); kept = 0
    for fp in dcm_paths:
        try:
            hdr = pydicom.dcmread(fp, stop_before_pixels=True)
        except Exception:
            stat['unreadable'] += 1; continue
        cat, bbox, meta = classify(hdr)
        stat[cat] += 1
        if cat != 'bmode_cine':
            continue
        try:
            ds = pydicom.dcmread(fp)                 # pikselli
            clip = process_clip(ds, bbox, meta['photometric'], size, n_frames)
        except Exception as e:
            stat['decode_fail'] += 1; continue
        di = os.path.basename(fp).replace('.dcm', '')
        outp = os.path.join(out_dir, f"{di}.npz")
        np.savez_compressed(
            outp, frames=clip,
            study_id=study_id, subject_id=label.get('subject_id', ''),
            grade_2025=label.get('grade_2025', ''), grade_2016=label.get('grade_2016', ''),
            confidence=label.get('confidence', ''), age=label.get('age', ''),
            gender=label.get('gender', ''), ef=label.get('EF', ''),
            n_orig_frames=meta['n_frames'], cine_rate=str(meta['cine_rate']))
        kept += 1
        if preview_dir and kept <= 2:               # study başı 2 önizleme PNG
            mid = clip[len(clip) // 2]
            cv2.imwrite(os.path.join(preview_dir, f"{di}_mid.png"), mid)
    stat['bmode_cine_kept'] = kept
    return stat

# --------------------------------------------------------------------------- #
# 5. GCS'ten study indir (VM modu)                                             #
# --------------------------------------------------------------------------- #
def gcs_download_study(gcs_uris, tmpdir, billing):
    paths = []
    for uri in gcs_uris:
        dst = os.path.join(tmpdir, os.path.basename(uri))
        r = subprocess.run(['gcloud', 'storage', 'cp', uri, dst,
                            f'--billing-project={billing}'],
                           capture_output=True)
        if r.returncode == 0:
            paths.append(dst)
    return paths

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--local-dir', help='İndirilmiş study klasörlerinin kökü (pilot)')
    ap.add_argument('--gcs', action='store_true', help='GCS stream modu (VM)')
    ap.add_argument('--manifest', help='manifest_study.csv (etiketler)')
    ap.add_argument('--manifest-dicom', help='manifest_dicom.csv (GCS modu için)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--billing-project', default=os.environ.get('GCP_BILLING_PROJECT', ''),
                    help='requester-pays için GCP proje kimliğiniz')
    ap.add_argument('--size', type=int, default=112)
    ap.add_argument('--n-frames', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0, help='0=hepsi (pilot için ör. 20)')
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    preview_dir = os.path.join(args.out_dir, '_preview') if args.preview else None
    if preview_dir: os.makedirs(preview_dir, exist_ok=True)
    labels = load_labels(args.manifest) if args.manifest else {}

    # study -> dcm listesi (lokal) veya gcs uri listesi
    jobs = []   # (study_id, kind, items)
    if args.local_dir:
        for d in sorted(glob.glob(os.path.join(args.local_dir, 's*'))):
            sid = os.path.basename(d).lstrip('s')
            dcms = sorted(glob.glob(os.path.join(d, '*.dcm')))
            if dcms: jobs.append((sid, 'local', dcms))
    elif args.gcs:
        by_study = defaultdict(list)
        with open(args.manifest_dicom, newline='') as f:
            for r in csv.DictReader(f):
                by_study[r['study_id']].append(r['gcs_uri'])
        for sid, uris in by_study.items():
            jobs.append((sid, 'gcs', uris))
    else:
        ap.error('--local-dir veya --gcs gerekli')

    if args.limit: jobs = jobs[:args.limit]
    print(f"İşlenecek study: {len(jobs)}")

    agg = defaultdict(int); n_out = 0; studies_with_output = 0
    for k, (sid, kind, items) in enumerate(jobs, 1):
        lab = labels.get(sid, {})
        if kind == 'gcs':
            with tempfile.TemporaryDirectory() as td:
                paths = gcs_download_study(items, td, args.billing_project)
                st = process_study(sid, paths, lab, args.out_dir, args.size,
                                   args.n_frames, preview_dir)
        else:
            st = process_study(sid, items, lab, args.out_dir, args.size,
                               args.n_frames, preview_dir)
        for kk, vv in st.items(): agg[kk] += vv
        if st.get('bmode_cine_kept', 0) > 0: studies_with_output += 1
        n_out += st.get('bmode_cine_kept', 0)
        print(f"[{k}/{len(jobs)}] s{sid} grade={lab.get('grade_2025','?'):<7} "
              f"B-mode-cine tutuldu={st.get('bmode_cine_kept',0):<3} "
              f"(doppler={st.get('color_doppler',0)}, still={st.get('bmode_still',0)})")

    # NPZ boyut özeti
    npz_sz = sum(os.path.getsize(p) for p in glob.glob(os.path.join(args.out_dir, '*.npz')))
    print("\n=== ÖZET ===")
    print(f"Study işlendi: {len(jobs)} (çıktısı olan: {studies_with_output})")
    print(f"Kategori toplamları: { {k:v for k,v in sorted(agg.items())} }")
    print(f"Üretilen NPZ: {n_out} | toplam boyut: {npz_sz/1e6:.1f} MB "
          f"| ort {npz_sz/max(n_out,1)/1e3:.0f} KB/klip")
    if n_out:
        # ekstrapolasyon: pilot -> 3097 study
        per_study = n_out / max(studies_with_output, 1)
        est_npz = npz_sz / max(studies_with_output, 1) * 3097 / 1e9
        print(f"Ekstrapolasyon: ~{per_study:.1f} klip/study -> 3097 study ≈ {est_npz:.1f} GB NPZ")
    if preview_dir:
        print(f"Önizleme PNG'ler: {preview_dir}")

if __name__ == '__main__':
    main()
