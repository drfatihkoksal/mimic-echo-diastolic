#!/usr/bin/env python3
"""
Faz B — View sınıflandırma (EchoPrime hazır view classifier ile).

concept.md §6.3 PİVOTU: PanEcho'nun view head'i YOK (view-agnostic; 40 klinik görev,
hiçbiri view değil — kendisi bile harici `simple_view_pred` kullanıyor). Bunun yerine
EchoPrime'ın BAĞIMSIZ view classifier'ı kullanılır (convnext_base, 11 kaba view;
ağırlık: model_data.zip -> view_classifier.pt). Sıfır eğitim, yerel RTX 5090.

Girdi: Faz A NPZ'leri (~/mimic-echo/npz/*.npz), her biri frames=(32,112,112) uint8 gri.
Çıktı: _views.csv (dicom_id, study_id, pred_view, prob, p_A4C, p_A2C, p_PLAX).
İstenen view'lar (diyastolik grading için, Nature 2026 seti): A4C, A2C, PLAX(=Parasternal_Long).

Ön-işleme EchoPrime ile BİREBİR (echo_prime/model.py + utils.crop_and_scale):
  gri -> 3 kanal kopya -> crop_and_scale(224, cubic, zoom=0.1) -> (x-mean)/std  [0-255 ölçeği].
View, klibin birkaç karesinden softmax-ortalama ile tahmin edilir (tek kareye göre daha gürbüz).

Kullanım:
  # doğrulama (rastgele N klip -> etiketli önizleme montajı, ground-truth yok):
  python3 view_classify.py --npz-dir ~/mimic-echo/npz --sample 40 --preview
  # tam koşu:
  python3 view_classify.py --npz-dir ~/mimic-echo/npz --out ~/mimic-echo/npz/_views.csv
"""
from __future__ import annotations
import argparse, csv, glob, os, sys, time
import numpy as np
import cv2
import torch
import torchvision

# EchoPrime kaba view sınıfları (utils.COARSE_VIEWS ile aynı sıra — head çıktısı bu sıraya bağlı)
COARSE_VIEWS = ['A2C', 'A3C', 'A4C', 'A5C', 'Apical_Doppler',
                'Doppler_Parasternal_Long', 'Doppler_Parasternal_Short',
                'Parasternal_Long', 'Parasternal_Short', 'SSN', 'Subcostal']
TARGETS = {'A4C': 'A4C', 'A2C': 'A2C', 'Parasternal_Long': 'PLAX'}  # tutulacak view'lar
ABBR = {'Parasternal_Long': 'PLAX', 'Parasternal_Short': 'PSAX',
        'Doppler_Parasternal_Long': 'D-PLAX', 'Doppler_Parasternal_Short': 'D-PSAX',
        'Apical_Doppler': 'A-Dopp', 'Subcostal': 'SubC'}  # önizleme etiketi için kısa ad
# EchoPrime normalizasyonu (0-255 ölçeğinde, /255 YOK) — echo_prime/model.py:67-68
MEAN = np.array([29.110628, 28.076836, 29.096405], dtype=np.float32)
STD  = np.array([47.989223, 46.456997, 47.20083], dtype=np.float32)

def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    """EchoPrime utils.crop_and_scale birebir kopyası (en-boy koru + %zoom kırp + resize)."""
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]; r_out = res[0] / res[1]
    if r_in > r_out:
        pad = int(round((in_res[0] - r_out * in_res[1]) / 2)); img = img[:, pad:-pad] if pad else img
    if r_in < r_out:
        pad = int(round((in_res[1] - in_res[0] / r_out) / 2)); img = img[pad:-pad] if pad else img
    if zoom != 0:
        px = round(int(img.shape[1] * zoom)); py = round(int(img.shape[0] * zoom))
        if px and py: img = img[py:-py, px:-px]
    return cv2.resize(img, res, interpolation=interpolation)

def preprocess_frame(gray112):
    """(112,112) uint8 gri -> (3,224,224) float32 (EchoPrime normalize)."""
    rgb = np.repeat(gray112[..., None], 3, axis=2)          # gri -> 3 kanal
    x = crop_and_scale(rgb).astype(np.float32)              # (224,224,3)
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))                       # (3,224,224)

def load_model(weights, device):
    sd = torch.load(weights, map_location='cpu', weights_only=True)
    m = torchvision.models.convnext_base()
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, len(COARSE_VIEWS))
    m.load_state_dict(sd)
    m.eval().to(device)
    return m

def frame_indices(n_total, n_take):
    if n_total <= n_take: return list(range(n_total))
    return list(np.linspace(0, n_total - 1, n_take).round().astype(int))

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--weights', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'models', 'view_classifier.pt'))
    ap.add_argument('--out', default=None, help='çıktı CSV (varsayılan: <npz-dir>/_views.csv)')
    ap.add_argument('--frames-per-clip', type=int, default=8, help='klip başına oylanacak kare')
    ap.add_argument('--batch', type=int, default=256, help='GPU batch (kare cinsinden)')
    ap.add_argument('--sample', type=int, default=0, help='>0: rastgele N klip (doğrulama)')
    ap.add_argument('--preview', action='store_true', help='sample modda etiketli montaj PNG yaz')
    ap.add_argument('--fp16', action='store_true', help='yarı-hassasiyet (daha hızlı)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_csv = args.out or os.path.join(args.npz_dir, '_views.csv')
    files = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
    if args.sample:
        rng = np.random.default_rng(0); files = list(rng.choice(files, size=min(args.sample, len(files)), replace=False))
    print(f"[{time.strftime('%H:%M:%S')}] Model yükleniyor ({device})... | klip: {len(files)} "
          f"| kare/klip: {args.frames_per_clip}", flush=True)
    model = load_model(args.weights, device)
    if args.fp16: model = model.half()

    # zaten işlenmişleri atla (resumable)
    done = set()
    if not args.sample and os.path.exists(out_csv):
        with open(out_csv, newline='') as f:
            for r in csv.DictReader(f): done.add(r['dicom_id'])
        print(f"  {len(done)} klip zaten _views.csv'de — atlanacak", flush=True)

    write_header = not (os.path.exists(out_csv) and done) and not args.sample
    fout = None if args.sample else open(out_csv, 'a', newline='')
    writer = None
    if fout:
        writer = csv.writer(fout)
        if write_header:
            writer.writerow(['dicom_id', 'study_id', 'pred_view', 'prob', 'p_A4C', 'p_A2C', 'p_PLAX', 'keep'])

    prev_rows = []                                   # sample/preview için
    counts = {}; t0 = time.time(); n = 0
    buf_x, buf_meta = [], []                          # meta: (dicom_id, study_id, frame_count)

    def flush_batch():
        nonlocal n
        if not buf_x: return
        x = torch.from_numpy(np.stack(buf_x)).to(device)
        if args.fp16: x = x.half()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        # klip başına kareleri grupla -> softmax ortalama
        off = 0
        for dicom_id, study_id, fc in buf_meta:
            p = probs[off:off + fc].mean(axis=0); off += fc
            top = int(p.argmax()); view = COARSE_VIEWS[top]
            p_a4c = float(p[COARSE_VIEWS.index('A4C')])
            p_a2c = float(p[COARSE_VIEWS.index('A2C')])
            p_plax = float(p[COARSE_VIEWS.index('Parasternal_Long')])
            keep = int(view in TARGETS)
            counts[view] = counts.get(view, 0) + 1
            if writer:
                writer.writerow([dicom_id, study_id, view, f"{p[top]:.4f}",
                                 f"{p_a4c:.4f}", f"{p_a2c:.4f}", f"{p_plax:.4f}", keep])
            if args.sample:
                prev_rows.append((dicom_id, view, float(p[top])))
            n += 1
        buf_x.clear(); buf_meta.clear()
        if fout: fout.flush()

    for fp in files:
        try:
            z = np.load(fp, allow_pickle=True)
            frames = z['frames']
            # MIMIC npz'lerinde study_id, EchoXFlow render'ında exam_id
            study_id = str(z['study_id']) if 'study_id' in z.files else str(z['exam_id'])
        except Exception:
            continue
        dicom_id = os.path.basename(fp).replace('.npz', '')
        if dicom_id in done: continue
        idx = frame_indices(frames.shape[0], args.frames_per_clip)
        for fi in idx:
            buf_x.append(preprocess_frame(frames[fi]))
        buf_meta.append((dicom_id, study_id, len(idx)))
        if len(buf_x) >= args.batch:
            flush_batch()
            if n and n % 2000 == 0:
                el = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] {n} klip | {n/el:.1f} klip/s | {dict(sorted(counts.items()))}", flush=True)
    flush_batch()
    if fout: fout.close()

    el = time.time() - t0
    kept = sum(counts.get(v, 0) for v in TARGETS)
    print(f"\n[{time.strftime('%H:%M:%S')}] BİTTİ. {n} klip / {el:.0f}s ({n/max(el,1):.1f} klip/s)")
    print(f"View dağılımı: {dict(sorted(counts.items(), key=lambda x:-x[1]))}")
    print(f"TUTULAN (A4C/A2C/PLAX): {kept}/{n} (%{100*kept/max(n,1):.1f})")
    if not args.sample:
        print(f"CSV: {out_csv}")

    # ---- doğrulama montajı (kardiyolog gözüyle kontrol) ----
    if args.sample and args.preview:
        prev_dir = os.path.join(args.npz_dir, '_view_preview'); os.makedirs(prev_dir, exist_ok=True)
        cols = 8; rows = (len(prev_rows) + cols - 1) // cols
        tile = 140; canvas = np.zeros((rows * tile, cols * tile), np.uint8)
        fmap = {os.path.basename(f).replace('.npz',''): f for f in files}
        for i, (did, view, prob) in enumerate(prev_rows):
            z = np.load(fmap[did]); fr = z['frames']; mid = fr[len(fr)//2]
            img = cv2.resize(mid, (tile, tile), interpolation=cv2.INTER_AREA)
            r, c = divmod(i, cols); canvas[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = img
            cv2.putText(canvas, f"{ABBR.get(view, view)[:9]} {prob:.2f}", (c*tile+3, r*tile+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
        outp = os.path.join(prev_dir, 'view_montage.png'); cv2.imwrite(outp, canvas)
        print(f"Doğrulama montajı: {outp}  ({len(prev_rows)} klip, etiketli)")

if __name__ == '__main__':
    main()
