#!/usr/bin/env python3
"""
Faz C — Adım 2: veri yükleyici (EchoClipDataset).

Birim = KLİP (per-klip eğitim; study etiketini miras alır). Study-level tahmin, çıkarımda
klipler study içinde toplanarak yapılır (bkz. train scripti).

Her klip (npz_hires/*.npz): frames=(T,224,224) uint8 gri, view + study meta gömülü.
Etiket & aux hedefler splits.csv + manifest_study.csv'den (study_id ile join).

Çıktı örneği (dict):
  x        : (3, clip_len, 224, 224) float32  — gri→3ch, ImageNet-norm (PanEcho girdisi)
  grade    : int64  (0=Normal,1=Grade1,2=Grade2,3=Grade3)
  aux      : (4,) float32  standardize [EF, LAVi, Ee_mean, TRvel]  (eksikse 0, maske ile atlanır)
  aux_mask : (4,) bool
  study_id, view, confidence

Örnekleme: clip_len kare. train=rastgele bitişik pencere + uniform (temporal augment);
eval=tüm klip boyunca uniform (deterministik). Augment (train): parlaklık/kontrast/gamma +
hafif afin (döndürme/ölçek/kaydırma) — L/R FLIP YOK (oda oryantasyonunu bozar).
"""
from __future__ import annotations
import csv, glob, os
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

GRADES = ['Normal', 'Grade1', 'Grade2', 'Grade3']
AUX_COLS = ['EF', 'LAVi', 'Ee_mean', 'TRvel']
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

DEFAULT_SPLITS = os.path.expanduser('~/mimic-echo/splits.csv')
DEFAULT_MANIFEST = os.path.expanduser('~/mimic-echo/manifest_study.csv')
DEFAULT_HIRES = os.path.expanduser('~/mimic-echo/npz_hires')


def _load_study_meta(splits_csv, manifest_csv, split):
    """study_id -> dict(grade, confidence, aux[list of float|nan]) — sadece `split`."""
    aux_raw = {}
    with open(manifest_csv, newline='') as f:
        for r in csv.DictReader(f):
            vec = []
            for c in AUX_COLS:
                try: vec.append(float(r.get(c, '').strip()))
                except Exception: vec.append(np.nan)
            aux_raw[r['study_id']] = vec
    meta = {}
    with open(splits_csv, newline='') as f:
        for r in csv.DictReader(f):
            if r['split'] != split:
                continue
            meta[r['study_id']] = {
                'grade': GRADES.index(r['grade_2025']),
                'confidence': r.get('confidence', ''),
                'aux': aux_raw.get(r['study_id'], [np.nan] * len(AUX_COLS)),
            }
    return meta


def compute_aux_stats(splits_csv=DEFAULT_SPLITS, manifest_csv=DEFAULT_MANIFEST):
    """Train split'inden aux hedeflerin (mean,std)'ını hesapla (standardizasyon için).
    Döner: (mean(4,), std(4,)) np.float32. Sadece BİR KEZ, train'den; val/test aynısını kullanır."""
    meta = _load_study_meta(splits_csv, manifest_csv, 'train')
    arr = np.array([m['aux'] for m in meta.values()], dtype=np.float64)  # (N,4), nan'lı
    mean = np.nanmean(arr, axis=0); std = np.nanstd(arr, axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _sample_indices(T, clip_len, train):
    """T kareden clip_len indeks. train: rastgele bitişik pencere + uniform; eval: uniform."""
    if T <= clip_len:
        idx = list(range(T)) + [T - 1] * (clip_len - T)      # kısa klip: son kareyi tekrarla
        return np.array(idx[:clip_len])
    if train:
        # ~1.5x pencere içinde rastgele bitişik aralık, sonra uniform örnekle (temporal augment)
        span = min(T, int(clip_len * np.random.uniform(1.0, 3.0)))
        start = np.random.randint(0, T - span + 1)
        return np.linspace(start, start + span - 1, clip_len).round().astype(int)
    return np.linspace(0, T - 1, clip_len).round().astype(int)


def _augment_clip(x, rot_deg=10, trans=0.06, scale=0.1, bc=0.2, gamma=0.2):
    """x: (3,T,H,W) float [0,1]. Klip boyunca TUTARLI hafif afin + foto augment. Yeni tensor döner."""
    C, T, H, W = x.shape
    # foto: parlaklık/kontrast/gamma (skaler, klip başına)
    g = float(np.exp(np.random.uniform(-gamma, gamma)))
    c = float(1 + np.random.uniform(-bc, bc)); b = float(np.random.uniform(-bc, bc)) * 0.5
    x = (x.clamp(1e-6, 1) ** g)
    x = (x * c + b).clamp(0, 1)
    # afin: döndürme+ölçek+kaydırma (tüm karelere aynı theta), grid_sample
    ang = np.deg2rad(np.random.uniform(-rot_deg, rot_deg))
    s = 1 + np.random.uniform(-scale, scale)
    tx = np.random.uniform(-trans, trans) * 2; ty = np.random.uniform(-trans, trans) * 2
    cos, sin = np.cos(ang) / s, np.sin(ang) / s
    theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32).unsqueeze(0).repeat(T, 1, 1)
    xt = x.permute(1, 0, 2, 3)                                # (T,3,H,W)
    grid = F.affine_grid(theta, [T, C, H, W], align_corners=False)
    xt = F.grid_sample(xt, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return xt.permute(1, 0, 2, 3).contiguous()               # (3,T,H,W)


_VIEW_IDX = None


def _view_index():
    """dicom_id -> görünüm. npz'leri açmak yerine Faz A sınıflandırma CSV'sinden okunur
    (67 bin dosyada dizin kurulumunu dakikalardan saniyeye indirir)."""
    global _VIEW_IDX
    if _VIEW_IDX is None:
        _VIEW_IDX = {}
        vp = os.path.join(os.path.dirname(DEFAULT_HIRES), 'npz', '_views.csv')
        if os.path.exists(vp):
            with open(vp, newline='') as f:
                for r in csv.DictReader(f):
                    _VIEW_IDX[r['dicom_id']] = r['pred_view']
    return _VIEW_IDX


def _view_of(fp):
    v = _view_index().get(os.path.basename(fp)[:-4])
    if v is None:                                   # son çare: npz'den oku
        try:
            v = str(np.load(fp, allow_pickle=True)['view'])
        except Exception:
            return None
    return {'Parasternal_Short': 'PSAX'}.get(v, v)


class EchoClipDataset(Dataset):
    def __init__(self, split, clip_len=16, train=False, aux_stats=None, single_frame=False,
                 splits_csv=DEFAULT_SPLITS, manifest_csv=DEFAULT_MANIFEST, hires_dir=DEFAULT_HIRES,
                 extra_dirs=(), views=None):
        """views: None = dizindeki her klip; aksi halde tutulacak görünüm adları kümesi
        (ör. {'A4C','A2C','PLAX'}). extra_dirs: ek görünüm klipleri için ilave dizinler."""
        self.clip_len = clip_len; self.train = train; self.single_frame = single_frame
        self.meta = _load_study_meta(splits_csv, manifest_csv, split)
        if aux_stats is None:
            aux_stats = compute_aux_stats(splits_csv, manifest_csv)
        self.aux_mean, self.aux_std = aux_stats
        # klip indeksi: bu split'teki study'lerin tüm npz'leri (ana dizin + ek dizinler)
        self.items = []
        for d in (hires_dir,) + tuple(extra_dirs):
            for fp in sorted(glob.glob(os.path.join(d, '*.npz'))):
                sid = os.path.basename(fp).split('_')[0]
                if sid not in self.meta:
                    continue
                if views is not None and _view_of(fp) not in views:
                    continue
                self.items.append((fp, sid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fp, sid = self.items[i]
        z = np.load(fp, allow_pickle=True)
        frames = z['frames']                                  # (T,224,224) uint8
        idx = _sample_indices(frames.shape[0], self.clip_len, self.train)
        clip = frames[idx].astype(np.float32) / 255.0         # (clip_len,224,224)
        x = torch.from_numpy(clip).unsqueeze(0).repeat(3, 1, 1, 1)   # (3,clip_len,224,224)
        if self.train:
            x = _augment_clip(x)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD                # ImageNet-norm (PanEcho)
        if self.single_frame:
            x = x[:, x.shape[1] // 2]                          # (3,224,224) — orta kare (temporal ablation)
        m = self.meta[sid]
        aux = np.array(m['aux'], dtype=np.float32)
        aux_mask = torch.from_numpy(~np.isnan(aux))
        aux_std = (np.nan_to_num(aux) - self.aux_mean) / self.aux_std
        aux_std = np.where(np.isnan(aux), 0.0, aux_std).astype(np.float32)
        return {
            'x': x, 'grade': torch.tensor(m['grade'], dtype=torch.long),
            'aux': torch.from_numpy(aux_std), 'aux_mask': aux_mask,
            'study_id': sid, 'view': str(z['view']), 'confidence': m['confidence'],
        }


if __name__ == '__main__':
    # hızlı duman testi
    stats = compute_aux_stats()
    print("aux mean:", stats[0], "std:", stats[1])
    ds = EchoClipDataset('val', clip_len=16, train=True, aux_stats=stats)
    print("val klip:", len(ds))
    b = ds[0]
    print("x:", tuple(b['x'].shape), b['x'].dtype, "| grade:", int(b['grade']),
          "| aux:", b['aux'].numpy().round(2), "mask:", b['aux_mask'].numpy(), "| view:", b['view'])
