#!/usr/bin/env python3
"""
Faz C.2 — Adım 1: C.1 backbone'undan klip embedding'leri çıkar (donuk temsil).

Multi-view füzyonu (attention-MIL) hızlı ve izole test etmek için: C.1'in fine-tune edilmiş
PanEcho backbone'unu kullanıp her klip için 768-dim embedding üret, diske yaz. Sonra MIL head
bu donuk embedding'ler üzerinde saniyeler içinde eğitilir (backbone tekrar koşmaz).

Çıktı: ~/mimic-echo/runs/c2_mil/emb_<split>.npz  (emb (N,768), study_id, grade, view, conf)
Kullanım: python3 embed_clips.py
"""
from __future__ import annotations
import os, numpy as np, torch
from torch.utils.data import DataLoader
from dataset import EchoClipDataset, compute_aux_stats, GRADES
from model import DiastolicModel

CKPT = os.path.expanduser('~/mimic-echo/runs/c1_panecho/best.pt')
OUT = os.path.expanduser('~/mimic-echo/runs/c2_mil'); os.makedirs(OUT, exist_ok=True)


@torch.no_grad()
def embed_split(model, split, aux_stats, batch=48, workers=10):
    ds = EchoClipDataset(split, clip_len=16, train=False, aux_stats=aux_stats)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    embs, sids, grades, views, confs = [], [], [], [], []
    for b in dl:
        x = b['x'].cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = model.backbone(x)                      # (B,768)
        embs.append(f.float().cpu().numpy())
        sids += list(b['study_id']); views += list(b['view']); confs += list(b['confidence'])
        grades += b['grade'].tolist()
    emb = np.concatenate(embs).astype(np.float32)
    np.savez(os.path.join(OUT, f'emb_{split}.npz'), emb=emb,
             study_id=np.array(sids), grade=np.array(grades, dtype=np.int64),
             view=np.array(views), conf=np.array(confs))
    print(f"  {split}: {emb.shape} → emb_{split}.npz", flush=True)


def main():
    dev = 'cuda'
    aux_stats = compute_aux_stats()
    model = DiastolicModel(clip_len=16, arch='panecho', pretrained=False).to(dev)  # ağırlık ckpt'ten
    sd = torch.load(CKPT, weights_only=False)['model']
    model.load_state_dict(sd)
    model.eval()
    print("C.1 backbone yüklendi; embedding çıkarılıyor...", flush=True)
    for split in ['train', 'val', 'test']:
        embed_split(model, split, aux_stats)
    print("BİTTİ.")


if __name__ == '__main__':
    main()
