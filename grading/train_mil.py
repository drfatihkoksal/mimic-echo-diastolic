#!/usr/bin/env python3
"""
Faz C.2 — Adım 2: multi-view attention-MIL füzyonu (donuk C.1 embedding'leri üzerinde).

Bag = study, örnekler = klip embedding'leri (768-dim, C.1 backbone'undan). Gated attention
(Ilse et al. 2018) klipleri ağırlıklandırır → study vektörü → CORN ordinal + aux head.
C.1'de füzyon = per-klip olasılık ortalaması (basit); burada ÖĞRENİLEN attention. Karşılaştırma:
  --pool attention  vs  --pool mean  (ikisi de embedding-uzayı; attention'ın katkısını izole eder)
Referans: C.1 klip-olasılık-ortalaması 3-sınıf QWK 0.512.

Girdi: ~/mimic-echo/runs/c2_mil/emb_<split>.npz + manifest (aux). Hızlı (dakikalar, küçük head).
Kullanım: python3 train_mil.py --pool attention   |   --pool mean
"""
from __future__ import annotations
import argparse, csv, json, math, os
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from dataset import compute_aux_stats, AUX_COLS, GRADES, DEFAULT_MANIFEST
from model import corn_loss, corn_predict, masked_mse

EMB = os.path.expanduser('~/mimic-echo/runs/c2_mil')


def load_aux():
    aux = {}
    with open(DEFAULT_MANIFEST, newline='') as f:
        for r in csv.DictReader(f):
            v = []
            for c in AUX_COLS:
                try: v.append(float(r.get(c, '').strip()))
                except Exception: v.append(np.nan)
            aux[r['study_id']] = v
    return aux


class BagDS(Dataset):
    """Study bag: (emb (n,768), grade, aux(4), aux_mask(4), study_id, conf)."""
    def __init__(self, split, aux_raw, aux_mean, aux_std):
        z = np.load(os.path.join(EMB, f'emb_{split}.npz'), allow_pickle=True)
        emb, sid, grade, conf = z['emb'], z['study_id'], z['grade'], z['conf']
        bags = defaultdict(list)
        for i, s in enumerate(sid):
            bags[s].append(i)
        self.items = []
        for s, idx in bags.items():
            g = int(grade[idx[0]])
            a = np.array(aux_raw.get(s, [np.nan] * len(AUX_COLS)), dtype=np.float32)
            amask = ~np.isnan(a)
            astd = np.where(np.isnan(a), 0.0, (np.nan_to_num(a) - aux_mean) / aux_std).astype(np.float32)
            self.items.append((emb[idx], g, astd, amask, s, str(conf[idx[0]])))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch):
    n = max(b[0].shape[0] for b in batch); B = len(batch)
    x = torch.zeros(B, n, 768); mask = torch.zeros(B, n, dtype=torch.bool)
    for i, (emb, *_ ) in enumerate(batch):
        k = emb.shape[0]; x[i, :k] = torch.from_numpy(emb); mask[i, :k] = True
    grade = torch.tensor([b[1] for b in batch], dtype=torch.long)
    aux = torch.from_numpy(np.stack([b[2] for b in batch]))
    amask = torch.from_numpy(np.stack([b[3] for b in batch]))
    return x, mask, grade, aux, amask, [b[4] for b in batch], [b[5] for b in batch]


class AttnMIL(nn.Module):
    def __init__(self, dim=768, hid=128, n_classes=4, n_aux=4, dropout=0.25, pool='attention'):
        super().__init__()
        self.pool = pool
        if pool == 'attention':
            self.V = nn.Linear(dim, hid); self.U = nn.Linear(dim, hid); self.w = nn.Linear(hid, 1)
        self.drop = nn.Dropout(dropout)
        self.ord = nn.Linear(dim, n_classes - 1)
        self.aux = nn.Linear(dim, n_aux)

    def forward(self, x, mask):                            # x (B,N,768), mask (B,N)
        if self.pool == 'attention':
            a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)  # (B,N)
            a = a.masked_fill(~mask, float('-inf'))
            alpha = torch.softmax(a, dim=1)
            z = (alpha.unsqueeze(-1) * x).sum(1)
        else:
            m = mask.float().unsqueeze(-1)
            z = (x * m).sum(1) / m.sum(1).clamp(min=1)
        z = self.drop(z)
        return self.ord(z), self.aux(z)


def metrics(recs, subset=None):
    r = [x for x in recs if subset is None or x['conf'] == subset]
    if len(r) < 2: return None
    t = np.array([x['true'] for x in r]); p = np.array([x['pred'] for x in r])
    s1 = np.array([x['p_ge1'] for x in r]); s2 = np.array([x['p_ge2'] for x in r])
    def blk(tt, pp, k):
        return {'qwk': round(float(cohen_kappa_score(tt, pp, weights='quadratic', labels=list(range(k)))), 4),
                'acc_pm1': round(float((np.abs(tt - pp) <= 1).mean()), 4),
                'bacc': round(float(balanced_accuracy_score(tt, pp)), 4),
                'macro_f1': round(float(f1_score(tt, pp, average='macro', labels=list(range(k)), zero_division=0)), 4)}
    out = {'n': len(r), '4class': blk(t, p, 4), '3class': blk(np.minimum(t, 2), np.minimum(p, 2), 3),
           'confusion4': confusion_matrix(t, p, labels=list(range(4))).tolist()}
    yb1 = (t >= 1).astype(int); yb2 = (t >= 2).astype(int)
    if len(np.unique(yb1)) == 2: out['AUROC_anyDD'] = round(float(roc_auc_score(yb1, s1)), 3)
    if len(np.unique(yb2)) == 2: out['AUROC_advDD'] = round(float(roc_auc_score(yb2, s2)), 3)
    return out


@torch.no_grad()
def evaluate(model, dl, dev, K):
    model.eval(); recs = []
    for x, mask, grade, aux, amask, sids, confs in dl:
        ol, _ = model(x.to(dev), mask.to(dev))
        pred, cum = corn_predict(ol.float())
        cum = cum.cpu().numpy(); pred = pred.cpu().numpy()
        for i, s in enumerate(sids):
            recs.append({'study_id': s, 'true': int(grade[i]), 'pred': int(pred[i]),
                         'conf': confs[i], 'p_ge1': float(cum[i, 0]), 'p_ge2': float(cum[i, 1])})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='attention', choices=['attention', 'mean'])
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--aux-weight', type=float, default=0.3)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    dev = 'cuda'; K = 4
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    aux_raw = load_aux(); am, asd = compute_aux_stats()
    tr = BagDS('train', aux_raw, am, asd); va = BagDS('val', aux_raw, am, asd); te = BagDS('test', aux_raw, am, asd)
    print(f"[MIL/{args.pool}] bags train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    # study-level dengeli örnekleme (grade)
    g = np.array([it[1] for it in tr.items]); cnt = np.bincount(g, minlength=K)
    w = (cnt.sum() / (K * np.maximum(cnt, 1)))[g]
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
    tl = DataLoader(tr, batch_size=args.batch, sampler=sampler, collate_fn=collate)
    vl = DataLoader(va, batch_size=64, shuffle=False, collate_fn=collate)
    tel = DataLoader(te, batch_size=64, shuffle=False, collate_fn=collate)

    model = AttnMIL(dropout=args.dropout, pool=args.pool).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    outdir = os.path.join(EMB, f'mil_{args.pool}'); os.makedirs(outdir, exist_ok=True)

    best = -2; best_ep = -1; bad = 0
    for ep in range(args.epochs):
        model.train()
        for x, mask, grade, aux, amask, *_ in tl:
            ol, ap_ = model(x.to(dev), mask.to(dev))
            loss = corn_loss(ol, grade.to(dev), K) + args.aux_weight * masked_mse(ap_, aux.to(dev), amask.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        M = metrics(evaluate(model, vl, dev, K)); qwk = M['4class']['qwk']
        if qwk > best:
            best = qwk; best_ep = ep; bad = 0
            torch.save(model.state_dict(), os.path.join(outdir, 'best.pt'))
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            print(f"  ep{ep} val QWK={qwk} 3cls={M['3class']['qwk']} advDD-AUC={M.get('AUROC_advDD')}"
                  f"{' *' if bad==0 else ''}", flush=True)
        if bad >= args.patience:
            print(f"[early-stop] en iyi ep{best_ep} val-QWK={best}", flush=True); break

    model.load_state_dict(torch.load(os.path.join(outdir, 'best.pt'), weights_only=True))
    Mt = metrics(evaluate(model, tel, dev, K)); Mte = metrics(evaluate(model, tel, dev, K), 'ecg_sinus')
    json.dump({'pool': args.pool, 'best_ep': best_ep, 'best_val_qwk': best, 'test': Mt, 'test_ecg_sinus': Mte},
              open(os.path.join(outdir, 'test_metrics.json'), 'w'), indent=1, ensure_ascii=False)
    print(f"\n=== MIL ({args.pool}) TEST ===")
    print(f"4-sınıf QWK={Mt['4class']['qwk']} ±1={Mt['4class']['acc_pm1']} bacc={Mt['4class']['bacc']}")
    print(f"3-sınıf QWK={Mt['3class']['qwk']} bacc={Mt['3class']['bacc']} macroF1={Mt['3class']['macro_f1']}")
    print(f"ikili AUROC: herhangi-DD={Mt.get('AUROC_anyDD')} ileri-DD={Mt.get('AUROC_advDD')}")


if __name__ == '__main__':
    main()
