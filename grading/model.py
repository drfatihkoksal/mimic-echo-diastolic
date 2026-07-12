#!/usr/bin/env python3
"""
Faz C — Adım 3: model (PanEcho backbone fine-tune + CORN ordinal head + aux regresyon).

- Backbone: PanEcho (echo-pretrained, 1.2M video) `backbone_only` → (B,768) video embedding.
  torch.hub'dan yüklenir (ilk sefer indirir, sonra ~/.cache'ten). timm gerekli.
- Ana head: CORN ordinal (K sınıf → K-1 mantık). Ordinal sıralamayı kullanır, Grade3 (n=54) için kritik.
- Aux head'ler: EF/LAVi/Ee_mean/TRvel regresyonu (grade'i tanımlayan ASE parametreleri; maskeli MSE).

CORN referansı: Shi X, Cao W, Raschka S. Deep neural networks for rank-consistent ordinal regression
based on conditional probabilities. Pattern Anal Appl 2023;26:941-955. doi:10.1007/s10044-023-01181-9
DİKKAT: Bu CORN'dur (koşullu olasılık şeması, P(y>r | y>=r)) — aşağıdaki corn_loss tam olarak bunu
uygular. CORAL (Cao W, Mirjalili V, Raschka S, Pattern Recognit Lett 2020) FARKLI bir yöntemdir
(çıkış katmanında ağırlık paylaşımı). İkisini karıştırmayın; makalede CORN atfı verilmelidir.
Loss & tahmin fonksiyonları aşağıda inline (bağımlılık yok).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

PANECHO_REPO = 'CarDS-Yale/PanEcho'


class DiastolicModel(nn.Module):
    """arch:
      'panecho'   — PanEcho video backbone (echo-pretrained) → 768. Girdi (B,3,T,224,224). [C.1]
      'panecho2d' — PanEcho 2D image-encoder (tek kare) → 768. Girdi (B,3,224,224). [temporal ablation]
      'r2plus1d'  — torchvision R(2+1)D-18 (Kinetics-init/random) → 512. Girdi (B,3,T,224,224). [pretraining ablation]
    """
    def __init__(self, clip_len=16, n_classes=4, n_aux=4, pretrained=True,
                 dropout=0.25, freeze_backbone=False, arch='panecho'):
        super().__init__()
        self.n_classes = n_classes; self.arch = arch
        self.single_frame = (arch == 'panecho2d')
        if arch == 'panecho':
            self.backbone = torch.hub.load(PANECHO_REPO, 'PanEcho', backbone_only=True,
                                           clip_len=clip_len, pretrained=pretrained, trust_repo=True)
            feat = 768
        elif arch == 'panecho2d':
            self.backbone = torch.hub.load(PANECHO_REPO, 'PanEcho', image_encoder_only=True,
                                           pretrained=pretrained, trust_repo=True)
            feat = 768
        elif arch == 'r2plus1d':
            import torchvision
            w = torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
            m = torchvision.models.video.r2plus1d_18(weights=w)
            feat = m.fc.in_features; m.fc = nn.Identity()
            self.backbone = m
        else:
            raise ValueError(f'bilinmeyen arch: {arch}')
        self.dropout = nn.Dropout(dropout)
        self.ord_head = nn.Linear(feat, n_classes - 1)        # CORN: K-1 koşullu mantık
        self.aux_head = nn.Linear(feat, n_aux)                # regresyon (standardize hedef)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        f = self.dropout(self.backbone(x))                    # (B,feat)
        return self.ord_head(f), self.aux_head(f)             # (B,K-1), (B,n_aux)


# --------------------------------------------------------------------------- #
# CORN ordinal loss & tahmin                                                    #
# --------------------------------------------------------------------------- #
def corn_loss(logits, targets, num_classes):
    """CORN koşullu ordinal loss (Cao 2020).
    logits: (B, K-1)  targets: (B,) int [0..K-1]. Her rank r için P(y>r | y>=r) ikili görevi,
    yalnızca y>=r örnekleri üzerinde. Döner: skaler."""
    total = 0.0; n_terms = 0
    for r in range(num_classes - 1):
        mask = targets >= r                                   # koşul: y >= r
        if mask.sum() == 0:
            continue
        lab = (targets[mask] > r).float()                     # hedef: y > r
        pred = logits[mask, r]
        # -[ log σ(pred)·lab + log(1-σ(pred))·(1-lab) ]  (sayısal-kararlı)
        loss = -(F.logsigmoid(pred) * lab + (F.logsigmoid(pred) - pred) * (1 - lab))
        total = total + loss.sum()
        n_terms += mask.sum().item()
    return total / max(n_terms, 1)


def corn_predict(logits):
    """CORN çıkarım. logits: (B,K-1) → (pred_label (B,), P(y>r) kümülatif (B,K-1)).
    P(y>=r+1) = Π_{j<=r} σ(logit_j); etiket = 0.5 eşiğini geçen rank sayısı."""
    probs = torch.sigmoid(logits)
    cum = torch.cumprod(probs, dim=1)                         # P(y>0), P(y>1), ...
    pred = (cum > 0.5).sum(dim=1)
    return pred, cum


def masked_mse(pred, target, mask):
    """Aux regresyon: sadece dolu (mask=True) hedeflerde MSE. Döner: skaler (hiç yoksa 0)."""
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask])


if __name__ == '__main__':
    # duman testi: model + loss + tek eğitim adımı
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = DiastolicModel(clip_len=16).to(dev)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    x = torch.rand(4, 3, 16, 224, 224, device=dev)
    y = torch.tensor([0, 1, 2, 3], device=dev)
    aux = torch.randn(4, 4, device=dev)
    aux_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool, device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    m.train()
    l0 = None
    for step in range(3):
        ord_logits, aux_pred = m(x)
        loss_ord = corn_loss(ord_logits, y, 4)
        loss_aux = masked_mse(aux_pred, aux, aux_mask)
        loss = loss_ord + 0.3 * loss_aux
        opt.zero_grad(); loss.backward(); opt.step()
        if l0 is None: l0 = loss.item()
    pred, cum = corn_predict(ord_logits)
    print(f"param(train): {n_train:.1f}M | ord_logits: {tuple(ord_logits.shape)} aux: {tuple(aux_pred.shape)}")
    print(f"loss: {l0:.3f} -> {loss.item():.3f} (ord {loss_ord.item():.3f} + aux {loss_aux.item():.3f})")
    print(f"corn_predict: {pred.tolist()} (hedef {y.tolist()})")
