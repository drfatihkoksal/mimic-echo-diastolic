#!/usr/bin/env python3
"""
Dış validasyon şekilleri — JASE biçimi (7,2 inç, 300 dpi, Arial-metrikli Liberation Sans).

Şekil: ayırt etme / kalibrasyon / klinik fayda üçlüsü. Üç kohort üç kategorik renkle;
renk kimliği taşır (kohort), çizgi stili stratejiyi taşır (model vs varsayılan).
Palet projenin mevcut mavi-turuncu çiftine üçüncü olarak mor eklenerek kuruldu ve
doğrulayıcıdan geçirildi (en kötü komşu CVD ΔE 23,7; normal görüş 27,8; hepsi >=3:1 kontrast).
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.expanduser('~/mimic-echo/figures')
os.makedirs(FIG, exist_ok=True)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d8d8d4'
CAT = ['#2a78d6', '#7a4fa3', '#eb6834']          # iç test / dış ASE LAP / dış E/e'
rng = np.random.default_rng(20260901)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'Arial', 'Nimbus Sans', 'DejaVu Sans'],
    'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': INK2,
    'axes.labelcolor': INK, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': INK2, 'ytick.color': INK2, 'legend.frameon': False,
})


def load():
    # Kohort tanımı da tek kaynaktan gelir (canonical_results.cohorts)
    sys.path.insert(0, HERE)
    from canonical_results import cohorts
    c = cohorts()
    return [('Internal test — ASE LAP',) + c['ic_test_ASE_LAP'][:2],
            ('External — ASE LAP',) + c['dis_ASE_LAP'][:2],
            ('External — E/e′ (primary)',) + c['dis_Ee_birincil'][:2]]


# Güven aralıkları YENİDEN HESAPLANMAZ: tek kaynak canonical_results.json (sabit tohum).
CANON = json.load(open(os.path.join(HERE, 'canonical_results.json')))
KEY = {'Internal test — ASE LAP': 'ic_test_ASE_LAP', 'External — ASE LAP': 'dis_ASE_LAP',
       'External — E/e′ (primary)': 'dis_Ee_birincil'}


def auc_ci(name):
    d = CANON['kohortlar'][KEY[name]]
    return d['auroc'], d['auroc_ga']


def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0, c - hw), min(1, c + hw)


def main():
    data = load()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))

    # --- A: ROC ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, ls=':', zorder=1)
    auc_txt = []
    for (name, y, p), c in zip(data, CAT):
        fpr, tpr, _ = roc_curve(y, p)
        au, ci = auc_ci(name)
        ax.plot(fpr, tpr, color=c, lw=1.6, solid_capstyle='round', zorder=3)
        auc_txt.append((name, au, ci, c))
    ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title('A  Discrimination', loc='left', fontsize=9, color=INK, pad=6)
    ax.text(0.97, 0.20, 'AUROC (95% CI)', ha='right', fontsize=6, color=INK2, transform=ax.transAxes)
    for i, (nm, au, ci, c) in enumerate(auc_txt):
        ax.text(0.97, 0.145 - i * 0.055, f'{au:.3f} ({ci[0]:.3f}–{ci[1]:.3f})', ha='right',
                fontsize=6.4, color=c, transform=ax.transAxes)

    # --- B: kalibrasyon ---
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, ls=':', zorder=1)
    for (name, y, p), c in zip(data, CAT):
        nb = 10 if len(y) > 400 else 5
        q = np.quantile(p, np.linspace(0, 1, nb + 1)); q[-1] += 1e-9
        idx = np.clip(np.digitize(p, q[1:-1]), 0, nb - 1)
        xs, ys, lo, hi = [], [], [], []
        for b in range(nb):
            m = idx == b
            if m.sum() < 5:
                continue
            k, n = int(y[m].sum()), int(m.sum())
            a, bnd = wilson(k, n)
            xs.append(p[m].mean()); ys.append(k / n); lo.append(a); hi.append(bnd)
        xs, ys = np.array(xs), np.array(ys)
        # Wilson aralığı ham orana göre kayıklıdır; k=0'da alt sınır ys'yi 1e-5 aşabiliyor
        lo_e = np.maximum(0.0, ys - np.array(lo)); hi_e = np.maximum(0.0, np.array(hi) - ys)
        ax.errorbar(xs, ys, yerr=[lo_e, hi_e], fmt='o-', color=c,
                    lw=1.3, ms=3.4, elinewidth=0.7, capsize=0, mec='white', mew=0.5, zorder=3)
    ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed frequency')
    ax.set_xlim(-0.03, 1.0); ax.set_ylim(-0.03, 1.0)
    ax.set_title('B  Calibration', loc='left', fontsize=9, color=INK, pad=6)
    # kuantil dilimler düşük olasılıkta yığıldığı için rule-out bölgesi büyütülür:
    # birincil son noktanın aşırı iyimserliği asıl olarak orada görünür
    ins = ax.inset_axes([0.55, 0.09, 0.42, 0.38])
    ins.plot([0, .25], [0, .25], color=GRID, lw=0.7, ls=':')
    for (name, y, p), c in zip(data, CAT):
        nb = 10 if len(y) > 400 else 5
        q = np.quantile(p, np.linspace(0, 1, nb + 1)); q[-1] += 1e-9
        idx = np.clip(np.digitize(p, q[1:-1]), 0, nb - 1)
        xs, ys = [], []
        for b in range(nb):
            m = idx == b
            if m.sum() >= 5 and p[m].mean() <= 0.25:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        if xs:
            ins.plot(xs, ys, 'o-', color=c, lw=1.1, ms=2.6, mec='white', mew=0.4)
    ins.set_xlim(-0.01, 0.26); ins.set_ylim(-0.02, 0.58)
    ins.tick_params(labelsize=5.2, length=2, pad=1)
    for sp in ('top', 'right'):
        ins.spines[sp].set_visible(True); ins.spines[sp].set_linewidth(0.5)
    ins.text(0.04, 0.90, 'rule-out region', transform=ins.transAxes, fontsize=5.6,
             color=INK2, va='top')

    # --- C: karar eğrisi ---
    ax = axes[2]
    dc = json.load(open(os.path.join(HERE, 'decision_curve.json')))
    keymap = {'Internal test — ASE LAP': 'iç test — ASE LAP',
              'External — ASE LAP': 'dış — ASE LAP',
              'External — E/e′ (primary)': 'dış — E/e′ (ön-tanımlı birincil)'}
    ax.axhline(0, color=GRID, lw=0.8, ls=':', zorder=1)
    for (name, _, _), c in zip(data, CAT):
        rows = dc[keymap[name]]['egri']
        pt = np.array([r['pt'] for r in rows])
        ax.plot(pt, [r['model'] for r in rows], color=c, lw=1.6, zorder=3)
        ax.plot(pt, [r['herkes'] for r in rows], color=c, lw=0.9, ls='--', alpha=0.75, zorder=2)
    ax.axvline(0.1286, color=INK2, lw=0.7, ls='-.', zorder=2)
    ax.text(0.1386, 0.285, 'locked\nthreshold', fontsize=5.8, color=INK2, va='top')
    ax.set_xlim(0.02, 0.60); ax.set_ylim(-0.10, 0.31)
    ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
    ax.set_title('C  Clinical utility', loc='left', fontsize=9, color=INK, pad=6)
    h = [plt.Line2D([], [], color=INK2, lw=1.6, label='Model'),
         plt.Line2D([], [], color=INK2, lw=0.9, ls='--', label='Refer all'),
         plt.Line2D([], [], color=GRID, lw=0.8, ls=':', label='Refer none')]
    ax.legend(handles=h, loc='upper right', fontsize=5.8, handlelength=1.6, labelspacing=0.4)

    handles = [plt.Line2D([], [], color=c, lw=2.0, label=n) for (n, _, _), c in zip(data, CAT)]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.14), handlelength=1.8, columnspacing=2.2)
    fig.subplots_adjust(wspace=0.34, bottom=0.20)
    for e in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'figure_external.{e}'))
    plt.close(fig)

    # --- ek şekil: kırpılma tersilleri ---
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, t in enumerate(CANON['kirpilma']['tersiller']):
        au, ci = t['auroc'], t['auroc_ga']
        ax.errorbar([i], [au], yerr=[[au - ci[0]], [ci[1] - au]], fmt='o', color=CAT[2],
                    ms=5, elinewidth=1.0, capsize=2.5, mec='white', mew=0.6)
        ax.text(i + 0.13, au, f"{au:.3f}", fontsize=6.5, color=INK, va='center')
    ref_au = CANON['ic_test_Ee_hedef_uyumsuz']['auroc']
    ax.axhline(ref_au, color=CAT[0], lw=1.0, ls='--')
    ax.text(0.5, ref_au + 0.006, f'internal E/e′ {ref_au:.3f}', fontsize=6, color=CAT[0],
            va='bottom', ha='center')   # T1-T2 arası, veri noktalarıyla çakışmaz
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['T1\ncleanest', 'T2', 'T3\nmost clipped'], fontsize=7)
    ax.set_ylabel('AUROC'); ax.set_ylim(0.52, 1.0); ax.set_xlim(-0.4, 2.5)
    ax.set_title('Reference-standard clipping', loc='left', fontsize=9, color=INK, pad=6)
    for e in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'figure_clipping.{e}'))
    print('yazıldı:', FIG)


if __name__ == '__main__':
    main()
