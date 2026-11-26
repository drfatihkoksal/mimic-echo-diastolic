#!/usr/bin/env python3
"""Şekiller: kilitli modelin performansı ve dış kohort akışı.

  Figure_1      (A) ROC, (B) kalibrasyon (beşte birlik dilimler, Wilson GA), (C) kilitli kuralla 1.000 çalışmada net sevk
                azalması (herkesi sevk etmeye göre), p_t %2–10. Üç küme: iç test, dış E/e′ (ön-tanımlı birincil), dış kılavuz
                hedefi ana kümesi. Kohort tanımları run_posthoc.py ile aynı işlevlerden gelir; AUROC ve GA sonuç dosyasından.
  Figure_1_data.xlsx   şeklin sayısal değerleri. Yalnız eğri / dilim düzeyinde toplu değerler.
  Figure_S1     dış kohort akış şeması (sayılar posthoc_results.json -> akis_dis).

Renkler: mavi / turuncu / mor, bu sırayla doğrulayıcıdan geçti (komşu çiftlerde CVD ΔE >= 23, normal görüş >= 27).
Çalıştırma (proje kökünden):  python3 external/posthoc/make_figures.py <çıktı klasörü>"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_posthoc as RR                                        # noqa: E402

INK, INK2, GRID = '#0b0b0b', '#52514e', '#d8d8d4'
CAT = ['#2a78d6', '#eb6834', '#7a4fa3']
MARK = ['o', 's', '^']
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.linewidth': 0.6, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': INK2, 'ytick.color': INK2, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def HU(v, n=3):
    """Yarım-yukarı yuvarlama; metindeki biçimle aynı (build_v3.F)."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP))


def sets():
    B = json.load(open(RR.H + 'runs/b2_binary/binary_results.json'))
    lo = B['gri_bolge']['rule_out_alti']
    IF = RR.load_internal(fix=True); X = RR.load_external()
    te = RR.pick(IF, 'test')
    ext = lambda sel: [r for r in X.values() if not r['af'] and r['pred'] and sel(r)]
    xe = ext(lambda r: r['y_ee'] is not None)
    xg = ext(lambda r: r['y_lap'] is not None and r['n_avail'] == 3)
    A = lambda rs, k: np.array([r[k] for r in rs])
    return lo, [('Internal test set, guideline target', 'ic_test', A(te, 'y'), A(te, 'p')),
                ('External, elevated E/e′', 'dis_Ee_birincil', A(xe, 'y_ee'), A(xe, 'p')),
                ('External, guideline target', 'dis_kilavuz_uc_olcum_ANA', A(xg, 'y_lap'), A(xg, 'p'))]


def reduction(y, pos, pt):
    y = y.astype(bool); n = len(y)
    return 1000 * ((~pos & ~y).sum() / n - ((~pos & y).sum() / n) * (1 - pt) / pt)


def figure1(out):
    lo, S = sets()
    R = json.load(open(RR.OUT))['kohortlar']
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.75))
    data = {'ROC': [], 'Calibration': [], 'Net reduction in referrals': []}

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, ls=':', zorder=1)
    for i, ((name, key, y, p), c) in enumerate(zip(S, CAT)):
        assert len(y) == R[key]['n'] and int(y.sum()) == R[key]['olay'], key
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, lw=1.6, solid_capstyle='round', zorder=3)
        a, g = R[key]['auroc'], R[key]['ga']
        yy = 0.19 - i * 0.065
        ax.plot([0.36, 0.43], [yy + 0.012, yy + 0.012], color=c, lw=1.8, transform=ax.transAxes, clip_on=False)
        ax.text(0.455, yy, f'{HU(a)} ({HU(g[0])}–{HU(g[1])})', fontsize=6.6, color=INK, transform=ax.transAxes)
        data['ROC'] += [(name, round(float(a_), 4), round(float(b_), 4)) for a_, b_ in zip(fpr, tpr)]
    ax.text(0.36, 0.255, 'AUROC (95% CI)', fontsize=6.2, color=INK2, transform=ax.transAxes)
    ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title('A  Discrimination', loc='left', fontsize=8.5, color=INK, pad=6)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, ls=':', zorder=1)
    for (name, key, y, p), c, m in zip(S, CAT, MARK):
        q = np.quantile(p, np.linspace(0, 1, 6)); q[-1] += 1e-9
        idx = np.clip(np.digitize(p, q[1:-1]), 0, 4)
        xs, ys, lo_e, hi_e = [], [], [], []
        for b in range(5):
            k, n = int(y[idx == b].sum()), int((idx == b).sum())
            w = RR.wilson(k, n)
            xs.append(p[idx == b].mean()); ys.append(k / n)
            lo_e.append(max(0.0, k / n - w[0])); hi_e.append(max(0.0, w[1] - k / n))
            data['Calibration'].append((name, b + 1, n, k, round(float(xs[-1]), 4), round(k / n, 4), w[0], w[1]))
        ax.errorbar(xs, ys, yerr=[lo_e, hi_e], fmt=m + '-', color=c, lw=1.2, ms=4.2, elinewidth=0.7, capsize=0,
                    mec='white', mew=0.6, zorder=3)
    ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed proportion')
    ax.set_xlim(-0.03, 1.0); ax.set_ylim(-0.03, 1.0)
    ax.set_title('B  Calibration', loc='left', fontsize=8.5, color=INK, pad=6)

    ax = axes[2]
    ax.axhline(0, color=INK2, lw=0.7, ls='--', zorder=1)
    ax.text(5.3, 18, 'refer every study', fontsize=6.2, color=INK2, ha='center', va='bottom')
    pts = np.round(np.arange(0.02, 0.1001, 0.005), 3)
    for (name, key, y, p), c in zip(S, CAT):
        red = [reduction(y, p > lo, pt) for pt in pts]
        for pt in (0.02, 0.05, 0.10):                    # metindeki değerlerle aynı olmalı
            ref = R[key]['karar_egrisi']['net_sevk_azalmasi_1000']['kilitli_dislama_siniri'][f'pt_{pt:.4f}']
            assert abs(reduction(y, p > lo, pt) - ref) < 0.06, (key, pt)
        ax.plot(pts * 100, red, color=c, lw=1.6, zorder=3)
        data['Net reduction in referrals'] += [(name, float(pt), round(float(r), 1)) for pt, r in zip(pts, red)]
    ax.set_xlim(1.8, 10.2); ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_xlabel('Threshold probability, %'); ax.set_ylabel('Net reduction in referrals\nper 1,000 studies')
    ax.set_title('C  Clinical utility of the locked rule', loc='left', fontsize=8.5, color=INK, pad=6)

    h = [plt.Line2D([], [], color=c, lw=1.8, marker=m, ms=4.2, mec='white', mew=0.6, label=f"{n} ({len(y)} studies, {int(y.sum())} events)")
         for (n, _, y, _), c, m in zip(S, CAT, MARK)]
    fig.legend(handles=h, loc='lower center', ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.11), handlelength=2.2, columnspacing=1.8)
    fig.subplots_adjust(wspace=0.44, bottom=0.19)
    fig.savefig(os.path.join(out, 'Figure_1.pdf'))
    fig.savefig(os.path.join(out, 'Figure_1.png'), dpi=300)
    fig.savefig(os.path.join(out, 'Figure_1.tif'), dpi=1000, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)

    import openpyxl
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    heads = {'ROC': ('Set', '1 - specificity', 'Sensitivity'),
             'Calibration': ('Set', 'Fifth of predicted probability', 'Studies', 'Events', 'Mean predicted probability',
                             'Observed proportion', 'Wilson 95% CI lower', 'Wilson 95% CI upper'),
             'Net reduction in referrals': ('Set', 'Threshold probability', 'Net reduction in referrals per 1,000 studies (locked rule vs referring every study)')}
    for nm, rows in data.items():
        ws = wb.create_sheet(nm[:31]); ws.append(heads[nm])
        for r in rows:
            ws.append(r)
    wb.save(os.path.join(out, 'Figure_1_data.xlsx'))


def figure_s1(out):
    r = json.load(open(RR.OUT)); f = r['akis_dis']; k = r['dis_kilavuz_eksik_degisken']; C = r['kohortlar']
    e = f['Ee_hesaplanamayan']; g = f['kilavuz_hedefi_karar_verilen']['toplam']
    fig, ax = plt.subplots(figsize=(7.4, 5.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    GREY = '#f4f4f1'

    def box(x, y, w, h, txt, fill='#ffffff', bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.2,rounding_size=1.2',
                                    fc=fill, ec=INK2, lw=0.8, zorder=3))
        ax.text(x, y, txt, ha='center', va='center', fontsize=7.2, color=INK, fontweight='bold' if bold else 'normal',
                linespacing=1.35, zorder=4)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9, shrinkA=0, shrinkB=0), zorder=2)

    XL, XR = 16, 84
    box(50, 94, 36, 8, f"EchoXFlow examinations\nn = {f['toplam_muayene']}")
    arrow(50, 89.8, 50, 80.4)
    arrow(50, 85, 67.8, 85)
    box(83, 85, 30, 8, f"Excluded: irregular rhythm\nn = {f['ritim_dislamasi']}", fill=GREY)
    box(50, 76, 36, 8, f"Regular rhythm\nn = {f['ritim_sonrasi']}")
    ax.plot([50, 50], [71.8, 68], color=INK2, lw=0.9); ax.plot([XL, XR], [68, 68], color=INK2, lw=0.9)

    # sol dal: E/e′ (ön-tanımlı birincil hedef)
    ax.text(XL + 1.5, 64.5, 'Elevated E/e′\n(prespecified primary target)', ha='left', va='center', fontsize=6.8, color=INK2, style='italic')
    d = C['dis_Ee_birincil']
    arrow(XL, 68, XL, 20.4)
    arrow(XL, 48, 23.3, 48)
    box(37, 48, 27, 17, f"Excluded: E/e′ not available\nn = {e['toplam']}\nneither E nor e′ measured, {e['E_ve_eprime_yok']}\n"
        f"E without e′, {e['yalniz_E_var_eprime_yok']}\ne′ without E, {e['yalniz_eprime_var_E_yok']}", fill=GREY)
    box(XL, 14, 30, 12, f"Evaluated against\nelevated E/e′, n = {d['n']}\n{d['olay']} with elevated E/e′", bold=True)

    # sağ dal: kılavuz hedefi
    ax.text(XR - 1.5, 64.5, 'Guideline-defined\nelevated LAP', ha='right', va='center', fontsize=6.8, color=INK2, style='italic')
    arrow(XR, 68, XR, 43.2)
    arrow(XR, 55, 78.7, 55)
    box(66, 55, 25, 10, f"Excluded: no decision from\nthe primary variables alone\nn = {f['ritim_sonrasi'] - g}", fill=GREY)
    box(XR, 38, 30, 10, f"Prespecified set\n(sensitivity analysis), n = {g}\n{C['dis_kilavuz_tum_duyarlilik']['olay']} with elevated LAP")
    arrow(XR, 32.8, XR, 20.4)
    arrow(XR, 26, 78.7, 26)
    d = C['dis_kilavuz_uc_olcum_ANA']
    box(64, 26, 29, 9.5, f"Excluded: fewer than three primary\nvariables measured, n = {g - d['n']}\n"
        f"no TR velocity, {k['TR_yok']}; no E/e′, {k['Ee_yok']}", fill=GREY)
    box(XR, 14, 30, 12, f"Main set: three primary\nvariables measured, n = {d['n']}\n{d['olay']} with elevated LAP", bold=True)
    assert g - d['n'] == k['TR_yok'] + k['Ee_yok'] and k['TR_yok_olay'] + k['Ee_yok_olay'] == 0
    fig.savefig(os.path.join(out, 'Figure_S1.png'), dpi=300)
    fig.savefig(os.path.join(out, 'Figure_S1.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    figure1(out); figure_s1(out)
    print('->', out)
