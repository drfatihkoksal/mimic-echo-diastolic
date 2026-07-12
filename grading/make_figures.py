#!/usr/bin/env python3
"""
JASE figürleri — 300 dpi, basılı dergi geleneklerine göre.

Renk kararları (dataviz doğrulayıcısıyla test edildi):
  * Diyastolik grade ORDİNAL bir şiddet ölçüsüdür → kategorik palet değil, SIRALI (tek-hue,
    açık→koyu) mavi rampa. Açıklık monoton azaldığı için figür GRİ BASKIDA da okunur.
  * 4 grade'i tek hue üzerinde hem ayırt edilebilir hem yeterli kontrastta tutmak mümkün
    değil → ikincil kodlama: farklı ÇİZGİ STİLLERİ + eğri ucunda doğrudan etiket.
  * ROC'taki 2 seri kategoriktir (kimlik, şiddet değil) → mavi/turuncu; CVD ΔE 96.7 (geçti).

Alt komutlar:
  fig3  — karışıklık matrisi + ROC eğrileri (test seti; OOF'tan bağımsız, kesin)
  fig5  — Kaplan-Meier (mortalite; OOF gerekir)
  fig6  — NT-proBNP kutu grafiği (OOF gerekir)

Kullanım: python3 make_figures.py fig3
"""
from __future__ import annotations
import argparse, csv, os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

GR = ['Normal', 'Grade1', 'Grade2', 'Grade3']
GRL = ['Normal', 'Grade 1', 'Grade 2', 'Grade 3']
H = os.path.expanduser('~/mimic-echo/')
FIG = os.path.join(H, 'figures')

# sıralı (şiddet) rampa — açık→koyu, gri baskıda ayrışır
SEQ = ['#cde2fb', '#6da7ec', '#256abf', '#0d366b']
SEQ_LINE = ['#86b6ef', '#2a78d6', '#1c5cab', '#0d366b']
STYLES = ['-', '--', '-.', ':']
# kategorik (kimlik) — doğrulayıcıdan geçti
CAT = {'any': '#2a78d6', 'adv': '#eb6834'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d8d8d4'

# JASE: "Arial, Courier, Times New Roman, Symbol, ya da benzeri görünen fontlar."
# Arial bu makinede kurulu değil; Liberation Sans Arial ile METRİK UYUMLU (aynı glif genişlikleri),
# standart ikamesidir → basımda Arial'e denk düşer.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'Arial', 'Nimbus Sans', 'DejaVu Sans'],
    'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.labelcolor': INK, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
})


def load_preds(path):
    t, p, cum = [], [], []
    with open(os.path.expanduser(path), newline='') as f:
        for r in csv.DictReader(f):
            c = ([float(x) for x in r['cum_probs'].split(';')] if 'cum_probs' in r
                 else [float(r['p_ge1']), float(r['p_ge2']), float(r['p_ge3'])])
            t.append(GR.index(r['true'])); p.append(GR.index(r['pred'])); cum.append(c)
    return np.array(t), np.array(p), np.array(cum)


def boot_roc(y, s, n=2000, seed=42):
    """ROC için bootstrap %95 bandı (ortak FPR ızgarasında TPR yüzdelikleri)."""
    grid = np.linspace(0, 1, 101)
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); tprs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[b], s[b])
        tprs.append(np.interp(grid, fpr, tpr))
    tprs = np.array(tprs)
    return grid, np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0)


def boot_auc_ci(y, s, n=2000, seed=42):
    """AUROC bootstrap %95 GA — binary_eval.py ile aynı yöntem/seed (sayılar tutsun)."""
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); a = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        a.append(roc_auc_score(y[b], s[b]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def fig3(args):
    t, p, cum = load_preds(args.preds)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4))

    # ── A) karışıklık matrisi: satır-normalize (sıralı dolgu + hücrede sayı)
    cm = confusion_matrix(t, p, labels=[0, 1, 2, 3])
    row = cm.sum(1, keepdims=True)
    frac = np.divide(cm, np.maximum(row, 1))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seq', ['#ffffff', SEQ[3]])
    ax1.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    for i in range(4):
        for j in range(4):
            # metin rengi dolguya göre — renk değil AÇIKLIK karar veriyor
            ax1.text(j, i, f"{cm[i,j]}", ha='center', va='center', fontsize=8,
                     color='white' if frac[i, j] > 0.55 else INK)
    ax1.set_xticks(range(4)); ax1.set_yticks(range(4))
    ax1.set_xticklabels(GRL, rotation=45, ha='right'); ax1.set_yticklabels(GRL)
    ax1.set_xlabel('Predicted grade'); ax1.set_ylabel('Reference grade (2025 ASE)')
    ax1.set_title('A  Grading agreement', loc='left', fontsize=9, color=INK, pad=8)
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.tick_params(length=0)

    # ── B) ROC: 2 kategorik seri + bootstrap bandı
    # NOT: 0.885 float'ta 0.8849... olduğu için f"{x:.2f}" 0.88 verir ama metinde 0.89 yazıyor.
    # Yarıyı-yukarı yuvarlayıp figür ile metnin AYNI sayıyı göstermesini garantiye alıyoruz.
    r2 = lambda x: f"{Decimal(str(x)).quantize(Decimal('0.01'), ROUND_HALF_UP)}"
    for key, (y, s, lab) in {
        'any': ((t >= 1).astype(int), cum[:, 0], 'Any dysfunction (≥Grade 1)'),
        'adv': ((t >= 2).astype(int), cum[:, 1], 'Advanced dysfunction (≥Grade 2)'),
    }.items():
        auc = roc_auc_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        g, lo, hi = boot_roc(y, s)
        alo, ahi = boot_auc_ci(y, s)
        ax2.fill_between(g, lo, hi, color=CAT[key], alpha=0.15, linewidth=0)
        ax2.plot(fpr, tpr, color=CAT[key], linewidth=2,
                 label=f"{lab}\nAUROC {r2(auc)} (95% CI {r2(alo)}–{r2(ahi)})")
    ax2.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle=':')
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.set_xlabel('1 − Specificity'); ax2.set_ylabel('Sensitivity')
    ax2.set_title('B  Screening performance', loc='left', fontsize=9, color=INK, pad=8)
    leg = ax2.legend(loc='lower right', frameon=False, fontsize=7, handlelength=1.6,
                     labelcolor=INK)
    ax2.set_aspect('equal')

    os.makedirs(FIG, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'figure3.{ext}'))
    print(f"→ {FIG}/figure3.pdf (+.png) | n={len(t)}")
    print(f"   shading = row-normalised recall; cell text = study count")


def fig4(args):
    """Figür 4 — dış doğrulama (OOF): A) Kaplan-Meier — tahmini grade'e göre 1-yıl sağkalım,
    B) NT-proBNP (log) — tahmini grade'e göre. Makalenin döngüsellik itirazına cevabı."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    from scipy.stats import kruskal, spearmanr

    # ── A) mortalite: analizle AYNI kohort (hasta başına en erken echo)
    first = {}
    for r in csv.DictReader(open(os.path.expanduser(args.mort))):
        s = r['subject']
        if s not in first or r['acq'] < first[s]['acq']:
            first[s] = r
    coh = list(first.values())
    T = np.array([float(r['time']) for r in coh])
    E = np.array([int(r['event']) for r in coh])
    PG = np.array([int(r['pred_grade']) for r in coh])
    lr = multivariate_logrank_test(T, PG, E)

    fig = plt.figure(figsize=(7.5, 4.4))
    # wspace geniş: A'nın eğri-ucu etiketleri panellerin ARASINA taşıyor, B'nin y başlığına binmesin
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], hspace=0.10, wspace=0.52)
    ax1 = fig.add_subplot(gs[0, 0])      # KM eğrileri
    axr = fig.add_subplot(gs[1, 0])      # numbers-at-risk
    ax2 = fig.add_subplot(gs[:, 1])      # NT-proBNP

    grid = [0, 90, 180, 270, 365]
    at_risk = []
    for g in range(4):
        m = PG == g
        km = KaplanMeierFitter().fit(T[m], E[m])
        km.plot_survival_function(ax=ax1, ci_show=False, color=SEQ_LINE[g],
                                  linestyle=STYLES[g], linewidth=1.8, legend=False)
        at_risk.append([int((T[m] >= t).sum()) for t in grid])

    # eğri ucunda DOĞRUDAN etiket → kimlik asla renk-tek-başına değil (gri baskı güvenli)
    for g in range(4):
        m = PG == g
        s_end = 1.0 - E[m].sum() / m.sum()
        ax1.annotate(GRL[g], xy=(365, s_end), xytext=(6, 0), textcoords='offset points',
                     va='center', fontsize=7.5, color=SEQ_LINE[g])
    ax1.set_xlim(0, 365); ax1.set_ylim(0.70, 1.005)
    # x ekseni ALT panelde (at-risk) — burada tick işareti bırakmak başıboş çizgiler üretir
    ax1.set_xticks(grid); ax1.set_xticklabels([])
    ax1.tick_params(axis='x', length=0)
    ax1.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax1.set_ylabel('Survival probability')
    ax1.set_xlabel('')          # lifelines varsayılan olarak 'timeline' yazıyor — x ekseni alt panelde
    ax1.grid(axis='y', color=GRID, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title('A  One-year survival by predicted grade', loc='left', fontsize=9,
                  color=INK, pad=8)
    # y ekseni 0.70'ten başlıyor — kesik olduğunu AÇIKÇA söyle (gizlenirse okuyucuyu yanıltır)
    ax1.text(0.03, 0.06, f"log-rank p = {lr.p_value:.1e}\nn = {len(coh)} patients, "
             f"{int(E.sum())} deaths\ny-axis truncated at 0.70",
             transform=ax1.transAxes, fontsize=7, color=INK2, va='bottom')

    # ── numbers at risk (KM'nin altında, aynı x ızgarasında)
    for g in range(4):
        for j, t in enumerate(grid):
            axr.text(t, 3 - g, f"{at_risk[g][j]}", ha='center', va='center',
                     fontsize=7, color=INK)
        axr.text(-28, 3 - g, GRL[g], ha='right', va='center', fontsize=7.5,
                 color=SEQ_LINE[g])
    axr.set_xlim(0, 365); axr.set_ylim(-0.7, 4.5)
    axr.set_xticks(grid); axr.set_yticks([])
    axr.set_xlabel('Days since echocardiogram')
    axr.text(0, 4.2, 'No. at risk', ha='left', va='center', fontsize=7.5, color=INK2)
    for s in axr.spines.values():
        s.set_visible(False)
    axr.tick_params(axis='y', length=0)

    # ── B) NT-proBNP (log ölçek) — tahmini grade'e göre
    rows = list(csv.DictReader(open(os.path.expanduser(args.bnp))))
    bnp = [[float(r['bnp']) for r in rows if int(r['pred_grade']) == g] for g in range(4)]
    risk = np.array([float(r['risk']) for r in rows])
    lb = np.log(np.array([float(r['bnp']) for r in rows]))
    rho, prho = spearmanr(risk, lb)
    kw = kruskal(*[b for b in bnp if len(b) > 1])

    bp = ax2.boxplot(bnp, positions=range(4), widths=0.62, patch_artist=True,
                     showfliers=False, medianprops=dict(color='white', linewidth=1.4),
                     whiskerprops=dict(color=INK2, linewidth=0.8),
                     capprops=dict(color=INK2, linewidth=0.8))
    for g, b in enumerate(bp['boxes']):
        b.set(facecolor=SEQ[g], edgecolor=INK2, linewidth=0.6)
    # tekil noktalar: n küçük olan grupta (tahmini Grade 3, n=11) kutu yanıltıcı olur
    rng = np.random.default_rng(42)
    for g in range(4):
        x = g + rng.uniform(-0.16, 0.16, len(bnp[g]))
        ax2.scatter(x, bnp[g], s=3.5, color=INK2, alpha=0.30, linewidths=0, zorder=3)
    ax2.axhline(125, color=GRID, linewidth=0.9, linestyle=':', zorder=1)
    ax2.annotate('125 pg/mL', xy=(3.45, 125), fontsize=6.5, color=INK2, va='bottom',
                 ha='right')
    ax2.set_yscale('log')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f"{GRL[g]}\n(n={len(bnp[g])})" for g in range(4)], fontsize=7.5)
    ax2.set_xlabel('Predicted grade')
    ax2.set_ylabel('NT-proBNP, pg/mL (log scale)', labelpad=6)
    ax2.grid(axis='y', color=GRID, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_title('B  NT-proBNP by predicted grade', loc='left', fontsize=9, color=INK,
                  pad=8)
    ax2.text(0.03, 0.955, f"Spearman ρ = {rho:.3f} (risk score vs. log NT-proBNP)\n"
             f"Kruskal–Wallis p = {kw.pvalue:.1e}   |   n = {len(rows)}",
             transform=ax2.transAxes, fontsize=7, color=INK2, va='top')

    os.makedirs(FIG, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'figure4.{ext}'))
    print(f"→ {FIG}/figure4.pdf (+.png)")
    print(f"   A: n={len(coh)} hasta, {int(E.sum())} ölüm | log-rank p={lr.p_value:.2e}")
    print(f"   B: n={len(rows)} | Spearman rho={rho:.3f} (p={prho:.1e}) | KW p={kw.pvalue:.2e}")
    print(f"   NOT: A'da y ekseni 0.70'ten başlıyor (kesik) — makale metninde belirt.")


def fig1(args):
    """Figür 1 — CONSORT tarzı çalışma akışı. TÜM sayılar diastolic_labels.csv / splits.csv'den
    canlı hesaplanır; hiçbiri elle yazılmaz (rakam kaymasın)."""
    lab = list(csv.DictReader(open(os.path.join(H, 'diastolic_labels.csv'))))
    esl = list(csv.DictReader(open(os.path.join(H, 'echo-study-list.csv'))))
    spl = list(csv.DictReader(open(os.path.join(H, 'splits.csv'))))
    pat = lambda rows: len({r['subject_id'] for r in rows})

    exc = [r for r in lab if r['excluded'] in ('True', '1', 'true')]
    keep = [r for r in lab if r not in exc]
    n_af = sum(1 for r in exc if 'AF' in r['excl_reason'].split(';'))
    n_mr = sum(1 for r in exc if 'MR:' in r['excl_reason'])
    n_ms = sum(1 for r in exc if 'MS:' in r['excl_reason'])
    n_mac = sum(1 for r in exc if 'MAC:' in r['excl_reason'])
    inc = [r for r in keep if r['grade_2025'] == 'Incomputable']
    amb = [r for r in keep if r['grade_2025'] == 'Grade2_or_3']
    grad = [r for r in keep if r['grade_2025'] in GR]
    gd = {g: sum(1 for r in spl if r['grade_2025'] == g) for g in GR}
    sd = {s: sum(1 for r in spl if r['split'] == s) for s in ['train', 'val', 'test']}
    cd = {c: sum(1 for r in spl if r['confidence'] == c) for c in ['ecg_sinus', 'assumed_sinus']}
    clips = sum(int(r['n_clips']) for r in spl)

    n_clip_all = 77069            # view_classify.log: sınıflandırılan toplam klip

    # ── Yerleşim İÇERİKTEN türetilir: kutu yüksekliği satır sayısından, dikey boşluk da
    # yanındaki dışlama kutusunun yüksekliğinden. Sabit koordinat vermeyince metin taşamaz.
    UNIT = 0.132                  # 1 satır ≈ 0.132 inç (7-8 pt yazı için rahat satır aralığı)
    PAD_M, PAD_E = 0.9, 0.7       # kutu içi üst/alt boşluk (satır cinsinden)

    # (ana kutu satırları, dışlama kutusu satırları)
    stages = [
        ([('MIMIC-IV-ECHO transthoracic studies', True),
          (f"{len(esl):,} studies · {pat(esl):,} patients", False)],
         [f"No linked structured measurement", f"report: {len(esl)-len(lab):,} studies"]),
        ([('Linked to structured measurements', True),
          (f"{len(lab):,} studies · {pat(lab):,} patients", False)],
         ['2025-ASE algorithm not applicable', '(assumes sinus rhythm, no significant',
          f"mitral disease) — {len(exc):,} studies:",
          f"   · atrial fibrillation/flutter: {n_af:,}",
          f"   · mitral regurgitation ≥3+: {n_mr:,}",
          f"   · mitral stenosis: {n_ms:,}",
          f"   · mitral annular calcification: {n_mac:,}",
          '   (categories overlap)']),
        ([('Eligible for 2025-ASE grading', True),
          (f"{len(keep):,} studies · {pat(keep):,} patients", False)],
         [f"Grade not assignable — {len(inc)+len(amb):,} studies:",
          "   · missing determinants (E/e′, LAVi,",
          f"     or TR velocity): {len(inc):,}",
          "   · Grade 2 vs 3 ambiguous",
          f"     (E/A unavailable): {len(amb):,}"]),
        # Klip view-seçimi TAM BURADA olur (kohorttan sonra değil) → notu bu adıma bağla,
        # yoksa figür "kohort ile split arasında klip eleniyor" gibi okunur.
        ([('Definite 2025-ASE grade assigned', True),
          (f"{len(grad):,} studies · {pat(grad):,} patients", False)],
         ['Clip view selection: color/spectral',
          'Doppler and still frames removed;',
          f"{clips:,} of {n_clip_all:,} classified clips",
          f"retained as A4C/A2C/PLAX ({100*clips/n_clip_all:.0f}%)",
          '',
          'No usable A4C/A2C/PLAX B-mode',
          f"cine: {len(grad)-len(spl):,} studies"]),
        ([('FINAL COHORT', True),
          (f"{len(spl):,} studies · {pat(spl):,} patients · {clips:,} clips", False),
          (f"Normal {gd['Normal']:,} · Grade 1 {gd['Grade1']:,} · Grade 2 {gd['Grade2']:,} · "
           f"Grade 3 {gd['Grade3']:,}", False),
          (f"ECG-confirmed sinus {cd['ecg_sinus']:,} · assumed sinus {cd['assumed_sinus']:,}",
           False)],
         []),
    ]
    split_line = (f"Patient-level split (no patient in two sets):   train {sd['train']:,}   ·   "
                  f"val {sd['val']:,}   ·   test {sd['test']:,}")

    hs = []                       # (ana_h, dışlama_h, boşluk)
    for main, ex in stages:
        hm = len(main) + 2 * PAD_M
        he = len(ex) + 2 * PAD_E if ex else 0
        hs.append((hm, he, max(3.4, he + 1.6)))
    total = sum(hm + gap for hm, _, gap in hs) + (1 + 2 * PAD_M) + 2.0   # + split kutusu

    fig, ax = plt.subplots(figsize=(7.2, total * UNIT + 0.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, total); ax.axis('off')
    XC, XM, WM, XE, WE = 30, 2, 56, 62, 37     # ana kutu merkezi/x/genişlik, dışlama x/genişlik

    y = total - 1.0
    for (main, ex), (hm, he, gap) in zip(stages, hs):
        is_final = main[0][0] == 'FINAL COHORT'
        ax.add_patch(plt.Rectangle((XM, y - hm), WM, hm, facecolor='#dcebfd' if is_final else '#eef4fd',
                                   edgecolor=SEQ[3] if is_final else SEQ_LINE[1],
                                   linewidth=0.9 if is_final else 0.7))
        for i, (txt, bold) in enumerate(main):     # satırlar kutunun İÇİNE, üstten sırayla
            ax.text(XC, y - PAD_M - 0.5 - i, txt, ha='center', va='center',
                    fontsize=7.8 if bold else 7.0, color=INK if bold else INK2,
                    fontweight='bold' if bold else 'normal')
        yb = y - hm
        ax.annotate('', xy=(XC, yb - gap), xytext=(XC, yb),
                    arrowprops=dict(arrowstyle='-|>', color=INK2, linewidth=0.9))
        if ex:
            ye = yb - gap / 2 - he / 2 + 0.5
            ax.plot([XC, XE], [ye + he / 2, ye + he / 2], color=INK2, linewidth=0.7)
            ax.add_patch(plt.Rectangle((XE, ye), WE, he, facecolor='#f4f4f2',
                                       edgecolor=INK2, linewidth=0.5))
            for i, line in enumerate(ex):
                ax.text(XE + 1.5, ye + he - PAD_E - 0.5 - i, line, ha='left', va='center',
                        fontsize=6.5, color=INK2)
        y = yb - gap

    hsp = 1 + 2 * PAD_M
    ax.add_patch(plt.Rectangle((XM, y - hsp), 96, hsp, facecolor='#ffffff',
                               edgecolor=INK2, linewidth=0.6))
    ax.text(XM + 48, y - hsp / 2, split_line, ha='center', va='center', fontsize=7.4, color=INK)

    os.makedirs(FIG, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'figure1.{ext}'))
    print(f"→ {FIG}/figure1.pdf (+.png)")
    print(f"   {len(esl)} → {len(lab)} → {len(keep)} → {len(grad)} → {len(spl)} study "
          f"| final {pat(spl)} hasta, {clips} klip")


def fig2(args):
    """Figür 2 — boru hattı/mimari şeması. Kodla BİREBİR olmalı:
      C.1  PanEcho video backbone (echo-pretrained), KLİP düzeyinde fine-tune;
           CORN ordinal (K-1=3 koşullu mantık) + aux regresyon (E/e′, LAVi, TRvel, EF),
           kayıp = CORN + 0.3·maskeli-MSE.
      C.2  (ASIL MODEL) backbone dondurulur → klip başına 768-d gömme → bag = study →
           gated attention MIL (Ilse 2018): α = softmax(w·[tanh(Vx) ⊙ σ(Ux)]), hid=128 →
           study vektörü (768) → CORN + aux.
      Çıkış: kümülatif P(≥G1), P(≥G2), P(≥G3); grade = 0.5'i geçen rank sayısı,
             risk skoru = P(≥Grade 2).
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    def blk(x, y, w, h, title, lines, fc='#eef4fd', ec=SEQ_LINE[1], ts=7.4):
        """Satırları kutunun İÇİNE eşit dağıtır (sabit ofset kullanınca son satırlar taşıyordu).
        lines: str ya da (str, fontsize)."""
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=0.8))
        ax.text(x + w / 2, y + h - 5.0, title, ha='center', va='center', fontsize=ts,
                fontweight='bold', color=INK)
        top, bot = y + h - 10.0, y + 2.5          # gövde metni bu bandın içinde kalır
        n = len(lines)
        for i, ln in enumerate(lines):
            txt, fs = ln if isinstance(ln, tuple) else (ln, 6.3)
            yy = top - (i + 0.5) * (top - bot) / n
            ax.text(x + w / 2, yy, txt, ha='center', va='center', fontsize=fs, color=INK2)

    def arr(x0, x1, y=50):
        ax.annotate('', xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle='-|>', color=INK2, linewidth=1.0))

    YT, HT = 44, 38                      # ana sıra: y ve yükseklik
    YM = YT + HT / 2                     # ana okların y'si

    # 1 — girdi
    blk(0, YT, 20, HT, 'Study', ['n clips (variable)', 'A4C · A2C · PLAX', '16 frames',
                                 '224 × 224 grayscale', ('B-mode only', 6.3)])
    arr(20, 25, YM)
    # 2 — C.1 backbone
    blk(25, YT, 22, HT, 'PanEcho encoder', ['echo-pretrained video', 'backbone, fine-tuned',
                                            'at clip level', '→ 768-d per clip'])
    ax.text(36, YT - 4, 'Stage C.1 — frozen after training', ha='center', fontsize=6.2,
            color=INK2, style='italic')
    arr(47, 52, YM)
    # 3 — MIL (asıl model)
    blk(52, YT, 26, HT, 'Gated attention MIL',
        ['bag = study, instances', '= clip embeddings',
         ('α = softmax(w·[tanh(Vx) ⊙ σ(Ux)])', 5.6),
         'hidden 128 (Ilse 2018)', '→ 768-d study vector'],
        fc='#dcebfd', ec=SEQ[3])
    ax.text(65, YT - 4, 'Stage C.2 — the reported model', ha='center', fontsize=6.2,
            color=SEQ[3], style='italic', fontweight='bold')
    arr(78, 81, YM)
    # 4 — CORN head (ana yol)
    blk(81, YT, 19, HT, 'CORN ordinal head', ['3 conditional logits',
                                              ('P(≥G1), P(≥G2), P(≥G3)', 5.7)], ts=7.0)

    # 5 — aux head: MIL'in ALTINDA ve yan yolda. Çıktı okunun ÜSTÜNDEN geçmemeli, yoksa
    # "aux çıktıya besleniyor" gibi okunur — oysa yalnızca eğitim sinyalidir.
    blk(52, 17, 26, 21, 'Auxiliary head', ['E/e′, LAVi, TR vel, EF',
                                           ('masked MSE — training only', 5.9)],
        fc='#f4f4f2', ec=INK2, ts=7.0)
    # kesikli ok sağa kaydırıldı: 'Stage C.2' altyazısının üstünden geçmesin
    ax.annotate('', xy=(72, 38), xytext=(72, YT),
                arrowprops=dict(arrowstyle='-|>', color=INK2, linewidth=0.8, linestyle=':'))

    # çıktı şeridi — ok CORN'dan iner, aux kutusuna DEĞMEZ. Metin İKİ satır: tek satırda
    # şeridin kenarlarından taşıyordu.
    ax.add_patch(plt.Rectangle((0, 0), 100, 15, facecolor='#f8fbff', edgecolor=SEQ_LINE[0],
                               linewidth=0.7))
    ax.text(3, 11.0, 'Outputs', fontsize=7.4, fontweight='bold', color=INK, va='center')
    ax.text(50, 7.5, 'grade = number of cumulative probabilities > 0.5', fontsize=6.4,
            color=INK2, va='center', ha='center')
    ax.text(50, 3.0, 'any dysfunction = P(≥Grade 1)      ·      advanced dysfunction '
            '(risk score) = P(≥Grade 2)', fontsize=6.4, color=INK2, va='center', ha='center')
    ax.annotate('', xy=(90.5, 15), xytext=(90.5, YT),
                arrowprops=dict(arrowstyle='-|>', color=INK2, linewidth=1.0))

    # eğitim kaybı notu
    ax.text(50, 88, 'Training loss = CORN ordinal  +  0.3 × masked MSE (guideline determinants)',
            ha='center', fontsize=6.8, color=INK2)
    ax.text(50, 96, 'No Doppler input at any stage', ha='center', fontsize=8.2,
            fontweight='bold', color=CAT['adv'])

    os.makedirs(FIG, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'figure2.{ext}'))
    print(f"→ {FIG}/figure2.pdf (+.png)")


def figS1(args):
    """Ek Figür S1 — TAHMİNİ grade'e göre kılavuz belirleyicileri (OOF, tüm kohort).

    DİKKAT (makale metninde de belirtildi): E/e′, LAVi ve TR hızı referans etiketi ÜRETEN
    ölçümlerdir. Bu figür bağımsız doğrulama DEĞİL; modelin görüntüden kılavuz belirleyicilerini
    geri kazandığını gösteren bir tutarlılık kontrolüdür. Bağımsız doğrulama Figür 5'tir
    (ölüm + NT-proBNP — ikisi de ekokardiyogramdan okunamaz).
    """
    from scipy.stats import kruskal, spearmanr

    man = {r['study_id']: r for r in csv.DictReader(open(os.path.join(H, 'manifest_study.csv')))}
    oof = list(csv.DictReader(open(os.path.expanduser(args.preds))))

    PANELS = [('Ee_mean', "E/e′ (mean)", None),
              ('LAVi', 'LA volume index, mL/m²', None),
              ('TRvel', 'TR velocity, m/s', None),
              ('EF', 'LVEF, %', None)]

    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.9))   # JASE: figür ≤7.25 inç geniş
    for ax, (col, lab, _) in zip(axs, PANELS):
        data, gi = [], []
        for g in range(4):
            v = [float(man[r['study_id']][col]) for r in oof
                 if r['pred'] == GR[g] and man.get(r['study_id'], {}).get(col, '').strip()]
            data.append(v); gi += [g] * len(v)
        allv = [x for v in data for x in v]
        rho, _ = spearmanr(gi, allv)
        kw = kruskal(*[v for v in data if len(v) > 1])

        bp = ax.boxplot(data, positions=range(4), widths=0.62, patch_artist=True,
                        showfliers=False, medianprops=dict(color='white', linewidth=1.3),
                        whiskerprops=dict(color=INK2, linewidth=0.7),
                        capprops=dict(color=INK2, linewidth=0.7))
        for g, b in enumerate(bp['boxes']):
            b.set(facecolor=SEQ[g], edgecolor=INK2, linewidth=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['N', 'G1', 'G2', 'G3'], fontsize=7)
        ax.set_xlabel('Predicted grade', fontsize=7.2)
        ax.set_ylabel(lab, fontsize=7.2)
        ax.tick_params(labelsize=6.8)
        ax.grid(axis='y', color=GRID, linewidth=0.5); ax.set_axisbelow(True)
        ax.set_title(f"ρ = {rho:.2f}   p = {kw.pvalue:.0e}", loc='left', fontsize=6.8,
                     color=INK2, pad=4)
        # her n'i KENDİ kutusunun altına hizala (ortalanmış tek satır yanıltıcı okunuyordu)
        for g, v in enumerate(data):
            ax.text(g, -0.20, f"{len(v)}", transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=6, color=INK2)
        ax.text(-0.02, -0.20, 'n', transform=ax.transAxes, ha='right', va='top',
                fontsize=6, color=INK2, style='italic')

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'figureS1.{ext}'))
    print(f"→ {FIG}/figureS1.pdf (+.png) | n={len(oof)} OOF study")
    print("   NOT: E/e′, LAVi, TRvel etiketi ÜRETEN ölçümler → bu tutarlılık kontrolü,")
    print("        bağımsız doğrulama değil (o Figür 4).")


def figS2(args):
    """Ek Figür S2 — attention ağırlıkları (yorumlanabilirlik).

    Eğitilmiş gated-attention MIL başlığını (runs/c2_mil/mil_attention/best.pt) donmuş test
    gömmelerine uygular ve her klibin α'sını çıkarır. Klip sayısı study'den study'ye değiştiği
    için ham α karşılaştırılamaz → her α'yı ÜNİFORM ağırlıkla normalize ediyoruz:
        rel = α_i × n_clip   (1.0 = model bu klibe 'ortalama kadar' bakıyor)
    Böylece "model hangi kesite, prevalansının ötesinde, fazladan ağırlık veriyor?" sorusu
    doğrudan okunur. GPU gerekmez — başlık küçük, CPU'da saniyeler sürer.
    """
    import torch
    import torch.nn.functional as F

    z = np.load(os.path.join(H, 'runs/c2_mil/emb_test.npz'), allow_pickle=True)
    emb, sid, view = z['emb'], z['study_id'], z['view']
    ck = torch.load(os.path.join(H, 'runs/c2_mil/mil_attention/best.pt'),
                    map_location='cpu', weights_only=True)

    x = torch.from_numpy(emb).float()
    a = F.linear(torch.tanh(F.linear(x, ck['V.weight'], ck['V.bias']))
                 * torch.sigmoid(F.linear(x, ck['U.weight'], ck['U.bias'])),
                 ck['w.weight'], ck['w.bias']).squeeze(-1).numpy()

    VIEWS = ['A4C', 'A2C', 'Parasternal_Long']
    VLAB = ['A4C', 'A2C', 'PLAX']
    rel = {v: [] for v in VIEWS}
    att_mass = {v: 0.0 for v in VIEWS}
    clip_n = {v: 0 for v in VIEWS}
    for s in np.unique(sid):
        m = sid == s
        alpha = np.exp(a[m] - a[m].max()); alpha /= alpha.sum()      # study içinde softmax
        n = m.sum()
        for al, v in zip(alpha, view[m]):
            if v in rel:
                rel[v].append(al * n)                                # 1.0 = üniform
                att_mass[v] += al
                clip_n[v] += 1
    n_std = len(np.unique(sid))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # A) klip başına göreli attention (1.0 = üniform)
    data = [rel[v] for v in VIEWS]
    bp = ax1.boxplot(data, positions=range(3), widths=0.6, patch_artist=True,
                     showfliers=False, medianprops=dict(color='white', linewidth=1.3),
                     whiskerprops=dict(color=INK2, linewidth=0.7),
                     capprops=dict(color=INK2, linewidth=0.7))
    for k, b in enumerate(bp['boxes']):
        b.set(facecolor=SEQ[k], edgecolor=INK2, linewidth=0.6)
    ax1.axhline(1.0, color=CAT['adv'], linewidth=1.0, linestyle='--', zorder=1)
    ax1.annotate('uniform', xy=(2.45, 1.0), fontsize=6.5, color=CAT['adv'], va='bottom',
                 ha='right')
    ax1.set_xticks(range(3)); ax1.set_xticklabels(VLAB, fontsize=7.5)
    ax1.set_ylabel('Attention relative to uniform\n(α × number of clips)', fontsize=7.2)
    ax1.set_xlabel('View', fontsize=7.2)
    ax1.tick_params(labelsize=6.8)
    ax1.grid(axis='y', color=GRID, linewidth=0.5); ax1.set_axisbelow(True)
    ax1.set_title('A  Per-clip attention by view', loc='left', fontsize=8.5, color=INK, pad=6)

    # B) toplam attention payı vs klip payı — model bir kesiti prevalansının ötesinde mi seçiyor?
    tot_clips = sum(clip_n.values()); tot_att = sum(att_mass.values())
    cs = [100 * clip_n[v] / tot_clips for v in VIEWS]
    as_ = [100 * att_mass[v] / tot_att for v in VIEWS]
    xx = np.arange(3)
    ax2.bar(xx - 0.19, cs, 0.36, color=GRID, edgecolor=INK2, linewidth=0.5, label='Clips')
    ax2.bar(xx + 0.19, as_, 0.36, color=SEQ[2], edgecolor=INK2, linewidth=0.5,
            label='Attention')
    for k in range(3):
        ax2.text(xx[k] + 0.19, as_[k] + 1.2, f"{as_[k]:.0f}%", ha='center', fontsize=6.5,
                 color=INK2)
    ax2.set_xticks(xx); ax2.set_xticklabels(VLAB, fontsize=7.5)
    ax2.set_ylabel('Share of total (%)', fontsize=7.2)
    ax2.set_xlabel('View', fontsize=7.2)
    ax2.tick_params(labelsize=6.8)
    ax2.set_ylim(0, 52)          # legend'e yer aç: 'upper right'ta çubukların üstüne biniyordu
    ax2.legend(frameon=False, fontsize=6.8, labelcolor=INK, loc='upper center', ncol=2)
    ax2.grid(axis='y', color=GRID, linewidth=0.5); ax2.set_axisbelow(True)
    ax2.set_title('B  Attention share vs. clip share', loc='left', fontsize=8.5, color=INK,
                  pad=6)

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'figureS2.{ext}'))
    print(f"→ {FIG}/figureS2.pdf (+.png) | {n_std} test study, {len(sid)} clip")
    for v, l in zip(VIEWS, VLAB):
        print(f"   {l:5s} klip payı {100*clip_n[v]/tot_clips:5.1f}%  "
              f"attention payı {100*att_mass[v]/tot_att:5.1f}%  "
              f"medyan göreli α {np.median(rel[v]):.2f}")


def central(args):
    """Central Illustration — JASE kuralları SERT ve mevcut tasarım ikisini de ihliyordu:
      * TAM 7.25 inç geniş (2175 px @300dpi), yükseklik 3.75–8.00 inç (1125–2400 px).
        Önceki sürüm 7.90 inç genişti → kabul edilmez.
      * Figür içindeki EN KÜÇÜK yazı 10 pt. Önceki sürümde metinler 6–8 pt idi.
        10 pt'ye sığmak için içerik SEYRELTİLDİ ve tuval uzatıldı (7.25 × 7.2 inç).
      * Tercih edilen dosya tipi TIFF/JPEG (PDF değil) → üçü de yazılır.
    Sonda bir doğrulayıcı, 10 pt'nin altında yazı kalmadığını ve piksel ölçüsünü denetler.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    FS_MIN = 10                     # JASE tabanı — hiçbir yazı bunun altına inemez
    FS_BODY, FS_HEAD, FS_TITLE = 10, 12, 15

    # Örnek vaka: --case ile değiştirilebilir. Varsayılan, makalede gösterilen test-seti
    # vakasıdır (EF %70, referans Grade 2, model P(≥G2)=0.94). MIMIC study_id'leri
    # de-identify edilmiş vekil anahtarlardır; yine de sabitlemek yerine parametreleştirildi.
    CASE = {v: f"{args.case}_{c}.npz" for v, c in
            zip(('A4C', 'A2C', 'PLAX'), args.case_clips.split(','))}

    fig = plt.figure(figsize=(7.25, 7.2))          # = 2175 × 2160 px @300 dpi
    gs = fig.add_gridspec(4, 2, height_ratios=[0.52, 1.85, 2.45, 0.32],
                          hspace=0.45, wspace=0.30,
                          left=0.085, right=0.975, top=0.975, bottom=0.045)

    # ── başlık
    axt = fig.add_subplot(gs[0, :]); axt.axis('off')
    axt.text(0.5, 0.72, 'Diastolic function graded from B-mode video alone',
             ha='center', va='center', fontsize=FS_TITLE, fontweight='bold', color=INK)
    axt.text(0.5, 0.26, 'MIMIC-IV-ECHO (open) · 3,065 studies · 2,518 patients · 2025 ASE labels',
             ha='center', va='center', fontsize=FS_BODY, color=INK2)

    # ── boru hattı şeridi
    axp = fig.add_subplot(gs[1, :]); axp.axis('off')
    axp.set_xlim(0, 100); axp.set_ylim(0, 100)
    for k, (v, fn) in enumerate(CASE.items()):
        d = np.load(os.path.join(H, 'npz_hires', fn)); fr = d['frames']
        b = axp.inset_axes([0.005 + k * 0.095, 0.28, 0.090, 0.68])
        b.imshow(fr[len(fr) // 3], cmap='gray', vmin=0, vmax=255, aspect='equal')
        b.set_xticks([]); b.set_yticks([])
        for s in b.spines.values():
            s.set_color(INK2); s.set_linewidth(0.6)
        b.set_title(v, fontsize=FS_BODY, color=INK2, pad=3)
    axp.text(14, 10, 'B-mode only — no Doppler', ha='center', va='center',
             fontsize=FS_BODY, color=CAT['adv'], fontweight='bold')
    axp.annotate('', xy=(35, 60), xytext=(28, 60),
                 arrowprops=dict(arrowstyle='-|>', color=INK2, linewidth=1.2))
    axp.add_patch(plt.Rectangle((36, 30), 30, 60, facecolor='#dcebfd', edgecolor=SEQ[3],
                                linewidth=0.9))
    axp.text(51, 73, 'PanEcho encoder', ha='center', va='center', fontsize=FS_BODY,
             fontweight='bold', color=INK)
    axp.text(51, 56, '+ multi-view attention', ha='center', va='center', fontsize=FS_BODY,
             color=INK2)
    axp.text(51, 39, 'over A4C · A2C · PLAX', ha='center', va='center', fontsize=FS_BODY,
             color=INK2)
    axp.annotate('', xy=(74, 60), xytext=(67, 60),
                 arrowprops=dict(arrowstyle='-|>', color=INK2, linewidth=1.2))
    axp.add_patch(plt.Rectangle((75, 30), 25, 60, facecolor='#eef4fd', edgecolor=SEQ_LINE[1],
                                linewidth=0.9))
    axp.text(87.5, 73, 'Grade + risk', ha='center', va='center', fontsize=FS_BODY,
             fontweight='bold', color=INK)
    axp.text(87.5, 56, 'Normal → Grade 3', ha='center', va='center', fontsize=FS_BODY,
             color=INK2)
    axp.text(87.5, 39, 'P(≥Grade 2)', ha='center', va='center', fontsize=FS_BODY, color=INK2)
    # kısaltıldı: uzun hâli soldaki 'no Doppler' etiketinin üstüne biniyordu
    axp.text(66, 10, 'Case: EF 70% → model: advanced dysfunction, P = 0.94',
             ha='center', va='center', fontsize=FS_BODY, color=INK)

    # ── sol: tarama ROC (kilitli test)
    t, p, cum = load_preds(args.preds)
    axs = fig.add_subplot(gs[2, 0])
    r2 = lambda x: f"{Decimal(str(x)).quantize(Decimal('0.01'), ROUND_HALF_UP)}"
    for key, (y, s, lab) in {
        'any': ((t >= 1).astype(int), cum[:, 0], 'Any DD'),
        'adv': ((t >= 2).astype(int), cum[:, 1], 'Advanced DD'),
    }.items():
        fpr, tpr, _ = roc_curve(y, s)
        axs.plot(fpr, tpr, color=CAT[key], linewidth=2.0,
                 label=f"{lab}  {r2(roc_auc_score(y, s))}")
    axs.plot([0, 1], [0, 1], color=GRID, linewidth=1.0, linestyle=':')
    axs.set_xlim(0, 1); axs.set_ylim(0, 1.02); axs.set_aspect('equal')
    axs.set_xlabel('1 − Specificity', fontsize=FS_BODY)
    axs.set_ylabel('Sensitivity', fontsize=FS_BODY)
    axs.set_xticks([0, 0.5, 1]); axs.set_yticks([0, 0.5, 1])
    axs.tick_params(labelsize=FS_BODY)
    axs.legend(loc='lower right', frameon=False, fontsize=FS_BODY, handlelength=1.2,
               labelcolor=INK, borderpad=0.2, title='AUROC', title_fontsize=FS_BODY)
    axs.set_title('Doppler-free screening\n(locked test set, n=438)', loc='left',
                  fontsize=FS_HEAD, fontweight='bold', color=INK, pad=8)

    # ── sağ: prognoz (OOF)
    first = {}
    for r in csv.DictReader(open(os.path.expanduser(args.mort))):
        s = r['subject']
        if s not in first or r['acq'] < first[s]['acq']:
            first[s] = r
    coh = list(first.values())
    T = np.array([float(r['time']) for r in coh]); E = np.array([int(r['event']) for r in coh])
    ADV = np.array([int(r['pred_grade']) >= 2 for r in coh])
    axk = fig.add_subplot(gs[2, 1])
    for m, col, st, lab in [(~ADV, SEQ_LINE[0], '-', 'Normal / Grade 1'),
                            (ADV, CAT['adv'], '--', 'Advanced')]:
        KaplanMeierFitter(label=lab).fit(T[m], E[m]).plot_survival_function(
            ax=axk, ci_show=False, color=col, linestyle=st, linewidth=2.0)
    axk.set_xlim(0, 365); axk.set_ylim(0.80, 1.005)
    axk.set_xticks([0, 180, 365]); axk.set_yticks([0.8, 0.9, 1.0])
    axk.set_xlabel('Days since echocardiogram', fontsize=FS_BODY)
    axk.set_ylabel('Survival probability', fontsize=FS_BODY)
    axk.tick_params(labelsize=FS_BODY)
    axk.grid(axis='y', color=GRID, linewidth=0.5); axk.set_axisbelow(True)
    axk.legend(loc='lower left', frameon=False, fontsize=FS_BODY, handlelength=1.4,
               labelcolor=INK, borderpad=0.2, title='Predicted', title_fontsize=FS_BODY)
    axk.set_title('Validated outside the echo\n(out-of-fold, n=2,518)', loc='left',
                  fontsize=FS_HEAD, fontweight='bold', color=INK, pad=8)
    axk.text(0.97, 0.97, 'HR 3.03 (1.48–6.18)\nadjusted for age, sex, EF\nNT-proBNP ρ = 0.58',
             transform=axk.transAxes, fontsize=FS_BODY, color=INK2, va='top', ha='right',
             linespacing=1.5)

    # ── alt şerit
    axb = fig.add_subplot(gs[3, :]); axb.axis('off')
    axb.add_patch(plt.Rectangle((0, 0.12), 1, 0.76, transform=axb.transAxes,
                                facecolor='#eef4fd', edgecolor='none'))
    axb.text(0.5, 0.5, 'Labels · splits · code · weights released', ha='center', va='center',
             fontsize=FS_HEAD, fontweight='bold', color=SEQ[3])

    # ── JASE denetimi: 10 pt altı yazı ve piksel ölçüsü
    small = [tx.get_fontsize() for tx in fig.findobj(matplotlib.text.Text)
             if tx.get_text().strip() and tx.get_fontsize() < FS_MIN]
    if small:
        raise SystemExit(f"JASE İHLALİ: {len(small)} yazı 10 pt altında: {sorted(set(small))}")

    os.makedirs(FIG, exist_ok=True)
    base = os.path.join(FIG, 'central_illustration')
    with plt.rc_context({'savefig.bbox': 'standard'}):
        for ext in ('pdf', 'png', 'tif', 'jpg'):       # JASE TIFF/JPEG tercih ediyor
            kw = {'pil_kwargs': {'quality': 95}} if ext == 'jpg' else {}
            fig.savefig(f"{base}.{ext}", dpi=300, **kw)
    from PIL import Image
    w, h = Image.open(f"{base}.png").size
    ok = (w == 2175) and (1125 <= h <= 2400)
    lr = logrank_test(T[~ADV], T[ADV], E[~ADV], E[ADV])
    print(f"→ {base}.tif / .jpg / .pdf / .png")
    print(f"   {w}×{h} px @300dpi = {w/300:.2f}×{h/300:.2f} in "
          f"→ JASE (2175 px geniş; 1125-2400 yüksek): {'OK' if ok else 'İHLAL'}")
    print(f"   en küçük yazı: {FS_MIN} pt (JASE tabanı) → OK")
    print(f"   ROC n={len(t)} | KM n={len(coh)}, log-rank p={lr.p_value:.1e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f1 = sub.add_parser('fig1'); f1.set_defaults(fn=fig1)
    f2 = sub.add_parser('fig2'); f2.set_defaults(fn=fig2)
    s2 = sub.add_parser('figS2'); s2.set_defaults(fn=figS2)
    s1 = sub.add_parser('figS1'); s1.set_defaults(fn=figS1)
    s1.add_argument('--preds', default='~/mimic-echo/runs/oof_predictions.csv')
    ci = sub.add_parser('central'); ci.set_defaults(fn=central)
    ci.add_argument('--preds', default='~/mimic-echo/runs/c2_mil/mil_attention/test_predictions.csv')
    ci.add_argument('--mort', default='~/mimic-echo/runs/mortality_cohort_oof.csv')
    ci.add_argument('--case', default='97850370', help='örnek vakanın study_id\'si')
    ci.add_argument('--case-clips', default='0036,0059,0002',
                    help='A4C,A2C,PLAX klip numaraları (virgülle)')
    f3 = sub.add_parser('fig3'); f3.set_defaults(fn=fig3)
    f3.add_argument('--preds', default='~/mimic-echo/runs/c2_mil/mil_attention/test_predictions.csv')
    f4 = sub.add_parser('fig4'); f4.set_defaults(fn=fig4)
    f4.add_argument('--mort', default='~/mimic-echo/runs/mortality_cohort_oof.csv')
    f4.add_argument('--bnp', default='~/mimic-echo/runs/ntprobnp_cohort_oof.csv')
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
