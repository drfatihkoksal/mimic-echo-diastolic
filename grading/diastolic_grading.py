#!/usr/bin/env python3
"""
Diyastolik disfonksiyon etiketleme (grading) modülü — MIMIC-IV-ECHO.

Metodolojinin KALBİ. İki kılavuzu da üretir:
  - ASE 2016 (Nagueh 2016) — karşılaştırma/duyarlılık için
  - ASE 2025 (Nagueh 2025, DOI 10.1016/j.echo.2025.03.011) — BİRİNCİL etiket, SADIK

Tasarım ilkeleri:
  - Saf grading fonksiyonları (grade_ase2016 / grade_ase2025) yan-etkisiz, test edilebilir.
  - Birim normalizasyonu + fizyolojik sınır filtresi (veri girişi hataları için).
  - Eksik parametrede 'Incomputable' döner (uydurma yok).
  - Hazır kardiyolog `diastolic_grade` harmonize edilir + türetme ile çapraz-doğrulanır.

MIMIC birim gerçekleri (profil doğrulandı):
  E, A, e' septal/lateral : m/s   (e' medyan 0.07 m/s = 7 cm/s)   -> kod cm/s'ye çevirir
  E/A, E/e'               : oransız
  la_vol                  : mL (cm3);  BSA: m2   -> LAVi = la_vol / BSA
  tr_velocity             : m/s   (PASP alanı `tv_est_pa_press_min` boş -> TR kullanılır)
  lvef                    : % ;  lvedd/septal_thickness/inf_lat_thickness: cm

Eksik (MIMIC'te YOK): LARS, IVRT, Pulmonary Vein S/D -> MIMIC'te 2025 ikincil kutu
  fiilen LAVi'ye dayanir. secondary_box_2025() kilavuzun TAM kutusunu uygular
  (LAVi / LARS / PV S/D, >=1 pozitif kurali; IVRT yedek). Bu degiskenleri tasiyan
  kohortlarda kutu eksiksiz calisir; MIMIC'te sonuc degismez.
"""
from __future__ import annotations
import argparse, csv, gzip, math, os, re, sys
from collections import defaultdict
from datetime import datetime

# --------------------------------------------------------------------------- #
# 0. Yollar (Fatih'in makinesi)                                               #
# --------------------------------------------------------------------------- #
ECHO_DIR   = os.path.expanduser("~/mimic-echo")
STUDY_LIST = os.path.join(ECHO_DIR, "echo-study-list.csv")
STRUCTURED = os.path.join(ECHO_DIR, "structured-measurement.csv.gz")
MIMIC_IV   = "/data/mimic 4 3 1/mimic-iv-3.1"          # echo v2.2 kaynaklı; v3.1 %100 kapsıyor
PATIENTS   = os.path.join(MIMIC_IV, "hosp", "patients.csv")
DIAGNOSES  = os.path.join(MIMIC_IV, "hosp", "diagnoses_icd.csv")
ECG_MM     = ("/data/identity/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
              "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv")
ECG_WINDOW_DAYS = 7    # eko-anı ritmi için en yakın ECG penceresi (temporal AF)
RECORD_LIST = os.path.join(ECHO_DIR, "echo-record-list.csv")   # DICOM envanteri
GCS_PREFIX  = "gs://mimic-iv-echo-1.0.physionet.org/"
BUCKET_TB, BUCKET_FILES = 1.7, 525422    # kaba boyut tahmini için (ort ~3.4 MB/dosya)

# --------------------------------------------------------------------------- #
# 1. Birim normalizasyonu + fizyolojik sınırlar                               #
# --------------------------------------------------------------------------- #
def parse_num(s):
    """'>2.8', '1.5-2.0', '72 %' -> float (aralık ise ortalama). Aksi None."""
    if s is None: return None
    s = s.strip()
    if not s: return None
    # Not: negatif değer beklenmez (hızlar/oranlar); tire = aralık ayıracı olarak ele alınır
    nums = re.findall(r'\d+\.?\d*', s.replace('<', ' ').replace('>', ' '))
    if not nums: return None
    f = [float(x) for x in nums]
    return sum(f) / len(f)

# Her parametre: (birimden cm/s vb. çarpanı, alt_sınır, üst_sınır) — sınır dışı -> None
# Değerler cm/s (hız) veya doğal birimde tutulur; oranlar oransız.
def _mps_to_cms(v): return v * 100.0

def norm_evel(v):      # E, A dalga hızı -> cm/s (m/s girdiden)
    if v is None: return None
    v = _mps_to_cms(v)
    return v if 5 <= v <= 250 else None

def norm_eprime(v):    # e' -> cm/s (m/s girdiden); fizyolojik 1-25 cm/s
    if v is None: return None
    v = _mps_to_cms(v)
    return v if 1 <= v <= 25 else None

def norm_ratio_ea(v):  # E/A oranı
    return v if (v is not None and 0.1 <= v <= 6) else None

def norm_ratio_ee(v):  # E/e' oranı
    return v if (v is not None and 2 <= v <= 70) else None

def norm_lavol(v):     # LA hacim (mL)
    return v if (v is not None and 5 <= v <= 300) else None

def norm_bsa(v):
    return v if (v is not None and 0.8 <= v <= 3.2) else None

def norm_tr(v):        # TR hızı (m/s)
    return v if (v is not None and 0.5 <= v <= 6.0) else None

def norm_ef(v):        # %
    return v if (v is not None and 5 <= v <= 90) else None

def norm_cm(v):        # duvar/boyut (cm)
    return v if (v is not None and 0.2 <= v <= 9) else None

def norm_decel(v):     # ms
    return v if (v is not None and 40 <= v <= 600) else None

# --- 2025 ikincil kutu degiskenleri (MIMIC'te yok; dis kohortlardan gelir) ---
def norm_lars(v):      # LA rezervuar strain, %
    return v if (v is not None and 1 <= v <= 70) else None

def norm_pvsd(v):      # pulmoner ven S/D orani, birimsiz
    return v if (v is not None and 0.1 <= v <= 5.0) else None

def norm_pvvel(v):     # pulmoner ven S veya D tepe hizi, cm/s
    return v if (v is not None and 5 <= v <= 150) else None

def norm_ivrt(v):      # izovolumik gevseme zamani, ms
    return v if (v is not None and 20 <= v <= 200) else None

# MIMIC measurement adı -> (kanonik anahtar, normalize fonksiyonu). Sadece TTE satırları.
PARAM_MAP = {
    'mv_peak_e':           ('E',        norm_evel),
    'mv_peak_a':           ('A',        norm_evel),
    'mv_peak_e_a':         ('EA',       norm_ratio_ea),
    'sept_e_prime':        ('e_sep',    norm_eprime),
    'lat_e_prime':         ('e_lat',    norm_eprime),
    'e_e_prime':           ('Ee_pre',   norm_ratio_ee),   # önceden hesaplı (kaynak belirsiz)
    'mean_E_to_eprime':    ('Ee_mean',  norm_ratio_ee),
    'septal_E_to_eprime':  ('Ee_sep',   norm_ratio_ee),
    'lateral_E_to_eprime': ('Ee_lat',   norm_ratio_ee),
    'la_vol':              ('LAvol',    norm_lavol),
    'body_surface_area':   ('BSA',      norm_bsa),
    'tr_velocity':         ('TRvel',    norm_tr),
    'lvef':                ('EF',       norm_ef),
    'lvedd':               ('LVIDd',    norm_cm),
    'septal_thickness':    ('IVSd',     norm_cm),
    'inf_lat_thickness':   ('PWTd',     norm_cm),
    'mv_e_decel':          ('DT',       norm_decel),
}
# Dışlama/kalite alanları (kategorik) ve hazır etiketler
CAT_MAP = {
    'mitral_regurg':   'MR',
    'mitral_stenosis': 'MS',
    'mac_severity':    'MAC',
    'diastolic_grade': 'ready_grade',
    'diastolic_fcn':   'ready_fcn',
}

# --------------------------------------------------------------------------- #
# 2. Türetilmiş değişkenler (E/e', LAVi, LV kütle indeksi)                     #
# --------------------------------------------------------------------------- #
def derive(p):
    """p: kanonik ham değerler dict'i. E/e' (sep/lat/ort), LAVi, LVMi ekler."""
    E = p.get('E')                      # cm/s
    es, el = p.get('e_sep'), p.get('e_lat')   # cm/s
    # E/e' bileşenden (E ve e' aynı birimde -> oran birimsiz). Önce türet, sonra hazır ile doldur.
    if E is not None and es: p.setdefault('Ee_sep_d', E / es)
    if E is not None and el: p.setdefault('Ee_lat_d', E / el)
    if E is not None and es and el: p.setdefault('Ee_mean_d', E / ((es + el) / 2.0))
    # Ortalama e' (cm/s)
    if es is not None and el is not None: p['e_avg'] = (es + el) / 2.0
    elif es is not None: p['e_avg'] = es
    elif el is not None: p['e_avg'] = el
    # Pulmoner ven S/D - hazir oran yoksa bilesen hizlardan turet
    pvs, pvd = p.get('PV_S'), p.get('PV_D')
    if pvs is not None and pvd:
        p.setdefault('PV_SD', pvs / pvd)
    # LAVi
    lav, bsa = p.get('LAvol'), p.get('BSA')
    if lav is not None and bsa: p['LAVi'] = lav / bsa
    # LV kütle indeksi (ASE küp formülü; cm -> g). LVM=0.8*(1.04*((LVIDd+PWTd+IVSd)^3 - LVIDd^3))+0.6
    d, pw, iv = p.get('LVIDd'), p.get('PWTd'), p.get('IVSd')
    if None not in (d, pw, iv):
        lvm = 0.8 * (1.04 * ((d + pw + iv) ** 3 - d ** 3)) + 0.6
        if bsa: p['LVMi'] = lvm / bsa
    return p

def ee_value(p, site):
    """E/e' değeri: önce türetilmiş (yüksek kapsama), sonra hazır. site: 'sep'|'lat'|'mean'."""
    d = {'sep': ('Ee_sep_d', 'Ee_sep'), 'lat': ('Ee_lat_d', 'Ee_lat'),
         'mean': ('Ee_mean_d', 'Ee_mean')}[site]
    for k in d:
        if p.get(k) is not None: return p[k]
    if site == 'mean' and p.get('Ee_pre') is not None:  # son çare önceden-hesaplı ortalama
        return p['Ee_pre']
    return None

# --------------------------------------------------------------------------- #
# 3. ASE 2025 — Table 6 yaş-özgü e' eşikleri (cm/s) + eşikler                  #
# --------------------------------------------------------------------------- #
def eprime_reduced_2025(p, age=None):
    """2025 'e' azalmış' kriteri. Yaş varsa Table 6, yoksa yaş-bağımsız (Fig 3 ana eşik)."""
    es, el, ea = p.get('e_sep'), p.get('e_lat'), p.get('e_avg')
    if age is not None:                                   # Table 6 (cm/s)
        if age < 40:   ts, tl, ta = 7, 10, 9
        elif age <= 65: ts, tl, ta = 6, 8, 7
        else:          ts, tl, ta = 6, 7, 6.5
        hits = [(es is not None and es < ts), (el is not None and el < tl),
                (ea is not None and ea < ta)]
    else:                                                  # yaş-bağımsız (septal≤6/lat≤7/ort≤6.5)
        hits = [(es is not None and es <= 6), (el is not None and el <= 7),
                (ea is not None and ea <= 6.5)]
    return any(hits), any(v is not None for v in (es, el, ea))

def ee_increased_2025(p):
    """E/e' artmış: septal≥15 VEYA lateral≥13 VEYA ortalama≥14."""
    checks = [(ee_value(p, 'sep'), 15), (ee_value(p, 'lat'), 13), (ee_value(p, 'mean'), 14)]
    avail = any(v is not None for v, _ in checks)
    hit   = any(v is not None and v >= t for v, t in checks)
    return hit, avail

def tr_increased_2025(p):
    """TR ≥2.8 m/s (PASP MIMIC'te boş)."""
    tr = p.get('TRvel')
    return (tr is not None and tr >= 2.8), (tr is not None)

# --- 2025 ikincil kutu esikleri (Nagueh 2025, JASE 38:537-569, Figure 3) -----
IK_LAVI = 34.0     # mL/m2 ; >  pozitif
IK_LARS = 18.0     # %     ; <= pozitif
IK_PVSD = 0.67     # oran  ; <= pozitif  (= %40 sistolik dolum fraksiyonu)
IK_IVRT = 70.0     # ms    ; <= pozitif  -- kilavuzda "Alternatively"

def secondary_box_2025(p, pv_sadece_dusuk_ef=False):
    """2025 ASE ikincil karar kutusu -- Nagueh 2025, Figure 3.

    UCLU OLCUT ve KURAL (Figure 3, dogrudan):
        Pulmonary Vein S/D <= 0.67  or  LARS <= 18%  or  LAVi > 34 mL/m2
        "None" -> Normal LAP        ">=1 present" -> Increased LAP
    Yani olculebilenlerden EN AZ BIRI pozitifse LAP yuksektir; hicbiri pozitif
    degilse LAP normaldir. Metin de aynisini soyler (s.555): "If LARS, pulmonary
    vein ... ratio, IVRT, and LAVi do not meet the cutoff threshold for elevated
    LAP, then LAP is likely normal."

    IVRT'nin konumu: Figure 3'te "Alternatively IVRT <= 70 ms" olarak uclunun
    ALTINDA yer alir; metinde de "LARS, pulmonary vein ... ratio, LAVi, or
    ALTERNATIVELY IVRT" denir. Bu yuzden es deger dorduncu olcut olarak sayilmaz:
    yalnizca ucluden hicbiri olculememisse devreye girer.

    PV S/D'nin kosullu kullanimi (s.555): oran LV sistolik disfonksiyonunda en
    guvenilirdir ve "should not be considered in normal subjects with normal
    echocardiographic results, when the ratio can be <= 0.67". Bu kutuya yalnizca
    birincil degiskenlerden en az biri anormalken inildigi icin degerlendirilen
    hasta tanim geregi "normal ekokardiyografik bulgulu normal denek" DEGILDIR;
    dolayisiyla varsayilan davranis PV S/D'yi olculdugu her yerde kullanmaktir.
    pv_sadece_dusuk_ef=True verilirse olcut yalnizca EF < %50 olan calismalarda
    uygulanir -- onceden tanimli duyarlilik analizi icin. Bu bir YORUM tercihidir
    ve makalede boyle beyan edilmelidir.

    GERIYE UYUMLULUK: yalniz LAVi mevcutken (MIMIC-IV-ECHO'nun durumu) karar eski
    surumle birebir aynidir; LAVi de yoksa ve baska hicbir degisken yoksa
    (False, False, ...) doner.

    Doner: (yuksek, mevcut, detay)
      detay: hangi olcut olculmus, hangisi pozitif, IVRT yedegine dusulmus mu.
    """
    lavi, lars, pvsd = p.get('LAVi'), p.get('LARS'), p.get('PV_SD')
    ef = p.get('EF')

    pv_kullanilabilir = pvsd is not None
    pv_bastirildi = False
    if pv_kullanilabilir and pv_sadece_dusuk_ef and not (ef is not None and ef < 50):
        pv_kullanilabilir, pv_bastirildi = False, True

    uclu = []                                        # (ad, pozitif_mi)
    if lavi is not None:
        uclu.append(('LAVi', lavi > IK_LAVI))
    if lars is not None:
        uclu.append(('LARS', lars <= IK_LARS))
    if pv_kullanilabilir:
        uclu.append(('PV_SD', pvsd <= IK_PVSD))

    detay = {'olculen': [a for a, _ in uclu],
             'pozitif': [a for a, poz in uclu if poz],
             'ivrt_yedegi': False,
             'pv_bastirildi': pv_bastirildi}

    if uclu:
        return any(poz for _, poz in uclu), True, detay

    # ucluden hicbiri yok -> kilavuzun "Alternatively IVRT" yedegi
    ivrt = p.get('IVRT')
    if ivrt is not None:
        detay.update(olculen=['IVRT'], ivrt_yedegi=True,
                     pozitif=['IVRT'] if ivrt <= IK_IVRT else [])
        return (ivrt <= IK_IVRT), True, detay

    return False, False, detay   # hicbir ikincil degisken yok -> karar verilemez

def grade_ase2025(p, age=None):
    """
    Nagueh 2025 Figure 3 (sinüs ritmi, ağır MR/MS/MAC dışlanmış varsayılır).
    Döner dict: grade, lap, dd_present, n_abnormal, flags, notes.
    """
    r = {'guideline': '2025'}
    e_red, e_av   = eprime_reduced_2025(p, age)
    ee_hi, ee_av  = ee_increased_2025(p)
    tr_hi, tr_av  = tr_increased_2025(p)
    EA = p.get('EA')

    # En az birincil 3 değişkenden yeterlisi yoksa hesaplanamaz
    if sum(int(x) for x in (e_av, ee_av, tr_av)) < 2:
        r.update(grade='Incomputable', lap=None, dd_present=None,
                 n_abnormal=None, reason='primer degiskenler yetersiz')
        return r

    abn = [e_red, ee_hi, tr_hi]
    n = sum(int(x) for x in abn)
    r['n_abnormal'] = n
    r['primary'] = {'e_reduced': e_red, 'ee_high': ee_hi, 'tr_high': tr_hi}

    def finalize_lap(lap):
        """LAP -> derece. Normal LAP: e' azalmışsa Grade1, değilse Normal DF."""
        if lap == 'elevated':
            if EA is None:
                return 'Grade2_or_3', 'E/A yok -> grade2/3 ayrimi yapilamadi'
            return ('Grade3' if EA >= 2 else 'Grade2'), None
        else:  # normal LAP
            return ('Grade1' if e_red else 'Normal'), None

    # Dallanma (Figure 3)
    if n == 0:
        lap = 'normal'
    elif n == 3:
        lap = 'elevated'
    elif e_red and not ee_hi and not tr_hi:            # SADECE e' azalmış
        if EA is not None and EA <= 0.8:
            lap = 'normal'                              # -> Grade 1
        else:                                           # E/A>0.8 (ya da E/A yok) -> ikincil kutu
            elev, av, ikd = secondary_box_2025(p)
            r['ikincil_kutu'] = ikd
            if not av:
                r.update(grade='Incomputable', lap=None, dd_present=e_red,
                         reason='ikincil kutu degiskeni yok')
                return r
            lap = 'elevated' if elev else 'normal'
    else:                                               # tek ↑E/e' / tek ↑TR / herhangi 2
        elev, av, ikd = secondary_box_2025(p)
        r['ikincil_kutu'] = ikd
        if not av:
            r.update(grade='Incomputable', lap=None, dd_present=None,
                     reason='ikincil kutu degiskeni yok')
            return r
        lap = 'elevated' if elev else 'normal'

    grade, note = finalize_lap(lap)
    r.update(grade=grade, lap=lap, dd_present=(grade != 'Normal'),
             reason=note)
    return r

# --------------------------------------------------------------------------- #
# 4. ASE 2016 (karşılaştırma) — Nagueh 2016 Fig 7 (varlık) + Fig 8 (grading)   #
# --------------------------------------------------------------------------- #
def _pos_of(checks):
    """checks: [(value, threshold, op), ...]. op '>' veya '<'. Döner (pozitif, mevcut)."""
    pos = avail = 0
    for v, t, op in checks:
        if v is None: continue
        avail += 1
        if (op == '>' and v > t) or (op == '<' and v < t):
            pos += 1
    return pos, avail

def _four_criteria_2016(p):
    """2016 normal-EF 4 kriteri: avg E/e'>14, e'(sep<7|lat<10), TR>2.8, LAVi>34."""
    ee = ee_value(p, 'mean')
    es, el = p.get('e_sep'), p.get('e_lat')
    eprime_abn = None
    if es is not None or el is not None:
        eprime_abn = (es is not None and es < 7) or (el is not None and el < 10)
    checks = []
    if ee is not None:        checks.append((ee > 14))
    if eprime_abn is not None: checks.append(eprime_abn)
    if p.get('TRvel') is not None: checks.append(p['TRvel'] > 2.8)
    if p.get('LAVi') is not None:  checks.append(p['LAVi'] > 34)
    avail = len(checks); pos = sum(int(x) for x in checks)
    return pos, avail

def _grade_2016(p):
    """2016 Fig 8: E/A tabanlı grading (DD var/EF düşük durumda)."""
    EA = p.get('EA'); E = p.get('E')
    ee = ee_value(p, 'mean')
    tr = p.get('TRvel'); lavi = p.get('LAVi')
    if EA is None:
        return 'Indeterminate', 'E/A yok'
    if EA <= 0.8 and (E is not None and E <= 50):
        return 'Grade1', None
    if EA >= 2:
        return 'Grade3', None
    # ara bölge -> 3 değişken (avg E/e'>14, TR>2.8, LAVi>34)
    checks = []
    if ee is not None: checks.append(ee > 14)
    if tr is not None: checks.append(tr > 2.8)
    if lavi is not None: checks.append(lavi > 34)
    avail = len(checks); pos = sum(int(x) for x in checks)
    if avail == 0: return 'Indeterminate', '3 degisken yok'
    frac = pos / avail
    if frac < 0.5:  return 'Grade1', None
    if frac > 0.5:  return 'Grade2', None
    return 'Indeterminate', '3 degisken 50%'

def grade_ase2016(p):
    r = {'guideline': '2016'}
    EF = p.get('EF')
    reduced_ef = (EF is not None and EF < 50)
    if reduced_ef:                              # DD varsayılır -> direkt grading
        g, note = _grade_2016(p)
        r.update(grade=g, dd_present=(g != 'Normal'), path='reducedEF', reason=note)
        return r
    # normal/bilinmeyen EF -> 4 kriter (varlık)
    pos, avail = _four_criteria_2016(p)
    if avail < 2:
        r.update(grade='Incomputable', dd_present=None, path='normalEF',
                 reason='4-kriterden <2 mevcut'); return r
    frac = pos / avail
    if frac < 0.5:
        r.update(grade='Normal', dd_present=False, path='normalEF'); return r
    if frac == 0.5:
        r.update(grade='Indeterminate', dd_present=None, path='normalEF'); return r
    # DD var -> grade
    g, note = _grade_2016(p)
    if g == 'Normal': g = 'Grade1'
    r.update(grade=g, dd_present=True, path='normalEF->grade', reason=note)
    return r

# --------------------------------------------------------------------------- #
# 5. Hazır kardiyolog grade harmonizasyonu                                     #
# --------------------------------------------------------------------------- #
READY_MAP = {
    '0 (nl)': 'Normal', 'I': 'Grade1', 'II': 'Grade2', 'III': 'Grade3',
    'IV': 'Grade3', 'III/IV': 'Grade3',            # eski IV -> mevcut max 3
    'DD Present': 'DD_present_ungraded',
    'Indeterminate': 'Indeterminate', 'Unable to assess': None,
}
def harmonize_ready(val):
    if val is None: return None
    return READY_MAP.get(val.strip(), None)

# --------------------------------------------------------------------------- #
# 6. Dışlama kriterleri (2025 Fig 3 ön-koşulu)                                 #
# --------------------------------------------------------------------------- #
MS_EXCLUDE  = re.compile(r'mild|moderate|severe', re.I)      # 'No valvular','Trivial','' hariç
MR_EXCLUDE  = re.compile(r'severe', re.I)                    # 'Severe (4+)','Moderate-severe (3+)'
MAC_EXCLUDE = re.compile(r'mod|severe', re.I)                # 'Mod MAC','Severe'

def exclusion_flags(cat, af=False):
    """Döner: (excluded_bool, sebep listesi). cat: MR/MS/MAC ham stringleri."""
    reasons = []
    ms = (cat.get('MS') or '').strip()
    if ms and ms.lower() not in ('no valvular', 'trivial') and MS_EXCLUDE.search(ms):
        reasons.append(f'MS:{ms}')
    mr = (cat.get('MR') or '').strip()
    if mr and MR_EXCLUDE.search(mr):
        reasons.append(f'MR:{mr}')
    mac = (cat.get('MAC') or '').strip()
    if mac and MAC_EXCLUDE.search(mac):
        reasons.append(f'MAC:{mac}')
    if af:
        reasons.append('AF')
    return (len(reasons) > 0), reasons

# --------------------------------------------------------------------------- #
# 7. Veri okuma + pipeline                                                     #
# --------------------------------------------------------------------------- #
def load_study_list():
    """measurement_id -> (subject_id, study_id, study_datetime)"""
    out = {}
    with open(STUDY_LIST, newline='') as f:
        for row in csv.DictReader(f):
            mid = (row.get('measurement_id') or '').strip()
            if mid:
                out[mid] = (row['subject_id'].strip(), row['study_id'].strip(),
                            (row.get('study_datetime') or '').strip())
    return out

def load_patients():
    """subject_id -> (gender, anchor_age, anchor_year)"""
    out = {}
    with open(PATIENTS, newline='') as f:
        for row in csv.DictReader(f):
            try:
                out[row['subject_id'].strip()] = (
                    row['gender'].strip(), int(row['anchor_age']), int(row['anchor_year']))
            except (ValueError, KeyError):
                continue
    return out

def load_af_subjects():
    """AF/flutter ICD tanısı olan subject_id kümesi (ICD9 4273x, ICD10 I48x). Fallback sinyali."""
    af = set()
    with open(DIAGNOSES, newline='') as f:
        for row in csv.DictReader(f):
            code = (row.get('icd_code') or '').strip().upper()
            if code.startswith('4273') or code.startswith('I48'):
                af.add(row['subject_id'].strip())
    return af

# ECG raporundan AF/flutter (afirmatif + possible; ventriküler fib hariç tutmak için 'atrial' şart)
_AF_RE = re.compile(r'atrial fib|a-?fib|atrial flutter|\bflutter\b', re.I)
def _parse_dt(s):
    try: return datetime.strptime((s or '')[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError): return None

def load_ecg_rhythm(study_map, window_days=ECG_WINDOW_DAYS):
    """
    Eko-anı ritmi: her measurement_id için, aynı hastanın eko tarihine EN YAKIN ECG'sini
    (pencere içinde) sınıfla. Döner: {mid: (rhythm, gap_gun, rapor0)} ; rhythm ∈ {'AF','sinus'}.
    Pencere dışı / ECG yok -> mid sözlükte yok (çağıran ICD fallback'e düşer).
    NOT: Aynı hastanın eko ve ECG zamanları AYNI tarih-kaydırmasıyla deidentifiye -> fark geçerli.
    """
    echo_subj = {v[0] for v in study_map.values()}
    ecg = defaultdict(list)   # subject -> [(dt, af_bool, report0)]
    with open(ECG_MM, newline='') as f:
        for row in csv.DictReader(f):
            s = row['subject_id'].strip()
            if s not in echo_subj: continue
            t = _parse_dt(row.get('ecg_time'))
            if t is None: continue
            reps = " | ".join((row.get(f'report_{i}') or '').strip() for i in range(18))
            ecg[s].append((t, bool(_AF_RE.search(reps)), (row.get('report_0') or '').strip()))
    out = {}
    for mid, (subj, sid, sdt) in study_map.items():
        t = _parse_dt(sdt)
        if t is None or subj not in ecg: continue
        best = min(ecg[subj], key=lambda x: abs((x[0] - t).total_seconds()))
        gap = abs((best[0] - t).total_seconds()) / 86400.0
        if gap <= window_days:
            out[mid] = ('AF' if best[1] else 'sinus', round(gap, 2), best[2])
    return out

def age_at_echo(pat, study_dt):
    """anchor_age + (study_year - anchor_year). Döner int veya None."""
    if pat is None: return None
    gender, aage, ayear = pat
    m = re.match(r'(\d{4})', study_dt or '')
    if not m: return None
    return aage + (int(m.group(1)) - ayear)

def build_params(study_map, tte_only=True):
    """structured-measurement -> {measurement_id: {'params':{...}, 'cat':{...}}} (sadece DICOM & TTE)."""
    wanted_mids = set(study_map)
    data = defaultdict(lambda: {'params': {}, 'cat': {}, 'test_types': set()})
    with gzip.open(STRUCTURED, 'rt', newline='') as f:
        for row in csv.DictReader(f):
            mid = (row.get('measurement_id') or '').strip()
            if mid not in wanted_mids: continue
            tt = (row.get('test_type') or '').strip()
            data[mid]['test_types'].add(tt)
            m = (row.get('measurement') or '').strip()
            res = row.get('result')
            if m in PARAM_MAP:
                if tte_only and tt != 'tte': continue
                key, norm = PARAM_MAP[m]
                v = norm(parse_num(res))
                if v is not None:
                    data[mid]['params'][key] = v
            elif m in CAT_MAP:
                if res and res.strip():
                    data[mid]['cat'][CAT_MAP[m]] = res.strip()
    return data

def build_dataset(out_csv=None, tte_only=True, ecg_window=ECG_WINDOW_DAYS):
    study_map = load_study_list()
    patients  = load_patients()
    af_subj   = load_af_subjects()
    ecg_rhythm = load_ecg_rhythm(study_map, window_days=ecg_window)
    data      = build_params(study_map, tte_only=tte_only)

    rows = []
    for mid, d in data.items():
        subj, study_id, sdt = study_map[mid]
        pat = patients.get(subj)
        age = age_at_echo(pat, sdt)
        gender = pat[0] if pat else None
        p = derive(dict(d['params']))
        # --- Ritim: TEMPORAL (eko-anı ECG) öncelik; yoksa ICD ever-AF fallback ---
        icd_af = subj in af_subj
        rh = ecg_rhythm.get(mid)          # (rhythm, gap, report0) | None
        if rh is not None:
            rhythm, ecg_gap, ecg_rep = rh
            rhythm_src = 'ecg'
            af = (rhythm == 'AF')          # eko-anı ritmi belirleyici; sinüs -> ICD'ye rağmen DAHİL
        else:
            rhythm = ('possible_AF_icd' if icd_af else 'assumed_sinus')
            ecg_gap, ecg_rep, rhythm_src = None, None, 'icd_fallback'
            af = icd_af                    # yakın ECG yok -> konservatif ICD fallback
        excluded, ex_reasons = exclusion_flags(d['cat'], af=af)

        g25 = grade_ase2025(p, age=age)
        g16 = grade_ase2016(p)
        ready = harmonize_ready(d['cat'].get('ready_grade'))

        row = {
            'subject_id': subj, 'study_id': study_id, 'measurement_id': mid,
            'test_types': '|'.join(sorted(t for t in d['test_types'] if t)),
            'age': age, 'gender': gender,
            'E_cms': p.get('E'), 'A_cms': p.get('A'), 'EA': p.get('EA'),
            'e_sep': p.get('e_sep'), 'e_lat': p.get('e_lat'), 'e_avg': p.get('e_avg'),
            'Ee_mean': ee_value(p, 'mean'), 'Ee_sep': ee_value(p, 'sep'),
            'Ee_lat': ee_value(p, 'lat'),
            'LAVi': p.get('LAVi'), 'TRvel': p.get('TRvel'), 'EF': p.get('EF'),
            'LVMi': p.get('LVMi'), 'DT': p.get('DT'),
            'grade_2025': g25['grade'], 'lap_2025': g25.get('lap'),
            'n_abn_2025': g25.get('n_abnormal'), 'reason_2025': g25.get('reason'),
            'grade_2016': g16['grade'], 'path_2016': g16.get('path'),
            'ready_grade_raw': d['cat'].get('ready_grade'), 'ready_grade': ready,
            'ready_fcn': d['cat'].get('ready_fcn'),
            'rhythm': rhythm, 'rhythm_src': rhythm_src, 'ecg_gap_days': ecg_gap,
            'ecg_report0': ecg_rep, 'icd_af': int(icd_af),
            'af': int(af), 'excluded': int(excluded), 'excl_reason': ';'.join(ex_reasons),
        }
        rows.append({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
    if out_csv:
        cols = list(rows[0].keys()) if rows else []
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
    return rows

# --------------------------------------------------------------------------- #
# 8. Özet + çapraz-doğrulama raporu                                            #
# --------------------------------------------------------------------------- #
def summarize(rows):
    def dist(key, pred=lambda r: True):
        c = defaultdict(int)
        for r in rows:
            if pred(r): c[r[key]] += 1
        return dict(sorted(c.items(), key=lambda x: -x[1]))
    incl = [r for r in rows if not r['excluded']]
    print(f"\n=== TOPLAM {len(rows)} study (dışlanmayan: {len(incl)}) ===")
    # Ritim kaynağı + temporal kurtarma
    print("Ritim kaynağı:", dist('rhythm_src'))
    print("Ritim etiketi:", dist('rhythm'))
    recovered = sum(1 for r in rows if r['icd_af']==1 and r['rhythm']=='sinus' and not r['excluded'])
    print(f"KURTARILAN (ICD ever-AF AMA eko-anı ECG sinüs, dışlanmadı): {recovered}")
    print("\n2025 grade dağılımı (dışlanmayan):", dist('grade_2025', lambda r: not r['excluded']))
    print("2016 grade dağılımı (dışlanmayan):", dist('grade_2016', lambda r: not r['excluded']))
    print("Dışlama sebepleri:", dist('excl_reason', lambda r: r['excluded']))
    print("Hazır grade (harmonize):", dist('ready_grade'))

    # 2025 vs 2016 çapraz (hesaplananlar)
    both = [r for r in incl if r['grade_2025'] not in ('Incomputable',)
            and r['grade_2016'] not in ('Incomputable', 'Indeterminate')]
    agree = sum(1 for r in both if r['grade_2025'] == r['grade_2016'])
    print(f"\n2025↔2016 (ikisi de hesaplanan {len(both)}): tam uyum {agree} "
          f"({100*agree/max(len(both),1):.1f}%)")

    # Türetme vs hazır kardiyolog grade (569-benzeri validasyon)
    val = [r for r in incl if r['ready_grade'] in ('Normal','Grade1','Grade2','Grade3')
           and r['grade_2025'] not in ('Incomputable', 'Grade2_or_3')]
    agree_r = sum(1 for r in val if r['grade_2025'] == r['ready_grade'])
    print(f"2025↔hazır kardiyolog grade (çakışan {len(val)}): tam uyum {agree_r} "
          f"({100*agree_r/max(len(val),1):.1f}%)")
    # ±1 derece tolerans
    ord_ = {'Normal':0,'Grade1':1,'Grade2':2,'Grade3':3}
    within1 = sum(1 for r in val if abs(ord_[r['grade_2025']]-ord_[r['ready_grade']])<=1)
    print(f"   ±1 derece içinde: {within1} ({100*within1/max(len(val),1):.1f}%)")

# --------------------------------------------------------------------------- #
# 9. Self-test (kılavuz ders-kitabı vakaları)                                  #
# --------------------------------------------------------------------------- #
def run_tests():
    T = 0; F = 0
    def chk(name, got, exp):
        nonlocal T, F
        if isinstance(got, float) and isinstance(exp, float):
            ok = abs(got - exp) < 1e-6
        else:
            ok = got == exp
        T += ok; F += (not ok)
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}: got={got} exp={exp}")

    # --- parse/normalize ---
    chk("parse '>2.8'", parse_num('>2.8'), 2.8)
    chk("parse '1.5-2.0'", parse_num('1.5-2.0'), 1.75)
    chk("norm e' 0.07 m/s -> 7 cm/s", norm_eprime(0.07), 7.0)
    chk("norm e' outlier 7.0 m/s -> None", norm_eprime(7.0), None)
    chk("norm TR 55 -> None", norm_tr(55), None)

    # --- 2025 Figure 3 vakaları (yaş-bağımsız) ---
    # 1) Hepsi normal: e' iyi, E/e' düşük, TR düşük -> Normal DF
    p = derive({'E':70,'e_sep':10,'e_lat':12,'TRvel':2.2,'EA':1.2,'LAvol':40,'BSA':2.0})
    chk("2025 all-normal", grade_ase2025(p)['grade'], 'Normal')
    # 2) Sadece e' azalmış + E/A<=0.8 -> Grade1
    p = derive({'E':40,'e_sep':5,'e_lat':6,'TRvel':2.2,'EA':0.7,'LAvol':40,'BSA':2.0})
    chk("2025 reduced-e' + E/A<=0.8 -> Grade1", grade_ase2025(p)['grade'], 'Grade1')
    # 3) 3/3 anormal, E/A<2 -> Grade2 (elevated LAP)
    p = derive({'E':100,'e_sep':4,'e_lat':5,'TRvel':3.0,'EA':1.5,'LAvol':80,'BSA':2.0})
    chk("2025 3-abnormal E/A<2 -> Grade2", grade_ase2025(p)['grade'], 'Grade2')
    # 4) 3/3 anormal, E/A>=2 -> Grade3
    p = derive({'E':120,'e_sep':4,'e_lat':5,'TRvel':3.2,'EA':2.4,'LAvol':90,'BSA':2.0})
    chk("2025 3-abnormal E/A>=2 -> Grade3", grade_ase2025(p)['grade'], 'Grade3')
    # 5) Sadece ↑E/e' + ikincil kutu(LAVi>34) pozitif -> elevated -> Grade2
    p = derive({'E':110,'e_sep':8,'e_lat':9,'TRvel':2.2,'EA':1.5,'LAvol':80,'BSA':2.0})  # E/e'sep=13.75<15,lat=12.2<13,mean~12.9<14 -> ee not high; adjust
    # e' iyi (>threshold), sadece E/e' yüksek istiyoruz: E büyük, e' orta
    p = derive({'E':120,'e_sep':7.5,'e_lat':8,'TRvel':2.2,'EA':1.5,'LAvol':80,'BSA':2.0})
    # Ee_sep=16>=15 -> ee_hi True; e' 7.5/8 >6/7 -> not reduced; TR 2.2 -> not high => sadece ↑E/e'
    chk("2025 ↑E/e'-only + LAVi>34 -> Grade2", grade_ase2025(p)['grade'], 'Grade2')
    # 6) Sadece ↑E/e' + LAVi<=34 -> normal LAP, e' normal -> Normal DF
    p = derive({'E':120,'e_sep':7.5,'e_lat':8,'TRvel':2.2,'EA':1.5,'LAvol':60,'BSA':2.0})  # LAVi=30
    chk("2025 ↑E/e'-only + LAVi<=34 -> Normal", grade_ase2025(p)['grade'], 'Normal')
    # 7) İkincil kutu değişkeni yok -> Incomputable
    p = derive({'E':120,'e_sep':7.5,'e_lat':8,'TRvel':2.2,'EA':1.5})  # LAVi yok
    chk("2025 ↑E/e'-only + LAVi yok -> Incomputable", grade_ase2025(p)['grade'], 'Incomputable')
    # 8) Yetersiz primer -> Incomputable
    chk("2025 bos -> Incomputable", grade_ase2025(derive({'E':80}))['grade'], 'Incomputable')

    # --- 2025 yaş-özgü e' (Table 6) ---
    # 70y, septal 6.2 cm/s: >65 eşiği <6 -> azalmış DEĞİL
    p = derive({'E':60,'e_sep':6.2,'e_lat':7.5,'TRvel':2.2,'EA':1.0,'LAvol':40,'BSA':2.0})
    chk("2025 age70 e_sep6.2 not reduced -> Normal", grade_ase2025(p, age=70)['grade'], 'Normal')
    # 30y, septal 6.5: <40 eşiği <7 -> azalmış (+E/A<=0.8) -> Grade1
    p = derive({'E':40,'e_sep':6.5,'e_lat':9,'TRvel':2.2,'EA':0.7,'LAvol':40,'BSA':2.0})
    chk("2025 age30 e_sep6.5 reduced -> Grade1", grade_ase2025(p, age=30)['grade'], 'Grade1')

    # --- 2016 ---
    # düşük EF -> DD var, E/A>=2 -> Grade3
    p = derive({'EF':30,'EA':2.2,'E':100,'Ee_mean':16,'TRvel':3.0,'LAvol':80,'BSA':2.0})
    chk("2016 lowEF E/A>=2 -> Grade3", grade_ase2016(p)['grade'], 'Grade3')
    # normal EF, 4 kriter hepsi negatif -> Normal
    p = derive({'EF':60,'E':70,'e_sep':10,'e_lat':12,'TRvel':2.0,'LAvol':40,'BSA':2.0,'EA':1.2})
    chk("2016 normalEF all-neg -> Normal", grade_ase2016(p)['grade'], 'Normal')
    # normal EF, 4 kriter hepsi pozitif -> DD -> grade
    p = derive({'EF':60,'E':100,'e_sep':5,'e_lat':6,'TRvel':3.0,'LAvol':80,'BSA':2.0,'EA':1.5})
    chk("2016 normalEF all-pos -> Grade2", grade_ase2016(p)['grade'], 'Grade2')

    # --- harmonize ---
    chk("harmonize 'I'", harmonize_ready('I'), 'Grade1')
    chk("harmonize '0 (nl)'", harmonize_ready('0 (nl)'), 'Normal')
    chk("harmonize 'Unable to assess'", harmonize_ready('Unable to assess'), None)

    # --- exclusion ---
    ex, rs = exclusion_flags({'MS':'Severe (<1.0 cm2) Valvular'})
    chk("exclude severe MS", ex, True)
    ex, rs = exclusion_flags({'MR':'Mild (1+)'})
    chk("keep mild MR", ex, False)
    ex, rs = exclusion_flags({'MAC':'Mod MAC'})
    chk("exclude mod MAC", ex, True)
    ex, rs = exclusion_flags({}, af=True)
    chk("exclude AF", ex, True)

    print(f"\n{'='*40}\nTESTLER: {T} geçti, {F} başarısız")
    return F == 0

# --------------------------------------------------------------------------- #
# 10. Manifest — seçili study'lerin DICOM yolları + etiket + güven katmanı      #
# --------------------------------------------------------------------------- #
GRADABLE = {'Normal', 'Grade1', 'Grade2', 'Grade3'}

def _confidence_tier(r):
    """ECG-teyitli sinüs = birincil; ICD-fallback assumed_sinus = genişletilmiş."""
    if r.get('rhythm') == 'sinus':      return 'ecg_sinus'      # yüksek güven
    if r.get('rhythm') == 'assumed_sinus': return 'assumed_sinus'  # ICD-fallback
    return 'other'

def build_manifest(labels_csv, out_dicom, out_study, include_grade23=False):
    """
    labels_csv (--build çıktısı) + echo-record-list.csv -> manifest.
    Seçim: dışlanmayan VE 2025-gradable study'ler. Her seçili study'nin TÜM DICOM'ları.
    (View bilgisi record-list'te YOK -> view filtreleme sonra VM'de.)
    """
    # 1) Seçili study'ler + etiketleri
    sel = {}   # study_id -> etiket dict
    with open(labels_csv, newline='') as f:
        for r in csv.DictReader(f):
            if r['excluded'] == '1': continue
            g = r['grade_2025']
            ok = g in GRADABLE or (include_grade23 and g == 'Grade2_or_3')
            if not ok: continue
            sel[r['study_id']] = {
                'subject_id': r['subject_id'], 'grade_2025': g,
                'grade_2016': r['grade_2016'], 'ready_grade': r['ready_grade'],
                'confidence': _confidence_tier(r), 'rhythm': r['rhythm'],
                'age': r['age'], 'gender': r['gender'], 'EF': r['EF'],
                'LAVi': r['LAVi'], 'Ee_mean': r['Ee_mean'], 'TRvel': r['TRvel'],
            }
    print(f"Seçili gradable study: {len(sel)}")

    # 2) record-list'i akıt, seçili study'lerin DICOM'larını yaz + study başı say
    ndicom = defaultdict(int)
    cols = ['subject_id', 'study_id', 'confidence', 'grade_2025', 'grade_2016',
            'ready_grade', 'age', 'gender', 'EF', 'acquisition_datetime',
            'dicom_filepath', 'gcs_uri']
    written = 0
    with open(RECORD_LIST, newline='') as f, open(out_dicom, 'w', newline='') as g:
        w = csv.DictWriter(g, fieldnames=cols); w.writeheader()
        for r in csv.DictReader(f):
            sid = r['study_id'].strip()
            m = sel.get(sid)
            if not m: continue
            ndicom[sid] += 1
            fp = r['dicom_filepath'].strip()
            w.writerow({'subject_id': r['subject_id'].strip(), 'study_id': sid,
                        'confidence': m['confidence'], 'grade_2025': m['grade_2025'],
                        'grade_2016': m['grade_2016'], 'ready_grade': m['ready_grade'],
                        'age': m['age'], 'gender': m['gender'], 'EF': m['EF'],
                        'acquisition_datetime': r['acquisition_datetime'].strip(),
                        'dicom_filepath': fp, 'gcs_uri': GCS_PREFIX + fp})
            written += 1

    # 3) study-düzeyi özet
    scols = ['study_id', 'subject_id', 'confidence', 'grade_2025', 'grade_2016',
             'ready_grade', 'age', 'gender', 'EF', 'LAVi', 'Ee_mean', 'TRvel', 'n_dicom']
    with open(out_study, 'w', newline='') as g:
        w = csv.DictWriter(g, fieldnames=scols); w.writeheader()
        for sid, m in sel.items():
            w.writerow({'study_id': sid, 'subject_id': m['subject_id'],
                        'confidence': m['confidence'], 'grade_2025': m['grade_2025'],
                        'grade_2016': m['grade_2016'], 'ready_grade': m['ready_grade'],
                        'age': m['age'], 'gender': m['gender'], 'EF': m['EF'],
                        'LAVi': m['LAVi'], 'Ee_mean': m['Ee_mean'], 'TRvel': m['TRvel'],
                        'n_dicom': ndicom.get(sid, 0)})

    # 4) özet + boyut tahmini
    with_dcm = sum(1 for s in sel if ndicom.get(s, 0) > 0)
    per_file_mb = BUCKET_TB * 1e6 / BUCKET_FILES     # ~MB/dosya
    est_gb = written * per_file_mb / 1000.0
    def dist(key):
        c = defaultdict(int)
        for m in sel.values(): c[m[key]] += 1
        return dict(sorted(c.items(), key=lambda x: -x[1]))
    print(f"DICOM'u olan study: {with_dcm}/{len(sel)} | toplam DICOM: {written}")
    print(f"Güven katmanı: {dist('confidence')}")
    print(f"2025 grade: {dist('grade_2025')}")
    print(f"Kaba indirme boyutu tahmini (tüm DICOM'lar, ham): ~{est_gb:.0f} GB "
          f"(view filtreleme öncesi; VM'de küçülecek)")
    print(f"\nYazıldı:\n  {out_dicom} ({written} DICOM satırı)\n  {out_study} ({len(sel)} study)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true', help='ders-kitabı vakalarını çalıştır')
    ap.add_argument('--build', action='store_true', help='etiket tablosu üret')
    ap.add_argument('--manifest', action='store_true', help='DICOM indirme manifesti üret')
    ap.add_argument('--out', default=os.path.join(ECHO_DIR, 'diastolic_labels.csv'))
    ap.add_argument('--include-nontte', action='store_true', help='TTE-dışını da dahil et')
    args = ap.parse_args()

    if args.selftest:
        ok = run_tests()
        if not args.build:
            sys.exit(0 if ok else 1)
    if args.build:
        rows = build_dataset(out_csv=args.out, tte_only=not args.include_nontte)
        summarize(rows)
        print(f"\nYazıldı: {args.out}  ({len(rows)} satır)")
    if args.manifest:
        build_manifest(args.out,
                       os.path.join(ECHO_DIR, 'manifest_dicom.csv'),
                       os.path.join(ECHO_DIR, 'manifest_study.csv'))
    if not (args.selftest or args.build or args.manifest):
        ap.print_help()

if __name__ == '__main__':
    main()
