#!/usr/bin/env python3
"""EchoXFlow: analiz edilen ilk dışa aktarım (şema EchoXFlow-1) ile yeniden yayımlanan sürümün (1.0.0, şema EchoXFlow-2)
spektral ölçüm değerlerinin karşılaştırması (analysis_log Ek 25 ve Ek 28; post hoc).

  python3 external/posthoc/release_compare.py fetch <exam_id> [...]   yeni sürümün küçük üyelerini indirir (HTTP aralık istekleri)
  python3 external/posthoc/release_compare.py compare                 -> external/posthoc/release_compare.json

Yeni sürümden yalnız üst veri ve küçük ölçüm/kalibrasyon dizileri okunur: ~/echoxflow-render/v2_meta/. Proje kökünden çalıştırılır."""
from __future__ import annotations
import collections as C, glob, io, json, os, sys, tarfile, time, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
V2 = os.path.expanduser('~/echoxflow-render/v2_meta')
REPO = 'https://huggingface.co/datasets/Ahus-AIM/EchoXFlow/resolve/{rev}/exams/{ex}.tar'
REV_OLD = 'd907c536cae16750b9e46f269a27befd4979ebdd'          # 2026-05-04; yerel kopya (indirme 2026-08-31)
REV_NEW = '696248bcb5cb0b3cd40e84af1f8343975e3632c3'          # 2026-10-18; EchoXFlow 1.0.0 (EchoXFlow-2026-10-15), son veri commit'i
TOL = 2e-3


class HTTPFile(io.RawIOBase):
    def __init__(s, url, block=1 << 17):
        s.pos = 0; s.block = block; s.cache = {}
        r = urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60)
        s.size = int(r.headers['Content-Length']); s.final = r.url
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, off, wh=0):
        s.pos = off if wh == 0 else (s.pos + off if wh == 1 else s.size + off); return s.pos
    def _blk(s, i):
        if i not in s.cache:
            a = i * s.block; b = min(s.size, a + s.block) - 1
            for k in range(5):
                try:
                    s.cache[i] = urllib.request.urlopen(urllib.request.Request(s.final, headers={'Range': f'bytes={a}-{b}'}), timeout=60).read(); break
                except Exception:
                    time.sleep(2 + 3 * k)
            if len(s.cache) > 48:
                s.cache.pop(next(iter(s.cache)))
        return s.cache[i]
    def read(s, n=-1):
        if n < 0:
            n = s.size - s.pos
        out = b''
        while n > 0 and s.pos < s.size:
            d = s._blk(s.pos // s.block)[s.pos % s.block: s.pos % s.block + n]
            if not d:
                break
            out += d; s.pos += len(d); n -= len(d)
        return out
    def readinto(s, b):
        d = s.read(len(b)); b[:len(d)] = d; return len(d)


def fetch(exams):
    for ex in exams:
        f = HTTPFile(REPO.format(rev=REV_NEW, ex=ex)); k = 0
        with tarfile.open(fileobj=f, mode='r:') as tf:
            for m in tf:
                if m.isfile() and m.size <= 200000 and ('/annotations/' in m.name or '/coordinates/' in m.name
                                                         or m.name.endswith(('.zattrs', '.zarray', '.zgroup', '.zmetadata'))):
                    dst = os.path.join(V2, m.name); os.makedirs(os.path.dirname(dst), exist_ok=True)
                    open(dst, 'wb').write(tf.extractfile(m).read()); k += 1
        print(ex, 'yazılan', k, flush=True)


def compare():
    sys.path.insert(0, os.path.join(ROOT, 'echoxflow-code', 'src'))
    import zarr
    cnt = C.Counter(); flags = C.Counter(); schema = C.Counter(); nex = 0
    for root in sorted(glob.glob(os.path.join(V2, 'exams', '*'))):
        ex = os.path.basename(root); nex += 1; loc = {}
        with tarfile.open(os.path.join(ROOT, 'echoxflow', 'exams', ex + '.tar')) as tf:
            for m in tf:
                if m.name.endswith('.zarr/.zattrs'):
                    loc[m.name.split('/')[-2]] = json.load(tf.extractfile(m))
        for zp in sorted(glob.glob(root + '/*.zarr')):
            rid = os.path.basename(zp)
            try:
                att = json.load(open(zp + '/.zattrs'))
            except Exception:
                continue
            schema[str(att.get('schema_version'))] += 1
            man = att['recording_manifest']
            for t in man.get('tracks') or []:
                if t.get('kind') == 'matrix' and 'nyquist_mps' in t:
                    flags[('velocity_reference', str(t.get('velocity_reference')))] += 1
                    flags[('angle_correction_applied', str(t.get('angle_correction_applied')))] += 1
            anns = [a for a in man.get('annotations') or [] if (a.get('links') or {}).get('coordinate_type') == 'spectral' and a.get('kind') == 'measurement']
            if not anns or rid not in loc:
                continue
            z = zarr.open(zp, mode='r'); old = C.defaultdict(list); otr = {}
            for t in loc[rid]['recording_manifest'].get('tracks') or []:
                for a in t.get('spectral_annotations') or []:
                    old[a['label']].append([p['velocity_mps'] for p in a['points']]); otr[a['label']] = t
            seen = C.Counter()
            for a in anns:
                lab = a['links'].get('measure_id'); i = seen[lab]; seen[lab] += 1
                try:
                    new = np.asarray(z[a['value']['zarr_path']]).reshape(-1, 2)[:, 1]
                except Exception:
                    cnt['okunamayan_olcum'] += 1; continue
                if i >= len(old.get(lab, [])) or len(old[lab][i]) != len(new):
                    cnt['eslesmeyen_olcum'] += 1; continue
                t = otr[lab]; cw = t.get('semantic_id') == 'continuous_wave'
                off = (t['spectral_row_baseline_frac'] - t['baseline_frac']) * 2.0 * t['nyquist_mps']
                grp = 'CW' if cw else ('PW_bf_farkli_rbf' if abs(off) > 1e-9 else 'PW_bf_esit_rbf')
                tr = 'TR_Vmax' if lab.endswith('TR Vmax') else None
                for vo, vn in zip(old[lab][i], new):
                    same, shifted = abs(vn - vo) < TOL, abs(vn - vo - off) < TOL
                    res = ('degismemis' if same and abs(off) > 1e-9 else 'eski_arti_ofset' if shifted and abs(off) > 1e-9
                           else 'degismemis' if same else 'sifirlanmis' if abs(vn) < 1e-9 else 'diger')
                    cnt[(grp, res)] += 1
                    if tr and cw:
                        cnt[('CW_TR_Vmax', res)] += 1
    out = {'uretim': 'external/posthoc/release_compare.py', 'not': 'POST HOC (analysis_log Ek 25, Ek 28). Rahatlık örneklemi: incelenen 30 muayeneden dokuzu.',
           'revizyon_analiz_edilen': REV_OLD, 'revizyon_yeni': REV_NEW, 'muayene': nex, 'sema_yeni': dict(schema),
           'okunamayan_olcum': cnt['okunamayan_olcum'], 'eslesmeyen_olcum': cnt['eslesmeyen_olcum'], 'gruplar': {},
           'yeni_ust_veri_isaretleri': {f'{k[0]}={k[1]}': v for k, v in sorted(flags.items())}}
    for g in ('CW', 'CW_TR_Vmax', 'PW_bf_esit_rbf', 'PW_bf_farkli_rbf'):
        d = {r: cnt[(g, r)] for r in ('eski_arti_ofset', 'degismemis', 'sifirlanmis', 'diger')}
        d['nokta'] = sum(d.values()); out['gruplar'][g] = d
    json.dump(out, open(os.path.join(HERE, 'release_compare.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    fetch(sys.argv[2:]) if sys.argv[1] == 'fetch' else compare()
