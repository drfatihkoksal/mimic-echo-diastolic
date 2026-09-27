#!/usr/bin/env python3
"""QC 2. tur: kör, karıştırılmış sıra, ayrı depolama anahtarı, CSV indirme. build_qc_page.py'den türetildi (2026-09-27)."""
import json, sys, html

CARDS = json.load(open(sys.argv[1]))
import random
random.Random(20260927).shuffle(CARDS)   # 1. turdan farklı, sabit tohumlu sıra
OUT = sys.argv[2]

HINT = {
 'MV E/A': 'Gate mitral kapak uçlarında mı? “E” erken diyastolik dalganın, “A” QRS’ten hemen önceki geç dalganın tepesinde mi?',
 'Eprime Septal': 'Gate <strong>septal</strong> mitral anulusta mı? İşaretler erken diyastolik anulus hareketinin tepesinde mi?',
 'Eprime Lateral': 'Gate <strong>lateral</strong> mitral anulusta mı? İşaretler erken diyastolik anulus hareketinin tepesinde mi?',
 'TR Vmax': 'Gate triküspit kapakta mı? İşaretler sistolik jetin dış kenarında mı?',
}
ORDER = ['MV E/A', 'Eprime Septal', 'Eprime Lateral', 'TR Vmax']


def num(v, unit='', d=2):
    return f'{v:.{d}f}{unit}' if isinstance(v, (int, float)) else '—'


def cal_line(c, kind):
    cm = 'Eprime' in kind
    v = c['v'] * (100 if cm else 1)
    unit = 'cm/s' if cm else 'm/s'
    tag = '<span class="cal-fix">düzeltilmiş</span>' if c.get('corrected') else ''
    return (f"<span class=\"cal-name\">{html.escape(c['name'])}</span>"
            f"<span class=\"cal-val\">{v:.2f} {unit}</span>"
            f"<span class=\"cal-t\">{c['time_s']:.2f} s</span>{tag}")


cards_html = []
for i, c in enumerate(CARDS, 1):
    panels = sorted(c['panels'], key=lambda p: ORDER.index(p['kind']) if p['kind'] in ORDER else 9)
    ph = []
    for p in panels:
        cals = ''.join(f'<li>{cal_line(x, p["kind"])}</li>' for x in p['calipers'])
        m = p['meta']
        ph.append(f"""
        <figure class="panel">
          <figcaption class="panel-head">
            <span class="panel-title">{html.escape(p['title'])}</span>
            <span class="panel-meta">gate {m['gate_depth_cm']} cm · {m['sweep_s']} s süpürme · {m['beats']} atım{'' if m['anchor_cv'] is None else f" · çapa sapması %{m['anchor_cv']}"}</span>
          </figcaption>
          <div class="panel-img"><img src="data:image/jpeg;base64,{p['img']}" alt="{html.escape(p['title'])} spektral izi" loading="lazy"></div>
          <ul class="cals">{cals}</ul>
          <p class="hint">{HINT.get(p['kind'], '')}</p>
          <div class="verdict panel-verdict" role="group" aria-label="Bu panel için karar" data-key="{c['exam_id']}::{p['kind']}">
            <button type="button" data-v="ok">Doğru</button>
            <button type="button" data-v="maybe">Şüpheli</button>
            <button type="button" data-v="bad">Yanlış</button>
          </div>
        </figure>""")

    ee = c['Ee_mean'] if c['Ee_mean'] is not None else (c['Ee_sep'] if c['Ee_sep'] is not None else c['Ee_lat'])
    cards_html.append(f"""
  <article class="card" id="{c['exam_id']}" data-exam="{c['exam_id']}">
    <header class="card-head">
      <div class="idwrap">
        <span class="seq">{i:02d}</span>
        <h2>{html.escape(c['exam_id'].replace('exam_', ''))}</h2>
      </div>
      <span class="rollup" data-exam="{c['exam_id']}" data-total="{len(panels)}">0/{len(panels)}</span>
    </header>
    <div class="card-body">
      <div class="measures">
        <table>
          <caption>Bizim çıkardığımız değerler</caption>
          <tbody>
            <tr><th>E</th><td>{num(c['E'],' cm/s',1)}</td></tr>
            <tr><th>A</th><td>{num(c['A'],' cm/s',1)}</td></tr>
            <tr><th>E/A</th><td>{num(c['EA'])}</td></tr>
            <tr class="sep"><th>e′ septal</th><td>{num(c['e_sep'],' cm/s',1)}</td></tr>
            <tr><th>e′ lateral</th><td>{num(c['e_lat'],' cm/s',1)}</td></tr>
            <tr><th>E/e′</th><td>{num(ee)}</td></tr>
            <tr class="sep"><th>TR Vmax</th><td>{num(c['TRvel'],' m/s')}</td></tr>
          </tbody>
        </table>
        <p class="ctx">{c['n_bmode']} B-mode kaydı · {num(c['fps'],' fps',0)}</p>
        <textarea class="note" rows="3" data-exam="{c['exam_id']}" placeholder="Not (isteğe bağlı) — hangi panel olduğunu yazabilirsin" aria-label="Not"></textarea>
      </div>
      <div class="panels">{''.join(ph)}</div>
    </div>
  </article>""")

CSS = """
:root{
  --ground:#f3f6f7; --surface:#ffffff; --surface-2:#eaeff1;
  --ink:#0f171b; --ink-2:#4a5c65; --ink-3:#7c8f98;
  --line:#d9e2e6; --line-strong:#bcc9cf;
  --accent:#0e7c86; --accent-soft:#d6ecee;
  --ok:#2c7a58; --warn:#9a6410; --bad:#a93a33;
  --shadow:0 1px 2px rgba(15,23,27,.06),0 8px 24px rgba(15,23,27,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0c1215; --surface:#121a1e; --surface-2:#182227;
    --ink:#e7eff2; --ink-2:#a2b3bb; --ink-3:#74878f;
    --line:#223037; --line-strong:#31434b;
    --accent:#4fb3bd; --accent-soft:#153338;
    --ok:#5aa886; --warn:#c69445; --bad:#d97a72;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --ground:#0c1215; --surface:#121a1e; --surface-2:#182227;
  --ink:#e7eff2; --ink-2:#a2b3bb; --ink-3:#74878f;
  --line:#223037; --line-strong:#31434b;
  --accent:#4fb3bd; --accent-soft:#153338;
  --ok:#5aa886; --warn:#c69445; --bad:#d97a72;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink); margin:0;
  font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; font-size:15px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
}
h1,h2,h3,.seq,.panel-title,th,caption,.verdict button,.topbar-title{
  font-family:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
}
.mono,td,.cal-val,.cal-raw,.panel-meta,.ctx,.count{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}

/* ---- üst şerit ---- */
.topbar{
  position:sticky; top:0; z-index:20; background:var(--surface); border-bottom:1px solid var(--line);
  display:flex; align-items:center; gap:20px; flex-wrap:wrap; padding:12px 28px;
}
.topbar-title{font-size:15px; font-weight:600; letter-spacing:.02em; margin:0}
.topbar-title span{color:var(--ink-3); font-weight:400}
.progress{display:flex; align-items:center; gap:10px; margin-left:auto; font-size:13px; color:var(--ink-2)}
.bar{width:150px; height:5px; background:var(--surface-2); border-radius:3px; overflow:hidden}
.bar>i{display:block; height:100%; width:0; background:var(--accent); transition:width .25s ease}
.tools{display:flex; gap:8px; align-items:center}
button.tool,label.tool{
  font:inherit; font-size:13px; padding:5px 11px; border:1px solid var(--line-strong);
  background:var(--surface); color:var(--ink-2); border-radius:5px; cursor:pointer;
}
button.tool:hover,label.tool:hover{border-color:var(--accent); color:var(--accent)}
label.tool{display:flex; gap:6px; align-items:center; user-select:none}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px}

/* ---- brief ---- */
.brief{max-width:74ch; margin:0 auto; padding:34px 28px 8px}
.brief h1{font-size:27px; line-height:1.2; margin:0 0 6px; text-wrap:balance; letter-spacing:-.01em}
.brief .sub{color:var(--ink-2); margin:0 0 20px; font-size:15px}
.brief p{margin:0 0 12px; color:var(--ink-2)}
.brief strong{color:var(--ink); font-weight:600}
.callout{
  border-left:3px solid var(--accent); background:var(--accent-soft);
  padding:12px 16px; border-radius:0 6px 6px 0; margin:18px 0 4px;
}
.callout p{color:var(--ink); margin:0}
.callout p+p{margin-top:8px}

/* ---- kart ---- */
main{max-width:1180px; margin:0 auto; padding:22px 28px 90px; display:flex; flex-direction:column; gap:20px}
.card{background:var(--surface); border:1px solid var(--line); border-radius:9px; box-shadow:var(--shadow); overflow:hidden}
.card.done{border-color:var(--accent)}
.card-head{
  display:flex; align-items:center; gap:14px; flex-wrap:wrap;
  padding:12px 18px; border-bottom:1px solid var(--line); background:var(--surface-2);
}
.idwrap{display:flex; align-items:baseline; gap:10px}
.seq{font-size:12px; color:var(--ink-3); letter-spacing:.08em}
.card-head h2{font-size:16px; margin:0; letter-spacing:.02em; font-weight:600}
.rollup{
  margin-left:auto; font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:12px;
  color:var(--ink-3); padding:3px 9px; border:1px solid var(--line); border-radius:20px;
}
.rollup.full{color:var(--accent); border-color:var(--accent)}
.lap{ font-size:12px; padding:3px 10px; border-radius:20px;
  border:1px solid var(--line-strong); color:var(--ink-3); background:var(--surface);
  filter:blur(5px); user-select:none; transition:filter .15s ease;
}
body.reveal .lap{filter:none; user-select:auto}
body.reveal .lap[data-lap="elevated"]{color:var(--warn); border-color:var(--warn)}
body.reveal .lap[data-lap="normal"]{color:var(--ok); border-color:var(--ok)}
.card-body{display:grid; grid-template-columns:216px minmax(0,1fr); gap:0}
.measures{padding:16px 18px; border-right:1px solid var(--line); display:flex; flex-direction:column; gap:12px}
.measures table{border-collapse:collapse; width:100%; font-size:13px}
.measures caption{
  text-align:left; font-size:11px; letter-spacing:.09em; text-transform:uppercase;
  color:var(--ink-3); padding-bottom:7px; font-weight:600;
}
.measures th{text-align:left; font-weight:400; color:var(--ink-2); padding:2px 0}
.measures td{text-align:right; padding:2px 0; color:var(--ink)}
.measures tr.sep th,.measures tr.sep td{padding-top:8px; border-top:1px solid var(--line)}
.ctx{font-size:11px; color:var(--ink-3); margin:0}
.verdict{display:flex; gap:5px}
.panel-verdict{margin-top:9px; max-width:330px}
.panel.rated .panel-img{outline:2px solid var(--accent); outline-offset:1px}
.panel.rated-bad .panel-img{outline-color:var(--bad)}
.panel.rated-maybe .panel-img{outline-color:var(--warn)}
.verdict button{
  flex:1; font-size:12.5px; padding:6px 4px; border-radius:5px; cursor:pointer;
  border:1px solid var(--line-strong); background:var(--surface); color:var(--ink-2);
}
.verdict button:hover{border-color:var(--ink-3)}
.verdict button[aria-pressed="true"][data-v="ok"]{background:var(--ok); border-color:var(--ok); color:#fff}
.verdict button[aria-pressed="true"][data-v="maybe"]{background:var(--warn); border-color:var(--warn); color:#fff}
.verdict button[aria-pressed="true"][data-v="bad"]{background:var(--bad); border-color:var(--bad); color:#fff}
.note{
  font:inherit; font-size:12.5px; resize:vertical; padding:6px 8px; border-radius:5px;
  border:1px solid var(--line); background:var(--ground); color:var(--ink); width:100%;
}
.panels{padding:14px 18px; display:flex; flex-direction:column; gap:20px; min-width:0}
.panel{margin:0; min-width:0}
.panel-head{display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:6px}
.panel-title{font-size:13.5px; font-weight:600; letter-spacing:.01em}
.panel-meta{font-size:11px; color:var(--ink-3)}
.panel-img{overflow-x:auto; background:#000; border-radius:5px; border:1px solid var(--line)}
.panel-img img{display:block; width:720px; max-width:none; height:auto}
@media (min-width:900px){.panel-img img{width:100%; max-width:720px}}
.cals{list-style:none; display:flex; flex-wrap:wrap; gap:6px 16px; margin:7px 0 0; padding:0; font-size:12.5px}
.cals li{display:flex; align-items:baseline; gap:7px}
.cal-name{color:var(--ink-3)}
.cal-val{color:var(--ink); font-weight:500}
.cal-t{color:var(--ink-3); font-size:11.5px}
.cal-fix{color:var(--accent); font-size:10.5px; letter-spacing:.05em; text-transform:uppercase}
.cal-arrow{color:var(--ink-3)}
.hint{font-size:12px; color:var(--ink-2); margin:6px 0 0}
.hint strong{color:var(--ink)}
.opts{font-size:12.5px; line-height:1.9; background:var(--surface); padding:10px 13px; border-radius:5px; border:1px solid var(--line)}
.opts .no{color:var(--bad)} .opts .yes{color:var(--ok)}
.brief em{font-style:normal; color:var(--ink); border-bottom:1px solid var(--accent)}
.fine{font-size:12.5px; color:var(--ink-2)}
.brief .g{color:var(--ok); font-weight:500}
.rubric{display:grid; grid-template-columns:auto 1fr; gap:6px 14px; margin:14px 0 16px; align-items:baseline}
.rubric dt{font-weight:600; font-size:12px; letter-spacing:.05em; text-transform:uppercase; padding:2px 9px; border-radius:20px; color:#fff}
.rubric dt.r-ok{background:var(--ok)} .rubric dt.r-maybe{background:var(--warn)} .rubric dt.r-bad{background:var(--bad)}
.rubric dd{margin:0; color:var(--ink-2)}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
@media (max-width:820px){
  .card-body{grid-template-columns:1fr}
  .measures{border-right:0; border-bottom:1px solid var(--line)}
  .brief,main{padding-left:16px; padding-right:16px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = r"""
const KEY='echoxflow-qc-tur2-v1';
const load=()=>{try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return {}}};
const save=s=>{try{localStorage.setItem(KEY,JSON.stringify(s))}catch(e){}};
let state=load();
const groups=[...document.querySelectorAll('.panel-verdict')];
const notes=[...document.querySelectorAll('.note')];

function paintGroup(g){
  const v=(state[g.dataset.key]||{}).v||null;
  g.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(v===b.dataset.v)));
  const fig=g.closest('.panel');
  fig.classList.toggle('rated', v==='ok');
  fig.classList.toggle('rated-maybe', v==='maybe');
  fig.classList.toggle('rated-bad', v==='bad');
}
function rollups(){
  document.querySelectorAll('.rollup').forEach(r=>{
    const ex=r.dataset.exam, total=+r.dataset.total;
    const done=groups.filter(g=>g.dataset.key.startsWith(ex+'::') && (state[g.dataset.key]||{}).v).length;
    r.textContent=done+'/'+total;
    r.classList.toggle('full', done===total);
    r.closest('.card').classList.toggle('done', done===total);
  });
}
function progress(){
  const done=groups.filter(g=>(state[g.dataset.key]||{}).v).length;
  document.getElementById('count').textContent=done+' / '+groups.length;
  document.querySelector('.bar>i').style.width=(100*done/groups.length)+'%';
  rollups();
}
groups.forEach(g=>{
  g.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
    const k=g.dataset.key, cur=(state[k]||{}).v;
    state[k]={v: cur===b.dataset.v ? null : b.dataset.v};
    save(state); paintGroup(g); progress();
  }));
  paintGroup(g);
});
notes.forEach(n=>{
  const k='note::'+n.dataset.exam;
  n.value=(state[k]||{}).t||'';
  n.addEventListener('input',e=>{state[k]={t:e.target.value}; save(state);});
});
progress();


document.getElementById('copy').addEventListener('click',async e=>{
  const rows=[['exam_id','panel','karar']];
  groups.forEach(g=>{const [ex,kind]=g.dataset.key.split('::');
    rows.push([ex,kind,(state[g.dataset.key]||{}).v||'']);});
  notes.forEach(n=>{const t=(state['note::'+n.dataset.exam]||{}).t||'';
    if(t.trim()) rows.push([n.dataset.exam,'NOT',t.replace(/[\r\n,]/g,' ')]);});
  const csv=rows.map(r=>r.join(',')).join('\n');
  try{ await navigator.clipboard.writeText(csv); e.target.textContent='Kopyalandı'; }
  catch(err){ e.target.textContent='Kopyalanamadı'; }
  setTimeout(()=>{e.target.textContent='Sonuçları kopyala';},1600);
});
function csvText(){
  const rows=[['exam_id','panel','karar']];
  groups.forEach(g=>{const [ex,kind]=g.dataset.key.split('::');
    rows.push([ex,kind,(state[g.dataset.key]||{}).v||'']);});
  notes.forEach(n=>{const t=(state['note::'+n.dataset.exam]||{}).t||'';
    if(t.trim()) rows.push([n.dataset.exam,'NOT',t.replace(/[\r\n,]/g,' ')]);});
  return rows.map(r=>r.join(',')).join('\n');
}
document.getElementById('download').addEventListener('click',()=>{
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([csvText()],{type:'text/csv'}));
  a.download='qc_tur2_kararlar.csv'; document.body.appendChild(a); a.click(); a.remove();
});
document.getElementById('dl1').addEventListener('click',e=>{
  let raw=null; try{raw=localStorage.getItem('echoxflow-qc-panel-v1');}catch(err){}
  if(!raw){e.target.textContent='1. tur kaydı bu tarayıcıda yok'; return;}
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([raw],{type:'application/json'}));
  a.download='qc_tur1_tarayici_kaydi.json'; document.body.appendChild(a); a.click(); a.remove();
  e.target.textContent='İndirildi (açmadan bana gönder)';
});
document.getElementById('reset').addEventListener('click',()=>{
  if(!window.__resetArmed){window.__resetArmed=true; const b=document.getElementById('reset'); b.textContent='Emin misin? Tekrar tıkla';
    setTimeout(()=>{window.__resetArmed=false; b.textContent='Sıfırla';},2500); return;}
  state={}; save(state); groups.forEach(paintGroup); notes.forEach(n=>n.value=''); progress();
});
"""

n_tr = sum(1 for c in CARDS for p in c['panels'] if p['kind'] == 'TR Vmax')
HTML = f"""<title>EchoXFlow QC 2. Tur</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>

<div class="topbar">
  <p class="topbar-title">EchoXFlow Doppler kontrolü <span>· 2. tur (kör)</span></p>
  <div class="tools">
    <button class="tool" id="download" type="button">Sonuçları indir (CSV)</button>
    <button class="tool" id="dl1" type="button" title="1. turun tarayıcıda kalan kayıtlarını ekranda göstermeden dosyaya indirir">1. tur kaydını indir</button>
    <button class="tool" id="copy" type="button">Sonuçları kopyala</button>
    <button class="tool" id="reset" type="button">Sıfırla</button>
  </div>
  <div class="progress"><span class="count" id="count">0 / {len(CARDS)}</span><span class="bar"><i></i></span></div>
</div>

<section class="brief">
  <h1>2. tur (kör): işaretler doğru yapıda, doğru dalgada ve tepe noktasında mı?</h1>
  <p class="sub">EchoXFlow birincil kohortundan 1. turdaki aynı {len(CARDS)} muayene, farklı ve sabit bir sırada.
  Bu tur gözlemci-içi uyum için yapılıyor: 1. turdaki kararlarını hatırlamaya çalışma, gördüğünü işaretle.
  ASE sınıflaması gizli.</p>
  <p><strong>Solda: hangi yapı?</strong> Kaydın kendi 2D görüntüsü, üstünde ışın hattı ve
  <span class="g">örnek hacim (gate)</span>. Gate mitral kapak uçlarında mı, septal mi lateral anulusta
  mı, triküspitte mi?</p>
  <p><strong>Altta: hangi faz?</strong> Kaydın <span class="g">EKG'si</span> spektrumla hizalı, QRS
  tetikleri sarı. E erken diyastolde mi, A QRS'ten hemen önce mi, TR sistolde mi?</p>
  <p class="fine">Yükseklik ölçeği her kaydın kendi kaliperlerinden sabitlenir; işaretlerin topluca
  yukarı-aşağı kayması bilgi taşımaz, bir işaretin kardeşlerine göre yanlış yerde olması taşır. Sapması
  %15'i aşan panellerde yükseklik çizilmez.</p>
  <dl class="rubric">
    <dt class="r-ok">Doğru</dt>
    <dd>Gate doğru yapıda, işaretler doğru dalgada ve kendi zarflarının tepesinde.</dd>
    <dt class="r-maybe">Şüpheli</dt>
    <dd>Bir şey oturmuyor ama emin olamadın.</dd>
    <dt class="r-bad">Yanlış</dt>
    <dd>Gate yanlış yapıda, işaret yanlış dalgada, ya da kardeşlerine göre bariz kaçık.</dd>
  </dl>
  <p>Kararlar tarayıcında saklanır. Bitirince <strong>“Sonuçları indir (CSV)”</strong> ile dosyayı kaydet
  (<span class="mono">qc_tur2_kararlar.csv</span>) ve bana yolunu söyle.</p>
</section>

<main>{''.join(cards_html)}</main>
<script>{JS}</script>
"""
open(OUT, 'w').write(HTML)
print(f'{OUT} yazıldı ({len(HTML)/1e6:.1f} MB)')
