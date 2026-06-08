#!/usr/bin/env python3
"""
patch_tier_filter2.py
1. WC filter + tier filter 統一用 marker.opacity（不用 visible）
2. Tier legend 文字顏色修正
3. 兩個 filter 可以疊加

用法：python3 patch_tier_filter2.py ~/UFC/index.html
"""
import sys, pathlib, shutil, datetime

TARGET = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("~/UFC/index.html").expanduser()
if not TARGET.exists():
    print(f"❌  找不到：{TARGET}"); sys.exit(1)

backup = TARGET.with_suffix(f".bak_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
shutil.copy2(TARGET, backup)
print(f"✅  備份：{backup}")

html = TARGET.read_text(encoding="utf-8")
original = html
results = []

def rep(old, new, label):
    global html
    count = html.count(old)
    if count == 0:
        results.append(f"⚠️   找不到（0次）：{label}"); return
    html = html.replace(old, new)
    results.append(f"✅  替換 {count:2d} 次：{label}")

# ══════════════════════════════════════════════════════════════════════
# PATCH 1 — 統一 filter 邏輯：WC + Tier 都用 opacity
# ══════════════════════════════════════════════════════════════════════
rep(
"""const wcTraceIdx={}; let ti=0;
Object.keys(byWC).forEach(wc=>{ wcTraceIdx[wc]=ti++; });
let activeWC=null;
function applyFilter(wc){
  activeWC=wc;
  Object.entries(wcTraceIdx).forEach(([w,i])=>Plotly.restyle('plot',{visible:wc===null?true:(w===wc)},i));
  document.querySelectorAll('.legend-item').forEach(item=>{
    const iw=item.dataset.wc;
    if(wc===null){item.classList.remove('muted','active-filter');}
    else if(iw===wc){item.classList.remove('muted');item.classList.add('active-filter');}
    else{item.classList.add('muted');item.classList.remove('active-filter');}
  });
  allBtn.classList.toggle('active-filter',wc===null);
}""",
"""const wcTraceIdx={}; let ti=0;
Object.keys(byWC).forEach(wc=>{ wcTraceIdx[wc]=ti++; });
let activeWC=null, activeTier=null;

function applyFilters(){
  Object.entries(wcTraceIdx).forEach(([wc,i])=>{
    const fighters = byWC[wc]||[];
    const opacities = fighters.map(f=>{
      const wcOk   = activeWC===null   || wc===activeWC;
      const tierOk = activeTier===null || f.tier===activeTier;
      return (wcOk && tierOk) ? tierOpacity(f) : 0;
    });
    Plotly.restyle('plot',{'marker.opacity':[opacities]},[i]);
  });
}

function applyFilter(wc){
  activeWC = activeWC===wc ? null : wc;
  applyFilters();
  document.querySelectorAll('.legend-item[data-wc]').forEach(item=>{
    const iw=item.dataset.wc;
    if(activeWC===null){item.classList.remove('muted','active-filter');}
    else if(iw===activeWC){item.classList.remove('muted');item.classList.add('active-filter');}
    else{item.classList.add('muted');item.classList.remove('active-filter');}
  });
  allBtn.classList.toggle('active-filter',activeWC===null);
}

function applyTierFilter(tier){
  activeTier = activeTier===tier ? null : tier;
  applyFilters();
  document.querySelectorAll('.legend-item[data-tier]').forEach(item=>{
    const it=item.dataset.tier;
    if(activeTier===null){item.classList.remove('muted','active-filter');}
    else if(it===activeTier){item.classList.remove('muted');item.classList.add('active-filter');}
    else{item.classList.add('muted');item.classList.remove('active-filter');}
  });
}""",
"統一 filter 邏輯用 opacity"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 2 — WC legend click 改用 applyFilter toggle
# ══════════════════════════════════════════════════════════════════════
rep(
"  item.addEventListener('click',()=>applyFilter(activeWC===wc?null:wc));",
"  item.addEventListener('click',()=>applyFilter(wc));",
"WC legend click 改 toggle"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 3 — Tier legend HTML 空容器 + margin
# ══════════════════════════════════════════════════════════════════════
rep(
"""    <div id="tier-legend" style="margin-bottom:10px">
      <div class="panel-title">Status</div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:10px;color:var(--text-lo)"><div style="width:8px;height:8px;border-radius:50%;background:#FFD700;flex-shrink:0"></div>Current Champion</div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:10px;color:var(--text-lo)"><div style="width:8px;height:8px;border-radius:50%;background:#C0A060;flex-shrink:0"></div>Former Champion</div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:10px;color:var(--text-lo)"><div style="width:8px;height:8px;border-radius:50%;background:#7F8FA6;flex-shrink:0"></div>Ranked Top 15</div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:10px;color:var(--text-lo)"><div style="width:8px;height:8px;border-radius:50%;background:#2a2a3a;flex-shrink:0"></div>Unranked</div>
    </div>""",
"""    <div id="tier-legend" style="margin-top:16px;margin-bottom:10px"></div>""",
"HTML: tier-legend 空容器"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 4 — JS：tier legend 動態建立（在 Universe search 之前）
# ══════════════════════════════════════════════════════════════════════
rep(
"// ── Universe search ──",
"""// ── Tier legend ──
const TIER_ITEMS=[
  {tier:'champion',   label:'Current Champion', color:'#FFD700'},
  {tier:'ex_champion',label:'Former Champion',  color:'#C0A060'},
  {tier:'ranked',     label:'Ranked Top 15',    color:'#7F8FA6'},
  {tier:'unranked',   label:'Unranked',         color:'#2a2a3a'},
];
const tierLegendEl=document.getElementById('tier-legend');
const tierTitle=document.createElement('div');
tierTitle.className='panel-title'; tierTitle.textContent='Status';
tierLegendEl.appendChild(tierTitle);
TIER_ITEMS.forEach(({tier,label,color})=>{
  const item=document.createElement('div');
  item.className='legend-item';
  item.dataset.tier=tier;
  item.style.cssText='display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:10px;color:var(--text-lo);cursor:pointer;';
  item.innerHTML=`<div style="width:8px;height:8px;border-radius:50%;background:${color};flex-shrink:0"></div>${label}`;
  item.addEventListener('click',()=>applyTierFilter(tier));
  tierLegendEl.appendChild(item);
});

// ── Universe search ──""",
"JS: tier legend 動態建立"
)

# ══════════════════════════════════════════════════════════════════════
print("\n".join(results))

if html == original:
    print("\n⚠️   沒有改動"); sys.exit(1)

TARGET.write_text(html, encoding="utf-8")
print(f"\n✅  已寫回：{TARGET}")

style_count = html.count("</style>")
print(f"   </style> 數量：{style_count}（應為 1）")
if style_count != 1:
    print("❌  異常！請從備份還原"); sys.exit(1)

print("\n🎉  完成！")
print("    git add -A && git commit -m 'feat: unified tier+wc filter via opacity' && git push")
