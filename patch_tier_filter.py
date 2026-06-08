#!/usr/bin/env python3
"""
patch_tier_filter.py
1. Status legend 加 tier filter 功能（click 過濾 3D 點）
2. tier-legend margin-top 對齊 weight classes spacing

用法：python3 patch_tier_filter.py ~/UFC/index.html
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
# PATCH 1 — tier-legend HTML：靜態改成動態 JS 生成，加間距
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
"HTML: tier-legend 改成空容器，由 JS 填充"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 2 — JS：在 applyFilter 後加 tier filter 邏輯
# ══════════════════════════════════════════════════════════════════════
rep(
"""let activeWC=null;
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
"""let activeWC=null, activeTier=null;

function applyFilter(wc){
  activeWC=wc;
  applyFilters();
  document.querySelectorAll('.legend-item').forEach(item=>{
    const iw=item.dataset.wc;
    if(wc===null){item.classList.remove('muted','active-filter');}
    else if(iw===wc){item.classList.remove('muted');item.classList.add('active-filter');}
    else{item.classList.add('muted');item.classList.remove('active-filter');}
  });
  allBtn.classList.toggle('active-filter',wc===null);
}

function applyTierFilter(tier){
  activeTier = activeTier===tier ? null : tier;
  applyFilters();
  document.querySelectorAll('.tier-legend-item').forEach(item=>{
    const it=item.dataset.tier;
    if(activeTier===null){item.classList.remove('muted','active-filter');}
    else if(it===activeTier){item.classList.remove('muted');item.classList.add('active-filter');}
    else{item.classList.add('muted');item.classList.remove('active-filter');}
  });
}

function applyFilters(){
  Object.entries(wcTraceIdx).forEach(([wc,i])=>{
    const fighters = byWC[wc]||[];
    const opacities = fighters.map(f=>{
      const wcMatch  = activeWC===null || wc===activeWC;
      const tierMatch= activeTier===null || f.tier===activeTier;
      return (wcMatch && tierMatch) ? tierOpacity(f) : 0;
    });
    const visible = activeWC===null ? true : (wc===activeWC);
    Plotly.restyle('plot',{'marker.opacity':opacities, visible:visible},[i]);
  });
}""",
"JS: tier filter logic + applyFilters"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 3 — JS：在 legend 建立後加 tier legend 建立
# ══════════════════════════════════════════════════════════════════════
rep(
"""// ── Universe search ──""",
"""// ── Tier legend ──
const TIER_ITEMS = [
  {tier:'champion',   label:'Current Champion', color:'#FFD700'},
  {tier:'ex_champion',label:'Former Champion',  color:'#C0A060'},
  {tier:'ranked',     label:'Ranked Top 15',    color:'#7F8FA6'},
  {tier:'unranked',   label:'Unranked',         color:'#2a2a3a'},
];
const tierLegendEl = document.getElementById('tier-legend');
const tierTitle = document.createElement('div');
tierTitle.className = 'panel-title';
tierTitle.textContent = 'Status';
tierLegendEl.appendChild(tierTitle);
TIER_ITEMS.forEach(({tier,label,color})=>{
  const item = document.createElement('div');
  item.className = 'tier-legend-item legend-item';
  item.dataset.tier = tier;
  item.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:10px;';
  item.innerHTML = `<div style="width:8px;height:8px;border-radius:50%;background:${color};flex-shrink:0"></div>${label}`;
  item.addEventListener('click',()=>applyTierFilter(tier));
  tierLegendEl.appendChild(item);
});

// ── Universe search ──""",
"JS: tier legend 動態建立，加 click handler"
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
print("    git add -A && git commit -m 'feat: tier status filter + spacing fix' && git push")
