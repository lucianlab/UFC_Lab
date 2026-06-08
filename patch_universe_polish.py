#!/usr/bin/env python3
"""
patch_universe_polish.py
1. 相機初始角度調整
2. Ranked 顏色改成中性灰（脫離藍色）
3. hlTrace 選中高亮改成 champagne 光暈
4. Status legend 加數量
5. TIER_ITEMS 顏色同步更新

用法：python3 patch_universe_polish.py ~/UFC/index.html
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
# PATCH 1 — Ranked 顏色：#7F8FA6 藍灰 → #9a9a9a 中性灰
# ══════════════════════════════════════════════════════════════════════
rep(
"  if(f.tier==='ranked') return '#7F8FA6';",
"  if(f.tier==='ranked') return '#9a9a9a';",
"tierColor ranked → 中性灰"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 2 — hlTrace 改成 champagne 光暈
# ══════════════════════════════════════════════════════════════════════
rep(
"const hlTrace={\n  type:'scatter3d', mode:'markers', x:[],y:[],z:[],text:[],\n  marker:{size:6,color:'#c0392b',opacity:0.95,line:{color:'#7f0000',width:1}},\n  hovertemplate:'<b>%{text}</b><extra></extra>', showlegend:false,\n};",
"const hlTrace={\n  type:'scatter3d', mode:'markers', x:[],y:[],z:[],text:[],\n  marker:{size:10,color:'rgba(201,169,110,0.15)',opacity:1,line:{color:'#c9a96e',width:2}},\n  hovertemplate:'<b>%{text}</b><extra></extra>', showlegend:false,\n};",
"hlTrace → champagne 光暈"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 3 — 初始相機角度
# ══════════════════════════════════════════════════════════════════════
rep(
"    camera:{eye:{x:1.5,y:1.5,z:1.2}},",
"    camera:{eye:{x:1.8,y:0.8,z:1.0}},",
"初始相機角度"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 4 — TIER_ITEMS：同步 ranked 顏色 + 加數量計算
# ══════════════════════════════════════════════════════════════════════
rep(
"""const TIER_ITEMS=[
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
});""",
"""const tierCounts={};
FIGHTERS.forEach(f=>{ const t=f.tier||'unranked'; tierCounts[t]=(tierCounts[t]||0)+1; });
const TIER_ITEMS=[
  {tier:'champion',   label:'Current Champion', color:'#FFD700'},
  {tier:'ex_champion',label:'Former Champion',  color:'#C0A060'},
  {tier:'ranked',     label:'Ranked Top 15',    color:'#9a9a9a'},
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
  const count=tierCounts[tier]||0;
  item.innerHTML=`<div style="width:8px;height:8px;border-radius:50%;background:${color};flex-shrink:0"></div>${label}<span style="color:var(--text-lo);margin-left:auto">${count}</span>`;
  item.addEventListener('click',()=>applyTierFilter(tier));
  tierLegendEl.appendChild(item);
});""",
"TIER_ITEMS 同步顏色 + 加數量"
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
print("    git add -A && git commit -m 'style: universe polish - camera, ranked color, champagne highlight, tier counts' && git push")
