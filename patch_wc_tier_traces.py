#!/usr/bin/env python3
"""
patch_wc_tier_traces.py
把 univTraces 從「per WC」改成「per WC × tier」
WC filter 和 Tier filter 都用 visible，支援雙重組合篩選

用法：python3 patch_wc_tier_traces.py ~/UFC/index.html
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
# PATCH 1 — univTraces 改成 WC × tier 分組
# ══════════════════════════════════════════════════════════════════════
rep(
"""const univTraces = Object.entries(byWC).map(([wc,fighters])=>({
  type:'scatter3d', mode:'markers', name:wc,
  x:fighters.map(f=>f.x), y:fighters.map(f=>f.y), z:fighters.map(f=>f.z),
  text:fighters.map(f=>f.name), customdata:fighters.map(f=>f.name),
  marker:{
    size:fighters.map(f=>tierSize(f)),
    color:fighters.map(f=>tierColor(f)),
    opacity:fighters.map(f=>tierOpacity(f)),
    line:{width:0},
  },
  hovertemplate:'<b>%{text}</b><extra></extra>',
}));""",
"""const TIER_ORDER = ['champion','ex_champion','ranked','unranked'];
let univTraces = [];
const traceIdx = {}; // traceIdx[wc][tier] = index
let _ti = 0;
Object.entries(byWC).forEach(([wc, fighters])=>{
  traceIdx[wc] = {};
  TIER_ORDER.forEach(tier=>{
    const group = fighters.filter(f=>(f.tier||'unranked')===tier);
    if(!group.length) return;
    traceIdx[wc][tier] = _ti++;
    univTraces.push({
      type:'scatter3d', mode:'markers', name:`${wc}__${tier}`,
      x:group.map(f=>f.x), y:group.map(f=>f.y), z:group.map(f=>f.z),
      text:group.map(f=>f.name), customdata:group.map(f=>f.name),
      marker:{
        size:group.map(f=>tierSize(f)),
        color:group.map(f=>tierColor(f)),
        opacity:group.map(f=>tierOpacity(f)),
        line:{width:0},
      },
      hovertemplate:'<b>%{text}</b><extra></extra>',
    });
  });
});""",
"univTraces 改成 WC × tier 分組"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 2 — applyFilters 改用 visible
# ══════════════════════════════════════════════════════════════════════
rep(
"""function applyFilters(){
  Object.entries(wcTraceIdx).forEach(([wc,i])=>{
    const fighters = byWC[wc]||[];
    const opacities = fighters.map(f=>{
      const wcMatch  = activeWC===null || wc===activeWC;
      const tierMatch= activeTier===null || f.tier===activeTier;
      return (wcMatch && tierMatch) ? tierOpacity(f) : 0;
    });
    Plotly.restyle('plot',{'marker.opacity':[opacities]},[i]);
  });
}""",
"""function applyFilters(){
  const indices = [], visibles = [];
  Object.entries(traceIdx).forEach(([wc, tierMap])=>{
    const wcMatch = activeWC===null || wc===activeWC;
    Object.entries(tierMap).forEach(([tier, idx])=>{
      const tierMatch = activeTier===null || tier===activeTier;
      indices.push(idx);
      visibles.push(wcMatch && tierMatch);
    });
  });
  Plotly.restyle('plot', {visible: visibles}, indices);
}""",
"applyFilters 改用 visible"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 3 — wcTraceIdx 不再需要，移除
# ══════════════════════════════════════════════════════════════════════
rep(
"""const wcTraceIdx={};\nObject.entries(byWC).forEach(([wc],i)=>{ wcTraceIdx[wc]=i; });""",
"",
"移除 wcTraceIdx"
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
print("    git add -A && git commit -m 'feat: WC×tier traces, visible filter' && git push")
