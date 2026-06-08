#!/usr/bin/env python3
"""
patch_filter_react.py
applyFilters 改用修改 univTraces opacity 再 Plotly.react，
解決 selectFighter 的 react 覆蓋 restyle 的問題
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
# PATCH 1 — univTraces 改成 let，方便後續修改
# ══════════════════════════════════════════════════════════════════════
rep(
"const univTraces = Object.entries(byWC).map(([wc,fighters])=>({",
"let univTraces = Object.entries(byWC).map(([wc,fighters])=>({",
"univTraces const→let"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 2 — applyFilters 改用修改 univTraces + Plotly.react
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
  univTraces = Object.entries(byWC).map(([wc,fighters])=>({
    type:'scatter3d', mode:'markers', name:wc,
    x:fighters.map(f=>f.x), y:fighters.map(f=>f.y), z:fighters.map(f=>f.z),
    text:fighters.map(f=>f.name), customdata:fighters.map(f=>f.name),
    marker:{
      size:fighters.map(f=>tierSize(f)),
      color:fighters.map(f=>tierColor(f)),
      opacity:fighters.map(f=>{
        const wcMatch  = activeWC===null || wc===activeWC;
        const tierMatch= activeTier===null || f.tier===activeTier;
        return (wcMatch && tierMatch) ? tierOpacity(f) : 0;
      }),
      line:{width:0},
    },
    hovertemplate:'<b>%{text}</b><extra></extra>',
  }));
  Plotly.react('plot',[...univTraces,hlTrace],univLayout,{responsive:true,displayModeBar:false});
}""",
"applyFilters 改用 univTraces + react"
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
print("    git add -A && git commit -m 'fix: filter use react instead of restyle' && git push")
