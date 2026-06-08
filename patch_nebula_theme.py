#!/usr/bin/env python3
"""
patch_nebula_theme.py
1. ✦ 符號切換鍵（plot 右上角）
2. CLASSIC / NEBULA 彩蛋文字（plot 左下角）
3. Nebula 配色：藍超巨星 / 玫瑰紅巨星 / 夢幻紫 / 深紫黑
4. 淡入淡出切換動畫

用法：python3 patch_nebula_theme.py ~/UFC/index.html
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
# PATCH 1 — CSS：✦ 按鈕 + 彩蛋文字
# ══════════════════════════════════════════════════════════════════════
rep(
"#plot { flex: 1; min-width: 0; }",
"""#plot { flex: 1; min-width: 0; position: relative; }

#theme-toggle {
  position: absolute;
  top: 12px; right: 14px;
  font-size: 16px;
  color: var(--text-lo);
  cursor: pointer;
  z-index: 10;
  opacity: 0.35;
  transition: opacity 0.2s, color 0.2s;
  user-select: none;
  line-height: 1;
}
#theme-toggle:hover { opacity: 1; }
#theme-toggle.nebula { color: #c77dff; opacity: 0.7; }

#theme-label {
  position: absolute;
  bottom: 14px; left: 16px;
  font-family: 'Cormorant Garamond', serif;
  font-size: 10px;
  letter-spacing: 0.22em;
  color: var(--text-hi);
  opacity: 0.12;
  z-index: 10;
  user-select: none;
  pointer-events: none;
  transition: opacity 0.4s;
}""",
"CSS: theme toggle + label"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 2 — HTML：在 plot div 裡加 ✦ 和彩蛋文字
# ══════════════════════════════════════════════════════════════════════
rep(
'  <div id="plot"></div>',
'  <div id="plot">\n    <div id="theme-toggle" title="Switch theme">✦</div>\n    <div id="theme-label">CLASSIC</div>\n  </div>',
"HTML: theme toggle + label"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 3 — JS：tierColor 改成支援雙主題
# ══════════════════════════════════════════════════════════════════════
rep(
"""function tierColor(f){
  if(f.tier==='champion') return '#FFD700';
  if(f.tier==='ex_champion') return '#C0A060';
  if(f.tier==='ranked') return '#7F8FA6';
  return '#2a2a3a';
}""",
"""let isNebula = false;
const CLASSIC_COLORS = {
  champion:   '#FFD700',
  ex_champion:'#C0A060',
  ranked:     '#9a9a9a',
  unranked:   '#2a2a3a',
};
const NEBULA_COLORS = {
  champion:   '#a0c4ff',
  ex_champion:'#ff8fa3',
  ranked:     '#c77dff',
  unranked:   '#1a0533',
};
function tierColor(f){
  const t = f.tier||'unranked';
  return isNebula ? (NEBULA_COLORS[t]||'#1a0533') : (CLASSIC_COLORS[t]||'#2a2a3a');
}""",
"JS: tierColor 雙主題"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 4 — JS：theme toggle 邏輯（在 boot 最後、Universe search 之前）
# ══════════════════════════════════════════════════════════════════════
rep(
"// ── Universe search ──",
"""// ── Theme toggle ──
function rebuildTraces(){
  univTraces = [];
  let _ti2 = 0;
  Object.keys(traceIdx).forEach(wc=>{ Object.keys(traceIdx[wc]).forEach(tier=>{ traceIdx[wc][tier]=_ti2++; }); });
  Object.entries(byWC).forEach(([wc, fighters])=>{
    TIER_ORDER.forEach(tier=>{
      const group = fighters.filter(f=>(f.tier||'unranked')===tier);
      if(!group.length) return;
      univTraces.push({
        type:'scatter3d', mode:'markers', name:wc+'__'+tier,
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
  });
}

const themeBtn  = document.getElementById('theme-toggle');
const themeLabel= document.getElementById('theme-label');
const plotEl    = document.getElementById('plot');

themeBtn.addEventListener('click',()=>{
  isNebula = !isNebula;
  themeBtn.classList.toggle('nebula', isNebula);
  themeLabel.textContent = isNebula ? 'NEBULA' : 'CLASSIC';
  plotEl.style.transition = 'opacity 0.3s';
  plotEl.style.opacity = '0';
  setTimeout(()=>{
    rebuildTraces();
    Plotly.react('plot',[...univTraces,hlTrace],univLayout,{responsive:true,displayModeBar:false});
    plotEl.style.opacity = '1';
  }, 300);
});

// ── Universe search ──""",
"JS: theme toggle 邏輯"
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
print("    git add -A && git commit -m 'feat: nebula theme toggle with easter egg' && git push")
