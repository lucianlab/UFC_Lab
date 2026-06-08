#!/usr/bin/env python3
"""
patch_blood_steel.py — Blood & Steel 配色全站替換

Red  dim=#7a1f28  accent=#b83040  (原 #8c1f1f / #c23030)
Blue dim=#1a3050  accent=#2a5090  (原 #1a3a6e / #2b5fb3 / #4a7fd4)

用法：python3 patch_blood_steel.py ~/UFC/index.html
"""

import sys, pathlib, shutil, datetime

TARGET = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("~/UFC/index.html").expanduser()
if not TARGET.exists():
    print(f"❌  找不到檔案：{TARGET}"); sys.exit(1)

backup = TARGET.with_suffix(f".bak_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
shutil.copy2(TARGET, backup)
print(f"✅  備份：{backup}\n")

html = TARGET.read_text(encoding="utf-8")
original = html
results = []

def replace_all(src, old, new, label):
    global html
    count = src.count(old)
    if count == 0:
        results.append(f"⚠️   找不到（{count}次）：{label}")
        return src
    src = src.replace(old, new)
    results.append(f"✅  替換 {count:2d} 次：{label}")
    return src

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — CSS variable 定義更新（:root 裡）
# ══════════════════════════════════════════════════════════════════════
html = replace_all(html,
    "  --crimson:     #8c1f1f;",
    "  --crimson:     #7a1f28;",
    ":root --crimson dim")

html = replace_all(html,
    "  --crimson-2:   #c23030;",
    "  --crimson-2:   #b83040;",
    ":root --crimson-2 accent")

html = replace_all(html,
    "  --crimson-dim: #3a1010;",
    "  --crimson-dim: #2e1018;",
    ":root --crimson-dim")

html = replace_all(html,
    "  --cobalt:      #1a3a6e;",
    "  --cobalt:      #1a3050;",
    ":root --cobalt dim")

html = replace_all(html,
    "  --cobalt-2:    #2b5fb3;",
    "  --cobalt-2:    #2a5090;",
    ":root --cobalt-2 accent")

html = replace_all(html,
    "  --cobalt-dim:  #0d1e3a;",
    "  --cobalt-dim:  #0d1828;",
    ":root --cobalt-dim")

html = replace_all(html,
    "  --red:         #c23030;",
    "  --red:         #b83040;",
    ":root --red alias")

html = replace_all(html,
    "  --blue:        #2b5fb3;",
    "  --blue:        #2a5090;",
    ":root --blue alias")

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — HTML inline style 硬碼顏色（精確字串替換）
# ══════════════════════════════════════════════════════════════════════

# Win probability 大數字
html = replace_all(html,
    'color:#c23030;font-family:\'Barlow Condensed\'',
    'color:#b83040;font-family:\'Barlow Condensed\'',
    "vs-prob-red-pct color")

html = replace_all(html,
    'color:#2b5fb3;font-family:\'Barlow Condensed\'',
    'color:#2a5090;font-family:\'Barlow Condensed\'',
    "vs-prob-blue-pct color")

# Win probability bar background
html = replace_all(html,
    'background:#8c1f1f;transition:width 0.6s ease',
    'background:#7a1f28;transition:width 0.6s ease',
    "vs-prob-bar-red background")

html = replace_all(html,
    'background:#1a3a6e;transition:width 0.6s ease',
    'background:#1a3050;transition:width 0.6s ease',
    "vs-prob-bar-blue background")

# Fighter name sub labels
html = replace_all(html,
    'color:rgba(194,48,48,0.55);font-family:\'DM Mono\'',
    'color:rgba(184,48,64,0.55);font-family:\'DM Mono\'',
    "vs-prob-red-name color")

html = replace_all(html,
    'color:rgba(43,95,179,0.55);font-family:\'DM Mono\'',
    'color:rgba(42,80,144,0.55);font-family:\'DM Mono\'',
    "vs-prob-blue-name color")

# DNA legend line swatches
html = replace_all(html,
    'style="background:#4a7fd4"',
    'style="background:#2a5090"',
    "DNA legend grappling line swatch")

html = replace_all(html,
    'style="background:#c23030"',
    'style="background:#b83040"',
    "DNA legend striking line swatch")

# DNA legend bar swatches (rgba)
html = replace_all(html,
    'style="background:rgba(43,95,179,0.38);border:1px solid rgba(43,95,179,0.55)"',
    'style="background:rgba(42,80,144,0.38);border:1px solid rgba(42,80,144,0.55)"',
    "DNA legend grappling bar swatch")

html = replace_all(html,
    'style="background:rgba(140,31,31,0.38);border:1px solid rgba(140,31,31,0.55)"',
    'style="background:rgba(122,31,40,0.38);border:1px solid rgba(122,31,40,0.55)"',
    "DNA legend striking bar swatch")

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — CSS 裡的硬碼 rgba（picker、fo-fill、sd-fill 等）
# ══════════════════════════════════════════════════════════════════════

# fo-fill.sub hardcoded blue
html = replace_all(html,
    '.fo-fill.sub { background:#4a7fd4; }',
    '.fo-fill.sub { background:#2a5090; }',
    ".fo-fill.sub background")

# vs-picker filled-red rgba backgrounds
html = replace_all(html,
    'rgba(140,31,31,0.12)',
    'rgba(122,31,40,0.12)',
    "vs-picker filled-red bg rgba")

html = replace_all(html,
    'rgba(26,58,110,0.18)',
    'rgba(26,48,80,0.18)',
    "vs-picker filled-blue bg rgba")

# vs-picker label/wc colors
html = replace_all(html,
    'rgba(194,48,48,0.5)',
    'rgba(184,48,64,0.5)',
    "vs-picker red label/wc rgba")

html = replace_all(html,
    'rgba(43,95,179,0.5)',
    'rgba(42,80,144,0.5)',
    "vs-picker blue label/wc rgba")

# Blueprint diff btn hard rgba
html = replace_all(html,
    'background:rgba(194,48,48,0.07)',
    'background:rgba(184,48,64,0.07)',
    "b-diff-btn hard background rgba")

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — JS 裡的硬碼顏色字串
# ══════════════════════════════════════════════════════════════════════

# Universe Plotly marker
html = replace_all(html,
    "textfont:{color:'#c23030',",
    "textfont:{color:'#b83040',",
    "Plotly textfont red")

html = replace_all(html,
    "marker:{size:9,color:'#8c1f1f',opacity:1,line:{color:'#c23030'",
    "marker:{size:9,color:'#7a1f28',opacity:1,line:{color:'#b83040'",
    "Plotly red marker")

html = replace_all(html,
    "marker:{size:9,color:'#1a3a6e',opacity:1,line:{color:'#9994a0'",
    "marker:{size:9,color:'#1a3050',opacity:1,line:{color:'#9994a0'",
    "Plotly blue marker")

# Radar chart (Chart.js)
html = replace_all(html,
    "borderColor: '#c23030', backgroundColor: 'rgba(194,48,48,0.1)', borderWidth: 1.5, pointBackgroundColor: '#c23030'",
    "borderColor: '#b83040', backgroundColor: 'rgba(184,48,64,0.1)', borderWidth: 1.5, pointBackgroundColor: '#b83040'",
    "radar chart red dataset")

html = replace_all(html,
    "borderColor: '#2b5fb3', backgroundColor: 'rgba(43,95,179,0.1)', borderWidth: 1.5, pointBackgroundColor: '#2b5fb3'",
    "borderColor: '#2a5090', backgroundColor: 'rgba(42,80,144,0.1)', borderWidth: 1.5, pointBackgroundColor: '#2a5090'",
    "radar chart blue dataset")

# Fight shape / SHAP accent color
html = replace_all(html,
    "const color = isRed ? '#8c1f1f' : '#1a3a6e';",
    "const color = isRed ? '#7a1f28' : '#1a3050';",
    "JS isRed dim color")

html = replace_all(html,
    "const accentColor = isRed ? '#c23030' : '#2b5fb3';",
    "const accentColor = isRed ? '#b83040' : '#2a5090';",
    "JS isRed accent color")

# Similar matchups fighter names
html = replace_all(html,
    '<span style="color:#c23030">${s.a}</span>',
    '<span style="color:#b83040">${s.a}</span>',
    "similar matchups red name")

html = replace_all(html,
    '<span style="color:#2b5fb3">${s.bName}</span>',
    '<span style="color:#2a5090">${s.bName}</span>',
    "similar matchups blue name")

html = replace_all(html,
    "winner===s.a ? '#c23030' : '#2b5fb3'",
    "winner===s.a ? '#b83040' : '#2a5090'",
    "similar matchups winner color")

# Fight verdict colors
html = replace_all(html,
    "'KO/TKO':     { verdict:'EXPLOSIVE FIREFIGHT', color:'#c23030' }",
    "'KO/TKO':     { verdict:'EXPLOSIVE FIREFIGHT', color:'#b83040' }",
    "verdict KO color")

html = replace_all(html,
    "'Submission': { verdict:'SUBMISSION HUNT',      color:'#4a7fd4' }",
    "'Submission': { verdict:'SUBMISSION HUNT',      color:'#2a5090' }",
    "verdict Submission color")

# Blueprint elite pip glow
html = replace_all(html,
    "'0 0 4px rgba(194,48,48,0.5)'",
    "'0 0 4px rgba(184,48,64,0.5)'",
    "blueprint pip elite glow")

html = replace_all(html,
    "elite ? 'rgba(194,48,48,0.45)'",
    "elite ? 'rgba(184,48,64,0.45)'",
    "blueprint glowColor elite")

html = replace_all(html,
    "elite ? 'rgba(194,48,48,0.35)'",
    "elite ? 'rgba(184,48,64,0.35)'",
    "blueprint borderColor elite")

# META winrate_diff color
html = replace_all(html,
    "? (d.winrate_diff > 0 ? '#c23030' : '#c9a96e')",
    "? (d.winrate_diff > 0 ? '#b83040' : '#c9a96e')",
    "META winrate_diff red color")

# META canvas drawLine colors
html = replace_all(html,
    "drawLine(grpPts.filter(Boolean), '#4a7fd4', 2);",
    "drawLine(grpPts.filter(Boolean), '#2a5090', 2);",
    "META drawLine grappling")

html = replace_all(html,
    "drawLine(strPts.filter(Boolean), '#c23030', 2);",
    "drawLine(strPts.filter(Boolean), '#b83040', 2);",
    "META drawLine striking")

# META canvas bar rgba
html = replace_all(html,
    "ctx.fillStyle   = 'rgba(140,31,31,0.38)';",
    "ctx.fillStyle   = 'rgba(122,31,40,0.38)';",
    "META canvas striking fill")

html = replace_all(html,
    "ctx.strokeStyle = 'rgba(140,31,31,0.60)';",
    "ctx.strokeStyle = 'rgba(122,31,40,0.60)';",
    "META canvas striking stroke")

html = replace_all(html,
    "ctx.fillStyle   = 'rgba(43,95,179,0.38)';",
    "ctx.fillStyle   = 'rgba(42,80,144,0.38)';",
    "META canvas grappling fill")

html = replace_all(html,
    "ctx.strokeStyle = 'rgba(43,95,179,0.60)';",
    "ctx.strokeStyle = 'rgba(42,80,144,0.60)';",
    "META canvas grappling stroke")

# ══════════════════════════════════════════════════════════════════════
# 結果輸出
# ══════════════════════════════════════════════════════════════════════
print("\n".join(results))

if html == original:
    print("\n⚠️   沒有任何改動，請確認上面的警告"); sys.exit(1)

TARGET.write_text(html, encoding="utf-8")
print(f"\n✅  已寫回：{TARGET}")

# 驗證
style_count = html.count("</style>")
print(f"   </style> 數量：{style_count}（應為 1）")

# 確認舊顏色已清乾淨（排除 :root 定義行以外）
import re
old_colors = ['#c23030','#8c1f1f','#2b5fb3','#1a3a6e','#4a7fd4']
for c in old_colors:
    lines = [i+1 for i,l in enumerate(html.split('\n'))
             if c in l and '--crimson' not in l and '--cobalt' not in l and '--red' not in l and '--blue' not in l]
    if lines:
        print(f"   ⚠️  {c} 仍殘留在行：{lines}")
    else:
        print(f"   ✅  {c} 已清除")

if style_count != 1:
    print("❌  </style> 數量異常！請從備份還原"); sys.exit(1)

print("\n🎉  Blood & Steel 配色套用完成！")
print("    git add -A && git commit -m 'style: Blood & Steel palette' && git push")
