#!/usr/bin/env python3
"""
patch_universe_style.py
1. Stat bar → 白色實色，移除 elite 顏色邏輯
2. 冠軍標示 → 名字保持白色，加 ◆ CHAMPION badge
3. Weight class → 冷暖連續色譜

用法：python3 patch_universe_style.py ~/UFC/index.html
"""

import sys, pathlib, shutil, datetime, re

TARGET = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("~/UFC/index.html").expanduser()
if not TARGET.exists():
    print(f"❌  找不到檔案：{TARGET}"); sys.exit(1)

backup = TARGET.with_suffix(f".bak_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
shutil.copy2(TARGET, backup)
print(f"✅  備份：{backup}\n")

html = TARGET.read_text(encoding="utf-8")
original = html
results = []

def rep(old, new, label):
    global html
    count = html.count(old)
    if count == 0:
        results.append(f"⚠️   找不到（0次）：{label}")
        return
    html = html.replace(old, new)
    results.append(f"✅  替換 {count:2d} 次：{label}")

# ══════════════════════════════════════════════════════════════════════
# PATCH 1 — Stat bar → 白色實色
# ══════════════════════════════════════════════════════════════════════

# 不管現在是黑金漸層版還是舊版，用 regex 捕捉
# 找 .stat-bar-wrap 整個 block
old_wrap_re = r'\.stat-bar-wrap \{[^}]+\}'
new_wrap = """.stat-bar-wrap {
  width: 100%;
  height: 2px;
  background: var(--border-2);
  border-radius: 1px;
  margin-top: 3px;
  margin-bottom: 10px;
  overflow: hidden;
  position: relative;
}"""
m = re.search(old_wrap_re, html, re.DOTALL)
if m:
    html = html[:m.start()] + new_wrap + html[m.end():]
    results.append("✅  替換  1 次：.stat-bar-wrap → 2px 灰底")
else:
    results.append("⚠️   找不到：.stat-bar-wrap")

# .stat-bar — 不管是黑色遮罩版還是藍色版，統一換成白色實色
old_bar_re = r'\.stat-bar \{[^}]+\}'
new_bar = """.stat-bar {
  height: 100%;
  border-radius: 1px;
  background: var(--text-hi);
  transition: width 0.5s cubic-bezier(.4,0,.2,1);
  position: absolute;
  left: 0;
  top: 0;
}"""
m = re.search(old_bar_re, html, re.DOTALL)
if m:
    html = html[:m.start()] + new_bar + html[m.end():]
    results.append("✅  替換  1 次：.stat-bar → 白色實色，left:0")
else:
    results.append("⚠️   找不到：.stat-bar")

# 移除 .stat-bar.elite（不再需要）
old_elite_re = r'\n\.stat-bar\.elite \{[^}]+\}'
m = re.search(old_elite_re, html, re.DOTALL)
if m:
    html = html[:m.start()] + html[m.end():]
    results.append("✅  移除：.stat-bar.elite")
else:
    results.append("⚠️   找不到：.stat-bar.elite（可能已移除）")

# setBar JS — 不管是遮罩版還是舊版，用 regex 統一改
old_setbar_re = r'function setBar\(sid,bid,val,display,max,thr\)\{.*?\}'
new_setbar = """function setBar(sid,bid,val,display,max,thr){
    const sv=document.getElementById(sid), bv=document.getElementById(bid);
    sv.textContent=display;
    bv.style.width=pct(val,max);
    sv.classList.remove('elite');
    bv.classList.remove('elite');
  }"""
m = re.search(old_setbar_re, html, re.DOTALL)
if m:
    html = html[:m.start()] + new_setbar + html[m.end():]
    results.append("✅  替換  1 次：setBar() → 白色 bar，移除 elite")
else:
    results.append("⚠️   找不到：setBar()")

# ══════════════════════════════════════════════════════════════════════
# PATCH 2 — 冠軍標示：名字白色，加小 badge
# ══════════════════════════════════════════════════════════════════════

# CSS：移除 is-champ 金色，加 champion-badge 樣式
rep(
"#fighter-wc.is-champ { color: var(--champagne); }",
"""#fighter-wc.is-champ { color: var(--text-md); }
.champion-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-family: 'DM Mono', monospace;
  font-size: 8px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--champagne);
  margin-bottom: 10px;
}
.champion-badge::before {
  content: '◆';
  font-size: 7px;
}""",
"CSS: is-champ → 移除金色名字，加 badge 樣式"
)

# HTML：加 champion-badge div（在 fighter-status 前）
rep(
'      <div id="fighter-status"',
'      <div id="champion-badge" class="champion-badge" style="display:none"></div>\n      <div id="fighter-status"',
"HTML: 加 champion-badge div"
)

# JS：名字不再染金，改為控制 badge 顯示
rep(
"  nameEl.style.color = (f.tier==='champion'||f.tier==='ex_champion') ? 'var(--champagne)' : 'var(--text-hi)';",
"""  nameEl.style.color = 'var(--text-hi)';
  const badgeEl = document.getElementById('champion-badge');
  if(badgeEl){
    if(f.tier==='champion'){ badgeEl.textContent='CHAMPION'; badgeEl.style.display='inline-flex'; }
    else if(f.tier==='ex_champion'){ badgeEl.textContent='FORMER CHAMPION'; badgeEl.style.display='inline-flex'; }
    else { badgeEl.style.display='none'; }
  }""",
"JS: 名字白色，badge 控制冠軍顯示"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 3 — Weight class 冷暖色譜
# ══════════════════════════════════════════════════════════════════════

rep(
"""const WC_COLORS = {
  'Strawweight':'#ff69b4',"Women's Strawweight":'#ff69b4',
  'Flyweight':'#ff8c00',"Women's Flyweight":'#ff8c00',
  'Bantamweight':'#ffd700',"Women's Bantamweight":'#ffd700',
  'Featherweight':'#adff2f',"Women's Featherweight":'#adff2f',
  'Lightweight':'#00ced1','Welterweight':'#1e90ff',
  'Middleweight':'#9370db','Light Heavyweight':'#ff4500',
  'Heavyweight':'#dc143c','Unknown':'#888',
};""",
"""const WC_COLORS = {
  'Strawweight':'#7b9ec4',"Women's Strawweight":'#7b9ec4',
  'Flyweight':'#6a9fa8',"Women's Flyweight":'#6a9fa8',
  'Bantamweight':'#5a9e8a',"Women's Bantamweight":'#5a9e8a',
  'Featherweight':'#6a9e6a',"Women's Featherweight":'#6a9e6a',
  'Lightweight':'#8fa855','Welterweight':'#b8a040',
  'Middleweight':'#c4824a','Light Heavyweight':'#b85e45',
  'Heavyweight':'#a03838','Unknown':'#666',
};""",
"WC_COLORS → 冷暖連續色譜"
)

# ══════════════════════════════════════════════════════════════════════
# 結果輸出
# ══════════════════════════════════════════════════════════════════════
print("\n".join(results))

if html == original:
    print("\n⚠️   沒有任何改動"); sys.exit(1)

TARGET.write_text(html, encoding="utf-8")
print(f"\n✅  已寫回：{TARGET}")

style_count = html.count("</style>")
print(f"   </style> 數量：{style_count}（應為 1）")
if style_count != 1:
    print("❌  </style> 異常！請從備份還原"); sys.exit(1)

print("\n🎉  完成！")
print("    git add -A && git commit -m 'style: white stat bar, champion badge, wc color spectrum' && git push")
