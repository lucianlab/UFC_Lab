#!/usr/bin/env python3
"""
patch_tabs_bars.py
1. 頂部導航：按鈕 → tab 底線風格
2. Finish Probability bars → dim 顏色
3. Model Confidence label → 統一灰色，移除漸變色邏輯

用法：python3 patch_tabs_bars.py ~/UFC/index.html
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

def rep(old, new, label):
    global html
    count = html.count(old)
    if count == 0:
        results.append(f"⚠️   找不到（0次）：{label}")
        return
    html = html.replace(old, new)
    results.append(f"✅  替換 {count:2d} 次：{label}")

# ══════════════════════════════════════════════════════════════════════
# PATCH 1 — 頂部導航改成 tab 底線風格
# ══════════════════════════════════════════════════════════════════════

rep(
"""#mode-toggle {
  display: flex;
  gap: 0;
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  flex-shrink: 0;
}

.mode-btn {
  padding: 0 18px;
  height: 28px;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  cursor: pointer;
  border: 1px solid var(--border-2);
  transition: all 0.18s ease;
  color: var(--text-lo);
  background: transparent;
  position: relative;
}

.mode-btn:first-child {
  border-radius: 3px 0 0 3px;
  border-right: none;
}
.mode-btn:not(:first-child):not(:last-child) {
  border-right: none;
}
.mode-btn:last-child {
  border-radius: 0 3px 3px 0;
}

.mode-btn.active {
  color: var(--champagne);
  background: rgba(201,169,110,0.07);
  border-color: var(--champagne-dim);
  border-color: var(--border-2);
}

/* btn-vs uses base active style */

.mode-btn:hover:not(.active) {
  color: var(--text-md);
  background: var(--surface-2);
}""",

"""#mode-toggle {
  display: flex;
  gap: 0;
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  flex-shrink: 0;
  height: 100%;
  align-items: stretch;
}

.mode-btn {
  padding: 0 20px;
  height: 100%;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  cursor: pointer;
  border: none;
  border-bottom: 2px solid transparent;
  transition: color 0.18s ease, border-color 0.18s ease;
  color: var(--text-lo);
  background: transparent;
  position: relative;
  border-radius: 0;
}

.mode-btn.active {
  color: var(--champagne);
  border-bottom-color: var(--champagne);
}

.mode-btn:hover:not(.active) {
  color: var(--text-md);
  border-bottom-color: var(--border-2);
}""",
"nav tab 底線風格"
)

# 移除重複的 border-right 規則（已不需要）
rep(
"/* ─── 3-button mode toggle ─── */\n.mode-btn:not(:first-child):not(:last-child){ border-right:none; border-radius:0; }",
"/* ─── nav tabs ─── */",
"移除舊 border-right 規則"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 2 — Finish Probability bars 全部改成 dim
# ══════════════════════════════════════════════════════════════════════

rep(
".fo-fill.ko  { background:var(--crimson-2); }",
".fo-fill.ko  { background:var(--crimson); }",
"fo-fill.ko → crimson dim"
)

rep(
".fo-fill.sub { background:#2a5090; }",
".fo-fill.sub { background:var(--cobalt); }",
"fo-fill.sub → cobalt dim"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 3 — Confidence label 統一灰色，移除動態漸變色
# ══════════════════════════════════════════════════════════════════════

rep(
"""        confLabelEl.textContent = data.confidence_label;
        const cf = Math.min(1, Math.max(0, data.confidence || 0));
        const cr = Math.round(26  + (201-26)  * cf);
        const cg = Math.round(24  + (169-24)  * cf);
        const cb = Math.round(36  + (110-36)  * cf);
        confLabelEl.style.color = `rgb(${cr},${cg},${cb})`;""",
"""        confLabelEl.textContent = data.confidence_label;
        confLabelEl.style.color = 'var(--text-lo)';""",
"confidence label → 固定灰色"
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
print("    git add -A && git commit -m 'style: tab nav, dim bars, grey confidence label' && git push")
