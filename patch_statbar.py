#!/usr/bin/env python3
"""
patch_statbar.py
把 Universe sidebar 的 Style Profile bar 改成黑金漸層效果。
用法：python3 patch_statbar.py ~/UFC/index.html
"""

import sys, re, pathlib, shutil, datetime

# ── 參數 ──────────────────────────────────────────────────────────────
TARGET = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("~/UFC/index.html").expanduser()

if not TARGET.exists():
    print(f"❌  找不到檔案：{TARGET}")
    sys.exit(1)

# 備份
backup = TARGET.with_suffix(f".bak_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
shutil.copy2(TARGET, backup)
print(f"✅  備份完成：{backup}")

html = TARGET.read_text(encoding="utf-8")
original = html  # 用來最後比對是否有改動

# ═══════════════════════════════════════════════════════════════════════
# PATCH 1 — .stat-bar-wrap  →  黑金漸層底層
# ═══════════════════════════════════════════════════════════════════════
OLD_WRAP = """.stat-bar-wrap {
  width: 100%;
  height: 1px;
  background: var(--border);
  border-radius: 1px;
  margin-top: 3px;
  margin-bottom: 10px;
}"""

NEW_WRAP = """.stat-bar-wrap {
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg, #09090d 0%, #2a1f0a 45%, #c9a96e 100%);
  border-radius: 2px;
  margin-top: 3px;
  margin-bottom: 10px;
  overflow: hidden;
  position: relative;
}"""

if OLD_WRAP in html:
    html = html.replace(OLD_WRAP, NEW_WRAP, 1)
    print("✅  PATCH 1 完成：.stat-bar-wrap 改為黑金漸層")
else:
    print("❌  PATCH 1 失敗：找不到 .stat-bar-wrap 原始內容，請手動確認")

# ═══════════════════════════════════════════════════════════════════════
# PATCH 2 — .stat-bar + .stat-bar.elite  →  右側黑色遮罩
# ═══════════════════════════════════════════════════════════════════════
OLD_BAR = """.stat-bar {
  height: 1px;
  border-radius: 1px;
  background: var(--cobalt-2);
  transition: width 0.5s cubic-bezier(.4,0,.2,1);
}

.stat-bar.elite {
  background: var(--crimson-2);
}"""

NEW_BAR = """.stat-bar {
  height: 100%;
  border-radius: 0;
  background: var(--obsidian);
  transition: width 0.5s cubic-bezier(.4,0,.2,1);
  position: absolute;
  right: 0;
  top: 0;
}"""

if OLD_BAR in html:
    html = html.replace(OLD_BAR, NEW_BAR, 1)
    print("✅  PATCH 2 完成：.stat-bar 改為右側黑色遮罩，移除 .elite 顏色")
else:
    print("❌  PATCH 2 失敗：找不到 .stat-bar 原始內容，請手動確認")

# ═══════════════════════════════════════════════════════════════════════
# PATCH 3 — setBar()  →  width 改為「剩餘黑色」比例
# ═══════════════════════════════════════════════════════════════════════
OLD_SETBAR = """  function setBar(sid,bid,val,display,max,thr){
    const sv=document.getElementById(sid), bv=document.getElementById(bid);
    sv.textContent=display; bv.style.width=pct(val,max);
    sv.classList.toggle('elite',val>=thr); bv.classList.toggle('elite',val>=thr);
  }"""

NEW_SETBAR = """  function setBar(sid,bid,val,display,max,thr){
    const sv=document.getElementById(sid), bv=document.getElementById(bid);
    sv.textContent=display;
    const p=Math.min(100,Math.max(0,(val/max)*100));
    bv.style.width=(100-p)+'%';
    sv.classList.toggle('elite',val>=thr);
    bv.classList.remove('elite');
  }"""

if OLD_SETBAR in html:
    html = html.replace(OLD_SETBAR, NEW_SETBAR, 1)
    print("✅  PATCH 3 完成：setBar() 改為右側遮罩寬度邏輯")
else:
    print("❌  PATCH 3 失敗：找不到 setBar() 原始內容，請手動確認")

# ═══════════════════════════════════════════════════════════════════════
# 寫回 & 驗證
# ═══════════════════════════════════════════════════════════════════════
if html == original:
    print("\n⚠️   檔案沒有任何改動，請確認上面哪個 PATCH 失敗了")
    sys.exit(1)

TARGET.write_text(html, encoding="utf-8")
print(f"\n✅  已寫回：{TARGET}")

# 快速驗證
style_count = html.count("</style>")
scripts = [m for m in html.split("<script") if "boot()" in m or "setBar" in m]
print(f"   </style> 數量：{style_count}（應為 1）")
print(f"   包含 setBar 的 script block 數：{len(scripts)}（應為 1）")

if style_count != 1:
    print("❌  </style> 數量異常！請從備份還原後手動修改")
    sys.exit(1)

print("\n🎉  全部完成！執行 git add -A && git commit -m 'style: stat-bar black-gold gradient' && git push 即可部署")
