#!/usr/bin/env python3
"""
patch_blueprint_layout2.py
1. Reach 移到左欄底部（Physical section）
2. 右欄：Boxing×2 + Jiu-Jitsu×3，無 Reach
3. Tier Reference：單欄 8 行，填滿右欄剩餘高度

用法：python3 patch_blueprint_layout2.py ~/UFC/index.html
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
# PATCH 1 — B_LEFT：加 Reach 到最底
# PATCH 2 — B_RIGHT：移除 Reach
# ══════════════════════════════════════════════════════════════════════
rep(
"""const B_LEFT = [
  {id:'pct_striking_power',    label:'KO Power',    sec:'Boxing'},
  {id:'pct_striking_accuracy', label:'Accuracy',    sec:null},
  {id:'pct_leg_kicks',         label:'Leg Kicks',   sec:'Muay Thai'},
  {id:'pct_clinch',            label:'Clinch',      sec:null},
  {id:'pct_body',              label:'Body',        sec:null},
  {id:'pct_td_frequency',      label:'TD Frequency',sec:'Wrestling'},
  {id:'pct_td_accuracy',       label:'TD Accuracy', sec:null},
  {id:'pct_td_defense',        label:'TD Defense',  sec:null},
];
const B_RIGHT = [
  {id:'pct_striking_volume',   label:'Volume',      sec:'Boxing'},
  {id:'pct_striking_defense',  label:'Defense',     sec:null},
  {id:'pct_submission',        label:'Submission',  sec:'Jiu-Jitsu'},
  {id:'pct_control',           label:'Control',     sec:null},
  {id:'pct_ground_pound',      label:'Grd & Pound', sec:null},
  {id:'pct_reach',             label:'Reach',       sec:'Physical'},""",
"""const B_LEFT = [
  {id:'pct_striking_power',    label:'KO Power',    sec:'Boxing'},
  {id:'pct_striking_accuracy', label:'Accuracy',    sec:null},
  {id:'pct_leg_kicks',         label:'Leg Kicks',   sec:'Muay Thai'},
  {id:'pct_clinch',            label:'Clinch',      sec:null},
  {id:'pct_body',              label:'Body',        sec:null},
  {id:'pct_td_frequency',      label:'TD Frequency',sec:'Wrestling'},
  {id:'pct_td_accuracy',       label:'TD Accuracy', sec:null},
  {id:'pct_td_defense',        label:'TD Defense',  sec:null},
  {id:'pct_reach',             label:'Reach',       sec:'Physical'},
];
const B_RIGHT = [
  {id:'pct_striking_volume',   label:'Volume',      sec:'Boxing'},
  {id:'pct_striking_defense',  label:'Defense',     sec:null},
  {id:'pct_submission',        label:'Submission',  sec:'Jiu-Jitsu'},
  {id:'pct_control',           label:'Control',     sec:null},
  {id:'pct_ground_pound',      label:'Grd & Pound', sec:null},""",
"B_LEFT 加 Reach，B_RIGHT 移除 Reach"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 3 — Tier Reference CSS：單欄 8 行
# ══════════════════════════════════════════════════════════════════════
rep(
""".b-tier-grid{
  display:grid; grid-template-columns:1fr 1fr; gap:0 10px;
  flex:1; min-height:0;
}
.b-tier-col{ display:flex; flex-direction:column; justify-content:space-between; gap:3px; }
.b-tier-row{ display:grid; grid-template-columns:26px 1fr; align-items:baseline; gap:3px; }
.b-tier-lbl{ font-family:'DM Mono',monospace; font-size:7.5px; font-weight:500; }
.b-tier-desc{ font-family:'Barlow',sans-serif; font-size:7.5px; color:var(--text-lo); line-height:1.3; }""",
""".b-tier-grid{
  display:flex; flex-direction:column;
  flex:1; min-height:0; gap:0;
}
.b-tier-col{ display:contents; }
.b-tier-row{
  flex:1; display:grid; grid-template-columns:28px 1fr;
  align-items:center; gap:4px;
  border-top:1px solid var(--border);
  padding:0 2px;
}
.b-tier-row:first-child{ border-top:none; }
.b-tier-lbl{ font-family:'DM Mono',monospace; font-size:7.5px; font-weight:600; color:var(--text-lo); }
.b-tier-desc{ font-family:'Barlow',sans-serif; font-size:7.5px; color:var(--text-lo); line-height:1.3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }""",
"CSS: Tier Reference 單欄 8 行"
)

# ══════════════════════════════════════════════════════════════════════
# PATCH 4 — TIER_LEFT/RIGHT JS：統一 text-lo，保留 desc
# ══════════════════════════════════════════════════════════════════════
rep(
"""  const TIER_LEFT = [
    {t:'S',  color:'#FFD700',          desc:'Gods of the sport. The one. The only.'},
    {t:'A+', color:'var(--champagne)', desc:'Legends of their era. Untouchable at their peak.'},
    {t:'A',  color:'var(--champagne)', desc:'Dynasty builders. Reigned supreme.'},
    {t:'B+', color:'var(--text-md)',   desc:'Champions who defended. Proved gold is earned, not given.'},
  ];
  const TIER_RIGHT = [
    {t:'B',  color:'var(--text-md)',   desc:'Once a champion, always a champion.'},
    {t:'C+', color:'var(--text-lo)',   desc:'Greatness needs no gold to be proven.'},
    {t:'C-D',color:'var(--text-lo)',   desc:'Historically ranked Top 5 / 10 / 15.'},
    {t:'E',  color:'var(--text-lo)',   desc:'World-class. The UFC is the sharkest tank on earth.'},
  ];""",
"""  const TIER_LEFT = [
    {t:'S',   color:'var(--text-lo)', desc:'Gods of the sport. The one. The only.'},
    {t:'A+',  color:'var(--text-lo)', desc:'Legends of their era. Untouchable at their peak.'},
    {t:'A',   color:'var(--text-lo)', desc:'Dynasty builders. Reigned supreme.'},
    {t:'B+',  color:'var(--text-lo)', desc:'Champions who defended. Proved gold is earned.'},
  ];
  const TIER_RIGHT = [
    {t:'B',   color:'var(--text-lo)', desc:'Once a champion, always a champion.'},
    {t:'C+',  color:'var(--text-lo)', desc:'Greatness needs no gold to be proven.'},
    {t:'C-D', color:'var(--text-lo)', desc:'Historically ranked Top 5 / 10 / 15.'},
    {t:'E',   color:'var(--text-lo)', desc:'World-class. The UFC is the sharkest tank.'},
  ];""",
"JS: Tier 統一 text-lo，desc 保留但縮短"
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
print("    git add -A && git commit -m 'style: reach to left col, tier ref single column 8 rows' && git push")
