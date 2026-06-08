#!/usr/bin/env python3
"""
patch_blueprint_cols.py
左欄：Muay Thai×3 + Wrestling×3 + Jiu-Jitsu×3
右欄：Boxing×5（含 Reach）+ Tier Reference

用法：python3 patch_blueprint_cols.py ~/UFC/index.html
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
  {id:'pct_reach',             label:'Reach',       sec:'Physical'},
];
const B_RIGHT = [
  {id:'pct_striking_volume',   label:'Volume',      sec:'Boxing'},
  {id:'pct_striking_defense',  label:'Defense',     sec:null},
  {id:'pct_submission',        label:'Submission',  sec:'Jiu-Jitsu'},
  {id:'pct_control',           label:'Control',     sec:null},
  {id:'pct_ground_pound',      label:'Grd & Pound', sec:null},
];""",
"""const B_LEFT = [
  {id:'pct_leg_kicks',         label:'Leg Kicks',   sec:'Muay Thai'},
  {id:'pct_clinch',            label:'Clinch',      sec:null},
  {id:'pct_body',              label:'Body',        sec:null},
  {id:'pct_td_frequency',      label:'TD Frequency',sec:'Wrestling'},
  {id:'pct_td_accuracy',       label:'TD Accuracy', sec:null},
  {id:'pct_td_defense',        label:'TD Defense',  sec:null},
  {id:'pct_submission',        label:'Submission',  sec:'Jiu-Jitsu'},
  {id:'pct_control',           label:'Control',     sec:null},
  {id:'pct_ground_pound',      label:'Grd & Pound', sec:null},
];
const B_RIGHT = [
  {id:'pct_striking_power',    label:'KO Power',    sec:'Boxing'},
  {id:'pct_striking_accuracy', label:'Accuracy',    sec:null},
  {id:'pct_striking_volume',   label:'Volume',      sec:null},
  {id:'pct_striking_defense',  label:'Defense',     sec:null},
  {id:'pct_reach',             label:'Reach',       sec:null},
];""",
"B_LEFT/B_RIGHT 重新分組"
)

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
print("    git add -A && git commit -m 'style: blueprint regroup cols muaythai+wrestling+jj / boxing+reach+tier' && git push")
