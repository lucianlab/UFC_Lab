#!/usr/bin/env python3
"""
patch_pip_color.py — pip 顏色改成灰→金，移除紅色
用法：python3 patch_pip_color.py ~/UFC/index.html
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

# ── bUpdateBox ──
old_update = """  const elite = val >= 8;
  const fillColor = elite ? 'var(--crimson-2)' : 'var(--champagne)';
  const glowColor = elite ? 'rgba(184,48,64,0.45)' : 'rgba(201,169,110,0.3)';
  for(let i=1;i<=10;i++){
    const pip=document.getElementById(`pip-${featId}-${i}`);
    if(!pip) continue;
    pip.style.background = i<=val ? fillColor : 'var(--border)';
    pip.style.boxShadow  = (i<=val&&elite) ? `0 0 3px ${glowColor}` : 'none';
  }
  const valEl=document.getElementById('val-'+featId);
  if(valEl){
    valEl.textContent=val;
    valEl.style.color = elite ? 'var(--crimson-2)' : 'var(--champagne)';
  }
  const box=document.getElementById('attr-'+featId);
  if(box) box.style.borderColor = elite ? 'rgba(184,48,64,0.35)' : 'var(--border)';"""

new_update = """  const elite = val >= 8;
  const fillColor = elite ? 'var(--champagne)' : 'var(--text-lo)';
  for(let i=1;i<=10;i++){
    const pip=document.getElementById(`pip-${featId}-${i}`);
    if(!pip) continue;
    pip.style.background = i<=val ? fillColor : 'var(--surface-3)';
    pip.style.boxShadow  = 'none';
  }
  const valEl=document.getElementById('val-'+featId);
  if(valEl){
    valEl.textContent=val;
    valEl.style.color = elite ? 'var(--champagne)' : 'var(--text-lo)';
  }
  const box=document.getElementById('attr-'+featId);
  if(box) box.style.borderColor = 'var(--border)';"""

# ── bFlashPip ──
old_flash = """  const elite = targetVal >= 8;
  const flashColor = elite ? 'var(--crimson-2)' : 'var(--champagne)';
  for(let i=1;i<=10;i++){
    const pip=document.getElementById(`pip-${featId}-${i}`);
    if(!pip) continue;
    pip.style.background = i<=targetVal ? flashColor : 'var(--border)';
    pip.style.boxShadow  = (i<=targetVal&&elite) ? '0 0 4px rgba(184,48,64,0.5)' : 'none';
  }
  const valEl=document.getElementById('val-'+featId);
  if(valEl){ valEl.textContent=targetVal; valEl.style.color=flashColor; }"""

new_flash = """  const elite = targetVal >= 8;
  const flashColor = elite ? 'var(--champagne)' : 'var(--text-lo)';
  for(let i=1;i<=10;i++){
    const pip=document.getElementById(`pip-${featId}-${i}`);
    if(!pip) continue;
    pip.style.background = i<=targetVal ? flashColor : 'var(--surface-3)';
    pip.style.boxShadow  = 'none';
  }
  const valEl=document.getElementById('val-'+featId);
  if(valEl){ valEl.textContent=targetVal; valEl.style.color=flashColor; }"""

for old, new, label in [(old_update, new_update, "bUpdateBox"), (old_flash, new_flash, "bFlashPip")]:
    if old in html:
        html = html.replace(old, new)
        print(f"✅  替換：{label}")
    else:
        print(f"⚠️   找不到：{label}")

if html == original:
    print("⚠️   沒有改動"); sys.exit(1)

TARGET.write_text(html, encoding="utf-8")
print(f"✅  已寫回：{TARGET}")
print("\n🎉  完成！")
print("    git add -A && git commit -m 'style: pip color grey->gold at 8+, remove red' && git push")
