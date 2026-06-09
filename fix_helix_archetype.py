from pathlib import Path
import json

f = Path('/home/angu/UFC/build_helix.py')
c = f.read_text(encoding='utf-8')

# 1. 加入讀取 fighters.json（在 fa_dict 之後）
old_block = """    fa_dict = {row["name"]: row.to_dict() for _, row in fa_df.iterrows()}
    print(f"    選手屬性: {len(fa_dict):,} 人")
except Exception as e:
    print(f"    WARNING: {e}")
    fa_dict = {}"""

new_block = """    fa_dict = {row["name"]: row.to_dict() for _, row in fa_df.iterrows()}
    print(f"    選手屬性: {len(fa_dict):,} 人")
except Exception as e:
    print(f"    WARNING: {e}")
    fa_dict = {}

# 讀 fighters.json 取得 archetype（PCA 八象限）
import json as _json
_fj_path = BASE / "data" / "fighters.json"
archetype_dict = {}
try:
    _fj = _json.load(open(_fj_path, encoding='utf-8'))
    archetype_dict = {f['name']: f.get('archetype','') for f in _fj}
    print(f"    archetype: {sum(1 for v in archetype_dict.values() if v):,} 人有資料")
except Exception as e:
    print(f"    WARNING archetype: {e}")"""

if old_block in c:
    c = c.replace(old_block, new_block)
    print("fighters.json block: OK")
else:
    print("fighters.json block: NOT FOUND")

# 2. 替換 get_style 函數
old_style = """def get_style(name):
    fa  = fa_dict.get(name,{})
    td  = float(fa.get("td_avg",0) or 0)
    sub = float(fa.get("sub_avg",0) or 0)
    spl = float(fa.get("splm",0) or 0)
    acc = float(fa.get("str_acc",0) or 0)
    if sub>1.5:              return "Submission"
    if td>3.0:               return "Wrestler"
    if td>1.5 and sub>0.5:   return "Grappler"
    if spl>5.0 and acc>50:   return "Striker"
    if spl>4.0:              return "Brawler"
    return "Balanced" """

new_style = """def get_style(name):
    arch = archetype_dict.get(name, '')
    if arch: return arch
    # fallback if not in fighters.json
    fa  = fa_dict.get(name,{})
    td  = float(fa.get("td_avg",0) or 0)
    sub = float(fa.get("sub_avg",0) or 0)
    spl = float(fa.get("splm",0) or 0)
    acc = float(fa.get("str_acc",0) or 0)
    if sub>1.5:              return "Submission Hunter"
    if td>3.0:               return "Chain Controller"
    if td>1.5 and sub>0.5:   return "Submission Hunter"
    if spl>5.0 and acc>50:   return "Pace Setter"
    if spl>4.0:              return "Pressure Swarm"
    return "Range Technician" """

if old_style in c:
    c = c.replace(old_style, new_style)
    print("get_style: OK")
else:
    print("get_style: NOT FOUND - trying stripped version")
    # try without trailing space
    old_style2 = old_style.rstrip()
    if old_style2 in c:
        c = c.replace(old_style2, new_style)
        print("get_style (stripped): OK")
    else:
        # find and show actual content
        idx = c.find('def get_style')
        print("Actual content:")
        print(repr(c[idx:idx+300]))

f.write_text(c, encoding='utf-8')
print("done")
