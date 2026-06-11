"""
patch_main_gemini.py — 在 main.py 加入 Gemini 2.0 Flash narrative endpoint
用法：python3 ~/UFC/patch_main_gemini.py ~/UFC/main.py
"""
import sys, shutil
from pathlib import Path
from datetime import datetime

if len(sys.argv) < 2:
    print("用法: python3 patch_main_gemini.py ~/UFC/main.py")
    sys.exit(1)

target = Path(sys.argv[1])
backup = target.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
shutil.copy2(target, backup)
print(f"備份: {backup}")

c = target.read_text(encoding="utf-8")

GEMINI_CODE = '''

# ══════════════════════════════════════════════════════
#  Gemini 2.0 Flash — VS Narrative
# ══════════════════════════════════════════════════════
import os

# SHAP feature → 可讀描述
SHAP_LABELS = {
    "delta_win_rate":          "win rate gap",
    "delta_ko_rate":           "KO rate gap",
    "delta_sub_rate":          "submission rate gap",
    "delta_finish_rate":       "finishing rate gap",
    "delta_sig_per_r":         "striking volume gap",
    "delta_sig_acc":           "striking accuracy gap",
    "delta_str_def":           "striking defense gap",
    "delta_td_per_r":          "takedown volume gap",
    "delta_td_avg_acc":        "takedown accuracy gap",
    "delta_td_def":            "takedown defense gap",
    "delta_ctrl_per_r":        "control time gap",
    "delta_sub_per_r":         "submission attempt gap",
    "delta_ground_pct":        "ground game gap",
    "delta_gas_tank":          "cardio gap",
    "delta_sos":               "strength of schedule gap",
    "delta_recent_win_rate":   "recent form gap",
    "delta_recent_sig_per_r":  "recent striking gap",
    "delta_age":               "age gap",
    "delta_reach":             "reach gap",
    "f1_td_threat_vs_f2_def":  "red's takedown threat vs blue's defense",
    "f2_td_threat_vs_f1_def":  "blue's takedown threat vs red's defense",
    "f1_str_threat_vs_f2_def": "red's striking threat vs blue's defense",
    "f2_str_threat_vs_f1_def": "blue's striking threat vs red's defense",
    "f1_sub_vs_f2_ctrl":       "red's submission vs blue's control",
    "f1_ko_vs_f2_absorb":      "red's KO power vs blue's durability",
}

def _shap_to_english(features: list, red_name: str, blue_name: str) -> str:
    parts = []
    for f in features[:3]:
        feat  = f.get("feature", "")
        shap  = f.get("shap", 0)
        label = SHAP_LABELS.get(feat, feat.replace("delta_","").replace("_"," "))
        favor = red_name.split()[-1] if shap > 0 else blue_name.split()[-1]
        parts.append(f"{favor} leads in {label}")
    return "; ".join(parts)

def _get_archetype(name: str) -> str:
    f = _by_name.get(name, {})
    return f.get("archetype", "Balanced")

def _fmt_stats(name: str) -> str:
    f = _by_name.get(name, {})
    sig  = f.get("sig_per_r", 0) or 0
    td   = f.get("td_per_r",  0) or 0
    ko   = round((f.get("ko_rate",  0) or 0) * 100)
    sub  = round((f.get("sub_rate", 0) or 0) * 100)
    sdef = round((f.get("str_def",  0) or 0) * 100)
    tdef = round((f.get("td_def",   0) or 0) * 100)
    return (f"{sig:.1f} sig.strikes/rd, {td:.1f} TDs/rd, "
            f"{ko}% KO rate, {sub}% sub rate, "
            f"{sdef}% str.def, {tdef}% TD def")

class NarrativeInput(BaseModel):
    red:              str
    blue:             str
    red_win_prob:     float
    method:           str
    pred_round:       int
    confidence_label: str
    top_features:     list = []

@app.post("/api/vs_narrative")
async def vs_narrative(inp: NarrativeInput):
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return {"narrative": ""}

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        winner     = inp.red  if inp.red_win_prob >= 0.5 else inp.blue
        loser      = inp.blue if inp.red_win_prob >= 0.5 else inp.red
        win_pct    = round(inp.red_win_prob * 100) if inp.red_win_prob >= 0.5 else round((1-inp.red_win_prob)*100)
        method_str = inp.method
        round_str  = f"R{inp.pred_round}" if inp.method != "Decision" else "decision"
        shap_str   = _shap_to_english(inp.top_features, inp.red, inp.blue)
        red_arch   = _get_archetype(inp.red)
        blue_arch  = _get_archetype(inp.blue)
        red_stats  = _fmt_stats(inp.red)
        blue_stats = _fmt_stats(inp.blue)

        prompt = f"""You are a UFC tactical analyst. Based ONLY on the statistics below, write exactly 2 sentences of fight analysis. Do NOT mention specific past fights or results. Only analyze based on the numbers provided.

FIGHTERS:
- {inp.red} ({red_arch}): {red_stats}
- {inp.blue} ({blue_arch}): {blue_stats}

MODEL PREDICTION: {winner} wins {win_pct}% probability via {method_str} ({round_str})
CONFIDENCE: {inp.confidence_label}
KEY FACTORS: {shap_str}

Write 2 concise sentences (max 30 words each) analyzing the tactical matchup and why the model favors {winner}. Be specific about styles and stats. No hedging phrases like "could" or "might"."""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=120,
                temperature=0.4,
            )
        )
        text = response.text.strip()
        # 確保不超過兩句
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        result = ". ".join(sentences[:2]) + ("." if sentences else "")
        return {"narrative": result}

    except Exception as e:
        print(f"Gemini error: {e}")
        return {"narrative": ""}
'''

if "vs_narrative" not in c:
    c = c.rstrip() + "\n" + GEMINI_CODE + "\n"
    target.write_text(c, encoding="utf-8")
    print("✅ Gemini endpoint 加入成功")
    print("\n下一步:")
    print("  1. pip install google-generativeai --break-system-packages")
    print("  2. 在 requirements.txt 加入 google-generativeai")
    print("  3. Railway 環境變數加入 GOOGLE_API_KEY")
    print("  4. git add main.py requirements.txt && git push")
else:
    print("⏭  已存在，跳過")
