from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 載入 pipeline 產出的資料（三軸座標 / 相似選手 / 分級 / 每回合統計）
# Load the pipeline output (axis coords / similar fighters / tier / per-round stats)
DATA_PATH = Path(__file__).parent / "data" / "fighters.json"
with open(DATA_PATH, encoding="utf-8") as f:
    FIGHTERS = json.load(f)

_by_name = {f["name"]: f for f in FIGHTERS}


@app.get("/api/fighters")
def get_fighters():
    return FIGHTERS


@app.get("/api/fighters/{name}")
def get_fighter(name: str):
    fighter = _by_name.get(name)
    if fighter is None:
        return {"error": "Fighter not found"}
    return fighter
