from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("data/clean/fighters_all.csv")

@app.get("/api/fighters")
def get_fighters():
    return df.fillna(0).to_dict(orient="records")

@app.get("/api/fighters/{name}")
def get_fighter(name: str):
    fighter = df[df["name"] == name]
    if fighter.empty:
        return {"error": "Fighter not found"}
    return fighter.fillna(0).to_dict(orient="records")[0]
