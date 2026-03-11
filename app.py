import joblib
import pandas as pd
import os
from fastapi import FastAPI
from pydantic import BaseModel

base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "pipeline.joblib")

app = FastAPI(title="Loan Default Calculator API")

pipe = joblib.load(file_path)

class LoanInput(BaseModel):
    data: dict  # 先用 dict 接，最省事

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: LoanInput):
    X = pd.DataFrame([payload.data])
    pd_score = float(pipe.predict_proba(X)[0, 1])
    band = "low" if pd_score < 0.15 else ("medium" if pd_score < 0.30 else "high")
    return {"pd": pd_score, "risk_band": band}
