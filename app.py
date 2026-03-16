"""
app.py — FastAPI runtime for RiskLens.
Loads models/artifacts.pkl (built by train.py) and serves the API.
"""

import os, pickle, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict
from pgmpy.inference import VariableElimination

# ── Load artifacts ─────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "models")
DATA_DIR  = os.path.join(BASE, "data")

with open(os.path.join(MODEL_DIR, "artifacts.pkl"), "rb") as f:
    arts = pickle.load(f)

nb_models        = arts["nb_models"]
bn_models        = arts["bn_models"]
encoders         = arts["encoders"]
DISEASE_FEATURES = arts["disease_features"]
ORDINAL_FEATURES = arts["ordinal_features"]
DISEASES         = arts["diseases"]

# Build inference engines at startup
bn_infer = {d: VariableElimination(bn_models[d]) for d in DISEASES}

recs = pd.read_excel(os.path.join(DATA_DIR, "recommendationsdoc_precise_detailed.xlsx"))

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RiskLens Health Screener API",
    description="Naive Bayes + Bayesian Network disease risk prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _age_group(age: int) -> str:
    if age <= 30:  return "Young"
    if age <= 45:  return "Adult"
    if age <= 60:  return "Middle"
    return "Senior"

def _encode(raw: dict) -> dict:
    encoded = {}
    encoded["AgeGroup"] = int(encoders["AgeGroup"].transform([_age_group(raw["age"])])[0])
    for col, val in raw.items():
        if col == "age":
            continue
        if col in ORDINAL_FEATURES:
            m = encoders[col]
            if val not in m:
                raise HTTPException(400, f"Invalid '{val}' for {col}. Options: {list(m.keys())}")
            encoded[col] = int(m[val])
        else:
            le = encoders[col]
            if val not in le.classes_:
                raise HTTPException(400, f"Invalid '{val}' for {col}. Options: {le.classes_.tolist()}")
            encoded[col] = int(le.transform([val])[0])
    return encoded

def _band(risk: float) -> str:
    if risk <= 20: return "0-20"
    if risk <= 40: return "21-40"
    if risk <= 60: return "41-60"
    if risk <= 80: return "61-80"
    return "81-100"

def _rec(disease: str, band: str) -> str:
    row = recs[(recs["Disease"] == disease) & (recs["RiskRangePercent"] == band)]
    return str(row["DoctorRecommendation"].values[0]) if not row.empty else "Consult a doctor."

# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    age:                    int = Field(..., ge=1, le=120, example=45)
    Gender:                 str = Field(..., example="Male")
    Smoking:                str = Field(..., example="No")
    Alcohol:                str = Field(..., example="No")
    PhysicalActivity:       str = Field(..., example="Moderate")
    DietQuality:            str = Field(..., example="Average")
    BloodPressure:          str = Field(..., example="Normal")
    FrequentUrination:      str = Field(..., example="No")
    ExcessiveThirst:        str = Field(..., example="No")
    FamilyHistoryDiabetes:  str = Field(..., example="No")
    Fatigue:                str = Field(..., example="No")
    ChestPain:              str = Field(..., example="No")
    FamilyHistoryHeart:     str = Field(..., example="No")
    SwellingAnkles:         str = Field(..., example="No")
    Wheezing:               str = Field(..., example="No")
    Breathlessness:         str = Field(..., example="No")
    Cough:                  str = Field(..., example="No")
    PaleSkin:               str = Field(..., example="No")
    WeightLoss:             str = Field(..., example="No")
    Dizziness:              str = Field(..., example="No")
    SugarLevel:             str = Field(..., example="Normal")

class DiseaseResult(BaseModel):
    disease:              str
    risk_percent:         float
    risk_band:            str
    recommendation:       str
    contributing_factors: Dict[str, float]

class PredictResponse(BaseModel):
    age_group: str
    results:   list[DiseaseResult]

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "RiskLens API is live 🟢"}

@app.get("/options")
def get_options():
    """Returns all valid values for every input field."""
    opts = {}
    for col in ORDINAL_FEATURES:
        opts[col] = ORDINAL_FEATURES[col]
    for col, enc in encoders.items():
        if col == "AgeGroup":
            continue
        if hasattr(enc, "classes_"):
            opts[col] = enc.classes_.tolist()
    return opts

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    encoded = _encode(req.model_dump())
    results = []

    for d in DISEASES:
        feats = DISEASE_FEATURES[d]

        # Naive Bayes
        X_nb = pd.DataFrame([{f: encoded[f] for f in feats}])
        p_nb = float(nb_models[d].predict_proba(X_nb)[0][1])

        # Bayesian Network
        ev = {f: int(encoded[f]) for f in feats}
        ev["AgeGroup"] = int(encoded["AgeGroup"])
        p_bn = float(bn_infer[d].query([d], evidence=ev).values[1])

        # Ensemble
        risk = round((0.6 * p_nb + 0.4 * p_bn) * 100, 2)
        band = _band(risk)

        # Contributing factors
        base_p = float(bn_infer[d].query([d], evidence={"AgeGroup": int(encoded["AgeGroup"])}).values[1])
        factors = {}
        for f in feats:
            q = bn_infer[d].query([d], evidence={"AgeGroup": int(encoded["AgeGroup"]), f: int(encoded[f])})
            factors[f] = round((float(q.values[1]) - base_p) * 100, 2)

        results.append(DiseaseResult(
            disease=d,
            risk_percent=risk,
            risk_band=band,
            recommendation=_rec(d, band),
            contributing_factors=factors,
        ))

    return PredictResponse(age_group=_age_group(req.age), results=results)
