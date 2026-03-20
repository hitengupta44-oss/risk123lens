"""
app.py — RiskLens Final Backend
Endpoints: /predict  /chat  /options  /health
"""

import os, pickle, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List
from pgmpy.inference import VariableElimination

# ── ML Artifacts ──────────────────────────────────────────────────────────────
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

bn_infer = {d: VariableElimination(bn_models[d]) for d in DISEASES}
recs     = pd.read_excel(os.path.join(DATA_DIR, "recommendationsdoc_precise_detailed.xlsx"))

# ── Chatbot ───────────────────────────────────────────────────────────────────
from local_chatbot import init_chatbot, chat as local_chat
init_chatbot()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="RiskLens API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

@app.options("/{full_path:path}")
async def preflight(request: Request, full_path: str):
    return JSONResponse(
        status_code=200,
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )

# ── Field normalisation ───────────────────────────────────────────────────────
def _fix(field: str, val: str) -> str:
    v = str(val).strip().lower()
    maps = {
        "Gender":           {"male":"Male","female":"Female","m":"Male","f":"Female"},
        "BloodPressure":    {"normal":"Normal","elevated":"Elevated","high":"High"},
        "PhysicalActivity": {"low":"Low","moderate":"Moderate","medium":"Moderate","high":"High"},
        "DietQuality":      {"poor":"Poor","average":"Average","good":"Good"},
        "SugarLevel":       {"normal":"Normal","high":"High"},
    }
    if field in maps:
        return maps[field].get(v, str(val).strip().title())
    yn = {"yes":"Yes","no":"No","true":"Yes","false":"No","1":"Yes","0":"No"}
    return yn.get(v, str(val).strip().title())

# ── Helpers ───────────────────────────────────────────────────────────────────
def _age_group(age: int) -> str:
    if age <= 30: return "Young"
    if age <= 45: return "Adult"
    if age <= 60: return "Middle"
    return "Senior"

def _encode(raw: dict) -> dict:
    enc = {}
    enc["AgeGroup"] = int(encoders["AgeGroup"].transform([_age_group(raw["age"])])[0])
    for col, val in raw.items():
        if col == "age":
            continue
        val = _fix(col, str(val))
        if col in ORDINAL_FEATURES:
            m = encoders[col]
            if val not in m:
                raise HTTPException(400, f"Bad value '{val}' for '{col}'. Allowed: {list(m.keys())}")
            enc[col] = int(m[val])
        else:
            if col not in encoders:
                continue
            le = encoders[col]
            if val not in le.classes_:
                raise HTTPException(400, f"Bad value '{val}' for '{col}'. Allowed: {le.classes_.tolist()}")
            enc[col] = int(le.transform([val])[0])
    return enc

def _band(risk: float) -> str:
    if risk <= 20: return "0-20"
    if risk <= 40: return "21-40"
    if risk <= 60: return "41-60"
    if risk <= 80: return "61-80"
    return "81-100"

def _rec(disease: str, band: str) -> str:
    row = recs[(recs["Disease"] == disease) & (recs["RiskRangePercent"] == band)]
    return str(row["DoctorRecommendation"].values[0]) if not row.empty else "Consult a doctor."

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    age:                    int = Field(..., ge=1, le=120)
    Gender:                 str
    Smoking:                str
    Alcohol:                str
    PhysicalActivity:       str
    DietQuality:            str
    BloodPressure:          str
    FrequentUrination:      str
    ExcessiveThirst:        str
    FamilyHistoryDiabetes:  str
    Fatigue:                str
    ChestPain:              str
    FamilyHistoryHeart:     str
    SwellingAnkles:         str
    Wheezing:               str
    Breathlessness:         str
    Cough:                  str
    PaleSkin:               str
    WeightLoss:             str
    Dizziness:              str
    SugarLevel:             str

class DiseaseResult(BaseModel):
    disease:              str
    risk_percent:         float
    risk_band:            str
    recommendation:       str
    contributing_factors: Dict[str, float]

class PredictResponse(BaseModel):
    age_group: str
    results:   List[DiseaseResult]

class ChatRequest(BaseModel):
    message:     str
    diseases:    List[str]        = []
    risk_scores: Dict[str, float] = {}
    history:     List[Dict]       = []

class ChatResponse(BaseModel):
    reply:   str
    sources: List[str]
    mode:    str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "RiskLens API is live 🟢"}

@app.get("/health")
def health():
    return {"status": "healthy", "diseases": DISEASES}

@app.get("/options")
def options():
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
    try:
        encoded = _encode(req.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Encoding error: {str(e)}")

    results = []
    for d in DISEASES:
        feats = DISEASE_FEATURES[d]
        try:
            X_nb   = pd.DataFrame([{f: encoded[f] for f in feats}])
            p_nb   = float(nb_models[d].predict_proba(X_nb)[0][1])
            ev     = {f: int(encoded[f]) for f in feats}
            ev["AgeGroup"] = int(encoded["AgeGroup"])
            p_bn   = float(bn_infer[d].query([d], evidence=ev).values[1])
            risk   = round((0.6 * p_nb + 0.4 * p_bn) * 100, 2)
            band   = _band(risk)
            base_p = float(bn_infer[d].query([d], evidence={"AgeGroup": int(encoded["AgeGroup"])}).values[1])
            factors = {}
            for f in feats:
                q = bn_infer[d].query([d], evidence={"AgeGroup": int(encoded["AgeGroup"]), f: int(encoded[f])})
                factors[f] = round((float(q.values[1]) - base_p) * 100, 2)
            results.append(DiseaseResult(
                disease=d, risk_percent=risk, risk_band=band,
                recommendation=_rec(d, band), contributing_factors=factors,
            ))
        except Exception:
            results.append(DiseaseResult(
                disease=d, risk_percent=0.0, risk_band="0-20",
                recommendation="Unable to calculate. Please consult a doctor.",
                contributing_factors={},
            ))

    return PredictResponse(age_group=_age_group(req.age), results=results)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    if len(req.message) > 500:
        raise HTTPException(400, "Message too long — max 500 characters")

    result = local_chat(
        message=req.message,
        diseases=req.diseases,
        risk_scores=req.risk_scores,
        history=req.history,
    )
    return ChatResponse(
        reply=result["reply"],
        sources=result["sources"],
        mode=result["mode"],
    )
