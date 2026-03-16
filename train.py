"""
train.py — Runs at Railway BUILD time.
Trains all models and saves to models/artifacts.pkl
"""

import os, pickle, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

BASE      = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DISEASES = ["Diabetes", "HeartDisease", "CKD", "Asthma", "Dyslipidemia", "Anemia"]

DISEASE_FEATURES = {
    "Diabetes":      ["SugarLevel", "FrequentUrination", "ExcessiveThirst",
                      "FamilyHistoryDiabetes", "Fatigue"],
    "HeartDisease":  ["ChestPain", "BloodPressure", "Smoking",
                      "FamilyHistoryHeart", "Alcohol"],
    "CKD":           ["SwellingAnkles", "FrequentUrination", "BloodPressure"],
    "Asthma":        ["Wheezing", "Breathlessness", "Cough", "Smoking"],
    "Dyslipidemia":  ["DietQuality", "PhysicalActivity", "Smoking", "Alcohol"],
    "Anemia":        ["PaleSkin", "Fatigue", "WeightLoss", "Dizziness"],
}

ORDINAL_FEATURES = {
    "BloodPressure":    ["Normal", "Elevated", "High"],
    "StressLevel":      ["Low", "Medium", "High"],
    "DietQuality":      ["Poor", "Average", "Good"],
    "PhysicalActivity": ["Low", "Moderate", "High"],
    "SaltIntake":       ["Low", "Medium", "High"],
}

STATE_NAMES = {k: list(range(len(v))) for k, v in ORDINAL_FEATURES.items()}

# ── Load & encode ──────────────────────────────────────────────────────────────
df = pd.read_excel(os.path.join(DATA_DIR, "diseasefinalset.xlsx"))

encoders = {}
for col in df.columns:
    if col in ORDINAL_FEATURES:
        mapping = {v: i for i, v in enumerate(ORDINAL_FEATURES[col])}
        df[col] = df[col].map(mapping)
        encoders[col] = mapping
    elif df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

df["AgeGroup"] = pd.cut(
    df["Age"], bins=[0, 30, 45, 60, 120],
    labels=["Young", "Adult", "Middle", "Senior"]
)
le_age = LabelEncoder()
df["AgeGroup"] = le_age.fit_transform(df["AgeGroup"].astype(str))
encoders["AgeGroup"] = le_age

imputer = SimpleImputer(strategy="most_frequent")
df[df.columns] = imputer.fit_transform(df)

# ── Train Naive Bayes ─────────────────────────────────────────────────────────
nb_models = {}
print("\n[NB] Training …")
for d in DISEASES:
    X = df[DISEASE_FEATURES[d]]
    y = df[d]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    nb = CategoricalNB()
    nb.fit(X_tr, y_tr)
    preds = nb.predict(X_te)
    print(f"  {d:15s}  acc={accuracy_score(y_te, preds):.3f}  f1={f1_score(y_te, preds, zero_division=0):.3f}")
    nb_models[d] = nb

# ── Train Bayesian Networks ───────────────────────────────────────────────────
bn_models = {}
print("\n[BN] Training …")
for d in DISEASES:
    feats = ["AgeGroup"] + DISEASE_FEATURES[d]
    data  = df[feats + [d]].copy().astype(int)
    edges = [(f, d) for f in feats]
    sn = {}
    for col in feats + [d]:
        sn[col] = STATE_NAMES[col] if col in STATE_NAMES else sorted(data[col].unique().tolist())
    model = DiscreteBayesianNetwork(edges)
    model.fit(data, estimator=MaximumLikelihoodEstimator, state_names=sn)
    bn_models[d] = model
    print(f"  {d:15s}  done")

# ── Save ──────────────────────────────────────────────────────────────────────
artifacts = {
    "nb_models": nb_models,
    "bn_models": bn_models,
    "encoders":  encoders,
    "disease_features": DISEASE_FEATURES,
    "ordinal_features": ORDINAL_FEATURES,
    "diseases": DISEASES,
}
with open(os.path.join(MODEL_DIR, "artifacts.pkl"), "wb") as f:
    pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\n✅  models/artifacts.pkl saved")
