"""
train.py — Runs at Railway BUILD time.
Trains all models and saves to models/artifacts.pkl
"""

import os
import pickle
import warnings
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

warnings.filterwarnings("ignore")

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")

DISEASES = ["Diabetes", "HeartDisease", "CKD", "Asthma", "Dyslipidemia", "Anemia"]

DISEASE_FEATURES = {
    "Diabetes": ["SugarLevel", "FrequentUrination", "ExcessiveThirst", "FamilyHistoryDiabetes", "Fatigue"],
    "HeartDisease": ["ChestPain", "BloodPressure", "Smoking", "FamilyHistoryHeart", "Alcohol"],
    "CKD": ["SwellingAnkles", "FrequentUrination", "BloodPressure"],
    "Asthma": ["Wheezing", "Breathlessness", "Cough", "Smoking"],
    "Dyslipidemia": ["DietQuality", "PhysicalActivity", "Smoking", "Alcohol"],
    "Anemia": ["PaleSkin", "Fatigue", "WeightLoss", "Dizziness"],
}

ORDINAL_FEATURES = {
    "BloodPressure": ["Normal", "Elevated", "High"],
    "StressLevel": ["Low", "Medium", "High"],
    "DietQuality": ["Poor", "Average", "Good"],
    "PhysicalActivity": ["Low", "Moderate", "High"],
    "SaltIntake": ["Low", "Medium", "High"],
}


def load_and_preprocess_data():
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

    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 30, 45, 60, 120], labels=["Young", "Adult", "Middle", "Senior"])
    le_age = LabelEncoder()
    df["AgeGroup"] = le_age.fit_transform(df["AgeGroup"].astype(str))
    encoders["AgeGroup"] = le_age

    imputer = SimpleImputer(strategy="most_frequent")
    df[df.columns] = imputer.fit_transform(df)

    return df, encoders


def train_and_evaluate_nb_model(df, d, encoders):
    X = df[DISEASE_FEATURES[d]]
    y = df[d]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []
    f1_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        nb = CategoricalNB()
        nb.fit(X_train, y_train)
        preds = nb.predict(X_test)

        acc_scores.append(accuracy_score(y_test, preds))
        f1_scores.append(f1_score(y_test, preds, zero_division=0))

    print(f"  {d:15s}  acc={sum(acc_scores) / len(acc_scores):.3f}  f1={sum(f1_scores) / len(f1_scores):.3f}")

    nb = CategoricalNB()
    nb.fit(X, y)

    return nb


def train_and_evaluate_bn_model(df, d, encoders):
    feats = ["AgeGroup"] + DISEASE_FEATURES[d]
    data = df[feats + [d]].copy().astype(int)
    edges = [(f, d) for f in feats]

    state_names = {k: list(range(len(v))) for k, v in ORDINAL_FEATURES.items()}

    model = DiscreteBayesianNetwork(edges)
    model.fit(data, estimator=MaximumLikelihoodEstimator, state_names=state_names)

    return model


def save_artifacts(nb_models, bn_models, encoders):
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifacts = {
        "nb_models": nb_models,
        "bn_models": bn_models,
        "encoders": encoders,
        "disease_features": DISEASE_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "diseases": DISEASES,
    }
    with open(os.path.join(MODEL_DIR, "artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    df, encoders = load_and_preprocess_data()

    nb_models = {}
    bn_models = {}

    print("\n[NB] Training and evaluating models...")
    for d in DISEASES:
        nb_models[d] = train_and_evaluate_nb_model(df, d, encoders)
        print(f"  {d:15s}  done")

    print("\n[BN] Training and evaluating models...")
    for d in DISEASES:
        bn_models[d] = train_and_evaluate_bn_model(df, d, encoders)
        print(f"  {d:15s}  done")

    save_artifacts(nb_models, bn_models, encoders)
    print("\n✅ models/artifacts.pkl saved")


if __name__ == "__main__":
    main()
