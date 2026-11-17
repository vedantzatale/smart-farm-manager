import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Load dataset (20,000 rows)
# --------------------------------------------------

data = pd.read_csv("../data/farm_disease_data_20000.csv")

print("Dataset loaded!")
print("Shape:", data.shape)


# --------------------------------------------------
# 2. Define all 20 features (must match backend)
# --------------------------------------------------

features = [
    "temperature",
    "humidity",
    "animal_age",
    "animal_count",
    "animal_density",
    "sick_animals",
    "mortality_today",
    "normal_mortality",
    "vaccination_compliance",
    "quarantine_compliance",
    "cleaning_frequency",
    "disinfection_frequency",
    "ppe_compliance",
    "visitor_count",
    "visitor_control",
    "animal_movement",
    "equipment_sharing",
    "pest_presence",
    "water_quality",
    "nearby_outbreak"
]

X = data[features]
y_raw = data["risk_level"]


# --------------------------------------------------
# 3. Encode risk labels: Low=0, Moderate=1, High=2
# --------------------------------------------------

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

print("\nRisk level classes:", encoder.classes_)
print("Label mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))


# --------------------------------------------------
# 4. Train / Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. Train Random Forest Classifier
# --------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))


# --------------------------------------------------
# 6. Train XGBoost Classifier
# --------------------------------------------------

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss",
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))


# --------------------------------------------------
# 7. Compare models
# --------------------------------------------------

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("\nRandom Forest  Accuracy:", round(rf_acc * 100, 2), "%")
print("XGBoost        Accuracy:", round(xgb_acc * 100, 2), "%")

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_model.predict(X_test), target_names=encoder.classes_))


# --------------------------------------------------
# 8. Select best model and save
# --------------------------------------------------

best_model = xgb_model if xgb_acc >= rf_acc else rf_model
best_name = "XGBoost" if xgb_acc >= rf_acc else "Random Forest"

print("\nBest model:", best_name)

joblib.dump(best_model, "../backend/disease_risk_model.pkl")
joblib.dump(encoder, "../backend/risk_level_encoder.pkl")

print("Model saved to: backend/disease_risk_model.pkl")
print("Encoder saved to: backend/risk_level_encoder.pkl")