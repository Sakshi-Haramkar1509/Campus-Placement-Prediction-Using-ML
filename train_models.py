"""
Train multiple models:
1. Placement Classification
2. Salary Regression
3. Job Role Classification

Save trained artifacts inside artifacts/ folder
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

# --------------------------------------------------
# Setup
# --------------------------------------------------

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DATA_PATH = "data/student_placement_dataset.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "student_placement_dataset.csv"

print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# Column Standardization
# --------------------------------------------------

if "Placement" in df.columns and "Placed" not in df.columns:
    df.rename(columns={"Placement": "Placed"}, inplace=True)

TARGET_PLACE = "Placed"
TARGET_SAL = "Salary" if "Salary" in df.columns else None
TARGET_ROLE = "JobRole" if "JobRole" in df.columns else None

# --------------------------------------------------
# Remove ID Columns (Important)
# --------------------------------------------------

ID_COLUMNS = ["StudentID"]

# --------------------------------------------------
# Define Feature Columns
# --------------------------------------------------

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove target & ID columns from feature lists
for col in [TARGET_PLACE, TARGET_SAL, TARGET_ROLE] + ID_COLUMNS:
    if col in num_cols:
        num_cols.remove(col)
    if col in cat_cols:
        cat_cols.remove(col)

# --------------------------------------------------
# ==============================
# 1️⃣ Placement Classification
# ==============================
# --------------------------------------------------

if TARGET_PLACE not in df.columns:
    raise ValueError("Placed column not found in dataset!")

df_pl = df.dropna(subset=[TARGET_PLACE]).copy()

X = df_pl.drop(columns=[TARGET_PLACE] + ID_COLUMNS, errors="ignore")
y = df_pl[TARGET_PLACE].astype(int)

# Handle Missing Values
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("missing")

# Train-Test Split
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# Models
log_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

rf_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
])

# Train
log_pipe.fit(Xtr, ytr)
rf_pipe.fit(Xtr, ytr)

# Evaluate
acc_log = accuracy_score(yte, log_pipe.predict(Xte))
acc_rf = accuracy_score(yte, rf_pipe.predict(Xte))

print(f"Logistic Accuracy: {acc_log:.4f}")
print(f"Random Forest Accuracy: {acc_rf:.4f}")

# Select Best Model
best_model = rf_pipe if acc_rf >= acc_log else log_pipe
joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "placement_pipe.pkl"))
print("✅ Placement model saved")

# --------------------------------------------------
# ==============================
# 2️⃣ Salary Regression
# ==============================
# --------------------------------------------------

if TARGET_SAL:
    df_sal = df[df[TARGET_SAL].notna() & (df[TARGET_SAL] > 0)].copy()

    if df_sal.shape[0] >= 30:
        Xs = df_sal.drop(columns=[TARGET_SAL, TARGET_PLACE] + ID_COLUMNS, errors="ignore")
        ys = df_sal[TARGET_SAL].astype(float)

        Xs[num_cols] = Xs[num_cols].fillna(Xs[num_cols].median())
        Xs[cat_cols] = Xs[cat_cols].fillna("missing")

        Xtrs, Xtes, ytrs, ytes = train_test_split(
            Xs, ys, test_size=0.2, random_state=42
        )

        reg_pipe = Pipeline([
            ("pre", preprocessor),
            ("reg", RandomForestRegressor(n_estimators=300, random_state=42))
        ])

        reg_pipe.fit(Xtrs, ytrs)
        r2 = r2_score(ytes, reg_pipe.predict(Xtes))

        print(f"Salary R² Score: {r2:.4f}")

        joblib.dump(reg_pipe, os.path.join(ARTIFACT_DIR, "salary_reg.pkl"))
        print("✅ Salary regression model saved")
    else:
        print("⚠ Not enough salary data to train regression model")

# --------------------------------------------------
# ==============================
# 3️⃣ Job Role Classification
# ==============================
# --------------------------------------------------

if TARGET_ROLE:
    df_role = df[df[TARGET_ROLE].notna()].copy()

    if df_role.shape[0] >= 30:

        le = LabelEncoder()
        df_role[TARGET_ROLE] = le.fit_transform(df_role[TARGET_ROLE].astype(str))

        Xr = df_role.drop(columns=[TARGET_ROLE, TARGET_SAL, TARGET_PLACE] + ID_COLUMNS, errors="ignore")
        yr = df_role[TARGET_ROLE]

        Xr[num_cols] = Xr[num_cols].fillna(Xr[num_cols].median())
        Xr[cat_cols] = Xr[cat_cols].fillna("missing")

        Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
            Xr, yr, test_size=0.2, stratify=yr, random_state=42
        )

        role_pipe = Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
        ])

        role_pipe.fit(Xtr_r, ytr_r)
        acc_role = accuracy_score(yte_r, role_pipe.predict(Xte_r))

        print(f"Job Role Accuracy: {acc_role:.4f}")

        joblib.dump(role_pipe, os.path.join(ARTIFACT_DIR, "role_clf.pkl"))
        joblib.dump(le, os.path.join(ARTIFACT_DIR, "le_jobrole.pkl"))

        print("✅ Job role classifier saved")
    else:
        print("⚠ Not enough role data to train role classifier")

print("\n🎉 Training completed successfully!")
print("Artifacts saved inside:", ARTIFACT_DIR)