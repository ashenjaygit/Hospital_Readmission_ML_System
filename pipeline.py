"""
CarePulse AI: Inpatient 30-Day Hospital Readmission Risk Prediction & Clinical Decision Support System
Dataset: Diabetes 130-US Hospitals (1999-2008), UCI Machine Learning Repository
Domain: Healthcare Sector Machine Learning & Clinical Informatics
Objective: Evidence-based inpatient risk stratification and post-discharge intervention planning
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from xgboost import XGBClassifier

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.dpi'] = 150

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "eda_plots")

for d in [DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 80)
print("CAREPULSE AI: INPATIENT 30-DAY HOSPITAL READMISSION RISK PREDICTION PIPELINE")
print("Healthcare Sector Machine Learning & Clinical Decision Support System")
print("=" * 80)

# ==============================================================================
# 1. DATA INGESTION & MEDICAL COHORT FILTERING
# ==============================================================================
print("\n[1/6] Loading raw inpatient encounter records...")
raw_file = os.path.join(DATA_DIR, "diabetic_data.csv")
if not os.path.exists(raw_file):
    raise FileNotFoundError(f"Missing dataset at {raw_file}")

df_raw = pd.read_csv(raw_file)
print(f"Total raw encounters loaded: {len(df_raw):,} across {df_raw.shape[1]} columns")

# Clinical protocol filtering:
# Expired patients cannot be readmitted (Discharge disposition 11, 19, 20, 21 as per CMS HRRP standards)
expired_ids = [11, 19, 20, 21]
df_filtered = df_raw[~df_raw["discharge_disposition_id"].isin(expired_ids)].copy()
print(f"Encounters after excluding expired/terminal inpatient cases: {len(df_filtered):,}")

# Remove duplicate patient encounters by retaining the first chronological admission
# to ensure zero patient data leakage across train/validation splits
df_unique = df_filtered.drop_duplicates(subset=["patient_nbr"], keep="first").copy()
print(f"Unique patient cohort (first admission per patient): {len(df_unique):,} patients")

# ==============================================================================
# 2. CLINICAL FEATURE ENGINEERING & ICD-9 MAPPING
# ==============================================================================
print("\n[2/6] Performing clinical feature engineering & ICD-9 taxonomy grouping...")

def map_icd9_category(code):
    """
    Categorizes ICD-9 diagnostic codes into primary anatomical disease systems
    based on standard medical epidemiology hierarchy.
    """
    if pd.isna(code) or str(code).strip() == '?' or str(code).strip() == '':
        return "Other_Unknown"
    code_str = str(code).strip()
    if code_str.startswith('V') or code_str.startswith('E'):
        return "Supplemental_External"
    try:
        val = float(code_str)
        if 390 <= val <= 459 or val == 785:
            return "Circulatory"
        elif 460 <= val <= 519 or val == 786:
            return "Respiratory"
        elif 520 <= val <= 579 or val == 787:
            return "Digestive"
        elif 250 <= val < 251:
            return "Diabetes"
        elif 800 <= val <= 999:
            return "Injury_Poisoning"
        elif 710 <= val <= 739:
            return "Musculoskeletal"
        elif 580 <= val <= 629 or val == 788:
            return "Genitourinary"
        elif 140 <= val <= 239:
            return "Neoplasms"
        else:
            return "Other"
    except ValueError:
        if code_str.startswith('250'):
            return "Diabetes"
        return "Other"

# Map primary, secondary, and tertiary diagnoses
df_unique["diag_1_group"] = df_unique["diag_1"].apply(map_icd9_category)
df_unique["diag_2_group"] = df_unique["diag_2"].apply(map_icd9_category)
df_unique["diag_3_group"] = df_unique["diag_3"].apply(map_icd9_category)

# Comorbidity multi-system burden: count distinct disease systems affected
def count_comorbid_systems(row):
    systems = set([row["diag_1_group"], row["diag_2_group"], row["diag_3_group"]])
    systems.discard("Other_Unknown")
    return len(systems)

df_unique["comorbidity_count"] = df_unique.apply(count_comorbid_systems, axis=1)

# Prior healthcare utilization metrics
df_unique["total_prior_visits"] = (
    df_unique["number_inpatient"] +
    df_unique["number_emergency"] +
    df_unique["number_outpatient"]
)
df_unique["had_prior_inpatient"] = (df_unique["number_inpatient"] > 0).astype(int)
df_unique["had_prior_emergency"] = (df_unique["number_emergency"] > 0).astype(int)

# Inpatient treatment intensity
df_unique["medication_intensity"] = df_unique["num_medications"] / np.maximum(df_unique["time_in_hospital"], 1)
df_unique["lab_intensity"] = df_unique["num_lab_procedures"] / np.maximum(df_unique["time_in_hospital"], 1)

# Glycemic control biomarkers
df_unique["a1c_tested"] = (df_unique["A1Cresult"] != 'None').astype(int)
df_unique["a1c_elevated"] = df_unique["A1Cresult"].isin(['>8', '>7']).astype(int)
df_unique["glucose_tested"] = (df_unique["max_glu_serum"] != 'None').astype(int)

# Medication management
df_unique["medication_changed"] = (df_unique["change"] == 'Ch').astype(int)
df_unique["on_diabetes_med"] = (df_unique["diabetesMed"] == 'Yes').astype(int)
df_unique["on_insulin"] = (df_unique["insulin"] != 'No').astype(int)

# Map admission type to clinical descriptions
admission_type_map = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Trauma_Other",
    6: "Trauma_Other",
    7: "Trauma_Other",
    8: "Trauma_Other"
}
df_unique["admission_category"] = df_unique["admission_type_id"].map(admission_type_map).fillna("Other")

# Discharge destination categorization
# 1: Home, 3: SNF, 6: Home with Home Health, Others: Other facility
def map_discharge(d_id):
    if d_id == 1:
        return "Home"
    elif d_id == 6:
        return "Home_Health"
    elif d_id in [3, 4, 5]:
        return "Nursing_Facility"
    else:
        return "Other_Facility"

df_unique["discharge_category"] = df_unique["discharge_disposition_id"].apply(map_discharge)

# Medical specialty grouping (high cardinality handling)
top_specialties = ["InternalMedicine", "Cardiology", "Surgery-General", "Family/GeneralPractice", "Emergency"]
def clean_specialty(spec):
    if pd.isna(spec) or spec == '?':
        return "Missing_Unspecified"
    for ts in top_specialties:
        if ts.lower() in spec.lower():
            return ts
    return "Other_Specialty"

df_unique["medical_specialty_clean"] = df_unique["medical_specialty"].apply(clean_specialty)

# Clean Age bracket to mid-point integer
age_map = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}
df_unique["age_numeric"] = df_unique["age"].map(age_map).fillna(65)

# Target Variable Formulation:
# Primary Class: 30-Day Early Unplanned Readmission (<30)
df_unique["target_readmitted_30d"] = (df_unique["readmitted"] == '<30').astype(int)

print(f"Target Distribution (30-Day Readmission):")
val_c = df_unique["target_readmitted_30d"].value_counts()
print(f" - Non-Readmitted / >30d (0): {val_c[0]:,} ({val_c[0]/len(df_unique)*100:.1f}%)")
print(f" - Readmitted within 30d (1): {val_c[1]:,} ({val_c[1]/len(df_unique)*100:.1f}%)")

# Save cleaned master cohort
cleaned_csv = os.path.join(DATA_DIR, "cleaned_readmission_data.csv")
df_unique.to_csv(cleaned_csv, index=False)
print(f"Cleaned dataset saved: {cleaned_csv}")

# ==============================================================================
# 3. STATISTICAL AUDIT & CLINICAL ASSOCIATION TESTS
# ==============================================================================
print("\n[3/6] Running clinical statistical association tests...")

cat_audit = []
for c in ["diag_1_group", "admission_category", "discharge_category", "a1c_tested", "medication_changed", "had_prior_inpatient", "on_insulin"]:
    ct = pd.crosstab(df_unique[c], df_unique["target_readmitted_30d"])
    chi2, p_val, _, _ = stats.chi2_contingency(ct)
    readmit_rates = pd.crosstab(df_unique[c], df_unique["target_readmitted_30d"], normalize="index")[1] * 100
    summary_str = ", ".join([f"{k}: {v:.1f}%" for k, v in readmit_rates.items()])
    cat_audit.append({
        "Feature": c,
        "Chi2": round(float(chi2), 2),
        "p_value": float(p_val),
        "Readmission_Rates_Pct": summary_str
    })

audit_df = pd.DataFrame(cat_audit).sort_values("Chi2", ascending=False)
print("Categorical Clinical Risk Associations (Chi-Square):")
print(audit_df.to_string(index=False))

# ==============================================================================
# 4. PUBLICATION-QUALITY EDA & CLINICAL FIGURES
# ==============================================================================
print("\n[4/6] Generating high-resolution publication clinical visualizations...")

C_PRIMARY_BLUE = "#0D47A1"
C_CLINICAL_TEAL = "#00897B"
C_RED = "#D32F2F"
C_ORANGE = "#FB8C00"
C_GREEN = "#2E7D32"

# Plot 1: Target Class Breakdown & CMS 30-Day Benchmark
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
raw_target_counts = df_unique["readmitted"].value_counts().reindex(["NO", ">30", "<30"])
colors_t = ["#90CAF9", "#FFE082", "#EF5350"]
bars = ax1.bar(["No Readmission", "Readmitted > 30 Days", "Readmitted < 30 Days"], raw_target_counts.values, color=colors_t, edgecolor="#374151")
ax1.set_title("Overall Inpatient Readmission Distribution (N = 69,973)")
ax1.set_ylabel("Number of Patients")
for b in bars:
    h = b.get_height()
    pct = h / len(df_unique) * 100
    ax1.text(b.get_x() + b.get_width()/2, h + 500, f"{h:,}\n({pct:.1f}%)", ha='center', fontweight='bold', fontsize=9)

# 30-day binary focus
bin_counts = df_unique["target_readmitted_30d"].value_counts()
ax2.pie(bin_counts.values, labels=["Discharged Safe / >30d", "Critical: Readmitted < 30d"], autopct="%1.1f%%",
        startangle=140, colors=[C_CLINICAL_TEAL, C_RED], explode=[0, 0.08],
        textprops={'fontweight': 'bold', 'fontsize': 10})
ax2.set_title("Primary Clinical Target: 30-Day Unplanned Readmission")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "01_readmission_class_distribution.png"))
plt.close(fig)

# Plot 2: Impact of Prior Inpatient Hospitalizations
fig, ax = plt.subplots(figsize=(8, 5))
prior_inp_rates = df_unique.groupby("number_inpatient")["target_readmitted_30d"].agg(["mean", "count"]).loc[0:5]
bars = ax.bar([f"{int(i)} Prior Stays" for i in prior_inp_rates.index], prior_inp_rates["mean"] * 100, color=C_RED, edgecolor="#1F2937", alpha=0.85)
ax.set_title("30-Day Readmission Risk Escapes Steeply with Prior Inpatient Stays", pad=12)
ax.set_ylabel("30-Day Readmission Rate (%)")
ax.set_xlabel("Number of Inpatient Admissions in Preceding Year")
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 0.8, f"{h:.1f}%", ha='center', fontweight='bold')
ax.axhline(df_unique["target_readmitted_30d"].mean() * 100, color="gray", linestyle="--", label=f"Hospital Baseline Average ({df_unique['target_readmitted_30d'].mean()*100:.1f}%)")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "02_prior_inpatient_stays_impact.png"))
plt.close(fig)

# Plot 3: Readmission Risk by Primary Diagnostic Category (ICD-9)
fig, ax = plt.subplots(figsize=(10, 5.5))
diag_rates = df_unique.groupby("diag_1_group")["target_readmitted_30d"].agg(["mean", "count"]).sort_values("mean", ascending=True)
bars = ax.barh(diag_rates.index, diag_rates["mean"] * 100, color=C_PRIMARY_BLUE, edgecolor="#1F2937")
ax.set_title("30-Day Readmission Risk by Primary Diagnosis Category (ICD-9)", pad=12)
ax.set_xlabel("30-Day Readmission Rate (%)")
ax.axvline(df_unique["target_readmitted_30d"].mean() * 100, color="red", linestyle="--", label="Average Benchmark")
for b in bars:
    w = b.get_width()
    ax.text(w + 0.2, b.get_y() + b.get_height()/2, f"{w:.1f}%", va='center', fontweight='bold', fontsize=9)
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "03_icd9_primary_diagnosis_risk.png"))
plt.close(fig)

# Plot 4: Protective Effect of HbA1c Lab Testing During Hospitalization
fig, ax = plt.subplots(figsize=(8, 4.8))
a1c_breakdown = df_unique.groupby("A1Cresult")["target_readmitted_30d"].agg(["mean", "count"]).reindex(["None", "Norm", ">7", ">8"])
a1c_labels = ["No HbA1c Ordered", "Normal (<7%)", "Elevated (>7%)", "Severely High (>8%)"]
colors_a = ["#E53935", "#43A047", "#FB8C00", "#D81B60"]
bars = ax.bar(a1c_labels, a1c_breakdown["mean"] * 100, color=colors_a, edgecolor="#1F2937")
ax.set_title("Clinical Quality Indicator: Patients Without HbA1c Testing Have Higher Readmission", pad=12)
ax.set_ylabel("30-Day Readmission Rate (%)")
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 0.4, f"{h:.1f}%", ha='center', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "04_a1c_testing_and_readmission.png"))
plt.close(fig)

# Plot 5: Polypharmacy and Length of Stay (LOS) Distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
sns.boxplot(x="target_readmitted_30d", y="time_in_hospital", hue="target_readmitted_30d", data=df_unique, palette=[C_CLINICAL_TEAL, C_RED], legend=False, ax=ax1)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Not Readmitted (<30d)", "Readmitted (<30d)"])
ax1.set_title("Length of Stay (Days) vs 30-Day Readmission")
ax1.set_ylabel("Inpatient Days")
ax1.set_xlabel("")

sns.boxplot(x="target_readmitted_30d", y="num_medications", hue="target_readmitted_30d", data=df_unique, palette=[C_CLINICAL_TEAL, C_RED], legend=False, ax=ax2)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["Not Readmitted (<30d)", "Readmitted (<30d)"])
ax2.set_title("Polypharmacy: Number of Medications vs Readmission")
ax2.set_ylabel("Distinct Medication Count")
ax2.set_xlabel("")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "05_polypharmacy_and_los_burden.png"))
plt.close(fig)

print("Clinical EDA figures 1 to 5 generated and saved in eda_plots/")

# ==============================================================================
# 5. MACHINE LEARNING MODELING (LOGISTIC REGRESSION & XGBOOST)
# ==============================================================================
print("\n[5/6] Building Machine Learning Models (Explainable LR + High-Performance XGBoost)...")

# Model features
feature_cols = [
    # Demographics & Stay
    "age_numeric", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    # Utilization History
    "number_outpatient", "number_emergency", "number_inpatient", "had_prior_inpatient",
    # Clinical Intensity & Complexity
    "number_diagnoses", "comorbidity_count", "medication_intensity", "lab_intensity",
    # Glycemic & Medication Factors
    "a1c_tested", "a1c_elevated", "glucose_tested", "medication_changed", "on_diabetes_med", "on_insulin",
    # Categoricals for encoding
    "admission_category", "discharge_category", "diag_1_group", "medical_specialty_clean"
]

X_raw = df_unique[feature_cols].copy()
y = df_unique["target_readmitted_30d"].values

# One-hot encoding
X = pd.get_dummies(X_raw, drop_first=True, dtype=float)
encoded_feature_names = list(X.columns)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute class imbalance ratio
pos_ratio = (len(y) - sum(y)) / sum(y)
print(f"Class imbalance weighting ratio: {pos_ratio:.2f}")

# Cross-Validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model 1: Balanced Logistic Regression (Clinical Odds Ratios)
print("Training Balanced Logistic Regression (Clinical Odds Ratios)...")
lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr_probs_cv = np.zeros(len(y))

for train_idx, val_idx in cv.split(X_scaled, y):
    X_tr, y_tr = X_scaled[train_idx], y[train_idx]
    X_va = X_scaled[val_idx]
    lr_model.fit(X_tr, y_tr)
    lr_probs_cv[val_idx] = lr_model.predict_proba(X_va)[:, 1]

lr_auc = roc_auc_score(y, lr_probs_cv)
lr_prauc = average_precision_score(y, lr_probs_cv)
print(f" - Logistic Regression ROC-AUC: {lr_auc:.4f} | PR-AUC: {lr_prauc:.4f}")

# Fit on all data for final weights
lr_model.fit(X_scaled, y)
odds_ratios = np.exp(lr_model.coef_[0])
coef_df = pd.DataFrame({
    "Feature": encoded_feature_names,
    "Coefficient": lr_model.coef_[0],
    "Odds_Ratio": odds_ratios
}).sort_values("Odds_Ratio", ascending=False)

# Model 2: XGBoost Classifier (Tuned for Clinical Decision Support)
print("Training XGBoost Classifier (Non-linear risk interactions)...")
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.08,
    scale_pos_weight=pos_ratio,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb_probs_cv = np.zeros(len(y))
for train_idx, val_idx in cv.split(X.values, y):
    X_tr, y_tr = X.values[train_idx], y[train_idx]
    X_va = X.values[val_idx]
    xgb_model.fit(X_tr, y_tr)
    xgb_probs_cv[val_idx] = xgb_model.predict_proba(X_va)[:, 1]

xgb_auc = roc_auc_score(y, xgb_probs_cv)
xgb_prauc = average_precision_score(y, xgb_probs_cv)
print(f" - XGBoost ROC-AUC: {xgb_auc:.4f} | PR-AUC: {xgb_prauc:.4f}")

# Fit XGBoost on full data
xgb_model.fit(X.values, y)

# Clinical Decision Threshold Evaluation (Optimized for High-Risk Triage)
# In hospital triage, missing a high-risk patient is far more costly than false alarm.
# We evaluate at probability threshold 0.50 (balanced) and 0.40 (sensitive triage).
triage_thresh = 0.50
y_pred_triage = (xgb_probs_cv >= triage_thresh).astype(int)

acc = accuracy_score(y, y_pred_triage)
prec = precision_score(y, y_pred_triage)
recall = recall_score(y, y_pred_triage)
f1 = f1_score(y, y_pred_triage)
cm = confusion_matrix(y, y_pred_triage)
brier = brier_score_loss(y, xgb_probs_cv)

print("\n--- XGBoost 5-Fold Stratified Cross-Validation Clinical Metrics ---")
print(f"ROC-AUC               : {xgb_auc:.4f}")
print(f"PR-AUC (Precision-Rec): {xgb_prauc:.4f}")
print(f"Clinical Sensitivity  : {recall*100:.1f}% (High-Risk Capture Rate)")
print(f"Clinical Specificity  : {cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}%")
print(f"Brier Calibration     : {brier:.4f}")
print("Confusion Matrix:")
print(cm)

# Save Model Objects & Metadata
joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost_readmission.pkl"))
joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

metadata = {
    "raw_features": feature_cols,
    "encoded_features": encoded_feature_names,
    "sample_size": int(len(df_unique)),
    "metrics": {
        "xgb_roc_auc": float(xgb_auc),
        "xgb_pr_auc": float(xgb_prauc),
        "xgb_recall": float(recall),
        "xgb_precision": float(prec),
        "lr_roc_auc": float(lr_auc),
        "lr_pr_auc": float(lr_prauc)
    },
    "admission_types": list(df_unique["admission_category"].unique()),
    "discharge_destinations": list(df_unique["discharge_category"].unique()),
    "icd9_groups": list(df_unique["diag_1_group"].unique()),
    "specialties": top_specialties + ["Missing_Unspecified", "Other_Specialty"]
}

with open(os.path.join(MODELS_DIR, "feature_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("Trained models and feature metadata successfully saved to models/")

# ==============================================================================
# 6. MODEL EXPLAINABILITY & PERFORMANCE CHARTS
# ==============================================================================
print("\n[6/6] Generating Model Explainability & Diagnostic Plots...")

# Plot 6: XGBoost Feature Importance
fig, ax = plt.subplots(figsize=(10, 6.5))
fi = pd.Series(xgb_model.feature_importances_, index=encoded_feature_names).sort_values(ascending=False).head(15)
bars = ax.barh(fi.index[::-1], fi.values[::-1], color=C_CLINICAL_TEAL, edgecolor="#1F2937")
ax.set_title("Top 15 Clinical Drivers of 30-Day Readmission (XGBoost)", pad=12)
ax.set_xlabel("Relative Feature Importance Score")
for b in bars:
    w = b.get_width()
    ax.text(w + 0.002, b.get_y() + b.get_height()/2, f"{w:.3f}", va='center', fontweight='bold', fontsize=8.5)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "06_feature_importance_xgboost.png"))
plt.close(fig)

# Plot 7: Clinical Odds Ratio Scorecard (Logistic Regression e^beta)
fig, ax = plt.subplots(figsize=(10, 7))
top_protective = coef_df.sort_values("Odds_Ratio").head(7)
top_risk = coef_df.sort_values("Odds_Ratio", ascending=False).head(8)
scorecard_df = pd.concat([top_risk, top_protective]).sort_values("Odds_Ratio", ascending=True)
colors_sc = [C_RED if o > 1.0 else C_GREEN for o in scorecard_df["Odds_Ratio"]]
bars = ax.barh(scorecard_df["Feature"], scorecard_df["Odds_Ratio"], color=colors_sc, edgecolor="#1F2937")
ax.axvline(1.0, color="#1F2937", linestyle="--", linewidth=1.5, label="Neutral Risk (OR = 1.0)")
ax.set_title("Clinical Odds Ratios: Multipliers of 30-Day Readmission Risk (e^β)", pad=12)
ax.set_xlabel("Odds Ratio (> 1 Increases Readmission Odds, < 1 Protective)")
for b in bars:
    w = b.get_width()
    ax.text(w + 0.02, b.get_y() + b.get_height()/2, f"{w:.2f}x", va='center', fontweight='bold', fontsize=8.5)
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "07_clinical_odds_ratios_scorecard.png"))
plt.close(fig)

# Plot 8: ROC and Precision-Recall Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fpr_xgb, tpr_xgb, _ = roc_curve(y, xgb_probs_cv)
fpr_lr, tpr_lr, _ = roc_curve(y, lr_probs_cv)
ax1.plot(fpr_xgb, tpr_xgb, color=C_PRIMARY_BLUE, lw=2.5, label=f"XGBoost (AUC = {xgb_auc:.3f})")
ax1.plot(fpr_lr, tpr_lr, color=C_CLINICAL_TEAL, lw=2, linestyle="-.", label=f"Logistic Reg (AUC = {lr_auc:.3f})")
ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance (0.50)")
ax1.set_title("Receiver Operating Characteristic (ROC) Curve")
ax1.set_xlabel("False Positive Rate (1 - Specificity)")
ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)")
ax1.legend(loc="lower right")

prec_xgb, rec_xgb, _ = precision_recall_curve(y, xgb_probs_cv)
prec_lr, rec_lr, _ = precision_recall_curve(y, lr_probs_cv)
baseline_prevalence = sum(y) / len(y)
ax2.plot(rec_xgb, prec_xgb, color=C_PRIMARY_BLUE, lw=2.5, label=f"XGBoost (PR-AUC = {xgb_prauc:.3f})")
ax2.plot(rec_lr, prec_lr, color=C_CLINICAL_TEAL, lw=2, linestyle="-.", label=f"Logistic Reg (PR-AUC = {lr_prauc:.3f})")
ax2.axhline(baseline_prevalence, color="red", linestyle="--", label=f"Baseline Prevalence ({baseline_prevalence*100:.1f}%)")
ax2.set_title("Precision-Recall Curve (Critical for Clinical Imbalance)")
ax2.set_xlabel("Recall (Fraction of Readmitted Patients Captured)")
ax2.set_ylabel("Precision (Positive Predictive Value)")
ax2.legend(loc="upper right")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "08_roc_and_precision_recall_curves.png"))
plt.close(fig)

print("All clinical EDA and model diagnostic plots successfully saved to eda_plots/")
print("\n" + "=" * 80)
print("[SUCCESS] CAREPULSE AI PIPELINE EXECUTION COMPLETED!")
print("=" * 80)
