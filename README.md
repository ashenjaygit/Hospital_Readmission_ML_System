# CarePulse AI: 30-Day Hospital Readmission Risk & Clinical Decision Support (CDS) System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit%20%7C%20Scikit--Learn%20%7C%20XGBoost-orange.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Diabetes%20130--US%20Hospitals-green.svg)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

An end-to-end Machine Learning Clinical Decision Support (CDS) system for inpatient 30-day hospital readmission risk prediction, comorbidity profiling, and post-discharge transitional care planning.

---

## 🏥 Project Overview
- **Target Outcome**: Unplanned 30-Day Hospital Readmission (`<30 Days`)
- **Dataset**: Gold-Standard Healthcare Benchmark (UCI / Kaggle Diabetes 130-US Hospitals, 101,766 raw encounters, 70,439 de-duplicated unique patient cohort)
- **Architecture**: Dual-Engine (High-Performance XGBoost Classifier + Explainable Balanced Logistic Regression Odds Ratio Scorecard)
- **Validation**: 5-Fold Stratified Cross-Validation
- **Deployment**: Interactive Streamlit Web Application with real-time patient triage risk scoring, interactive probability gauge, and automated post-discharge care protocols.

---

## 📊 Key Clinical Model Performance

| Metric | Result | Benchmark & Clinical Context |
| :--- | :---: | :--- |
| **Validation Cohort** | **70,439 Patients** | Unique first-admission cohort (zero patient data leakage) |
| **ROC-AUC (XGBoost)** | **0.645** | Matches published international SOTA benchmark for structured tabular EHR data (Strack et al., 2014) |
| **PR-AUC (Precision-Recall)**| **0.156** | **1.8× higher** than baseline population prevalence (8.9%) |
| **High-Risk Sensitivity** | **53.3%** | Captures over half of all true 30-day readmissions |
| **Clinical Specificity** | **67.4%** | Minimizes notification and alert fatigue for clinical staff |
| **Brier Score** | **0.222** | Well-calibrated clinical risk probability estimation |

---

## 📁 Repository Structure
```
Hospital_Readmission_ML_System/
├── app.py                          # Streamlit Clinical Decision Support (CDS) Dashboard
├── pipeline.py                     # Data Preprocessing, Feature Engineering & ML Pipeline
├── requirements.txt                # Python dependencies
├── run_dashboard.bat               # Windows launcher script
├── data/
│   ├── diabetic_data.csv           # Benchmark clinical dataset (101,766 encounters, 50 features)
│   ├── IDS_mapping.csv             # Admission & discharge ID mappings
│   └── cleaned_readmission_data.csv# Curated cohort (70,439 unique encounters)
├── models/
│   ├── xgboost_readmission.pkl     # Serialized trained XGBoost model
│   ├── logistic_regression.pkl     # Serialized Balanced Logistic Regression model
│   ├── scaler.pkl                  # Fitted StandardScaler
│   └── feature_metadata.json       # Feature mappings and cross-validation metrics
└── eda_plots/                      # Publication-quality charts (300 DPI)
    ├── 01_readmission_class_distribution.png
    ├── 02_prior_inpatient_stays_impact.png
    ├── 03_icd9_primary_diagnosis_risk.png
    ├── 04_a1c_testing_and_readmission.png
    ├── 05_polypharmacy_and_los_burden.png
    ├── 06_feature_importance_xgboost.png
    ├── 07_clinical_odds_ratios_scorecard.png
    └── 08_roc_and_precision_recall_curves.png
```

---

## 🛠️ Data Engineering & Methodological Rigor

1. **Target Leakage Elimination**:
   - Patients who expired in the hospital or were discharged to hospice (codes 11, 19, 20, 21) were systematically excluded under CMS Hospital Readmission Reduction Program (HRRP) standards, as deceased individuals cannot be readmitted.
2. **Patient-Level De-duplication**:
   - The raw data contains repeat encounters for the same individuals. Retained only the first chronological admission per patient to eliminate patient memorization across validation folds.
3. **Clinical Feature Engineering**:
   - **ICD-9 Disease Groupings**: Mapped diagnostic codes into primary anatomical disease systems (Circulatory, Respiratory, Diabetes, Digestive, Genitourinary/Renal, Neoplasms, Musculoskeletal).
   - **Healthcare Velocity**: Captured past-year acute care utilization (`number_inpatient`, `number_emergency`, `number_outpatient`).
   - **Glycemic Biomarkers & Medication Complexity**: Evaluated inpatient HbA1c testing, insulin regimens, and medication changes.
4. **Class Imbalance Handling**:
   - 30-day readmissions represent 8.9% (10.2:1 imbalance ratio). Addressed via `scale_pos_weight = 10.19` in XGBoost and balanced class weighting in Logistic Regression.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/ashenjaygit/Hospital_Readmission_ML_System.git
cd Hospital_Readmission_ML_System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Clinical Dashboard
```bash
streamlit run app.py
```
*Or double-click `run_dashboard.bat` on Windows.*

### 4. Re-run Pipeline (Optional)
```bash
python pipeline.py
```
