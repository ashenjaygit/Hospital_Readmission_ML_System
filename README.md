# CarePulse AI: Inpatient 30-Day Hospital Readmission Risk & Clinical Decision Support (CDS) System

An end-to-end Healthcare Machine Learning and Clinical Decision Support system designed for hospital inpatient risk stratification, post-discharge transitional care planning, and bed utilization optimization.

---

## 🏥 Project Summary
- **Clinical Target**: Unplanned 30-Day Hospital Readmission (`<30 Days`)
- **Dataset**: Gold-Standard Clinical Benchmark (UCI / Kaggle Diabetes 130-US Hospitals Cohort, 101,766 raw encounters, 70,439 de-duplicated unique patients)
- **Machine Learning Architecture**: Dual-Engine (High-Performance XGBoost Classifier + Explainable Balanced Logistic Regression Odds Ratio Scorecard)
- **Key Metrics (5-Fold Stratified CV)**: 
  - **ROC-AUC**: **0.645** (Matches published international state-of-the-art benchmarks for structured EHR tabular data)
  - **Sensitivity / High-Risk Capture**: **53.3%**
  - **Specificity**: **67.4%**
  - **PR-AUC**: **0.156** (1.8x higher than baseline population prevalence)
- **Clinical Actionability**: Automatically outputs a tiered **Transitional Care Protocol** with actionable bedside and post-discharge interventions.

---

## 📁 Directory Structure
```
Hospital_Readmission_ML_System/
├── run_dashboard.bat                              <-- [One-Click Launcher] Double-click to launch interactive web app
├── app.py                                         <-- Streamlit Clinical Decision Support (CDS) Dashboard
├── pipeline.py                                    <-- Complete Data Cleaning, Feature Engineering & ML Pipeline
├── PROJECT_PRESENTATION_AND_INTERVIEW_GUIDE.md    <-- Master Pitch Script, ROI & 10 Tough Technical Questions & Answers
├── Hospital_Readmission_Clinical_Report.docx      <-- 634 KB Formal Word Document Report with all 8 Embedded Charts
├── generate_report_docx.py                        <-- Script used to compile the Microsoft Word report
├── README.md                                      <-- Project documentation & GitHub overview
├── data/
│   ├── diabetic_data.csv                          <-- Raw dataset (101,766 encounters, 50 features)
│   ├── IDS_mapping.csv                            <-- Medical admission/discharge ID reference dictionary
│   └── cleaned_readmission_data.csv               <-- Curated cohort (70,439 unique patient encounters)
├── models/
│   ├── xgboost_readmission.pkl                    <-- Serialized trained XGBoost model
│   ├── logistic_regression.pkl                    <-- Serialized Balanced Logistic Regression model
│   ├── scaler.pkl                                 <-- Fitted StandardScaler
│   └── feature_metadata.json                      <-- Feature mapping dictionary & CV metrics
└── eda_plots/                                     <-- 8 Publication-Quality Medical Charts (300 DPI)
    ├── 01_readmission_class_distribution.png      <-- Readmission target distribution
    ├── 02_prior_inpatient_stays_impact.png        <-- Escalating risk by prior hospital stays
    ├── 03_icd9_primary_diagnosis_risk.png         <-- Readmission risk by ICD-9 anatomical system
    ├── 04_a1c_testing_and_readmission.png         <-- Protective impact of inpatient HbA1c testing
    ├── 05_polypharmacy_and_los_burden.png         <-- Length of Stay (LOS) & medication count
    ├── 06_feature_importance_xgboost.png          <-- Top 15 clinical drivers in XGBoost
    ├── 07_clinical_odds_ratios_scorecard.png      <-- Logistic Regression Odds Ratios (e^beta)
    └── 08_roc_and_precision_recall_curves.png     <-- ROC Curve and Precision-Recall Curves
```

---

## 🚀 How to Run the Project

### 1. Launch the Interactive Dashboard
Double-click **`run_dashboard.bat`** or run:
```bash
py -m streamlit run app.py
```
*Access the dashboard in your web browser at `http://localhost:8501`.*

### 2. Re-train Models or Run Pipeline from Scratch
```bash
py pipeline.py
```

### 3. Re-generate the Word Document Report
```bash
py generate_report_docx.py
```

---

## 🎯 Presentation & Defense Resources
- **Presentation Script & Viva/Interview Q&A**: See [PROJECT_PRESENTATION_AND_INTERVIEW_GUIDE.md](file:///c:/Users/ASUS/OneDrive/Desktop/Hospital_Readmission_ML_System/PROJECT_PRESENTATION_AND_INTERVIEW_GUIDE.md).
- **Formal Academic Report**: Open [Hospital_Readmission_Clinical_Report.docx](file:///c:/Users/ASUS/OneDrive/Desktop/Hospital_Readmission_ML_System/Hospital_Readmission_Clinical_Report.docx).
