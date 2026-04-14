<div align="center">

<br/>

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/SHAP-XAI-00D4FF?style=for-the-badge" />
<img src="https://img.shields.io/badge/License-MIT-34D399?style=for-the-badge" />

<br/><br/>

# 🏥 Smart Healthcare Prediction System

### *AI-Powered Clinical Decision Support with Explainable Machine Learning*

<p align="center">
  <em>A multi-module Streamlit application that brings ML-driven health risk assessment, medical NLP, and SHAP explainability into a single clinical-grade interface.</em>
</p>

<br/>

---

</div>

<br/>

## ✦ Overview

The **Smart Healthcare Prediction System** is a full-stack data science portfolio project that simulates a clinical decision-support tool. It leverages trained machine learning models, rule-based medical NLP, and SHAP explainability to provide transparent risk assessments across four clinical use cases — all from a single, beautifully designed Streamlit app.

> ⚠️ **Disclaimer:** This application is built for **educational and portfolio purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment.

<br/>

---

<br/>

## 🧩 Modules

<br/>

### `Module 1` — 🫀 Heart Disease Prediction
> Predicts the probability of coronary heart disease based on clinical vitals and lab values.

- **Inputs:** Age, resting BP, cholesterol, ECG results, max heart rate, ST depression, chest pain type, and more
- **Output:** Risk probability with a visual gauge meter
- **Model:** Trained classifier with feature scaling (`heart_model.pkl`, `heart_scaler.pkl`)
- **Explainability:** SHAP force plots showing per-feature contribution

<br/>

### `Module 2` — 🩸 Diabetes Risk Classifier
> Assesses diabetic risk using patient metabolic markers.

- **Inputs:** Glucose, BMI, insulin levels, pregnancies, blood pressure, skin thickness, age
- **Output:** Diabetic / Non-diabetic classification with probability score
- **Model:** Trained classifier with feature scaling (`diabetes_model.pkl`, `diabetes_scaler.pkl`)
- **Explainability:** SHAP waterfall & summary plots

<br/>

### `Module 3` — 🏨 ICU Readmission Risk (LACE Score)
> Estimates 30-day hospital readmission risk using LACE-inspired scoring.

- **Inputs:** ICU stay duration, prior admissions, number of diagnoses, creatinine, WBC, hemoglobin, discharge disposition
- **Output:** Composite LACE score (0–70), risk tier (Low / Moderate / High), and a risk factor breakdown bar chart

<br/>

### `Module 4` — 🧠 Medical NLP Report Analyzer
> Extracts clinical entities and scores risk from free-text clinical notes.

- **Inputs:** Raw clinical notes / discharge summaries (paste or type)
- **NER Tags:** Diseases, Symptoms, Medications, Anatomy, Vitals & Labs
- **Output:** Annotated report with color-coded entities, clinical risk score, and highlighted text
- **Technique:** Rule-based keyword NLP (extendable to spaCy / scispaCy)

<br/>

### `Module 5` — 📋 Patient History & Analytics
> Tracks and visualizes all predictions made in the current session.

- Session-level prediction log with timestamps
- Aggregated risk distribution chart (High / Moderate / Low)
- One-click history clear

<br/>

---

<br/>

## 🖥️ App Preview

<br/>

| Feature | Detail |
|---|---|
| **UI Theme** | Dark premium — deep navy background with blue/cyan accent glows |
| **Typography** | Playfair Display · DM Sans · DM Mono · Outfit |
| **Layout** | Wide layout with collapsible sidebar navigation |
| **Charts** | Matplotlib (dark-themed), SHAP plots, risk gauge SVG |
| **Responsiveness** | Fluid column layouts, metric cards, hover interactions |

<br/>

---

<br/>

## 🗂️ Project Structure

```
smart-healthcare-prediction/
│
├── app.py                    # Main Streamlit application (all 5 modules)
│
├── models/
│   ├── heart_model.pkl       # Trained heart disease classifier
│   ├── heart_scaler.pkl      # StandardScaler for heart features
│   ├── heart_features.pkl    # Feature name list for heart module
│   ├── diabetes_model.pkl    # Trained diabetes classifier
│   ├── diabetes_scaler.pkl   # StandardScaler for diabetes features
│   └── diabetes_features.pkl # Feature name list for diabetes module
│
└── requirements.txt          # Python dependencies
```

<br/>

---

<br/>

## ⚙️ Installation & Setup

<br/>

**1. Clone the repository**

```bash
git clone https://github.com/your-username/smart-healthcare-prediction.git
cd smart-healthcare-prediction
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the app**

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

<br/>

---

<br/>

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
shap>=0.44.0
joblib>=1.2.0
```

<br/>

---

<br/>

## 🔬 Technical Stack

<br/>

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit + Custom HTML/CSS |
| **ML Models** | Scikit-learn (classification) |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **NLP** | Rule-based entity extraction (keyword matching) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Serialization** | Joblib `.pkl` files |

<br/>

---

<br/>

## 🧠 ML + XAI Highlights

- **SHAP Integration** — Every prediction from heart disease and diabetes modules comes with a SHAP force plot, letting clinicians see *exactly* which features drove the risk score up or down.
- **Probability Calibration** — Models output calibrated probability scores, not just binary labels.
- **Visual Risk Gauges** — SVG-based semicircular gauge meters provide intuitive risk visualization.
- **LACE Score Logic** — ICU readmission module implements a clinically-inspired composite scoring formula using key discharge metrics.

<br/>

---

<br/>

## 📌 Key Features at a Glance

- ✅ 4 clinical prediction modules in one app
- ✅ SHAP explainability for every ML prediction
- ✅ Medical NLP with named entity recognition (NER)
- ✅ Session-based patient history tracking
- ✅ Premium dark UI with custom fonts & CSS animations
- ✅ Risk gauges, charts, and annotated report views
- ✅ Modular codebase — easy to extend with new models

<br/>

---

<br/>

## 🚀 Future Roadmap

- [ ] Add real-time DICOM / EHR data integration
- [ ] Upgrade NLP to spaCy + scispaCy biomedical model
- [ ] Deploy to Streamlit Cloud with persistent storage
- [ ] Add PDF report export for predictions
- [ ] Integrate LLM-based clinical reasoning (Claude / GPT)
- [ ] Multi-user support with login and history persistence

<br/>

---

<br/>

## 👤 Author

<br/>

**Aayush**
Final Year B.Tech — Information Technology
Chandigarh Engineering College, Landran | Class of 2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/your-username)

<br/>

---

<br/>

<div align="center">

**⭐ If you find this project useful, please consider starring the repository!**

<br/>

*Smart Healthcare Prediction System · v2.0 · Built with ❤️ for clinical AI education*

</div>
