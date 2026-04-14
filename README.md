🏥 Smart Healthcare Prediction System (HealthAI)

An advanced AI-powered healthcare prediction system for early risk detection and clinical insights.

🔗 Live Demo:
👉 https://smart-healthcare-prediction-system-jni2e7swd9ejwje4gpqvct.streamlit.app/

🚀 Overview

HealthAI is a multi-functional healthcare prediction platform that uses Machine Learning + NLP to analyze patient data and provide:

Disease risk prediction
Clinical decision support
Explainable AI insights
✨ Key Features
🫀 Heart Disease Prediction
Predicts cardiac risk using clinical parameters
Displays probability, confidence & insights
🩸 Diabetes Prediction
Estimates diabetes probability
Calculates approximate HbA1c levels
🏨 ICU Readmission Analysis
Uses LACE-based scoring system
Predicts 30-day readmission risk
📄 Medical NLP Analyzer
Extracts:
Diseases
Symptoms
Medications
Vitals
Generates clinical risk score
📊 Explainable AI (SHAP)
Visualizes feature importance
Helps understand model decisions
📈 Interactive Dashboard
Modern UI with animations
Real-time metrics & gauge charts
Patient history tracking
🧠 Tech Stack
Category	Technologies Used
Frontend	Streamlit
Backend	Python
Machine Learning	Scikit-learn
Data Processing	Pandas, NumPy
Visualization	Matplotlib
Explainability	SHAP
Model Storage	Joblib
📂 Project Structure
Smart-Healthcare-Prediction-System/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
│
├── models/                 # Trained ML models
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── diabetes_features.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── heart_features.pkl
│
└── README.md               # Project documentation
⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-username/Smart-Healthcare-Prediction-System.git
cd Smart-Healthcare-Prediction-System
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Application
streamlit run app.py
📊 How It Works
User enters medical data
Data is preprocessed & scaled
ML models generate predictions
Results displayed with:
Risk probability
Confidence score
Visual gauge
SHAP explanations
📸 Modules Summary
Module	Description
🫀 Heart Disease	Predicts cardiac risk
🩸 Diabetes	Predicts diabetes + HbA1c
🏨 ICU	Readmission risk analysis
📄 NLP	Clinical text analysis
⚠️ Disclaimer

This project is intended for educational purposes only.
It is not a substitute for professional medical advice.

👨‍💻 Author

Aayush Sharma
💡 AI & Data Science Enthusiast

⭐ Contribution

Contributions are welcome!

Fork the repository
Create a new branch
Submit a Pull Request
💡 Future Improvements
Add more disease prediction models
Integrate real-time healthcare APIs
Deploy with authentication system
Mobile-friendly UI
