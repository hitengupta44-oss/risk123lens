# RiskLens
Project Overview
RiskLens is a full-stack AI health screening web application that predicts a user's risk for 6 major diseases using an ensemble of Naive Bayes and Bayesian Network models. Users sign in with Google, answer 21 health questions, and receive an instant personalised risk report with doctor recommendations.

Component	Technology	Hosting
ML Backend	FastAPI + scikit-learn + pgmpy	Hugging Face Spaces (Free)
Frontend	Next.js 14 + Tailwind CSS	Vercel (Free)
Authentication	Firebase Auth (Google)	Firebase (Free)
ML Models	CategoricalNB + Bayesian Network	Trained at build time

🧬 Diseases Screened
Disease	Key Features Used	Risk Factors
 Diabetes	Sugar Level, Frequent Urination, Excessive Thirst	Family history, Fatigue
 Heart Disease	Chest Pain, Blood Pressure, Smoking	Family history, Alcohol
 CKD	Swelling Ankles, Frequent Urination	Blood Pressure
 Asthma	Wheezing, Breathlessness, Cough	Smoking
 Dyslipidemia	Diet Quality, Physical Activity	Smoking, Alcohol
 Anemia	Pale Skin, Fatigue, Weight Loss	Dizziness

🏗️ Architecture
ML Model Pipeline
•	Data loaded from diseasefinalset.xlsx (40,000 patient records)
•	Ordinal encoding for Blood Pressure, Diet Quality, Physical Activity
•	Label encoding for binary Yes/No features
•	Age grouped into: Young (≤30), Adult (≤45), Middle (≤60), Senior (60+)
•	CategoricalNB trained per disease — 80/20 train/test split
•	Bayesian Network trained per disease using Maximum Likelihood Estimation
•	Ensemble: Final Risk % = (0.6 × Naive Bayes) + (0.4 × Bayesian Network)
•	All models pickled to models/artifacts.pkl at Docker build time

Risk Bands
Risk %	Band	Colour	Action
0 – 20%	Very Low	🟢 Green	Maintain healthy lifestyle
21 – 40%	Low	🟢 Green	Annual screening recommended
41 – 60%	Moderate	🟡 Amber	Consult doctor for tests
61 – 80%	High	🔴 Red	Prompt medical consultation
81 – 100%	Very High	🔴 Red	Urgent medical attention

<img width="1916" height="876" alt="image" src="https://github.com/user-attachments/assets/526ee259-be23-4cb4-b89b-67a0594162a6" />

<img width="1918" height="873" alt="image" src="https://github.com/user-attachments/assets/fd84a968-d3fe-4538-bb3b-c7c75c18905b" />

<img width="1918" height="871" alt="image" src="https://github.com/user-attachments/assets/c333b49f-11d1-4472-94b7-38368b3d9a8d" />


