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

<img width="1918" height="863" alt="image" src="https://github.com/user-attachments/assets/d8ea524b-b16b-4f21-b4ff-88087763490b" />

<img width="1918" height="862" alt="image" src="https://github.com/user-attachments/assets/d518dd8f-5784-4176-bc3d-ecb1995dd324" />

<img width="1918" height="868" alt="image" src="https://github.com/user-attachments/assets/54f11cd3-f507-4d28-83b7-521b238f1676" />

<img width="1918" height="865" alt="image" src="https://github.com/user-attachments/assets/c548b20f-9982-476f-b7e2-3f826a30d89f" />

Prototype: https://risklens123.vercel.app/
