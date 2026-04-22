# 🦟 Dengue Surveillance System
**Tracking & Analyzing Dengue Cases for Improved Outbreak Monitoring — Agusan del Sur**

> ISELEC 104 · Ive Jane B. Sabando · Python · Streamlit · Scikit-learn · Plotly

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📌 About

A full-featured **Dengue Surveillance Dashboard** built entirely in Python with Streamlit.  
It applies machine learning, time-series forecasting, and exploratory data analysis to dengue surveillance data for improved outbreak monitoring and early warning detection in Agusan del Sur.

---

## 🚀 Features

| Tab | What it does |
|-----|-------------|
| 📋 **Data & EDA** | Upload CSV/Excel · cleaned data preview · distributions · correlation heatmap |
| 🗺️ **Geographic** | Cases per municipality · scatter plots · High / Medium / Low alert system |
| 📈 **Time Series** | Multi-year trend lines · seasonal heatmap · rainfall & temperature overlay |
| 🔬 **Clustering** | K-Means risk profiling · elbow method · silhouette score · PCA scatter |
| 🤖 **Prediction** | Random Forest · Gradient Boosting · Logistic Regression · ARIMA forecast |
| 📊 **Model Comparison** | 5-fold cross-validation · radar chart · metric bar comparison |

---

## 🛠️ How to Run Locally

### 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/dengue-surveillance.git
cd dengue-surveillance
```

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Run the app
```bash
streamlit run app.py
```

Opens at **http://localhost:8501** 🎉

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub (public)
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Click **New app**
4. Select your repo · branch: `main` · main file: `app.py`
5. Click **Deploy** — live in ~2 minutes!

---

## 📂 File Structure

```
dengue-surveillance/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 📊 Expected Data Format

Upload a **CSV or Excel** file with these columns (sample data is built-in if no file is uploaded):

| Column | Type | Example |
|--------|------|---------|
| `Year` | int | 2025 |
| `Month` | str | February |
| `Month_Num` | int | 2 |
| `Municipality` | str | Bunawan |
| `Barangay` | str | Barangay 3 |
| `Age_Group` | str | 20-29 |
| `Sex` | str | Female |
| `Clinical_Classification` | str | Dengue With Warning Signs |
| `Outcome` | str | Recovered |
| `Hospitalized` | int | 0 or 1 |
| `Deaths` | int | 0 or 1 |
| `Families_Affected` | int | 2 |
| `Rainfall_mm` | float | 145.3 |
| `Temperature_C` | float | 31.2 |
| `Humidity_pct` | float | 78.5 |
| `Water_Level` | float | 2.4 |
| `Weather` | str | Rainy |

---

## 🤖 Models Used

| Model | Purpose |
|-------|---------|
| **K-Means** | Municipality risk profiling & clustering |
| **Random Forest** | Hospitalization outbreak prediction |
| **Gradient Boosting** | Hospitalization outbreak prediction |
| **Logistic Regression** | Baseline classifier |
| **Decision Tree** | Interpretable prediction |
| **Naive Bayes** | Probabilistic baseline |
| **ARIMA** | Monthly case count forecasting |

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
statsmodels>=0.14.0
openpyxl>=3.1.0
```

---

## 👩‍💻 Author

**Ive Jane B. Sabando**  
ISELEC 104 — Information Systems Elective  
Academic Year 2025–2026

---

## 📄 License

MIT License — Free to use and modify for academic and research purposes.
