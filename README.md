# 🏦 Bank Marketing Intelligence System (PRISM)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green?style=for-the-badge)

## 📖 Project Overview
**PRISM (Predictive Real-time Intelligence for Strategic Marketing)** is an end-to-end Machine Learning solution designed to optimize bank telemarketing campaigns. 

Instead of random calling ("spray and pray"), this system analyzes customer demographics, financial history, and **macroeconomic indicators** to predict the likelihood of a customer subscribing to a term deposit.

### 🚀 Key Business Impact
Based on the validation results in `CapstoneProject_PRISM.ipynb`:
* **5x Efficiency:** The model is 5x more effective than random calling.
* **Pareto Principle:** By targeting only the **Top 20%** of scored leads, the bank captures **~70%** of all interested customers.
* **High Recall:** Optimized to ensure potential customers are not missed (Recall ~88.4%).

---

## 🛠️ Key Features

### 1. Machine Learning Engine
* **Algorithm:** Optimized classification model (Pipeline with StandardScaler, OneHotEncoder).
* **Threshold Tuning:** Custom decision threshold set at **0.27** to prioritize Recall over Precision.
* **Fatigue Analysis:** Automatically flags clients who have been contacted too frequently ("Spam Risk").

### 2. FastAPI Deployment (`main.py`)
The model is served via a REST API with the following capabilities:
* **`POST /predict`**: Real-time single prediction for walk-in or live-call scenarios.
* **`POST /predict_batch`**: Bulk processing via CSV upload for daily campaign generation.
* **Economic Context:** Automatically fetches/integrates current economic indicators (e.g., interest rates, CPI) into the prediction logic.

---

## 📊 Model Performance

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Recall** | **88.4%** | The model successfully identifies ~88% of all actual positive leads. |
| **True Positives** | 820 | Successfully captured interested customers in the test set. |
| **Threshold** | 0.27 | Lowered from 0.5 to maximize opportunity capture. |

> *"With this model, the marketing team can focus on the 'High Potential' leads, ignoring the bottom 50% of prospects who are unlikely to convert."*

---

## 💻 Tech Stack
* **Core:** Python 3.9+
* **ML & Data:** Scikit-Learn, Pandas, NumPy, Imbalanced-learn (SMOTE).
* **Web Framework:** FastAPI, Uvicorn.
* **Utils:** Joblib (Model serialization), Pydantic (Data validation).

---

## 📂 Project Structure
```bash
├── CapstoneProject_PRISM.ipynb  # Model training, EDA, and evaluation notebook
├── main.py                      # FastAPI application entry point
├── models/
│   └── model_bank_lead_scoring.joblib  # Serialized trained model
├── services/
│   └── economic_data.py         # Helper to fetch economic indicators
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```
## ⚡ How to Run

### 1. Setup Environment
```bash
# Clone repository
git clone [https://github.com/RPriago/bank_lead_ML.git](https://github.com/RPriago/bank_lead_ML.git)
cd bank_lead_ML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


