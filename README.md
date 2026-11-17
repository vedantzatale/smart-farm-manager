# 🌾 Smart Farm Biosecurity AI Manager

An intelligent farm biosecurity assessment and disease risk prediction system powered by **XGBoost Machine Learning**, a **FastAPI** backend, and a modern **React + Vite** frontend dashboard.

Designed for livestock and poultry farm operators to monitor environmental conditions, track biosecurity compliance, predict outbreak risks in real time, and receive actionable prevention recommendations.

---

## 🚀 Key Features

- **🤖 AI-Powered Disease Risk Prediction**:
  - Multi-class classification (**Low**, **Moderate**, **High**) with detailed class probability distributions.
  - Trained on 20,000 farm records using an optimized **XGBoost Classifier**.
  - Evaluates **20 critical indicators** including environmental conditions, animal density, mortality spikes, visitor traffic, and biosecurity protocols.

- **🛡️ Algorithmic Biosecurity Scoring**:
  - Rule-based weighted scoring engine evaluating:
    - Vaccination Compliance (30%)
    - Cleaning & Sanitation Frequency (20%)
    - PPE Adherence (15%)
    - Visitor Controls & Biosecurity Gates (15%)
    - Mortality vs. Baseline Normal Mortality (15%)
    - Disinfection, Water Quality, & Pest Management
  - Classifies farm health into **Excellent (85–100)**, **Good (70–84)**, **Fair (50–69)**, or **Critical (<50)**.

- **💡 Dynamic Action Recommendations**:
  - Automatically generates prioritized medical, sanitation, quarantine, and operational recommendations tailored to daily risk assessments.

- **📊 Comprehensive Historical Analytics**:
  - SQLite-backed historical logging of all assessments.
  - Track trends in disease risk, biosecurity scores, temperature, humidity, and animal mortality over time.

- **💻 Modern Interactive Dashboard**:
  - Built with **React 19**, **Vite**, and **Lucide React**.
  - Interactive assessment forms, metric cards, progress rings, and historical log views.

---

## 🏗️ System Architecture

```
farm-biosecurity-ai-main/
│
├── backend/
│   ├── main.py                  # FastAPI server & API endpoints
│   ├── biosecurity.py           # Biosecurity score calculation engine
│   ├── recommendations.py       # Recommendation rule engine
│   ├── database.py              # SQLite connection & table schemas
│   ├── disease_risk_model.pkl   # Serialized XGBoost model
│   ├── risk_level_encoder.pkl   # Scikit-learn LabelEncoder
│   └── requirements.txt         # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main Dashboard UI & views
│   │   ├── App.css              # Dashboard styles
│   │   ├── main.jsx             # React entrypoint
│   │   └── assets/              # Logos and graphics
│   ├── index.html               # Vite HTML template
│   ├── package.json             # Node dependencies & scripts
│   └── vite.config.js           # Vite configuration
│
├── ml/
│   ├── generate_dataset.py      # Synthetic dataset generator (20,000 samples)
│   └── train_model.py           # Model training & evaluation pipeline
│
└── data/
    ├── farm_disease_data.csv
    └── farm_disease_data_20000.csv
```

---

## 🛠️ Tech Stack

- **Machine Learning & Data**: Python 3.10+, XGBoost, Scikit-learn, Pandas, NumPy, Joblib
- **Backend API**: FastAPI, Uvicorn, Pydantic, SQLite3
- **Frontend**: React 19, Vite, Lucide React, Modern CSS3
- **DevOps & Version Control**: Git, GitHub

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/vedantzatale/smart-farm-manager.git
cd smart-farm-manager
```

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI backend server
uvicorn main:app --reload --port 8001
```
*Backend will be running at:* `http://127.0.0.1:8001`  
*Interactive Swagger API Docs:* `http://127.0.0.1:8001/docs`

### 3. Frontend Setup
In a new terminal window:
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start development server
npm run dev
```
*Frontend will be running at:* `http://localhost:5173`

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status, version, and model specifications |
| `POST` | `/farms` | Register a new farm profile |
| `GET` | `/farms/{farm_id}` | Fetch farm details and configuration |
| `PUT` | `/farms/{farm_id}` | Update existing farm profile |
| `POST` | `/farms/{farm_id}/assessment` | Submit daily readings & run AI prediction |
| `GET` | `/farms/{farm_id}/history` | Retrieve historical assessments for a farm |

---

## 🧠 Machine Learning Model Training

To retrain the disease risk model:
```bash
cd ml
python train_model.py
```
This evaluates both Random Forest and XGBoost classifiers against the dataset and exports updated `.pkl` model files to `backend/`.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
