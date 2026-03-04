# 🚢 KPA Vehicle Traffic & Congestion Analytics Dashboard

> **Kenya Ports Authority — Policy and Research Section**
> Research Report on Vehicle Traffic and Congestion at KPA Gates · June 2025

---

## 📌 Overview

This is an interactive **Streamlit analytics dashboard** built on top of the Kenya Ports Authority (KPA) research survey conducted in May–June 2025. The study investigated vehicle traffic congestion at KPA access gates — Port of Mombasa and ICD Nairobi — and collected responses from **879 respondents** across five stakeholder groups.

The dashboard provides:
- **Exploratory data analysis** of all five survey datasets
- **Machine learning models** to predict congestion risk and gate wait time
- **Interactive prediction tool** for individual driver profiling
- **Recommendation matrix** from the official KPA research report

---

## 📁 Project Structure
```
📦 kenya-port/
├── port.py                    # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── COMBINED_DATASETS.csv      # Combined survey data (upload in app)
```

### Original Source Datasets (upload in app)

| File | Respondents | Description |
|------|------------|-------------|
| `TRUCK_EXCEL_DATASET.xlsx` | 714 | Truck drivers — gate usage, wait times, congestion causes |
| `CLEARING_AGENTS_EXCEL_ANALYSIS.xlsx` | ~124 | Clearing agents — documentation, gate usage |
| `KPA_STAFF_EXCEL_DATASET.xlsx` | 22 | KPA staff — operations, staffing, infrastructure |
| `CUSTOM_EXCEL_DATASET.xlsx` | 10 | Customs/KRA officials — scanning, documentation |
| `TRAFFIC_POLICE_EXCEL_DATASET.xlsx` | 9 | Traffic police — congestion causes, solutions |

---

## 🚀 Getting Started

### 1. Clone or Download the Project
```bash
git clone https://github.com/your-org/kenya-port.git
cd kenya-port
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run port.py
```

### 4. Upload Your Data

On the landing screen, upload either:

- **Option A** — The combined `COMBINED_DATASETS.csv` file (fastest)
- **Option B** — All 5 original `.xlsx` Excel files (auto-merged on upload)

---

## 📦 Requirements
```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
openpyxl>=3.1.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

> **Python version:** 3.9 or higher recommended

---

## 📊 Dashboard Pages

### 1. 📊 Executive Dashboard
High-level KPIs and summary charts including:
- Container throughput (2.005M TEUs in 2024, +14% YoY)
- Truck waiting time distribution (33% of drivers wait >5 hours)
- Congestion frequency breakdown
- Monthly traffic volumes January–June 2025

### 2. 👥 Demographics
Respondent profile analysis:
- Nationality distribution (88.4% Kenyan truck drivers)
- Gender breakdown across all stakeholder groups
- Years of experience distribution
- Gate visit frequency patterns

### 3. 🚦 Traffic Patterns
Gate usage and temporal congestion patterns:
- Most used gates (Gate 24: 48.1%, Gate 18: 31.2% of truck drivers)
- Time-of-day congestion (40% afternoon, 30% evening)
- Weekly entry/exit traffic trends
- Operational shift analysis and ICD Nairobi vehicle movements

### 4. ⚠️ Congestion Causes
Multi-stakeholder analysis of congestion drivers:
- Top causes: slow gate processing (24.6%), security checks (21.1%), KRA gadget delays (17.7%)
- Impact on operations: longer working hours, increased fuel costs, missed deliveries
- Stakeholder-proposed solutions ranked by recommendation rate

### 5. 🤖 ML Predictive Models
Four machine learning models trained and compared:

| Model | Purpose |
|-------|---------|
| Random Forest | Primary predictor (highest AUC) |
| Gradient Boosting | Ensemble benchmark |
| Logistic Regression | Baseline linear model |
| Decision Tree | Interpretable rule-based model |

**Two prediction targets:**
- **High Congestion Risk** — predicts whether a driver will *always/often* experience congestion
- **Long Wait Time Risk** — predicts whether a driver will wait more than 2 hours at a gate

### 6. 🔍 Predict for New Driver
Interactive prediction form. Enter a driver's profile to get:
- Congestion risk probability (%)
- Wait time risk probability (%)
- Overall operational risk gauge (0–100)
- Personalised mitigation recommendations

### 7. 📋 Recommendations
Full recommendation matrix with problem, recommendation, responsible agency, priority level, and implementation timeline.

---

## 🤖 Machine Learning Details

### Feature Engineering

| Variable | Encoding |
|---------|---------|
| Waiting time | 0 (< 30 min) → 4 (> 5 hrs) |
| Congestion frequency | 0 (Never) → 4 (Always) |
| Years of experience | 0 (< 1 yr) → 3 (> 10 yrs) |
| Visit frequency | 0 (Rarely) → 4 (Daily) |

### Target Variables
- `high_congestion` — 1 if driver reports "Always" or "Often" experiencing congestion
- `long_wait` — 1 if driver reports waiting more than 2 hours per gate visit

---

## 📋 Key Findings

| Metric | Value |
|--------|-------|
| 2024 Container Throughput | 2.005M TEUs (+14%) |
| Truck Turnaround Target | ≤ 30 minutes |
| Actual Truck Turnaround | 90–180 minutes |
| Drivers Waiting > 5 hrs | 33% |
| Clearing Agents Always Congested | 59% |
| Peak Congestion Window | 2 PM – 6 PM (40%) |
| Most Used Gate | Gate 24 (48.1%) |

---

## 📜 Data Sources

- **Fieldwork period:** May – June 2025
- **Study area:** Port of Mombasa & ICD Nairobi
- **Methodology:** Mixed-methods (quantitative surveys + qualitative interviews)
- **Theoretical framework:** Kerner's Three-Phase Traffic Flow Theory
- **Tool:** SPSS, Excel, Python

---

## 👥 Research Team

| Name | Role |
|------|------|
| Weldon Korir | Chair — Corporate Research & Policy |
| Sipei Ntome | Technical Member |
| Mary Munyi | Technical Member |
| Denis Simon | Technical Member |
| Phylis Muthee | Security Services |
| Umi Bakari | Secretariat |

---

## 📄 License

Property of **Kenya Ports Authority**. For internal research and operational use only.

---

*Built with Streamlit · Pandas · Plotly · scikit-learn*
```

---

## 📄 `requirements.txt`
```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
openpyxl>=3.1.0
```

---

## 🛠️ Step-by-step instructions

**Step 1** — Create a new folder on your computer:
```
kenya-port/
