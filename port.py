"""
KPA Vehicle Traffic & Congestion Analytics Dashboard
======================================================
Research Report: Vehicle Traffic and Congestion at KPA Gates (June 2025)
Kenya Ports Authority – Policy and Research Section

HOW TO RUN:
    pip install streamlit pandas scikit-learn matplotlib seaborn plotly openpyxl
    streamlit run kpa_traffic_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings("ignore")

# ── ML imports ──────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
from sklearn.inspection import permutation_importance

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="KPA Traffic Analytics",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #003087 0%, #0056b3 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { font-size: 2rem; margin: 0; font-weight: 700; }
    .main-header p  { font-size: 0.95rem; margin: 0.4rem 0 0; opacity: 0.85; }

    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #003087;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    }
    .metric-card h2 { color: #003087; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #555; font-size: 0.85rem; margin: 0.2rem 0 0; }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #003087;
        border-bottom: 2px solid #003087;
        padding-bottom: 6px;
        margin-bottom: 1rem;
    }
    .insight-box {
        background: #f0f6ff;
        border-left: 4px solid #003087;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.6rem 0;
        font-size: 0.9rem;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.6rem 0;
        font-size: 0.9rem;
    }
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.6rem 0;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e8ecf2;
        border-radius: 6px 6px 0 0;
        padding: 6px 18px;
        font-weight: 600;
        color: #003087;
    }
    .stTabs [aria-selected="true"] {
        background-color: #003087 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING — FILE UPLOAD
# ============================================================
@st.cache_data
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data
def combine_excels(file_bytes_dict):
    """Combine multiple raw Excel files into one dataset."""
    source_map = {
        "clearing": "CLEARING_AGENTS",
        "custom":   "CUSTOM",
        "kpa":      "KPA_STAFF",
        "traffic":  "TRAFFIC_POLICE",
        "truck":    "TRUCK",
    }
    dfs = []
    for key, file_bytes in file_bytes_dict.items():
        source = source_map.get(key, key.upper())
        df = pd.read_excel(io.BytesIO(file_bytes), header=None)
        df.insert(0, "Source_Dataset", source)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    # Promote first data row as header for non-clearing sheets
    return combined

def show_upload_screen():
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#003087,#0056b3);padding:2rem;
                border-radius:14px;color:white;text-align:center;margin-bottom:1.5rem;">
        <img src="data:image/png;base64,{KPA_LOGO_B64}"
             style="width:90px;border-radius:50%;margin-bottom:0.8rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.3);">
        <h1 style="margin:0;font-size:2rem;">KPA Traffic Analytics</h1>
        <p style="opacity:.85;margin:.5rem 0 0;">Kenya Ports Authority · June 2025 Research Report</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Upload Your Dataset to Get Started")

    tab_csv, tab_excel = st.tabs(["📄 Upload Combined CSV", "📊 Upload Raw Excel Files"])

    with tab_csv:
        st.markdown("""
        <div style="background:#f0f6ff;border-left:4px solid #003087;padding:.8rem 1rem;
                    border-radius:6px;margin-bottom:1rem;">
        Upload the <b>COMBINED_DATASETS.csv</b> file generated from the five original Excel datasets.
        </div>
        """, unsafe_allow_html=True)
        uploaded_csv = st.file_uploader(
            "Drop COMBINED_DATASETS.csv here",
            type=["csv"],
            key="csv_uploader",
            help="The combined CSV file from all 5 datasets"
        )
        if uploaded_csv:
            with st.spinner("Loading dataset..."):
                df = load_csv(uploaded_csv.read())
            st.success(f"✅ Loaded **{len(df):,} rows × {len(df.columns)} columns** from {uploaded_csv.name}")
            st.session_state["df"] = df
            st.rerun()

    with tab_excel:
        st.markdown("""
        <div style="background:#f0f6ff;border-left:4px solid #003087;padding:.8rem 1rem;
                    border-radius:6px;margin-bottom:1rem;">
        Upload all <b>5 original Excel files</b> together. They will be auto-combined.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            f_truck    = st.file_uploader("🚛 Truck Dataset",          type=["xlsx"], key="truck")
            f_clearing = st.file_uploader("📋 Clearing Agents Dataset", type=["xlsx"], key="clearing")
            f_custom   = st.file_uploader("🏛️ Custom/KRA Dataset",      type=["xlsx"], key="custom")
        with col2:
            f_kpa      = st.file_uploader("👷 KPA Staff Dataset",       type=["xlsx"], key="kpa")
            f_traffic  = st.file_uploader("🚔 Traffic Police Dataset",  type=["xlsx"], key="traffic")

        files_uploaded = {k: f for k, f in {
            "truck": f_truck, "clearing": f_clearing,
            "custom": f_custom, "kpa": f_kpa, "traffic": f_traffic
        }.items() if f is not None}

        if files_uploaded:
            st.info(f"📁 {len(files_uploaded)}/5 files uploaded: {', '.join(files_uploaded.keys())}")

        if len(files_uploaded) == 5:
            if st.button("🔗 Combine & Load All Files", use_container_width=True):
                with st.spinner("Combining datasets..."):
                    file_bytes_dict = {k: f.read() for k, f in files_uploaded.items()}
                    df = combine_excels(file_bytes_dict)
                st.success(f"✅ Combined **{len(df):,} rows × {len(df.columns)} columns** from 5 Excel files")
                st.session_state["df"] = df
                st.rerun()
        elif files_uploaded:
            st.warning(f"Please upload all 5 files ({5 - len(files_uploaded)} remaining).")

    # Demo note
    st.markdown("""
    ---
    <div style="background:#fff8e1;border-left:4px solid #ff9800;padding:.8rem 1rem;
                border-radius:6px;font-size:.88rem;">
    <b>💡 Tip:</b> The Combined CSV is preferred for fastest loading.
    You can generate it from the 5 Excel files using the <b>Upload Raw Excel Files</b> tab above,
    or by running the provided <code>generate_csv.py</code> script locally.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Route: show uploader or load from session ────────────────────────────────
if "df" not in st.session_state:
    show_upload_screen()

df_raw = st.session_state["df"]

@st.cache_data
def get_truck_data(df):
    trucks = df[df["Source_Dataset"] == "TRUCK"].copy()
    trucks = trucks[[
        "Nationality","Gender","Yearsexperience","Visitfrequency",
        "Gate18","Gate24","Gates9","Gate12","Gate15","Gate16","ICDGATES",
        "Containerized","Empty","Bulk","Breakbulk","Refridgerated",
        "Morning","Midday","Afternoon","Evening",
        "Averagewaitingtime","Trafficcongestionfrequency",
        "Toomanytrucks","clearance","securitychecks","Gateprocessing",
        "Trackinggadgets","Roadconditions","Gatelanes","Truckscheduling",
        "Fuelcost","Increasedtunaruondtimes","misseddeliveryschedules",
        "longerworkinghours","Increaseddemurrage","Delayinstacking",
        "Increasedstoragefees","stressorfatigue","Nosignificantimpact"
    ]].reset_index(drop=True)
    return trucks

@st.cache_data
def get_all_sources(df):
    return {
        "TRUCK"          : df[df["Source_Dataset"] == "TRUCK"],
        "CUSTOM"         : df[df["Source_Dataset"] == "CUSTOM"],
        "KPA_STAFF"      : df[df["Source_Dataset"] == "KPA_STAFF"],
        "TRAFFIC_POLICE" : df[df["Source_Dataset"] == "TRAFFIC_POLICE"],
        "CLEARING_AGENTS": df[df["Source_Dataset"] == "CLEARING_AGENTS"],
    }

# ============================================================
# ML FEATURE ENGINEERING
# ============================================================
@st.cache_data
def prepare_ml_features(trucks):
    df = trucks.copy()

    # Binary encode selected / not selected columns
    binary_cols = [
        "Gate18","Gate24","Gates9","Gate12","Gate15","Gate16","ICDGATES",
        "Containerized","Empty","Bulk","Breakbulk","Refridgerated",
        "Morning","Midday","Afternoon","Evening",
        "Toomanytrucks","clearance","securitychecks","Gateprocessing",
        "Trackinggadgets","Roadconditions","Gatelanes","Truckscheduling",
        "Fuelcost","Increasedtunaruondtimes","misseddeliveryschedules",
        "longerworkinghours","Increaseddemurrage","Delayinstacking",
        "Increasedstoragefees","stressorfatigue","Nosignificantimpact"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: 1 if str(x).strip().lower() == "selected" else 0)

    # Encode ordinals
    wait_map = {
        "Less than 30 mins":0, "30 min-1 hr":1, "1-2 hrs":2, "2-5 hrs":3, "over 5 hrs":4
    }
    df["wait_encoded"] = df["Averagewaitingtime"].map(wait_map)

    freq_map = {"Never":0,"Rarely":1,"Sometimes":2,"often":3,"Often":3,"Always":4}
    df["congestion_encoded"] = df["Trafficcongestionfrequency"].map(freq_map)

    exp_map = {"Less than 1 year":0,"1-5 years":1,"6-10 yeras":2,"Over 10 years":3}
    df["exp_encoded"] = df["Yearsexperience"].map(exp_map)

    visit_map = {
        "Rarely less than once per month":0,
        "A few times a month,1-3 times":1,
        "Once a week":2,
        "several times a week,2-4 times":3,
        "Daily":4
    }
    df["visit_encoded"] = df["Visitfrequency"].map(visit_map)

    df["is_kenyan"] = (df["Nationality"] == "Kenya").astype(int)
    df["is_male"]   = (df["Gender"] == "Male").astype(int)

    # Target: High Congestion (Always/Often = 1)
    df["high_congestion"] = df["congestion_encoded"].apply(lambda x: 1 if x >= 3 else 0)

    # Target: Long Wait (over 2 hrs = 1)
    df["long_wait"] = df["wait_encoded"].apply(lambda x: 1 if x >= 3 else 0)

    feature_cols = [
        "is_kenyan","is_male","exp_encoded","visit_encoded",
        "Gate18","Gate24","Gates9","Gate12","Gate16","ICDGATES",
        "Morning","Midday","Afternoon","Evening",
        "Containerized","Empty","Bulk","Toomanytrucks",
        "clearance","securitychecks","Gateprocessing","Trackinggadgets",
        "Roadconditions","Gatelanes","Truckscheduling"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="main-header">
    <img src="data:image/png;base64,{KPA_LOGO_B64}"
         style="width:70px;border-radius:50%;margin-bottom:0.6rem;
                box-shadow:0 2px 10px rgba(0,0,0,0.3);vertical-align:middle;">
    <h1>KPA Vehicle Traffic &amp; Congestion Analytics</h1>
    <p>Research Report · Port of Mombasa &amp; ICD Nairobi · June 2025 · Kenya Ports Authority</p>
</div>
""", unsafe_allow_html=True)

# Load
# ── Initialize data from session state ──────────────────────────────────────
trucks   = get_truck_data(df_raw)
sources  = get_all_sources(df_raw)
df_ml, feature_cols = prepare_ml_features(trucks)

# ============================================================
# SIDEBAR
# ============================================================
# ── Embedded KPA Logo (base64) ──────────────────────────────────────────────
KPA_LOGO_B64 = "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"

with st.sidebar:
    st.markdown(
        f'''<div style="text-align:center; padding: 0.8rem 0 0.4rem;">
            <img src="data:image/png;base64,{KPA_LOGO_B64}"
                 style="width:140px; border-radius:50%;
                        box-shadow:0 2px 10px rgba(0,48,135,0.3);">
            <div style="font-size:0.8rem; color:#003087; font-weight:700;
                        margin-top:0.4rem; letter-spacing:0.5px;">
                KENYA PORTS AUTHORITY
            </div>
        </div>''',
        unsafe_allow_html=True
    )
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Executive Dashboard",
        "👥 Demographics",
        "🚦 Traffic Patterns",
        "⚠️  Congestion Causes",
        "🤖 ML Predictive Models",
        "🔍 Predict for New Driver",
        "📋 Recommendations"
    ])
    st.markdown("---")
    st.markdown("**Dataset Summary**")
    st.markdown(f"- Total records: **{len(df_raw):,}**")
    st.markdown(f"- Truck Drivers: **{len(sources['TRUCK']):,}**")
    st.markdown(f"- Clearing Agents: **{len(sources['CLEARING_AGENTS'])-5:,}** (~survey)")
    st.markdown(f"- KPA Staff: **{len(sources['KPA_STAFF'])}**")
    st.markdown(f"- Customs Officials: **{len(sources['CUSTOM'])}**")
    st.markdown(f"- Traffic Police: **{len(sources['TRAFFIC_POLICE'])}**")
    st.markdown("---")
    if st.button("🔄 Upload New Dataset", use_container_width=True):
        del st.session_state["df"]
        st.rerun()

# ============================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# ============================================================
if page == "📊 Executive Dashboard":
    st.markdown('<div class="section-title">📊 Executive Dashboard</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("2.005M TEUs", "2024 Container Throughput"),
        ("14%", "YoY Cargo Growth"),
        ("33%", "Drivers Wait >5 hrs"),
        ("59%", "Agents: Always Congested"),
        ("90-180 min", "Avg Truck Turnaround"),
    ]
    for col, (val, lbl) in zip([c1,c2,c3,c4,c5], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <h2>{val}</h2>
            <p>{lbl}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Waiting Time Distribution – Truck Drivers</div>', unsafe_allow_html=True)
        wait_counts = trucks["Averagewaitingtime"].value_counts()
        order = ["Less than 30 mins","30 min-1 hr","1-2 hrs","2-5 hrs","over 5 hrs"]
        wait_counts = wait_counts.reindex(order)
        colors = ["#2ecc71","#f1c40f","#e67e22","#e74c3c","#8e44ad"]
        fig = px.bar(x=wait_counts.index, y=wait_counts.values,
                     color=wait_counts.index,
                     color_discrete_sequence=colors,
                     labels={"x":"Wait Time","y":"Drivers"},
                     text=wait_counts.values)
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=320, plot_bgcolor="white",
                          xaxis_title="", yaxis_title="Number of Drivers")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Congestion Frequency – All Drivers</div>', unsafe_allow_html=True)
        cf = trucks["Trafficcongestionfrequency"].value_counts()
        order2 = ["Never","Rarely","Sometimes","often","Always"]
        cf = cf.reindex(order2).fillna(0)
        fig2 = px.pie(values=cf.values, names=cf.index,
                      color_discrete_sequence=["#2ecc71","#f1c40f","#e67e22","#e74c3c","#8e44ad"],
                      hole=0.4)
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(height=320, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Monthly Traffic Volumes (Jan–Jun 2025)</div>', unsafe_allow_html=True)
    monthly_data = {
        "Month":   ["Jan","Feb","Mar","Apr","May","Jun"],
        "Transit": [13056,11865,13119,12370,13559,6922],
        "Local/CFS":[15187,12661,15501,15600,15648,8445],
        "Export":  [30362,27183,31680,30867,33429,15665],
    }
    mdf = pd.DataFrame(monthly_data)
    fig3 = px.line(mdf, x="Month", y=["Transit","Local/CFS","Export"],
                   markers=True, title="",
                   color_discrete_map={"Transit":"#003087","Local/CFS":"#0078d4","Export":"#e74c3c"})
    fig3.update_layout(height=300, plot_bgcolor="white", yaxis_title="Trucks",
                       legend_title="Truck Type")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class="warning-box">⚠️  <b>Key Finding:</b> Export trucks consistently dominate volumes (avg 28,000+/month), 
    with May 2025 reaching a peak of 33,429. June shows a partial-month dip.</div>
    <div class="insight-box">📌 <b>Target Miss:</b> Truck turnaround target is ≤30 min, but actual averages 90–180 minutes — 
    3–6× above the performance contract target.</div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 2 — DEMOGRAPHICS
# ============================================================
elif page == "👥 Demographics":
    st.markdown('<div class="section-title">👥 Respondent Demographics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Nationality Distribution – Truck Drivers**")
        nat = trucks["Nationality"].value_counts().reset_index()
        nat.columns = ["Country","Count"]
        fig = px.bar(nat, x="Count", y="Country", orientation="h",
                     color="Count", color_continuous_scale="Blues",
                     text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=320, plot_bgcolor="white",
                          coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Gender Distribution Across Stakeholder Groups**")
        gender_data = {
            "Category":    ["Truck Drivers","Clearing Agents","Custom Officials","KPA Staff","Traffic Police"],
            "Male %":      [98.0, 96.0, 70.0, 63.6, 66.7],
            "Female %":    [2.0,  4.0,  30.0, 36.4, 33.3],
        }
        gdf = pd.DataFrame(gender_data)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Male", x=gdf["Category"], y=gdf["Male %"],
                              marker_color="#003087"))
        fig2.add_trace(go.Bar(name="Female", x=gdf["Category"], y=gdf["Female %"],
                              marker_color="#e74c3c"))
        fig2.update_layout(barmode="stack", height=320, plot_bgcolor="white",
                           yaxis_title="%", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Years of Experience – Truck Drivers**")
        exp = trucks["Yearsexperience"].value_counts()
        order = ["Less than 1 year","1-5 years","6-10 yeras","Over 10 years"]
        exp = exp.reindex(order)
        fig3 = px.bar(x=exp.index, y=exp.values,
                      color_discrete_sequence=["#0056b3"],
                      text=exp.values,
                      labels={"x":"Experience","y":"Count"})
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=300, plot_bgcolor="white",
                           showlegend=False, xaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**Visit Frequency – Truck Drivers**")
        vf = trucks["Visitfrequency"].value_counts().reset_index()
        vf.columns = ["Frequency","Count"]
        fig4 = px.pie(vf, values="Count", names="Frequency",
                      color_discrete_sequence=px.colors.sequential.Blues_r,
                      hole=0.35)
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class="insight-box">👤 <b>Workforce Profile:</b> 88.4% of truck drivers are Kenyan nationals. 
    45.9% have over 10 years of experience — a highly experienced but gender-imbalanced workforce (98% male).</div>
    <div class="insight-box">🌍 <b>Regional Hub:</b> 11.6% of truck drivers are from Uganda, Tanzania, DRC Congo, 
    Rwanda, Burundi & South Sudan, confirming the Port's East African gateway role.</div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 3 — TRAFFIC PATTERNS
# ============================================================
elif page == "🚦 Traffic Patterns":
    st.markdown('<div class="section-title">🚦 Traffic Patterns & Gate Usage</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Mostly Used Gates – Truck Drivers vs Clearing Agents**")
        gate_data = {
            "Gate":           ["9/10 Main","12/13 Shimanzi","Gate 15","Gate 16","Gate 18","Gate 24","ICD Gates"],
            "Truck Drivers":  [93, 16, 1, 30, 308, 475, 93],
            "Clearing Agents":[27,  0, 0,  0,  63,  74,  21],
        }
        gdf = pd.DataFrame(gate_data)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Truck Drivers",  x=gdf["Gate"], y=gdf["Truck Drivers"],  marker_color="#003087"))
        fig.add_trace(go.Bar(name="Clearing Agents",x=gdf["Gate"], y=gdf["Clearing Agents"],marker_color="#e74c3c"))
        fig.update_layout(barmode="group", height=330, plot_bgcolor="white",
                          yaxis_title="Responses", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Congestion Time of Day (All Respondents)**")
        tod_data = {"Period":["Morning","Midday","Afternoon","Evening"],"Count":[150,219,480,368]}
        tdf = pd.DataFrame(tod_data)
        colors_tod = ["#2ecc71","#f1c40f","#e74c3c","#8e44ad"]
        fig2 = px.pie(tdf, values="Count", names="Period",
                      color_discrete_sequence=colors_tod, hole=0.4)
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(height=330)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Weekly Traffic Entry & Exit Pattern (May 2025)**")
    weekly = {
        "Day":        ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        "Entry Total":[4923, 5386, 6733, 7896, 7664, 6654, 4262],
        "Exit Total": [5129, 5482, 5245, 6603, 6799, 6051, 2720],
    }
    wdf = pd.DataFrame(weekly)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=wdf["Day"], y=wdf["Entry Total"], name="Entry Total",
                              mode="lines+markers", line=dict(color="#003087", width=3),
                              marker=dict(size=8)))
    fig3.add_trace(go.Scatter(x=wdf["Day"], y=wdf["Exit Total"],  name="Exit Total",
                              mode="lines+markers", line=dict(color="#e74c3c", width=3),
                              marker=dict(size=8)))
    fig3.update_layout(height=280, plot_bgcolor="white",
                       yaxis_title="Trucks", xaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**May 2025 — Containerised Trucks Breakdown**")
        may_data = {"Type":["Transit","Local/CFS","Export","Loose Cargo","Empty"],
                    "Count":[13559, 15648, 6230, 9418, 16395]}
        mdf2 = pd.DataFrame(may_data)
        fig4 = px.bar(mdf2, x="Type", y="Count", text="Count",
                      color="Type",
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        fig4.update_traces(textposition="outside")
        fig4.update_layout(height=310, plot_bgcolor="white",
                           showlegend=False, xaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.markdown("**Shift Distribution — Mombasa Port**")
        shift_data = {
            "Category":   ["Transit","Local/CFS","Export","Loose Cargo","Empty"],
            "1st Shift":  [108, 175, 82, 392, 364],
            "2nd Shift":  [246, 235, 88, 270, 321],
            "3rd Shift":  [142,  62, 91, 347, 247],
        }
        sdf = pd.DataFrame(shift_data).set_index("Category")
        fig5 = px.bar(sdf.T.reset_index().rename(columns={"index":"Shift"}),
                      x="Shift", y=sdf.columns.tolist(),
                      barmode="stack", height=310,
                      color_discrete_sequence=["#003087","#0056b3","#e74c3c","#f1c40f","#2ecc71"])
        fig5.update_layout(plot_bgcolor="white", xaxis_title="", yaxis_title="Trucks",
                           legend_title="Type")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    <div class="insight-box">🕑 <b>Peak Congestion:</b> Afternoon (2–6 PM) accounts for 40% of congestion incidents; 
    combined afternoon + evening = 70%. Gate 24 and Gate 18 handle 79.3% of all truck traffic.</div>
    <div class="warning-box">📅 <b>Peak Days:</b> Thursday and Friday record the highest weekly traffic volumes 
    (7,896 and 7,664 entries), requiring enhanced staffing on these days.</div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 4 — CONGESTION CAUSES
# ============================================================
elif page == "⚠️  Congestion Causes":
    st.markdown('<div class="section-title">⚠️  Causes of Traffic Congestion at KPA Gates</div>', unsafe_allow_html=True)

    causes_truck = {
        "Cause": ["Slow Gate Processing","Slow Security Checks","KRA Gadget Delays",
                  "Documentation Delays","Too Many Trucks","Limited Gate Lanes",
                  "Lack of Truck Scheduling","Insufficient Port Space","Incomplete Docs","Poor Road Conditions"],
        "Truck Drivers": [440,377,316,302,214,30,42,34,16,18],
        "Clearing Agents":[73, 49, 69, 76, 54, 13, 8, 9,12,11],
    }
    cdf = pd.DataFrame(causes_truck)
    cdf["Total"] = cdf["Truck Drivers"] + cdf["Clearing Agents"]
    cdf = cdf.sort_values("Total", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(y=cdf["Cause"], x=cdf["Truck Drivers"],  name="Truck Drivers",
                         orientation="h", marker_color="#003087"))
    fig.add_trace(go.Bar(y=cdf["Cause"], x=cdf["Clearing Agents"],name="Clearing Agents",
                         orientation="h", marker_color="#e74c3c"))
    fig.update_layout(barmode="stack", height=430, plot_bgcolor="white",
                      xaxis_title="Response Count", yaxis_title="",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Impact on Truck Drivers**")
        impact_d = {
            "Impact":["Longer Working Hours","Increased Storage Fees","Increased Fuel Costs",
                      "Missed Delivery Schedules","Stress / Fatigue","Increased Turnaround",
                      "Increased Demurrage","Delay in Stacking"],
            "Score":[26.7, 18.3, 14.5, 12.0, 10.5, 8.7, 5.5, 3.8]
        }
        idf = pd.DataFrame(impact_d).sort_values("Score", ascending=True)
        fig2 = px.bar(idf, x="Score", y="Impact", orientation="h",
                      color="Score", color_continuous_scale="Reds",
                      text="Score")
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(height=330, plot_bgcolor="white",
                           coloraxis_showscale=False, xaxis_title="%", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("**Stakeholder-Proposed Solutions**")
        sol_data = {
            "Solution":["Truck Scheduling System","Enhance KPA Coordination",
                        "Expand Gate Capacity","Automate Clearance",
                        "Upgrade Inspection Tech","Increase Customs Staff",
                        "Faster Document Processing","Pre-Clearance Process",
                        "Staff Training"],
            "Priority":[37.5, 33.3, 25.0, 21.6, 23.8, 16.7, 21.4, 16.7, 13.7]
        }
        sdf2 = pd.DataFrame(sol_data).sort_values("Priority", ascending=True)
        fig3 = px.bar(sdf2, x="Priority", y="Solution", orientation="h",
                      color="Priority", color_continuous_scale="Greens",
                      text="Priority")
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig3.update_layout(height=330, plot_bgcolor="white",
                           coloraxis_showscale=False, xaxis_title="% Recommending", yaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

    # Driver congestion cause heatmap from binary columns
    st.markdown("**Congestion Cause Selection Rate by Truck Drivers (%)**")
    cause_cols = {
        "Toomanytrucks":"Too Many Trucks","clearance":"Clearance Delays",
        "securitychecks":"Security Checks","Gateprocessing":"Gate Processing",
        "Trackinggadgets":"KRA Gadgets","Roadconditions":"Road Conditions",
        "Gatelanes":"Limited Gate Lanes","Truckscheduling":"No Truck Scheduling"
    }
    rates = {}
    for col, label in cause_cols.items():
        if col in df_ml.columns:
            rates[label] = round(df_ml[col].mean() * 100, 1)

    rdf = pd.DataFrame(list(rates.items()), columns=["Cause","Selection Rate %"])
    rdf = rdf.sort_values("Selection Rate %", ascending=False)
    fig4 = px.bar(rdf, x="Cause", y="Selection Rate %", text="Selection Rate %",
                  color="Selection Rate %", color_continuous_scale="Blues")
    fig4.update_traces(textposition="outside")
    fig4.update_layout(height=330, plot_bgcolor="white",
                       coloraxis_showscale=False, xaxis_title="")
    st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# PAGE 5 — ML MODELS
# ============================================================
elif page == "🤖 ML Predictive Models":
    st.markdown('<div class="section-title">🤖 Machine Learning Predictive Models</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎯 Congestion Level Predictor", "⏱ Long Wait Time Predictor"])

    for tab, target_col, target_label in [
        (tab1, "high_congestion", "High Congestion (Always/Often)"),
        (tab2, "long_wait",       "Long Wait Time (>2 hrs)")
    ]:
        with tab:
            # Prepare
            model_df = df_ml[feature_cols + [target_col]].dropna()
            X = model_df[feature_cols]
            y = model_df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y)

            models = {
                "Random Forest":         RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"),
                "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Logistic Regression":   LogisticRegression(max_iter=500, random_state=42, class_weight="balanced"),
                "Decision Tree":         DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
            }

            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc    = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
                try:
                    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
                except:
                    auc = 0.0
                results[name] = {
                    "model": model, "acc": acc, "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(), "auc": auc, "y_pred": y_pred
                }

            # Best model
            best_name = max(results, key=lambda k: results[k]["auc"])
            best = results[best_name]

            col1, col2, col3 = st.columns(3)
            col1.metric("Best Model", best_name)
            col2.metric("Test Accuracy", f"{best['acc']*100:.1f}%")
            col3.metric("ROC-AUC", f"{best['auc']:.3f}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Model Comparison**")
                comp_df = pd.DataFrame({
                    "Model"     : list(results.keys()),
                    "Accuracy"  : [r["acc"]*100 for r in results.values()],
                    "CV Mean"   : [r["cv_mean"]*100 for r in results.values()],
                    "ROC-AUC"   : [r["auc"] for r in results.values()],
                })
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(name="Test Accuracy %", x=comp_df["Model"],
                                          y=comp_df["Accuracy"], marker_color="#003087"))
                fig_comp.add_trace(go.Bar(name="CV Mean %", x=comp_df["Model"],
                                          y=comp_df["CV Mean"], marker_color="#0078d4"))
                fig_comp.update_layout(barmode="group", height=310, plot_bgcolor="white",
                                       yaxis_title="%", xaxis_title="")
                st.plotly_chart(fig_comp, use_container_width=True)

            with col_b:
                st.markdown("**Confusion Matrix — Best Model**")
                cm = confusion_matrix(y_test, best["y_pred"])
                fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                                   color_continuous_scale="Blues",
                                   labels=dict(x="Predicted", y="Actual"),
                                   x=["Low/Moderate","High"],
                                   y=["Low/Moderate","High"])
                fig_cm.update_layout(height=310, title=f"Confusion Matrix: {target_label}")
                st.plotly_chart(fig_cm, use_container_width=True)

            # Feature importance
            st.markdown("**Feature Importances — Best Model**")
            rf_best = results[best_name]["model"]
            if hasattr(rf_best, "feature_importances_"):
                importances = rf_best.feature_importances_
            else:
                pi = permutation_importance(rf_best, X_test, y_test, n_repeats=5, random_state=42)
                importances = pi.importances_mean

            fi_df = pd.DataFrame({
                "Feature"   : feature_cols,
                "Importance": importances
            }).sort_values("Importance", ascending=True).tail(15)

            fig_fi = px.bar(fi_df, x="Importance", y="Feature",
                            orientation="h",
                            color="Importance", color_continuous_scale="Blues",
                            text=fi_df["Importance"].round(3))
            fig_fi.update_traces(textposition="outside")
            fig_fi.update_layout(height=420, plot_bgcolor="white",
                                 coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig_fi, use_container_width=True)

            # Classification report
            with st.expander("📋 Full Classification Report"):
                cr = classification_report(y_test, best["y_pred"],
                                           target_names=["Low/Moderate","High"],
                                           output_dict=True)
                cr_df = pd.DataFrame(cr).transpose().round(3)
                st.dataframe(cr_df, use_container_width=True)

# ============================================================
# PAGE 6 — PREDICT FOR NEW DRIVER
# ============================================================
elif page == "🔍 Predict for New Driver":
    st.markdown('<div class="section-title">🔍 Predict Congestion & Wait Risk for a New Driver</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">Enter a truck driver's profile to predict their likely congestion experience 
    and gate wait time at KPA. Useful for onboarding risk briefings and gate scheduling.</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        nationality = st.selectbox("Nationality", ["Kenya","Uganda","Tanzania","DRC-Congo","Rwanda","Burundi","South Sudan"])
        gender      = st.selectbox("Gender", ["Male","Female"])
        experience  = st.selectbox("Years of Experience", ["Less than 1 year","1-5 years","6-10 yeras","Over 10 years"])
        visit_freq  = st.selectbox("Visit Frequency", ["Daily","several times a week,2-4 times",
                                                        "Once a week","A few times a month,1-3 times",
                                                        "Rarely less than once per month"])
    with col2:
        gate18 = st.checkbox("Uses Gate 18", value=True)
        gate24 = st.checkbox("Uses Gate 24", value=True)
        gate9  = st.checkbox("Uses Gate 9/10")
        gate12 = st.checkbox("Uses Gate 12")
        gate16 = st.checkbox("Uses Gate 16")
        icd    = st.checkbox("Uses ICD Gates")
    with col3:
        morning   = st.checkbox("Arrives Morning (6am–10am)")
        midday    = st.checkbox("Arrives Midday (10am–2pm)")
        afternoon = st.checkbox("Arrives Afternoon (2pm–6pm)", value=True)
        evening   = st.checkbox("Arrives Evening (6pm+)", value=True)
        st.markdown("**Congestion Causes Experienced:**")
        toomany   = st.checkbox("Too Many Trucks", value=True)
        clearance = st.checkbox("Clearance Delays", value=True)
        security  = st.checkbox("Security Check Delays")
        gateproc  = st.checkbox("Slow Gate Processing", value=True)
        gadgets   = st.checkbox("KRA Gadget Delays")
        road      = st.checkbox("Poor Road Conditions")
        lanes     = st.checkbox("Limited Gate Lanes")
        sched     = st.checkbox("No Truck Scheduling")

    if st.button("🔮 Run Prediction", use_container_width=True):
        exp_map   = {"Less than 1 year":0,"1-5 years":1,"6-10 yeras":2,"Over 10 years":3}
        visit_map = {"Rarely less than once per month":0,"A few times a month,1-3 times":1,
                     "Once a week":2,"several times a week,2-4 times":3,"Daily":4}
        input_vec = pd.DataFrame([{
            "is_kenyan"       : 1 if nationality=="Kenya" else 0,
            "is_male"         : 1 if gender=="Male" else 0,
            "exp_encoded"     : exp_map[experience],
            "visit_encoded"   : visit_map[visit_freq],
            "Gate18"          : int(gate18),
            "Gate24"          : int(gate24),
            "Gates9"          : int(gate9),
            "Gate12"          : int(gate12),
            "Gate16"          : int(gate16),
            "ICDGATES"        : int(icd),
            "Morning"         : int(morning),
            "Midday"          : int(midday),
            "Afternoon"       : int(afternoon),
            "Evening"         : int(evening),
            "Containerized"   : 1,
            "Empty"           : 0,
            "Bulk"            : 0,
            "Toomanytrucks"   : int(toomany),
            "clearance"       : int(clearance),
            "securitychecks"  : int(security),
            "Gateprocessing"  : int(gateproc),
            "Trackinggadgets" : int(gadgets),
            "Roadconditions"  : int(road),
            "Gatelanes"       : int(lanes),
            "Truckscheduling" : int(sched),
        }])[feature_cols]

        # Train models on full data
        model_df = df_ml[feature_cols + ["high_congestion","long_wait"]].dropna()
        X_all = model_df[feature_cols]

        rf_c = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
        rf_c.fit(X_all, model_df["high_congestion"])
        rf_w = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
        rf_w.fit(X_all, model_df["long_wait"])

        cong_prob = rf_c.predict_proba(input_vec)[0][1]
        wait_prob = rf_w.predict_proba(input_vec)[0][1]
        cong_pred = rf_c.predict(input_vec)[0]
        wait_pred = rf_w.predict(input_vec)[0]

        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        rc1, rc2 = st.columns(2)
        with rc1:
            cong_color = "#e74c3c" if cong_pred == 1 else "#2ecc71"
            cong_label = "HIGH CONGESTION RISK" if cong_pred == 1 else "LOW/MODERATE RISK"
            st.markdown(f"""
            <div style="background:{cong_color}22; border:2px solid {cong_color};
                        border-radius:12px; padding:1.2rem; text-align:center;">
                <h2 style="color:{cong_color}; margin:0;">{cong_label}</h2>
                <p style="font-size:1.4rem; margin:0.5rem 0;"><b>Probability: {cong_prob*100:.1f}%</b></p>
                <p style="color:#555; font-size:0.85rem;">Likelihood of Always/Often experiencing congestion</p>
            </div>""", unsafe_allow_html=True)
        with rc2:
            wait_color = "#e74c3c" if wait_pred == 1 else "#2ecc71"
            wait_label = "HIGH WAIT TIME RISK" if wait_pred == 1 else "MANAGEABLE WAIT TIME"
            st.markdown(f"""
            <div style="background:{wait_color}22; border:2px solid {wait_color};
                        border-radius:12px; padding:1.2rem; text-align:center;">
                <h2 style="color:{wait_color}; margin:0;">{wait_label}</h2>
                <p style="font-size:1.4rem; margin:0.5rem 0;"><b>Probability: {wait_prob*100:.1f}%</b></p>
                <p style="color:#555; font-size:0.85rem;">Likelihood of waiting >2 hours at gate</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        overall_risk = (cong_prob + wait_prob) / 2
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall_risk * 100,
            title={"text":"Overall Operational Risk Score"},
            gauge={
                "axis": {"range":[0,100]},
                "bar":  {"color":"#003087"},
                "steps":[
                    {"range":[0,33],  "color":"#2ecc71"},
                    {"range":[33,66], "color":"#f1c40f"},
                    {"range":[66,100],"color":"#e74c3c"},
                ],
                "threshold":{"line":{"color":"black","width":4},"thickness":0.75,"value":66}
            }
        ))
        fig_gauge.update_layout(height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Personalised recommendations
        st.markdown("### 💡 Personalised Recommendations")
        recs = []
        if afternoon or evening:
            recs.append("⏰ **Shift arrival time** to morning (6–10 AM) to avoid peak congestion windows.")
        if gate18 and gate24:
            recs.append("🚪 **Diversify gate use** — Gates 18 & 24 are most congested. Consider Gate 9/10 where eligible.")
        if clearance or gateproc:
            recs.append("📄 **Pre-clear documentation** before gate arrival to reduce processing time.")
        if gadgets:
            recs.append("📡 **Coordinate KRA gadget fitting** in advance to avoid waiting at the gate.")
        if not sched:
            recs.append("📅 **Book time slots** via KPA's upcoming Truck Appointment System (TAS) when available.")
        if not recs:
            recs.append("✅ This driver's profile indicates lower operational risk — maintain current practices.")
        for r in recs:
            st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 7 — RECOMMENDATIONS
# ============================================================
elif page == "📋 Recommendations":
    st.markdown('<div class="section-title">📋 Report Recommendations & Implementation Matrix</div>', unsafe_allow_html=True)

    rec_data = [
        {"#":1, "Problem":"Unregulated truck arrivals","Recommendation":"Implement Truck Appointment System (TAS)",
         "Responsibility":"KPA","Priority":"🔴 Critical","Timeline":"Short-term"},
        {"#":2, "Problem":"No staging space for trucks","Recommendation":"Establish Off-Port Truck Marshalling Yard",
         "Responsibility":"KPA","Priority":"🔴 Critical","Timeline":"Medium-term"},
        {"#":3, "Problem":"No coordinated traffic policy","Recommendation":"Develop Comprehensive Traffic Management Policy",
         "Responsibility":"KPA, KRA, Traffic Police, MoT","Priority":"🟠 High","Timeline":"Short-term"},
        {"#":4, "Problem":"Manual gate processes","Recommendation":"Expand & Digitize Gate Infrastructure (RFID, ANPR)",
         "Responsibility":"KPA","Priority":"🔴 Critical","Timeline":"Medium-term"},
        {"#":5, "Problem":"Insufficient peak-hour staffing","Recommendation":"Enhance HR Deployment at Gates",
         "Responsibility":"KPA, KRA","Priority":"🟠 High","Timeline":"Immediate"},
        {"#":6, "Problem":"Weak supervision & accountability","Recommendation":"Strengthen Staff Supervision & Appraisal",
         "Responsibility":"KPA, KRA","Priority":"🟡 Medium","Timeline":"Short-term"},
        {"#":7, "Problem":"Narrow roads & poor drainage","Recommendation":"Expand Access Roads & Internal Circulation",
         "Responsibility":"KPA, KURA","Priority":"🟠 High","Timeline":"Long-term"},
        {"#":8, "Problem":"Over-reliance on road transport","Recommendation":"Promote Modal Shift to SGR Rail",
         "Responsibility":"KPA, KRC","Priority":"🟡 Medium","Timeline":"Long-term"},
    ]
    rdf = pd.DataFrame(rec_data)
    st.dataframe(rdf.set_index("#"), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Priority Distribution**")
        pri_counts = rdf["Priority"].value_counts().reset_index()
        pri_counts.columns = ["Priority","Count"]
        colors_pri = {"🔴 Critical":"#e74c3c","🟠 High":"#e67e22","🟡 Medium":"#f1c40f"}
        fig_pri = px.pie(pri_counts, values="Count", names="Priority",
                         color_discrete_sequence=["#e74c3c","#e67e22","#f1c40f"])
        fig_pri.update_layout(height=300)
        st.plotly_chart(fig_pri, use_container_width=True)
    with col2:
        st.markdown("**Responsibility Distribution**")
        resp_counts = rdf["Responsibility"].value_counts().reset_index()
        resp_counts.columns = ["Agency","Count"]
        fig_resp = px.bar(resp_counts, x="Count", y="Agency", orientation="h",
                          color="Count", color_continuous_scale="Blues")
        fig_resp.update_layout(height=300, plot_bgcolor="white",
                               coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_resp, use_container_width=True)

    st.markdown("""
    <div class="success-box">✅ <b>Quick Wins (Immediate):</b> 
    Increase peak-hour staffing at Gates 18 & 24; mandate pre-arrival documentation submission.</div>
    <div class="warning-box">⚠️ <b>Strategic Priority:</b> 
    A fully operational Truck Appointment System (TAS) is the single highest-impact intervention to 
    reduce clustering — endorsed by 37.5% of traffic police and 21.6% of KPA staff.</div>
    <div class="insight-box">📌 <b>Performance Gap:</b> 
    Current turnaround of 90–180 min vs target of ≤30 min represents a 3–6× miss on the 
    2024/25 Performance Contract — requiring urgent systemic intervention, not just operational tweaks.</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 📁 Data Sources")
    st.markdown("""
    | Dataset | Respondents | Coverage |
    |---------|------------|---------|
    | Truck Drivers Survey | 714 | Gate usage, wait times, congestion causes, impacts |
    | Clearing Agents Survey | ~124 | Documentation, gate usage, congestion frequency |
    | KPA Staff Survey | 22 | Gate operations, staffing, infrastructure |
    | Custom/KRA Officials | 10 | Documentation, scanning, staffing |
    | Traffic Police | 9 | Congestion causes, solutions |
    | KPA Traffic Records | Jan–Jun 2025 | Monthly truck volumes by type and gate |
    """)
