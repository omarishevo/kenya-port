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
    st.markdown("""
    <div style="background:linear-gradient(135deg,#003087,#0056b3);padding:2rem;
                border-radius:14px;color:white;text-align:center;margin-bottom:1.5rem;">
        <h1 style="margin:0;font-size:2rem;">🚢 KPA Traffic Analytics</h1>
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
st.markdown("""
<div class="main-header">
    <h1>🚢 KPA Vehicle Traffic & Congestion Analytics</h1>
    <p>Research Report · Port of Mombasa & ICD Nairobi · June 2025 · Kenya Ports Authority</p>
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
with st.sidebar:
    st.markdown("### 🚢 KPA Analytics")
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
