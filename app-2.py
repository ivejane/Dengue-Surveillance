"""
============================================================
  DENGUE SURVEILLANCE SYSTEM
  Agusan del Sur — Tracking & Analyzing Dengue Cases
  for Improved Outbreak Monitoring
============================================================
  Author  : Ive Jane B. Sabando
  Course  : ISELEC 104
  Stack   : Python · Streamlit · Plotly · Scikit-learn
============================================================

HOW TO RUN LOCALLY:
    pip install -r requirements.txt
    streamlit run app.py

HOW TO DEPLOY:
    Push to GitHub → connect at share.streamlit.io
============================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, confusion_matrix,
    roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# ── Statsmodels ───────────────────────────────────────────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dengue Surveillance System — Agusan del Sur",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"]        { font-family: 'DM Sans', sans-serif; }
.stApp                             { background: #0b1120; color: #e2e8f0; }

/* ── Header ── */
.dss-header {
    background: linear-gradient(135deg, #b91c1c 0%, #7f1d1d 60%, #450a0a 100%);
    border-radius: 14px;
    padding: 1.8rem 2.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 40px rgba(185,28,28,.45);
    border: 1px solid rgba(255,255,255,.06);
    position: relative; overflow: hidden;
}
.dss-header::after {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px;
    background:rgba(255,255,255,.04); border-radius:50%;
}
.dss-header h1 {
    font-family:'Syne',sans-serif; font-size:1.9rem;
    font-weight:800; color:#fff; margin:0; letter-spacing:-.4px;
}
.dss-header p  { color:rgba(255,255,255,.65); margin:.3rem 0 0 0; font-size:.87rem; }

/* ── KPI cards ── */
.kpi-card {
    background:rgba(255,255,255,.04);
    border:1px solid rgba(255,255,255,.08);
    border-radius:12px; padding:1.1rem 1rem;
    text-align:center; transition:transform .2s;
}
.kpi-card:hover { transform:translateY(-3px); }
.kpi-val   { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#ef4444; line-height:1; }
.kpi-lbl   { font-size:.72rem; color:rgba(255,255,255,.45); text-transform:uppercase; letter-spacing:.7px; margin-top:.25rem; }
.kpi-sub   { font-size:.78rem; color:#f59e0b; margin-top:.15rem; }

/* ── Section titles ── */
.sec-title {
    font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700;
    color:#ef4444; border-left:4px solid #ef4444;
    padding-left:.7rem; margin:1.4rem 0 .7rem 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(255,255,255,.03); border-radius:10px;
    padding:4px; gap:4px; border:1px solid rgba(255,255,255,.07);
}
.stTabs [data-baseweb="tab"] {
    border-radius:8px; color:rgba(255,255,255,.5);
    font-family:'Syne',sans-serif; font-weight:600;
    font-size:.83rem; padding:8px 16px; transition:all .2s;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#b91c1c,#ef4444) !important;
    color:#fff !important; box-shadow:0 4px 14px rgba(239,68,68,.35);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:rgba(255,255,255,.02) !important; border-right:1px solid rgba(255,255,255,.06) !important; }

/* ── Alert boxes ── */
.alert-r { background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.35); border-radius:8px; padding:.75rem 1rem; color:#fca5a5; }
.alert-y { background:rgba(245,158,11,.1);  border:1px solid rgba(245,158,11,.3);  border-radius:8px; padding:.75rem 1rem; color:#fcd34d; }
.alert-g { background:rgba(34,197,94,.08);  border:1px solid rgba(34,197,94,.25);  border-radius:8px; padding:.75rem 1rem; color:#86efac; }

/* ── Metric override ── */
[data-testid="metric-container"] { background:transparent !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — consistent Plotly dark theme
# ══════════════════════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    title_font_size=13,
    legend=dict(bgcolor="rgba(0,0,0,.3)", borderwidth=1,
                bordercolor="rgba(255,255,255,.1)"),
)
GRID = dict(showgrid=True, gridcolor="rgba(255,255,255,.05)")
NOGRID = dict(showgrid=False)
RED_SCALE = ["#450a0a","#7f1d1d","#b91c1c","#ef4444","#fca5a5"]


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def generate_sample_data() -> pd.DataFrame:
    """Generate realistic synthetic dengue surveillance data."""
    rng = np.random.default_rng(42)

    MUNICIPALITIES = [
        "Bunawan","Trento","Esperanza","Bayugan","Prosperidad",
        "San Luis","Santa Josefa","Sibagat","Talacogon","La Paz",
        "Loreto","Veruela","Rosario","San Francisco","Cortes",
    ]
    AGE_GROUPS   = ["0-4","5-9","10-14","15-19","20-29","30-39","40-49","50-59","60+"]
    AGE_PROBS    = [.08,.10,.12,.13,.18,.15,.10,.08,.06]
    CLINICAL     = ["Dengue Without Warning Signs","Dengue With Warning Signs","Severe Dengue"]
    CLIN_PROBS   = [.55,.35,.10]
    OUTCOMES     = ["Recovered","Hospitalized","Died"]
    OUT_PROBS    = [.70,.28,.02]
    WEATHER      = ["Sunny","Cloudy","Rainy","Partly Cloudy","Thunderstorm"]
    MONTHS       = ["January","February","March","April","May","June",
                    "July","August","September","October","November","December"]

    rows = []
    for year in [2023, 2024, 2025]:
        for m_idx, month in enumerate(MONTHS):
            # Seasonal sine wave — dengue peaks during rainy season (Jun–Sep)
            season = 1 + 2.8 * max(0, np.sin((m_idx - 2) * np.pi / 6))
            for mun in MUNICIPALITIES:
                n = int(rng.poisson(season * 9))
                for _ in range(n):
                    clinical = rng.choice(CLINICAL, p=CLIN_PROBS)
                    outcome  = rng.choice(OUTCOMES,  p=OUT_PROBS)
                    hosp     = 1 if outcome == "Hospitalized" else int(rng.random() < .15)
                    died     = 1 if outcome == "Died"         else 0
                    rows.append({
                        "Year":                   year,
                        "Month":                  month,
                        "Month_Num":              m_idx + 1,
                        "Municipality":           mun,
                        "Barangay":               f"Barangay {rng.integers(1,10)}",
                        "Age_Group":              rng.choice(AGE_GROUPS, p=AGE_PROBS),
                        "Sex":                    rng.choice(["Male","Female"]),
                        "Clinical_Classification": clinical,
                        "Outcome":                outcome,
                        "Hospitalized":           hosp,
                        "Deaths":                 died,
                        "Families_Affected":      int(rng.integers(0, 6)),
                        "Rainfall_mm":            round(float(rng.uniform(0, 320)), 1),
                        "Temperature_C":          round(float(rng.uniform(24, 37)), 1),
                        "Humidity_pct":           round(float(rng.uniform(55, 96)), 1),
                        "Water_Level":            round(float(rng.uniform(0.5, 6.0)), 2),
                        "Weather":                rng.choice(WEATHER),
                    })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month_Num"].astype(str).str.zfill(2) + "-01"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_data(upload) -> tuple[pd.DataFrame, bool]:
    if upload is not None:
        try:
            df = pd.read_csv(upload) if upload.name.endswith(".csv") else pd.read_excel(upload)
            return df, True
        except Exception as exc:
            st.error(f"❌ Could not read file: {exc}")
    return generate_sample_data(), False


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="dss-header">
  <h1>🦟 DENGUE SURVEILLANCE SYSTEM</h1>
  <p>Agusan del Sur &nbsp;·&nbsp; Tracking &amp; Analyzing Dengue Cases for Improved Outbreak Monitoring
     &nbsp;·&nbsp; ISELEC 104 &nbsp;·&nbsp; Ive Jane B. Sabando</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Data Upload")
    upload = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv","xlsx","xls"],
        help="Upload your dengue dataset or use the built-in sample data.",
    )
    st.caption("Accepted columns: Year, Month, Municipality, Barangay, Age_Group, Sex, Clinical_Classification, Outcome, Hospitalized, Deaths, Rainfall_mm, Temperature_C, Humidity_pct, Water_Level, Families_Affected")
    st.markdown("---")

df_raw, user_file = load_data(upload)

with st.sidebar:
    st.markdown("### ⚙️ Filters")
    years  = sorted(df_raw["Year"].unique().tolist()) if "Year" in df_raw.columns else []
    munis  = sorted(df_raw["Municipality"].unique().tolist()) if "Municipality" in df_raw.columns else []

    sel_years = st.multiselect("Year(s)", years, default=years)
    sel_munis = st.multiselect("Municipality", munis,
                               default=munis[:6] if len(munis) > 6 else munis)

    st.markdown("---")
    st.markdown("### 🔬 Model Settings")
    n_clusters   = st.slider("K-Means Clusters (K)", 2, 8, 4)
    forecast_per = st.slider("Forecast Periods (months)", 4, 24, 12)
    test_size    = st.slider("Train/Test Split (test %)", 10, 40, 25)

    st.markdown("---")
    st.caption("ISELEC 104 · Dengue Surveillance · Ive Jane B. Sabando")

# Apply sidebar filters
df = df_raw.copy()
if sel_years and "Year" in df.columns:
    df = df[df["Year"].isin(sel_years)]
if sel_munis and "Municipality" in df.columns:
    df = df[df["Municipality"].isin(sel_munis)]


# ══════════════════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
total   = len(df)
deaths  = int(df["Deaths"].sum())         if "Deaths"       in df.columns else 0
hosp    = int(df["Hospitalized"].sum())   if "Hospitalized" in df.columns else 0
cfr     = round(deaths / total * 100, 2) if total > 0 else 0
severe  = len(df[df["Clinical_Classification"] == "Severe Dengue"]) \
          if "Clinical_Classification" in df.columns else 0
n_munis = df["Municipality"].nunique()   if "Municipality"  in df.columns else 0

for col, val, lbl, sub in zip(
    st.columns(6),
    [f"{total:,}", str(deaths), str(hosp), f"{cfr}%", str(severe), str(n_munis)],
    ["Total Cases","Deaths","Hospitalized","Case Fatality Rate","Severe Dengue","Municipalities"],
    ["🦟 Reported","⚠️ Fatalities","🏥 Admitted","📊 CFR","🔴 Critical","📍 Affected"],
):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📋 Data & EDA",
    "🗺️ Geographic",
    "📈 Time Series",
    "🔬 Clustering",
    "🤖 Prediction",
    "📊 Model Comparison",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA & EDA
# ──────────────────────────────────────────────────────────────────────────────
with t1:
    st.markdown('<div class="sec-title">Data Cleaning & Exploratory Data Analysis (EDA)</div>',
                unsafe_allow_html=True)

    if user_file:
        st.success(f"✅ Custom dataset loaded — {len(df_raw):,} rows · {df_raw.shape[1]} columns")
    else:
        st.info("ℹ️ Using built-in sample data. Upload your own CSV/Excel in the sidebar.")

    st.markdown("**Dataset Preview (first 10 rows after cleaning):**")
    st.dataframe(df.head(10), use_container_width=True, height=270)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="sec-title">Dataset Overview</div>', unsafe_allow_html=True)
        overview = {
            "Total Records":    f"{len(df):,}",
            "Columns":          str(df.shape[1]),
            "Missing Values":   str(df.isnull().sum().sum()),
            "Duplicate Rows":   str(df.duplicated().sum()),
            "Year Range":       f"{df['Year'].min()} – {df['Year'].max()}" if "Year" in df.columns else "—",
            "Municipalities":   str(df["Municipality"].nunique()) if "Municipality" in df.columns else "—",
        }
        st.dataframe(
            pd.DataFrame(overview.items(), columns=["Metric","Value"]),
            use_container_width=True, hide_index=True,
        )

    with col_b:
        st.markdown('<div class="sec-title">Descriptive Statistics</div>', unsafe_allow_html=True)
        num_cols = [c for c in ["Rainfall_mm","Temperature_C","Humidity_pct","Water_Level"] if c in df.columns]
        if num_cols:
            st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

    # ── Distribution charts ──
    st.markdown('<div class="sec-title">Distribution Analysis</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)

    with d1:
        if "Clinical_Classification" in df.columns:
            fig = px.pie(
                df["Clinical_Classification"].value_counts().reset_index(),
                names="Clinical_Classification", values="count",
                title="Clinical Classification",
                color_discrete_sequence=["#ef4444","#f59e0b","#7f1d1d"],
                hole=.45,
            )
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with d2:
        if "Age_Group" in df.columns:
            order = ["0-4","5-9","10-14","15-19","20-29","30-39","40-49","50-59","60+"]
            age_df = (df["Age_Group"].value_counts()
                      .reindex(order, fill_value=0).reset_index())
            fig = px.bar(age_df, x="Age_Group", y="count",
                         title="Cases by Age Group",
                         color="count", color_continuous_scale=RED_SCALE)
            fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
            fig.update_xaxes(tickangle=-35, **NOGRID)
            fig.update_yaxes(**GRID)
            st.plotly_chart(fig, use_container_width=True)

    with d3:
        if "Sex" in df.columns:
            sex_df = df["Sex"].value_counts().reset_index()
            fig = px.bar(sex_df, x="Sex", y="count", title="Cases by Sex",
                         color="Sex",
                         color_discrete_map={"Male":"#3b82f6","Female":"#ef4444"})
            fig.update_layout(**PLOT_LAYOUT, showlegend=False)
            fig.update_xaxes(**NOGRID); fig.update_yaxes(**GRID)
            st.plotly_chart(fig, use_container_width=True)

    # ── Outcome + Correlation ──
    st.markdown('<div class="sec-title">Outcome & Correlation Analysis</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)

    with e1:
        if "Outcome" in df.columns:
            out_df = df["Outcome"].value_counts().reset_index()
            fig = px.funnel(out_df, x="count", y="Outcome",
                            title="Case Outcomes Funnel",
                            color_discrete_sequence=["#ef4444","#f59e0b","#22c55e"])
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with e2:
        corr_cols = [c for c in
                     ["Rainfall_mm","Temperature_C","Humidity_pct","Water_Level","Hospitalized","Deaths"]
                     if c in df.columns]
        if len(corr_cols) >= 3:
            corr = df[corr_cols].corr().round(2)
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap",
                            color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — GEOGRAPHIC
# ──────────────────────────────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="sec-title">Geographic Distribution of Dengue Cases</div>',
                unsafe_allow_html=True)

    if "Municipality" not in df.columns:
        st.warning("Municipality column not found in dataset.")
    else:
        muni_sum = df.groupby("Municipality").agg(
            Cases        =("Municipality","count"),
            Deaths       =("Deaths","sum"),
            Hospitalized =("Hospitalized","sum"),
            Avg_Rainfall =("Rainfall_mm","mean"),
            Avg_Temp     =("Temperature_C","mean"),
        ).reset_index()
        muni_sum["CFR_pct"]    = (muni_sum["Deaths"] / muni_sum["Cases"] * 100).round(2)
        muni_sum["Hosp_pct"]   = (muni_sum["Hospitalized"] / muni_sum["Cases"] * 100).round(1)

        g1, g2 = st.columns([3, 2])
        with g1:
            fig = px.bar(
                muni_sum.sort_values("Cases"),
                x="Cases", y="Municipality", orientation="h",
                color="CFR_pct", color_continuous_scale="Reds",
                title="Cases by Municipality (color = CFR %)",
                text="Cases",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(**PLOT_LAYOUT, height=480, yaxis_title="")
            fig.update_xaxes(**NOGRID); fig.update_yaxes(**NOGRID)
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            fig2 = px.scatter(
                muni_sum, x="Avg_Rainfall", y="Cases",
                size="Deaths", color="Avg_Temp",
                hover_name="Municipality",
                color_continuous_scale="Oranges",
                title="Rainfall vs Cases (size=Deaths)",
                labels={"Avg_Rainfall":"Avg Rainfall (mm)",
                        "Avg_Temp":"Avg Temp (°C)"},
            )
            fig2.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sec-title">Municipality Summary Table</div>', unsafe_allow_html=True)
        st.dataframe(
            muni_sum.sort_values("Cases", ascending=False)
            .style
            .background_gradient(subset=["Cases"], cmap="Reds")
            .background_gradient(subset=["CFR_pct"], cmap="OrRd")
            .format({"CFR_pct":"{:.2f}%","Hosp_pct":"{:.1f}%",
                     "Avg_Rainfall":"{:.1f}","Avg_Temp":"{:.1f}"}),
            use_container_width=True, height=380,
        )

        # Alert system
        st.markdown('<div class="sec-title">Outbreak Alert Status</div>', unsafe_allow_html=True)
        high   = muni_sum[muni_sum["CFR_pct"] >  3]
        medium = muni_sum[(muni_sum["CFR_pct"] >= 1) & (muni_sum["CFR_pct"] <= 3)]
        low    = muni_sum[muni_sum["CFR_pct"] <  1]

        a1, a2, a3 = st.columns(3)
        with a1:
            h_list = ", ".join(high["Municipality"].tolist()) if len(high) else "None"
            st.markdown(f'<div class="alert-r">🔴 <b>HIGH ALERT</b> ({len(high)})<br>{h_list}</div>',
                        unsafe_allow_html=True)
        with a2:
            m_list = ", ".join(medium["Municipality"].tolist()[:5]) if len(medium) else "None"
            st.markdown(f'<div class="alert-y">🟡 <b>MEDIUM ALERT</b> ({len(medium)})<br>{m_list}</div>',
                        unsafe_allow_html=True)
        with a3:
            l_list = ", ".join(low["Municipality"].tolist()[:5]) if len(low) else "None"
            st.markdown(f'<div class="alert-g">🟢 <b>LOW RISK</b> ({len(low)})<br>{l_list}</div>',
                        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — TIME SERIES
# ──────────────────────────────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="sec-title">Dengue Trend & Seasonal Analysis</div>',
                unsafe_allow_html=True)

    if "Month_Num" not in df.columns or "Year" not in df.columns:
        st.warning("Year / Month_Num columns required for time-series analysis.")
    else:
        monthly = (df.groupby(["Year","Month","Month_Num"])
                   .agg(Cases=("Municipality","count"),
                        Deaths=("Deaths","sum"),
                        Hospitalized=("Hospitalized","sum"))
                   .reset_index()
                   .sort_values(["Year","Month_Num"]))

        # Multi-year trend
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Monthly Cases by Year", "Deaths & Hospitalizations"),
            row_heights=[.65, .35], vertical_spacing=.12,
        )
        yr_colors = ["#ef4444","#f59e0b","#3b82f6","#22c55e"]
        for i, yr in enumerate(sorted(monthly["Year"].unique())):
            yd = monthly[monthly["Year"] == yr]
            fig.add_trace(go.Scatter(
                x=yd["Month"], y=yd["Cases"], name=str(yr), mode="lines+markers",
                line=dict(color=yr_colors[i % len(yr_colors)], width=2.5),
                marker=dict(size=6),
            ), row=1, col=1)

        all_m = (df.groupby(["Month","Month_Num"])
                 .agg(Deaths=("Deaths","sum"), Hosp=("Hospitalized","sum"))
                 .reset_index().sort_values("Month_Num"))
        fig.add_trace(go.Bar(x=all_m["Month"], y=all_m["Deaths"],
                             name="Deaths", marker_color="rgba(239,68,68,.8)"), row=2, col=1)
        fig.add_trace(go.Bar(x=all_m["Month"], y=all_m["Hosp"],
                             name="Hospitalized", marker_color="rgba(59,130,246,.7)"), row=2, col=1)
        fig.update_layout(**PLOT_LAYOUT, height=520, barmode="group")
        fig.update_xaxes(**NOGRID); fig.update_yaxes(**GRID)
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        st.markdown('<div class="sec-title">Seasonal Heatmap (Year × Month)</div>',
                    unsafe_allow_html=True)
        MONTH_ORDER = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        if "Month" in df.columns:
            heat = df.groupby(["Year","Month"]).size().reset_index(name="Cases")
            heat_piv = (heat.pivot(index="Year", columns="Month", values="Cases")
                        .fillna(0)
                        .reindex(columns=[m for m in MONTH_ORDER if m in heat.columns], fill_value=0))
            fig2 = px.imshow(heat_piv, text_auto=True,
                             color_continuous_scale="YlOrRd",
                             title="Cases Heatmap — Year × Month", aspect="auto")
            fig2.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        # Env factors overlay
        st.markdown('<div class="sec-title">Environmental Factors vs Case Load</div>',
                    unsafe_allow_html=True)
        if all(c in df.columns for c in ["Rainfall_mm","Temperature_C"]):
            env = (df.groupby(["Year","Month_Num"])
                   .agg(Cases=("Municipality","count"),
                        Rain=("Rainfall_mm","mean"),
                        Temp=("Temperature_C","mean"))
                   .reset_index())
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(go.Bar(x=env["Month_Num"], y=env["Cases"],
                                  name="Cases", marker_color="rgba(239,68,68,.7)"),
                           secondary_y=False)
            fig3.add_trace(go.Scatter(x=env["Month_Num"], y=env["Rain"],
                                      name="Avg Rainfall (mm)",
                                      line=dict(color="#3b82f6", width=2)),
                           secondary_y=True)
            fig3.add_trace(go.Scatter(x=env["Month_Num"], y=env["Temp"],
                                      name="Avg Temp (°C)",
                                      line=dict(color="#f59e0b", width=2, dash="dot")),
                           secondary_y=True)
            fig3.update_layout(**PLOT_LAYOUT,
                               title="Cases vs Environmental Factors (all selected years)", height=380)
            fig3.update_xaxes(tickvals=list(range(1,13)),
                              ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                                        "Jul","Aug","Sep","Oct","Nov","Dec"])
            st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────
with t4:
    st.markdown('<div class="sec-title">K-Means Clustering — Municipality Risk Profiling</div>',
                unsafe_allow_html=True)

    if "Municipality" not in df.columns:
        st.warning("Municipality column required.")
    else:
        cdf = df.groupby("Municipality").agg(
            Cases        =("Municipality","count"),
            Deaths       =("Deaths","sum"),
            Hospitalized =("Hospitalized","sum"),
            Avg_Rain     =("Rainfall_mm","mean"),
            Avg_Temp     =("Temperature_C","mean"),
            Avg_Humid    =("Humidity_pct","mean"),
            Avg_Water    =("Water_Level","mean"),
        ).reset_index()
        cdf["CFR"]      = (cdf["Deaths"] / cdf["Cases"]).fillna(0)
        cdf["Hosp_Rate"]= (cdf["Hospitalized"] / cdf["Cases"]).fillna(0)

        FEAT = ["Cases","CFR","Hosp_Rate","Avg_Rain","Avg_Temp","Avg_Humid","Avg_Water"]
        X    = cdf[FEAT].values
        sc   = StandardScaler()
        Xs   = sc.fit_transform(X)

        k_range = range(2, min(9, len(cdf)))
        inertias, silhs = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(Xs)
            inertias.append(km.inertia_)
            silhs.append(silhouette_score(Xs, km.labels_))

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(x=list(k_range), y=inertias,
                                       mode="lines+markers",
                                       line=dict(color="#ef4444", width=2.5),
                                       marker=dict(size=7)))
            fig.update_layout(**PLOT_LAYOUT, title="Elbow Method",
                              xaxis_title="K", yaxis_title="Inertia")
            fig.update_xaxes(**NOGRID); fig.update_yaxes(**GRID)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = go.Figure(go.Scatter(x=list(k_range), y=silhs,
                                        mode="lines+markers",
                                        line=dict(color="#f59e0b", width=2.5),
                                        marker=dict(size=7)))
            fig2.update_layout(**PLOT_LAYOUT, title="Silhouette Score",
                               xaxis_title="K", yaxis_title="Score")
            fig2.update_xaxes(**NOGRID); fig2.update_yaxes(**GRID)
            st.plotly_chart(fig2, use_container_width=True)

        # Final clusters
        k_use = min(n_clusters, len(cdf) - 1)
        km_f  = KMeans(n_clusters=k_use, random_state=42, n_init=10)
        cdf["Cluster"] = km_f.fit_predict(Xs).astype(str)

        pca = PCA(n_components=2, random_state=42)
        Xp  = pca.fit_transform(Xs)
        cdf["PC1"], cdf["PC2"] = Xp[:,0], Xp[:,1]

        st.markdown('<div class="sec-title">Cluster Visualization (PCA 2D)</div>',
                    unsafe_allow_html=True)
        fig3 = px.scatter(
            cdf, x="PC1", y="PC2", color="Cluster", size="Cases",
            hover_name="Municipality",
            hover_data={"Cases":True,"CFR":":.2%","Cluster":True,"PC1":False,"PC2":False},
            title=f"K-Means Clusters (K={k_use}) — Municipality Risk Profiles",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig3.update_layout(**PLOT_LAYOUT, height=420)
        fig3.update_xaxes(**GRID); fig3.update_yaxes(**GRID)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="sec-title">Cluster Profile Summary</div>',
                    unsafe_allow_html=True)
        profile = (cdf.groupby("Cluster")
                   .agg(Count=("Municipality","count"),
                        Municipalities=("Municipality",lambda x: ", ".join(sorted(x))),
                        Avg_Cases=("Cases","mean"),
                        Avg_CFR=("CFR","mean"),
                        Avg_Hosp=("Hosp_Rate","mean"),
                        Avg_Rain=("Avg_Rain","mean"))
                   .reset_index())
        st.dataframe(
            profile.style.format({
                "Avg_Cases":"{:.0f}", "Avg_CFR":"{:.2%}",
                "Avg_Hosp":"{:.2%}",  "Avg_Rain":"{:.1f}",
            }),
            use_container_width=True, hide_index=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
with t5:
    st.markdown('<div class="sec-title">Outbreak Prediction — Machine Learning Models</div>',
                unsafe_allow_html=True)

    FEAT_COLS = [c for c in
                 ["Month_Num","Water_Level","Rainfall_mm","Temperature_C","Humidity_pct","Families_Affected"]
                 if c in df.columns]

    if not FEAT_COLS or "Hospitalized" not in df.columns:
        st.warning("Need feature columns + Hospitalized column for prediction.")
    else:
        pdf = df[FEAT_COLS + ["Hospitalized"]].dropna()
        Xp  = pdf[FEAT_COLS].values
        yp  = pdf["Hospitalized"].values

        ts  = test_size / 100
        Xtr, Xte, ytr, yte = train_test_split(Xp, yp, test_size=ts, random_state=42)

        models = {
            "Random Forest":     RandomForestClassifier(n_estimators=150, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        }
        results = {}
        for name, mdl in models.items():
            mdl.fit(Xtr, ytr)
            pred  = mdl.predict(Xte)
            prob  = mdl.predict_proba(Xte)[:,1]
            results[name] = {
                "model": mdl, "pred": pred, "prob": prob,
                "acc":   (pred == yte).mean(),
                "auc":   roc_auc_score(yte, prob),
            }

        # Performance table
        perf = pd.DataFrame([
            {"Model": k, "Accuracy": f"{v['acc']:.2%}", "AUC-ROC": f"{v['auc']:.3f}"}
            for k, v in results.items()
        ])
        st.markdown('<div class="sec-title">Model Performance</div>', unsafe_allow_html=True)
        st.dataframe(perf, use_container_width=True, hide_index=True)

        best_name = max(results, key=lambda k: results[k]["auc"])
        best      = results[best_name]
        st.success(f"✅ Best Model: **{best_name}**  |  AUC = {best['auc']:.3f}  |  Accuracy = {best['acc']:.2%}")

        p1, p2 = st.columns(2)
        with p1:
            cm  = confusion_matrix(yte, best["pred"])
            fig = px.imshow(cm, text_auto=True,
                            title=f"Confusion Matrix — {best_name}",
                            labels=dict(x="Predicted",y="Actual"),
                            color_continuous_scale="Reds", aspect="auto")
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with p2:
            rf  = results["Random Forest"]["model"]
            fi  = pd.DataFrame({"Feature": FEAT_COLS,
                                 "Importance": rf.feature_importances_}).sort_values("Importance")
            fig2 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                          title="Random Forest — Feature Importance",
                          color="Importance", color_continuous_scale="Reds")
            fig2.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
            fig2.update_xaxes(**GRID); fig2.update_yaxes(**NOGRID)
            st.plotly_chart(fig2, use_container_width=True)

        # ── ARIMA Forecast ────────────────────────────────────────────────────
        st.markdown('<div class="sec-title">ARIMA Time Series Forecast</div>',
                    unsafe_allow_html=True)
        ts_data = (df.groupby(["Year","Month_Num"]).size()
                   .reset_index(name="Cases")
                   .sort_values(["Year","Month_Num"]))
        series  = ts_data["Cases"].values

        if ARIMA_AVAILABLE and len(series) >= 12:
            try:
                arima = ARIMA(series, order=(2, 1, 2)).fit()
                fc    = arima.forecast(steps=forecast_per)
                fc    = np.maximum(fc, 0)           # no negative cases
                hist_idx = list(range(len(series)))
                fc_idx   = list(range(len(series), len(series) + forecast_per))

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=hist_idx, y=series, mode="lines",
                                          name="Historical",
                                          line=dict(color="#ef4444", width=2)))
                fig3.add_trace(go.Scatter(x=fc_idx, y=fc, mode="lines+markers",
                                          name=f"Forecast ({forecast_per} months)",
                                          line=dict(color="#f59e0b", width=2, dash="dash"),
                                          marker=dict(size=5, symbol="diamond")))
                fig3.add_vrect(x0=len(series)-.5, x1=len(series)+forecast_per,
                               fillcolor="rgba(245,158,11,.06)", line_width=0,
                               annotation_text="Forecast Zone",
                               annotation_font_color="#f59e0b",
                               annotation_position="top left")
                fig3.update_layout(**PLOT_LAYOUT,
                                   title=f"ARIMA Forecast — Next {forecast_per} Months",
                                   xaxis_title="Time Period (months)",
                                   yaxis_title="Predicted Cases", height=380)
                fig3.update_xaxes(**NOGRID); fig3.update_yaxes(**GRID)
                st.plotly_chart(fig3, use_container_width=True)

                fc_tbl = pd.DataFrame({
                    "Period":          [f"Month +{i+1}" for i in range(forecast_per)],
                    "Predicted Cases": np.round(fc, 1),
                })
                st.dataframe(fc_tbl.T, use_container_width=True)

            except Exception as exc:
                st.warning(f"ARIMA fitting note: {exc}. Showing moving average instead.")
                _ma_fallback(series, st)
        else:
            # ── Simple MA fallback ────────────────────────────────────────────
            ma3 = pd.Series(series).rolling(3, min_periods=1).mean().values
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=list(range(len(series))), y=series,
                                  name="Cases", marker_color="rgba(239,68,68,.75)"))
            fig3.add_trace(go.Scatter(x=list(range(len(series))), y=ma3,
                                      name="3-Month MA",
                                      line=dict(color="#f59e0b", width=2)))
            fig3.update_layout(**PLOT_LAYOUT,
                               title="Monthly Cases with 3-Month Moving Average",
                               xaxis_title="Period", yaxis_title="Cases", height=360)
            st.plotly_chart(fig3, use_container_width=True)
            if not ARIMA_AVAILABLE:
                st.info("Install statsmodels (`pip install statsmodels`) to enable ARIMA forecasting.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
with t6:
    st.markdown('<div class="sec-title">Model Comparison — 5-Fold Cross-Validation</div>',
                unsafe_allow_html=True)

    FEAT_COLS2 = [c for c in
                  ["Month_Num","Water_Level","Rainfall_mm","Temperature_C","Humidity_pct","Families_Affected"]
                  if c in df.columns]

    if not FEAT_COLS2 or "Hospitalized" not in df.columns:
        st.warning("Need feature columns + Hospitalized column.")
    else:
        cmp_df = df[FEAT_COLS2 + ["Hospitalized"]].dropna()
        Xc = StandardScaler().fit_transform(cmp_df[FEAT_COLS2].values)
        yc = cmp_df["Hospitalized"].values

        all_models = {
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
            "Naive Bayes":         GaussianNB(),
        }

        with st.spinner("Running 5-fold cross-validation on all models…"):
            rows_cv = []
            for mname, mobj in all_models.items():
                acc = cross_val_score(mobj, Xc, yc, cv=5, scoring="accuracy").mean()
                auc = cross_val_score(mobj, Xc, yc, cv=5, scoring="roc_auc").mean()
                f1  = cross_val_score(mobj, Xc, yc, cv=5, scoring="f1").mean()
                rows_cv.append({"Model": mname,
                                 "CV Accuracy": round(acc, 4),
                                 "CV AUC-ROC":  round(auc, 4),
                                 "CV F1 Score": round(f1,  4)})

        comp_tbl = pd.DataFrame(rows_cv).sort_values("CV AUC-ROC", ascending=False)

        st.dataframe(
            comp_tbl.style
            .background_gradient(subset=["CV Accuracy","CV AUC-ROC","CV F1 Score"], cmap="Reds")
            .format({"CV Accuracy":"{:.2%}","CV AUC-ROC":"{:.3f}","CV F1 Score":"{:.3f}"}),
            use_container_width=True, hide_index=True,
        )

        r1, r2 = st.columns(2)
        # ── Radar ─────────────────────────────────────────────────────────────
        with r1:
            metrics  = ["CV Accuracy","CV AUC-ROC","CV F1 Score"]
            cats     = metrics + [metrics[0]]
            clrs     = ["#ef4444","#f59e0b","#3b82f6","#22c55e","#a855f7"]
            fig_rad  = go.Figure()
            for i, (_, row) in enumerate(comp_tbl.iterrows()):
                vals = [row[m] for m in metrics] + [row[metrics[0]]]
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals, theta=cats, fill="toself", name=row["Model"],
                    line=dict(color=clrs[i % len(clrs)], width=2),
                ))
            fig_rad.update_layout(
                **PLOT_LAYOUT,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0.4, 1.0],
                                   gridcolor="rgba(255,255,255,.1)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,.1)"),
                ),
                title="Model Performance Radar",
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        # ── Bar comparison ────────────────────────────────────────────────────
        with r2:
            melted = comp_tbl.melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig_bar = px.bar(
                melted, x="Model", y="Score", color="Metric", barmode="group",
                title="Metric Comparison",
                color_discrete_sequence=["#ef4444","#f59e0b","#3b82f6"],
            )
            fig_bar.update_layout(**PLOT_LAYOUT)
            fig_bar.update_xaxes(tickangle=-20, **NOGRID)
            fig_bar.update_yaxes(range=[0.4, 1.0], **GRID)
            st.plotly_chart(fig_bar, use_container_width=True)

        best_row = comp_tbl.iloc[0]
        st.markdown(f"""
        <div class="alert-g" style="margin-top:1rem;">
        ✅ <b>Best Model: {best_row['Model']}</b> &nbsp;|&nbsp;
        AUC-ROC: <b>{best_row['CV AUC-ROC']:.3f}</b> &nbsp;|&nbsp;
        Accuracy: <b>{best_row['CV Accuracy']:.2%}</b> &nbsp;|&nbsp;
        F1: <b>{best_row['CV F1 Score']:.3f}</b><br>
        <small>Recommended for dengue hospitalization outbreak prediction in Agusan del Sur.</small>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:rgba(255,255,255,.2);font-size:.76rem;padding:.8rem 0;'>
🦟 Dengue Surveillance System · Agusan del Sur · ISELEC 104 · Ive Jane B. Sabando<br>
Built with Python · Streamlit · Plotly · Scikit-learn · Statsmodels
</div>
""", unsafe_allow_html=True)
