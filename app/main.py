import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import asyncio
import io
from datetime import datetime

# Insert project root to sys.path so imports of 'core' resolve properly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import core engines
from core.fda_client import FDAClient
from core.stats_engine import analyze_all_pairs, calculate_contingency_matrix, calculate_disproportionality_metrics
from core.network_builder import build_safety_network, generate_network_plot, map_adr_to_soc
from core.interaction_matrix import build_coprescription_matrix, generate_heatmap
from core.temporal_analysis import (
    compute_quarterly_signal_counts,
    compute_quarterly_soc_counts,
    generate_trend_line_chart,
    generate_stacked_area_chart
)
from app.pdf_generator import generate_clinical_pdf
from app.pharmacoaudit_tab import render_pharmacoaudit_tab
from app.econsim_tab import render_econsim_tab
from app.nlp_tab import render_nlp_tab

# Set page configurations
logo_path = "app/assets/logo.png"
page_icon = logo_path if os.path.exists(logo_path) else "🧬"
st.set_page_config(
    page_title="VigiSignal-X | Computational Pharmacovigilance Engine",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== PREMIUM CSS THEME ========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography with System Fallbacks */
    html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stMarkdown {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Page Intro Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stApp {
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    .animate-fade-in {
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Hero Header with animated gradient */
    .hero-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 40%, #3B82F6 70%, #6366F1 100%);
        background-size: 200% 200%;
        animation: gradient-shift 8s ease infinite;
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        border-radius: 50%;
        background: rgba(255,255,255,0.03);
    }

    .hero-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }

    .hero-header p {
        margin: 0.75rem 0 0 0;
        font-size: 1rem;
        opacity: 0.85;
        font-weight: 300;
        line-height: 1.5;
    }

    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism KPI Container & Cards */
    .kpi-container {
        display: flex;
        gap: 16px;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }

    .kpi-card {
        flex: 1 1 calc(25% - 16px);
        min-width: 220px;
        padding: 1.25rem 1.5rem;
        border-radius: 16px;
        color: #F1F5F9;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.15);
    }

    .kpi-card .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94A3B8;
        margin: 0;
    }

    .kpi-card .kpi-value {
        font-size: 2.25rem;
        font-weight: 800;
        margin: 0.25rem 0 0 0;
        line-height: 1;
        background: linear-gradient(to right, #FFFFFF, #E2E8F0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .kpi-card .kpi-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin: 0.25rem 0 0 0;
    }

    /* Transparent Colored Glass Highlights */
    .kpi-1 { background: rgba(59, 130, 246, 0.08); border-color: rgba(59, 130, 246, 0.2); }
    .kpi-2 { background: rgba(16, 185, 129, 0.08); border-color: rgba(16, 185, 129, 0.2); }
    .kpi-3 { background: rgba(245, 158, 11, 0.08); border-color: rgba(245, 158, 11, 0.2); }
    .kpi-4 { background: rgba(139, 92, 246, 0.08); border-color: rgba(139, 92, 246, 0.2); }

    /* Alert Pulse Animation for Flags */
    @keyframes pulse-glow {
        0% { box-shadow: 0 8px 32px rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.25); background: rgba(239, 68, 68, 0.08); }
        50% { box-shadow: 0 8px 32px rgba(239, 68, 68, 0.35); border-color: rgba(239, 68, 68, 0.5); background: rgba(239, 68, 68, 0.12); }
        100% { box-shadow: 0 8px 32px rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.25); background: rgba(239, 68, 68, 0.08); }
    }

    .kpi-pulse {
        animation: pulse-glow 2s infinite ease-in-out;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B0F19 0%, #111827 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #E2E8F0 !important;
    }

    /* Form control / inputs styling */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    span[data-baseweb="tag"] {
        background-color: #3B82F6 !important;
        color: white !important;
        border-radius: 6px !important;
    }

    /* Button and Download Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #2563EB, #4F46E5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        width: 100% !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4) !important;
        color: white !important;
        border: none !important;
    }

    .stDownloadButton>button {
        background: linear-gradient(135deg, #065F46, #10B981) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
        width: 100% !important;
    }

    .stDownloadButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4) !important;
        color: white !important;
        border: none !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
        color: #94A3B8;
        background-color: transparent;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #F1F5F9;
    }

    .stTabs [aria-selected="true"] {
        color: #3B82F6 !important;
        background-color: rgba(59, 130, 246, 0.08) !important;
        border-bottom: 2px solid #3B82F6 !important;
    }

    /* Dark Mode Info Cards */
    .info-card {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #F1F5F9 !important;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        border-left: 4px solid !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        border-color: rgba(255, 255, 255, 0.15) !important;
    }

    .info-card h4 {
        margin: 0 0 0.5rem 0;
        font-weight: 700;
        font-size: 1rem;
    }

    .info-card p {
        margin: 0;
        font-size: 0.85rem;
        color: #94A3B8 !important;
        line-height: 1.5;
    }

    .welcome-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-top: 1.5rem;
    }

    .stAlert {
        border-radius: 10px !important;
        background-color: rgba(30, 41, 59, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================== HELPER FUNCTIONS ========================
def fetch_data_sync(api_key):
    client = FDAClient(api_key=api_key)
    raw_data = asyncio.run(client.fetch_all_reports())
    df = client.process_and_filter_reports(raw_data)
    client.save_data(raw_data, df)
    return df

# ======================== SESSION STATE ========================
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "reports_df" not in st.session_state:
    st.session_state.reports_df = None

RAW_DATA_PATH = "data/raw/raw_reports.json"
PROCESSED_DATA_PATH = "data/processed/processed_reports.csv"

# ======================== SIDEBAR ========================
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_column_width=True)
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <h2 style="margin: 0.5rem 0 0 0; font-weight: 800; letter-spacing: -0.02em; color: #E2E8F0;">VigiSignal-X</h2>
        <p style="font-size: 0.75rem; color: #94A3B8; margin: 0.25rem 0 0 0; letter-spacing: 0.05em; text-transform: uppercase;">Computational Pharmacovigilance</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">🧬</span>
        <h2 style="margin: 0.5rem 0 0 0; font-weight: 800; letter-spacing: -0.02em; color: #E2E8F0;">VigiSignal-X</h2>
        <p style="font-size: 0.75rem; color: #94A3B8; margin: 0.25rem 0 0 0; letter-spacing: 0.05em; text-transform: uppercase;">Computational Pharmacovigilance</p>
    </div>
    """, unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 Dataset Configuration")
data_mode = st.sidebar.radio(
    "Data Source Mode",
    options=["🟢 Sample Cohort (Pre-Cached)", "🌐 openFDA Live Fetch"],
    index=0,
    help="Sample Cohort mode runs instantly offline using the pre-cached geriatric polypharmacy cohort."
)

if data_mode == "🟢 Sample Cohort (Pre-Cached)":
    if not st.session_state.data_loaded or st.session_state.reports_df is None:
        if os.path.exists(PROCESSED_DATA_PATH):
            try:
                st.session_state.reports_df = pd.read_csv(PROCESSED_DATA_PATH)
                st.session_state.data_loaded = True
            except Exception as e:
                st.sidebar.error(f"Error loading cached dataset: {e}")
        else:
            st.sidebar.error("Pre-cached sample cohort dataset not found in data/processed/processed_reports.csv.")
else:
    # Live Fetch mode. Keep data if loaded, but if nothing is loaded, state is False.
    pass

api_key = st.sidebar.text_input("openFDA API Key (Optional)", type="password", help="Keyless access uses limit=100/page. An API key unlocks limit=1000/page.")

if st.sidebar.button("🔄 Fetch Fresh FAERS Data"):
    with st.spinner("Downloading safety reports from openFDA..."):
        try:
            df = fetch_data_sync(api_key)
            if not df.empty:
                st.session_state.reports_df = df
                st.session_state.data_loaded = True
                st.sidebar.success(f"Fetched & cached {len(df)} polypharmacy reports!")
                st.rerun()
            else:
                st.sidebar.warning("No records matched the geriatric polypharmacy criteria.")
        except Exception as e:
            st.sidebar.error(f"Fetch error: {e}")

st.sidebar.markdown("### 🎛️ Analysis Filters")

# Sidebar filters
if st.session_state.data_loaded and st.session_state.reports_df is not None:
    df = st.session_state.reports_df

    all_drugs_flat = []
    for d_str in df["drugs"].dropna():
        all_drugs_flat.extend(str(d_str).split(";"))

    drug_counts = pd.Series(all_drugs_flat).value_counts()
    drug_options = list(drug_counts.index)
    drug_select_options = [f"{d} (N={drug_counts[d]})" for d in drug_options]
    drug_to_freq_map = {f"{d} (N={drug_counts[d]})": d for d in drug_options}

    # Pre-load top demo drugs in Sample Cohort mode
    default_selection = []
    if data_mode == "🟢 Sample Cohort (Pre-Cached)":
        default_demo_drugs = ["ASPIRIN", "LISINOPRIL", "METFORMIN", "IBUPROFEN", "PREDNISONE"]
        for d in default_demo_drugs:
            match = [opt for opt in drug_select_options if opt.startswith(d + " ")]
            if match:
                default_selection.append(match[0])
    
    if not default_selection:
        default_selection = drug_select_options[:5] if len(drug_select_options) >= 5 else drug_select_options

    selected_drugs_raw = st.sidebar.multiselect(
        "Target Medications",
        options=drug_select_options,
        default=default_selection,
        help="Select specific drugs to analyze."
    )
    selected_drugs = [drug_to_freq_map[s] for s in selected_drugs_raw]

    min_cases = st.sidebar.slider("Min Safety Cases (a)", 1, 30, 3,
                                   help="Minimum co-occurrence count to flag a signal.")
    ror_threshold = st.sidebar.slider("ROR Threshold", 0.5, 5.0, 1.0, 0.1,
                                       help="Flag signals only if ROR exceeds this value.")
else:
    st.sidebar.info("Fetch data to enable filters.")
    selected_drugs = []
    min_cases = 3
    ror_threshold = 1.0

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align:center; padding: 0.5rem 0;">
    <p style="font-size: 0.7rem; color: #64748B; margin: 0;">
        Built with Streamlit • openFDA FAERS<br>
        © 2026 VigiSignal-X Research
    </p>
</div>
""", unsafe_allow_html=True)

# ======================== MAIN CONTENT ========================

# Hero Header
st.markdown("""
<div class="hero-header">
    <h1>🧬 VigiSignal-X Computational Pharmacovigilance</h1>
    <p>Next-generation adverse drug reaction signal detection and bipartite biorisk network mapping
    in geriatric polypharmacy cohorts (Age ≥ 60, Concomitant Medications > 4)</p>
</div>
""", unsafe_allow_html=True)

if data_mode == "🟢 Sample Cohort (Pre-Cached)":
    st.markdown("""
    <div style="background-color: rgba(16, 185, 129, 0.15); border-left: 5px solid #10B981; padding: 12px 15px; border-radius: 8px; margin-bottom: 1.5rem;">
        <span style="font-weight: 700; color: #10B981;">🟢 SAMPLE COHORT MODE ACTIVE</span>
        <span style="color: #94A3B8; font-size: 0.9rem; margin-left: 10px;">Using pre-cached sample cohort dataset (data/processed/processed_reports.csv)</span>
    </div>
    """, unsafe_allow_html=True)

# ======================== WELCOME SCREEN ========================
if not st.session_state.data_loaded or st.session_state.reports_df is None:
    st.info("👋 **Welcome to VigiSignal-X!** Click 'Fetch Fresh FAERS Data' in the sidebar to begin.")

    st.markdown("""
    <div class="welcome-grid animate-fade-in">
        <div class="info-card" style="border-left-color: #3B82F6;">
            <h4 style="color: #3B82F6;">Async Fetching</h4>
            <p>Concurrent openFDA queries with rate-limiting & exponential backoff.</p>
        </div>
        <div class="info-card" style="border-left-color: #10B981;">
            <h4 style="color: #10B981;">Contingency Analytics</h4>
            <p>2×2 matrix with ROR, 95% CI, and Haldane correction.</p>
        </div>
        <div class="info-card" style="border-left-color: #F59E0B;">
            <h4 style="color: #F59E0B;">Network Mapping</h4>
            <p>Bipartite drug→SOC graphs with Plotly interactivity.</p>
        </div>
        <div class="info-card" style="border-left-color: #8B5CF6;">
            <h4 style="color: #8B5CF6;">Clinical Reports</h4>
            <p>Offline PDF generation with ReportLab for clinical audit.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ======================== DATA LOADED — RUN ANALYSIS ========================
df_active = st.session_state.reports_df

if selected_drugs:
    mask = df_active["drugs"].apply(lambda x: any(sd in str(x).split(";") for sd in selected_drugs))
    df_filtered_cohort = df_active[mask].reset_index(drop=True)
else:
    df_filtered_cohort = df_active

# Run Stats Engine
signals_df = analyze_all_pairs(df_filtered_cohort, min_cases=min_cases, ror_threshold=ror_threshold)

# Summary metrics
total_cohort_reports = len(df_active)
filtered_reports = len(df_filtered_cohort)
num_signals = len(signals_df[signals_df["is_signal"] == True]) if not signals_df.empty else 0
unique_drugs_analyzed = len(selected_drugs) if selected_drugs else len(set(";".join(df_active["drugs"].dropna()).split(";")))

# KPI Cards
pulse_class = " kpi-pulse" if num_signals > 0 else ""
st.markdown(f"""
<div class="kpi-container animate-fade-in">
    <div class="kpi-card kpi-1">
        <p class="kpi-label">Total Cohort Reports</p>
        <p class="kpi-value">{total_cohort_reports:,}</p>
        <p class="kpi-sub">Geriatric polypharmacy profiles</p>
    </div>
    <div class="kpi-card kpi-2">
        <p class="kpi-label">Filtered for Analysis</p>
        <p class="kpi-value">{filtered_reports:,}</p>
        <p class="kpi-sub">Matching selected medications</p>
    </div>
    <div class="kpi-card kpi-3{pulse_class}">
        <p class="kpi-label">Safety Signals Flagged</p>
        <p class="kpi-value">{num_signals}</p>
        <p class="kpi-sub">ROR &gt; {ror_threshold}, CI<sub>lower</sub> &gt; 1.0 &amp; IC<sub>025</sub> &gt; 0.0</p>
    </div>
    <div class="kpi-card kpi-4">
        <p class="kpi-label">Drugs Analyzed</p>
        <p class="kpi-value">{unique_drugs_analyzed}</p>
        <p class="kpi-sub">Unique medicinal products</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================== TABS ========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Safety Signals",
    "🕸️ Toxicity Network",
    "🔥 Drug Interactions",
    "📈 Temporal Trends",
    "📋 Analytics Table",
    "📄 Clinical Report",
    "📚 Methodology & Insights",
    "📋 PharmacoAudit-AI",
    "💸 EconSim-Health",
    "🛡️ Clinical NLP & De-Identification"
])

# ==================== TAB 1: SAFETY SIGNALS DASHBOARD ====================
with tab1:
    st.subheader("Disproportionality Signal Scores")
    st.markdown("Horizontal bar charts of the highest Reporting Odds Ratios (ROR) for target drug-ADR pairs, color-coded by System Organ Class.")

    if not signals_df.empty:
        active_signals = signals_df[signals_df["is_signal"] == True].copy()
        if selected_drugs:
            active_signals = active_signals[active_signals["drug"].isin(selected_drugs)]

        if active_signals.empty:
            st.info("No statistically significant safety signals detected for the selected drugs/thresholds.")
        else:
            import plotly.express as px
            top_plot = active_signals.head(20).sort_values(by="ror", ascending=True).copy()
            top_plot["SOC"] = top_plot["adr"].apply(map_adr_to_soc)

            fig = px.bar(
                top_plot, x="ror", y="adr", color="SOC",
                facet_col="drug", facet_col_wrap=2, orientation="h",
                labels={"ror": "Reporting Odds Ratio (ROR)", "adr": "Adverse Reaction (ADR)", "drug": "Medication"},
                title="Top 20 Drug-ADR Safety Signals (Grouped by Medication)",
                hover_data={"a": True, "ci_lower": ":.2f", "ci_upper": ":.2f", "triage_tier": True},
                color_discrete_map={
                    "Cardiotoxicity/Vascular": "#EF4444",
                    "Gastrointestinal": "#F59E0B",
                    "Renal/Urinary": "#3B82F6",
                    "Respiratory": "#10B981",
                    "Nervous System": "#8B5CF6",
                    "Other/Unclassified": "#9CA3AF"
                }
            )
            fig.update_layout(
                template="plotly_dark",
                height=500 + (len(top_plot["drug"].unique()) // 2) * 150,
                xaxis_title="Reporting Odds Ratio (ROR)", yaxis_title="",
                hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_color="#E2E8F0"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F1F5F9", family="Inter, sans-serif"),
                title=dict(font=dict(color="#F1F5F9"))
            )
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="Baseline")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available to plot.")

# ==================== TAB 2: TOXICITY NETWORK GRAPH ====================
with tab2:
    st.subheader("Bipartite Drug-Toxicity Network")
    st.markdown("Interactive mapping of drug nodes to System Organ Class toxicities. Edge thickness = unique ADR count, color intensity = ROR.")

    col_ctrl, col_graph = st.columns([1, 4])
    with col_ctrl:
        layout_choice = st.radio("Layout",
                                  ["Circular", "Bipartite (Columns)", "Spring (Dynamic)"], index=0)
        layout_map = {"Circular": "circular", "Bipartite (Columns)": "bipartite", "Spring (Dynamic)": "spring"}
        layout_type = layout_map[layout_choice]

        st.markdown("""
        **Legend:**
        - 🔵 **Blue** = Target Drugs
        - 🟠 **Orange** = Organ Systems
        - 🔴 **Red edges** = Signal strength
        """)

    with col_graph:
        if not signals_df.empty:
            active_signals = signals_df[signals_df["is_signal"] == True].copy()
            if selected_drugs:
                active_signals = active_signals[active_signals["drug"].isin(selected_drugs)]
            G = build_safety_network(active_signals)
            fig_net = generate_network_plot(G, layout_type=layout_type)
            st.plotly_chart(fig_net, use_container_width=True)
        else:
            st.info("No data to construct the network.")

# ==================== TAB 3: DRUG INTERACTION HEATMAP ====================
with tab3:
    st.subheader("Drug Co-Prescription Interaction Heatmap")
    st.markdown("Reveals which medications are most frequently co-prescribed in the geriatric polypharmacy cohort. Hotspots (bright red) indicate high-frequency drug combinations that warrant clinical scrutiny for potential drug-drug interactions.")

    if df_filtered_cohort is not None and not df_filtered_cohort.empty:
        top_n = st.slider("Number of Top Drugs to Analyze", 5, 25, 15, key="heatmap_top_n")

        co_matrix, pair_details = build_coprescription_matrix(df_filtered_cohort, top_n_drugs=top_n)

        # Heatmap
        fig_heat = generate_heatmap(co_matrix)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Pair details table
        st.markdown("#### Top Co-Prescribed Drug Pairs")
        if not pair_details.empty:
            display_pairs = pair_details.head(20).copy()
            display_pairs = display_pairs.rename(columns={
                "drug_a": "Drug A",
                "drug_b": "Drug B",
                "co_count": "Co-Prescriptions",
                "avg_reactions_per_report": "Avg ADRs/Report",
                "top_adrs": "Top Adverse Events",
                "affected_systems": "Affected Organ Systems"
            })
            st.dataframe(display_pairs, use_container_width=True, hide_index=True)
        else:
            st.info("No co-prescription pairs found.")
    else:
        st.info("No data loaded.")

# ==================== TAB 4: TEMPORAL TRENDS ====================
with tab4:
    st.subheader("Temporal Safety Signal Evolution")
    st.markdown("Tracks how adverse drug reaction frequencies change over time (quarterly aggregation). Use this to spot emerging safety signals that may indicate new pharmacovigilance concerns.")

    if df_filtered_cohort is not None and not df_filtered_cohort.empty:
        viz_type = st.radio("Visualization Type",
                             ["ADR Trend Lines", "Organ System Burden (Stacked Area)"],
                             horizontal=True, key="temporal_viz")

        if viz_type == "ADR Trend Lines":
            top_n_adrs = st.slider("Top N Adverse Reactions to Track", 3, 15, 8, key="temporal_top_n")
            pivot = compute_quarterly_signal_counts(df_filtered_cohort, selected_drugs, top_n_adrs=top_n_adrs)
            omission_pct = pivot.attrs.get("omission_pct", 0.0)
            
            if omission_pct < 100.0:
                fig_trend = generate_trend_line_chart(pivot, title="ADR Reporting Frequency Over Time")
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.error("❌ Cannot render timeline: 100% of records lack valid date documentation.")
        else:
            pivot_soc = compute_quarterly_soc_counts(df_filtered_cohort, selected_drugs)
            omission_pct = pivot_soc.attrs.get("omission_pct", 0.0)
            
            if omission_pct < 100.0:
                fig_area = generate_stacked_area_chart(pivot_soc, title="System Organ Class Toxicity Burden Over Time")
                st.plotly_chart(fig_area, use_container_width=True)
            else:
                st.error("❌ Cannot render timeline: 100% of records lack valid date documentation.")

        # Show warning/info if some records were excluded due to missing dates
        if omission_pct > 0.0:
            st.warning(f"⚠️ **Data Integrity Warning:** {omission_pct:.1f}% of cohort records were excluded from this temporal trend analysis due to incomplete or missing date documentation in the source FAERS reports.")

        # Summary insight
        st.markdown("""
        > **Clinical Insight:** Increasing trends in specific ADR categories over consecutive quarters
        may indicate emerging safety concerns that warrant regulatory attention or prescribing guideline updates.
        """)
    else:
        st.info("No data loaded.")

# ==================== TAB 5: CONTINGENCY ANALYTICS TABLE ====================
with tab5:
    st.subheader("Drug-ADR Contingency Calculations")
    st.markdown("Searchable table of raw frequency parameters and computed disproportionality metrics.")

    if not signals_df.empty:
        search_query = st.text_input("🔍 Search by Medication or ADR:", "", key="table_search").strip().upper()

        table_df = signals_df.copy()
        if search_query:
            table_df = table_df[
                table_df["drug"].str.contains(search_query) |
                table_df["adr"].str.contains(search_query)
            ]

        styled_df = table_df.copy()
        styled_df["is_signal"] = styled_df["is_signal"].apply(lambda x: "✅ SIGNAL" if x else "❌ NO SIGNAL")
        
        # Format triage tier with emojis
        triage_map = {
            "High Priority": "🔴 High Priority",
            "Moderate Priority": "🟠 Moderate Priority",
            "Weak Priority": "🟡 Weak Priority",
            "Not Significant": "⚪ Not Significant"
        }
        styled_df["triage_tier"] = styled_df["triage_tier"].map(triage_map)
        
        styled_df = styled_df.rename(columns={
            "drug": "Medication", "adr": "Adverse Reaction (ADR)",
            "a": "Cases (a)", "b": "Other Drug Event (b)",
            "c": "Drug No Event (c)", "d": "Other No Event (d)",
            "ror": "ROR Score", "se": "SE ln(ROR)",
            "ci_lower": "95% CI Lower", "ci_upper": "95% CI Upper",
            "haldane_applied": "Haldane Correction", "is_signal": "Status",
            "triage_tier": "Triage Priority"
        })

        # Display CSV export button
        col_dl, col_spacer = st.columns([1, 3])
        with col_dl:
            csv_data = table_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Export Table as CSV",
                data=csv_data,
                file_name=f"VigiSignal_X_Contingency_Table_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No data loaded to show table.")

# ==================== TAB 6: CLINICAL TRIAGE REPORT ====================
with tab6:
    st.subheader("Clinical Triage Audit & PDF Report")
    st.markdown("Generate and download a professional pharmacovigilance audit PDF.")

    if not signals_df.empty:
        os.makedirs("data/reports", exist_ok=True)
        report_path = f"data/reports/vigisignal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Live preview
        st.markdown("---")
        st.markdown("### 📋 Executive Summary Preview")

        active_sig_list = signals_df[signals_df["is_signal"] == True]
        if selected_drugs:
            active_sig_list = active_sig_list[active_sig_list["drug"].isin(selected_drugs)]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Clinical Metadata:**
            - **Cohort:** Geriatric (Age ≥ 60), >4 medications
            - **Medications:** {", ".join(selected_drugs) if selected_drugs else "All"}
            - **Signals Identified:** `{len(active_sig_list)}`
            - **Analysis Date:** `{datetime.now().strftime('%Y-%m-%d')}`
            """)
        with col2:
            st.markdown("""
            **Triage Action Plan:**
            - Reconcile polypharmacy profiles
            - Monitor cardiovascular metrics for cardiotoxicity
            - Adjust dosing based on eGFR for renal signals
            - Report novel ADRs to FDA MedWatch
            """)

        report_format = st.radio(
            "Report Format",
            options=["📄 One-Page Executive Summary", "📋 Full Clinical Audit Report"],
            index=1,
            horizontal=True,
            help="Select whether to generate a single-page quick summary or a detailed multi-page clinical report."
        )

        st.markdown("---")

        if st.button("📁 Compile Clinical PDF Report"):
            try:
                if report_format == "📄 One-Page Executive Summary":
                    from app.pdf_generator import generate_one_page_summary
                    generate_one_page_summary(
                        output_path=report_path,
                        signals_df=signals_df,
                        target_drugs=selected_drugs,
                        total_cohort_reports=total_cohort_reports,
                        min_cases=min_cases,
                        ror_threshold=ror_threshold
                    )
                    file_name = f"VigiSignal_X_Summary_{datetime.now().strftime('%Y%m%d')}.pdf"
                    btn_label = "⬇️ Download Executive Summary PDF"
                else:
                    generate_clinical_pdf(
                        output_path=report_path,
                        signals_df=signals_df,
                        target_drugs=selected_drugs,
                        total_cohort_reports=total_cohort_reports,
                        min_cases=min_cases,
                        ror_threshold=ror_threshold
                    )
                    file_name = f"VigiSignal_X_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
                    btn_label = "⬇️ Download Clinical Audit PDF"
                    
                with open(report_path, "rb") as f:
                    pdf_bytes = f.read()
                st.success("✅ PDF compiled successfully!")
                st.download_button(
                    label=btn_label,
                    data=pdf_bytes,
                    file_name=file_name,
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Error compiling PDF: {e}")
    else:
        st.info("No data loaded. Fetch data before generating reports.")

# ==================== TAB 7: RESEARCH METHODOLOGY & INSIGHTS ====================
with tab7:
    st.subheader("📚 Research Methodology & Clinical Insights")
    
    st.markdown("""
    This section outlines the analytical framework, mathematical formulas, and validation strategies behind **VigiSignal-X**, demonstrating its applicability for international-standard pharmacovigilance research.
    """)
    
    col_math, col_tiers = st.columns(2)
    
    with col_math:
        st.markdown("""
        ### 🧮 Mathematical Engine
        VigiSignal-X computes disproportionality metrics using a **2x2 contingency matrix** for a target drug and adverse reaction (ADR):
        
        | | Target ADR | Other ADRs |
        |---|---|---|
        | **Target Drug** | $a$ (Exposed with Event) | $c$ (Exposed without Event) |
        | **Other Drugs** | $b$ (Unexposed with Event) | $d$ (Unexposed without Event) |
        
        #### 1. Reporting Odds Ratio (ROR)
        $$ROR = \\frac{a \\times d}{b \\times c}$$
        
        #### 2. Standard Error of $\\ln(ROR)$
        $$SE = \\sqrt{\\frac{1}{a} + \\frac{1}{b} + \\frac{1}{c} + \\frac{1}{d}}$$
        
        #### 3. 95% Confidence Interval (CI)
        $$95\\% \\text{ CI} = e^{\\ln(ROR) \\pm 1.96 \\times SE}$$
        
        #### 4. Haldane Correction
        If any cell ($a, b, c, d$) is $0$, a Haldane correction of $+0.5$ is added to all cells to prevent division-by-zero or undefined standard error calculations.
        """)
        
    with col_tiers:
        st.markdown("""
        ### 🚦 Clinical Triage Tiers
        To assist clinical pharmacologists in triaging signal significance, VigiSignal-X classifies detected relationships into four tiers:
        
        * 🔴 **High Priority (Tier 1):** ROR $\\ge 3.0$, lower bound of $95\\%$ CI $> 1.5$, and case count $a \\ge 5$. Represents strong, statistically robust signal co-occurrences.
        * 🟠 **Moderate Priority (Tier 2):** ROR $\\ge 2.0$, lower bound of $95\\%$ CI $> 1.0$, and case count $a \\ge 3$. Represents significant signals with moderate strength.
        * 🟡 **Weak Priority (Tier 3):** ROR $> 1.0$, lower bound of $95\\%$ CI $> 1.0$, and case count $a \\ge 3$. Represents early/weak disproportionality.
        * ⚪ **Not Significant (Tier 4):** Case count $a < 3$, ROR $\\le 1.0$, or lower bound of $95\\%$ CI $\\le 1.0$.
        """)
        
    st.markdown("---")
    
    st.markdown("### 🧬 Literature-Based Positive Control Validation")
    
    st.markdown("""
    To validate the signal-detection pipeline, the engine's outputs are cross-referenced with established pharmacotherapy literature and FDA-approved product labels. 
    """)
    
    validation_data = [
        {
            "Medication": "LISINOPRIL",
            "Adverse Event (ADR)": "ANGIOEDEMA",
            "Triage Priority": "🔴 High Priority",
            "Clinical Mechanism / Label Status": "Accumulation of bradykinin (ACE inhibition). FDA black-box warning.",
            "Citation (PMID)": "PMID: 19808381"
        },
        {
            "Medication": "ASPIRIN",
            "Adverse Event (ADR)": "GASTROINTESTINAL HAEMORRHAGE",
            "Triage Priority": "🔴 High Priority",
            "Clinical Mechanism / Label Status": "COX-1 inhibition reduces cytoprotective prostaglandins. Standard label warning.",
            "Citation (PMID)": "PMID: 11095507"
        },
        {
            "Medication": "OMNISCAN (GADODIAMIDE)",
            "Adverse Event (ADR)": "NEPHROGENIC SYSTEMIC FIBROSIS",
            "Triage Priority": "🔴 High Priority",
            "Clinical Mechanism / Label Status": "Gadolinium accumulation in severe renal impairment. FDA black-box warning.",
            "Citation (PMID)": "PMID: 18055615"
        },
        {
            "Medication": "METFORMIN",
            "Adverse Event (ADR)": "LACTIC ACIDOSIS",
            "Triage Priority": "🟠 Moderate Priority",
            "Clinical Mechanism / Label Status": "Inhibition of mitochondrial oxidative phosphorylation. FDA black-box warning.",
            "Citation (PMID)": "PMID: 27150179"
        }
    ]
    validation_df = pd.DataFrame(validation_data)
    st.dataframe(validation_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 Case Study: Geriatric Polypharmacy Cascades")
    st.markdown("""
    Geriatric polypharmacy cohorts are at high risk of drug-drug-adverse-event cascades. Common combinations include:
    * **ACE Inhibitor (Lisinopril) + NSAID (Ibuprofen):** Concomitant use increases renal strain (triple whammy effect if a diuretic is present), which manifests as a sharp rise in **Acute Kidney Injury** signals.
    * **Antiplatelet (Aspirin) + NSAID (Ibuprofen):** Co-administration of Aspirin and Ibuprofen significantly amplifies mucosal irritation, producing high ROR signals for **Gastrointestinal Haemorrhage**.
    """)
        
    st.markdown("---")
    
    st.markdown("""
    ### ⚠️ Limitations of FAERS Data
    When interpreting these safety signals, it is crucial to remain scientifically objective and acknowledge the following baseline limitations:
    1. **No Causality Proof:** Disproportionality metrics indicate reporting associations, not direct biological causality.
    2. **Reporting Bias:** Spontaneous reporting systems are subject to significant underreporting, duplication, and media-driven reporting spikes.
    3. **No Denominator:** FAERS contains only adverse event reports; the total number of patients exposed to the drug is unknown, preventing direct incidence calculation.
    """)

# ==================== TAB 8: PHARMACOAUDIT-AI ====================
with tab8:
    render_pharmacoaudit_tab()

# ==================== TAB 9: ECONSIM-HEALTH ====================
with tab9:
    render_econsim_tab()

# ==================== TAB 10: CLINICAL NLP & DE-IDENTIFICATION ====================
with tab10:
    render_nlp_tab()
