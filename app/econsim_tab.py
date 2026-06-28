import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core.simulation_engine import simulate_deprescribing_heor, simulate_sensitivity_analysis

def render_econsim_tab() -> None:
    """
    Renders the EconSim-Health tab UI.
    """
    st.subheader("💸 EconSim-Health")
    st.markdown("Health Economics & Outcomes Research (HEOR) Decision-Analytic Simulator for Geriatric Polypharmacy Deprescribing Interventions")
    
    # 🎛️ Render Simulation Parameters inside the tab in an expander with structured columns
    with st.expander("🎛️ Model Simulation Parameters", expanded=True):
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            st.markdown("##### **Cohort & Intervention**")
            cohort_size = st.number_input(
                "Cohort Size (Patients)", 100, 10000, 1000, step=100, key="econsim_cohort_size"
            )
            deprescribe_success = st.slider(
                "Deprescribing Success Rate (%)", 10, 100, 60, key="econsim_deprescribe_success"
            ) / 100.0
            cost_intervention = st.number_input(
                "Intervention Cost per Patient ($)", 10, 1000, 150, key="econsim_cost_intervention"
            )
            
        with col_p2:
            st.markdown("##### **Clinical Event Probabilities**")
            prob_ade_control = st.slider(
                "Prob. of Adverse Event (Control) (%)", 5, 50, 20, key="econsim_prob_ade_control"
            ) / 100.0
            prob_ade_interv = st.slider(
                "Prob. of Adverse Event (Intervention) (%)", 1, 30, 8, key="econsim_prob_ade_interv"
            ) / 100.0
            cost_ade = st.number_input(
                "Cost of Treating Adverse Event ($)", 1000, 50000, 12000, step=1000, key="econsim_cost_ade"
            )
            
        with col_p3:
            st.markdown("##### **Utility (QALY) Weights**")
            qaly_healthy = st.slider(
                "QALY Weight (Healthy State)", 0.7, 1.0, 0.9, step=0.05, key="econsim_qaly_healthy"
            )
            qaly_ade = st.slider(
                "QALY Weight (Adverse Event State)", 0.2, 0.7, 0.5, step=0.05, key="econsim_qaly_ade"
            )
            
    # Perform Calculations
    results = simulate_deprescribing_heor(
        cohort_size=cohort_size,
        deprescribe_success=deprescribe_success,
        cost_intervention=cost_intervention,
        prob_ade_control=prob_ade_control,
        prob_ade_interv=prob_ade_interv,
        cost_ade=cost_ade,
        qaly_healthy=qaly_healthy,
        qaly_ade=qaly_ade
    )
    
    inc_cost = results["inc_cost"]
    inc_qaly = results["inc_qaly"]
    effective_ade_prob = results["effective_ade_prob"]
    total_cost_control = results["total_cost_control"]
    icer = results["icer"]

    # Display KPI cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-label">Incremental Cost</p>
            <p class="kpi-value">${inc_cost:,.2f}</p>
            <p style="font-size:0.75rem; color:#94A3B8; margin:0.25rem 0 0 0;">Difference in total health costs</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-label">Incremental QALYs Gained</p>
            <p class="kpi-value">{inc_qaly:,.2f}</p>
            <p style="font-size:0.75rem; color:#94A3B8; margin:0.25rem 0 0 0;">Quality-Adjusted Life Years gained</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        if np.isnan(icer):
            icer_str = "N/A"
        elif icer < 0:
            icer_str = "COST SAVING (Dominant)"
        else:
            icer_str = f"${icer:,.2f} / QALY"
        st.markdown(f"""
        <div class="kpi-card" style="border-color: #10B981;">
            <p class="kpi-label">ICER Score</p>
            <p class="kpi-value" style="font-size:1.5rem; color:#10B981;">{icer_str}</p>
            <p style="font-size:0.75rem; color:#94A3B8; margin:0.25rem 0 0 0;">Incremental Cost-Effectiveness Ratio</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📊 Cost Comparison")
        
        cost_df = pd.DataFrame({
            "Group": ["Control (Standard Care)", "Clinical Intervention"],
            "Program Costs ($)": [0, cohort_size * cost_intervention],
            "Adverse Event Treatment ($)": [total_cost_control, cohort_size * effective_ade_prob * cost_ade]
        })
        
        fig = px.bar(
            cost_df, x="Group", y=["Program Costs ($)", "Adverse Event Treatment ($)"],
            title="Aggregated Health System Expenditures",
            color_discrete_map={"Program Costs ($)": "#2563EB", "Adverse Event Treatment ($)": "#EF4444"}
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F1F5F9", family="Inter, sans-serif"),
            barmode="stack"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### 🎯 Cost-Effectiveness Scatter Plane")
        
        sim_df = simulate_sensitivity_analysis(
            inc_cost=inc_cost,
            inc_qaly=inc_qaly,
            cost_ade=cost_ade,
            cohort_size=cohort_size,
            iterations=250,
            deprescribe_success=deprescribe_success,
            cost_intervention=cost_intervention,
            prob_ade_control=prob_ade_control,
            prob_ade_interv=prob_ade_interv,
            qaly_healthy=qaly_healthy,
            qaly_ade=qaly_ade
        )
        
        fig_scatter = px.scatter(
            sim_df, x="Incremental QALYs", y="Incremental Cost ($)",
            title="Monte Carlo Sensitivity Analysis (WTP threshold = $50,000/QALY)",
            labels={"Incremental QALYs": "Incremental QALYs Gained", "Incremental Cost ($)": "Incremental Cost ($)"}
        )
        
        # Add Willingness-to-pay line
        x_line = np.linspace(min(sim_df["Incremental QALYs"]), max(sim_df["Incremental QALYs"]), 10)
        y_line = x_line * 50000
        
        fig_scatter.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines", name="WTP Threshold ($50k)",
            line=dict(color="orange", dash="dash")
        ))
        
        fig_scatter.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F1F5F9", family="Inter, sans-serif")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Interpretation Box
    st.markdown("### 📋 Clinical & Economic Interpretation")
    if icer < 0:
        st.success("🎉 **Dominant Intervention:** The pharmacist-led deprescribing program is **cost-saving and health-improving**. It costs less than standard care while generating higher quality-adjusted life years for the cohort. Implementation is highly recommended.")
    elif icer <= 50000:
        st.info(f"✅ **Cost-Effective Intervention:** The program has an ICER of **${icer:,.2f}/QALY**, which is below the common willingness-to-pay threshold of **$50,000/QALY**. The health benefits justify the investment.")
    else:
        st.warning(f"⚠️ **Not Cost-Effective:** The program has an ICER of **${icer:,.2f}/QALY**, which exceeds the willingness-to-pay threshold of **$50,000/QALY**. Consider optimizing intervention costs or targeting higher-risk cohorts.")

    st.markdown("---")
    with st.expander("ℹ️ Model Methodology", expanded=True):
        st.markdown("""
        * **Design:** Decision-Analytic Decision Tree Model
        * **Cohort:** Hypertensive geriatric patients (Age >= 60, Medications > 4)
        * **Goal:** Evaluate the cost-utility of a clinical pharmacist reviewing medication profiles to deprescribe NSAIDs or other high-risk medication triggers.
        """)
