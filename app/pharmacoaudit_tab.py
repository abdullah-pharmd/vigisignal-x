import streamlit as st
from core.protocol_parser import parse_protocol

# Sample Clinical Trial Protocol Text
SAMPLE_PROTOCOL = """CLINICAL TRIAL PROTOCOL: PROTOCOL-L-4725
TITLE: A Phase III Randomized, Double-Blind, Placebo-Controlled Study of Lisinopril for Cardioprotection in Older Hypertensive Patients with History of Polypharmacy.

1. OBJECTIVES AND ENDPOINTS
The primary objective of this study is to evaluate the clinical efficacy of Lisinopril compared to placebo in reducing major adverse cardiovascular events (MACE).
The primary endpoint is the occurrence of cardiovascular death, non-fatal myocardial infarction, or non-fatal stroke within a 24-month follow-up period.
The secondary endpoint includes the rate of all-cause hospitalization, change in systolic blood pressure, and renal clearance metrics.

2. PATIENT ELIGIBILITY CRITERIA
Patients must meet all of the following inclusion criteria:
- Male or female aged 60 years or older.
- Documented essential hypertension with systolic blood pressure ≥ 140 mmHg.
- Currently taking at least 4 concomitant medications (geriatric polypharmacy profile).
- Patient must provide written informed consent.

Patients meeting any of the following exclusion criteria will be excluded:
- History of angioedema associated with prior ACE inhibitor treatment.
- Severe renal impairment defined as estimated glomerular filtration rate (eGFR) < 30 mL/min/1.73m².
- Active hyperkalemia with serum potassium > 5.5 mEq/L.
- Pregnant or lactating women.

3. SAFETY BOUNDARIES AND STOPPING RULES
Safety monitoring will be conducted weekly. The following safety stopping boundaries are established:
- Grade 3 or 4 Hyperkalemia (Serum Potassium ≥ 6.0 mEq/L) will require immediate treatment discontinuation.
- Severe symptomatic angioedema or airway compromise requires permanent withdrawal of Lisinopril.
- A confirmed increase in serum creatinine of ≥ 50% above baseline values will trigger dosage reduction and a renal safety review.
"""

def render_pharmacoaudit_tab() -> None:
    """
    Renders the PharmacoAudit-AI tab UI.
    """
    st.subheader("📋 PharmacoAudit-AI")
    st.markdown("Automated Natural Language Processing (NLP) Parser for Extracting Safety Boundaries & Eligibility Criteria from Clinical Trial Protocols")
    
    # Initialize session state for protocol input
    if "pharmacoaudit_text" not in st.session_state:
        st.session_state.pharmacoaudit_text = ""
        
    col_left, col_right = st.columns([2, 3])
    
    with col_left:
        st.markdown("### 📥 Protocol Input")
        st.write("Paste your clinical trial protocol text below or load the pre-configured sample protocol.")
        
        if st.button("📄 Load Sample Protocol", key="pharmacoaudit_load_sample_btn"):
            st.session_state.pharmacoaudit_text = SAMPLE_PROTOCOL
            
        protocol_input = st.text_area(
            "Clinical Trial Protocol Text",
            value=st.session_state.pharmacoaudit_text,
            placeholder="Paste protocol here...",
            height=350,
            key="pharmacoaudit_text_area"
        )
        # Update session state to capture user changes
        st.session_state.pharmacoaudit_text = protocol_input
        
        # Display About section inside the tab to avoid sidebar clutter
        st.markdown("---")
        with st.expander("ℹ️ About PharmacoAudit-AI", expanded=True):
            st.markdown("""
            * **Field:** Regulatory Science & Clinical Trial Audits
            * **Methodology:** Regular Expression Parser & Semantic Keyword Triage
            * **Goal:** Automate protocol parsing to identify safety-stopping boundaries and eligibility criteria.
            """)

    with col_right:
        st.markdown("### 📊 Extracted Safety & Protocol Boundaries")
        
        if protocol_input.strip():
            with st.spinner("Parsing protocol structure..."):
                extracted = parse_protocol(protocol_input)
                
            if extracted:
                tab1, tab2, tab3 = st.tabs(["📋 Eligibility Criteria", "🎯 Objectives & Endpoints", "🚨 Safety Stopping Boundaries"])
                
                with tab1:
                    st.markdown("#### **Inclusion Criteria**")
                    if extracted.get("inclusion"):
                        for item in extracted["inclusion"]:
                            st.markdown(f"✅ {item}")
                    else:
                        st.info("No inclusion criteria detected.")
                        
                    st.markdown("---")
                    st.markdown("#### **Exclusion Criteria**")
                    if extracted.get("exclusion"):
                        for item in extracted["exclusion"]:
                            st.markdown(f"❌ {item}")
                    else:
                        st.info("No exclusion criteria detected.")
                        
                with tab2:
                    st.markdown("#### **Protocol Objectives & Endpoints**")
                    if extracted.get("endpoints"):
                        for item in extracted["endpoints"]:
                            st.markdown(f"🎯 {item}")
                    else:
                        st.info("No endpoints detected.")
                        
                with tab3:
                    st.markdown("#### **Clinical Stopping Rules & Safety Boundaries**")
                    if extracted.get("safety_boundaries"):
                        for item in extracted["safety_boundaries"]:
                            st.markdown(f"⚠️ {item}")
                    else:
                        st.info("No safety stopping boundaries detected.")
                        
                # Export Markdown Report
                st.markdown("---")
                audit_date = st.date_input("Audit Date", key="pharmacoaudit_audit_date")
                md_output = f"""# PharmacoAudit-AI Protocol Safety Report
Report Generated: {audit_date}

## 1. Eligibility Criteria
### Inclusion
{chr(10).join(['- ' + i for i in extracted.get("inclusion", [])])}

### Exclusion
{chr(10).join(['- ' + e for e in extracted.get("exclusion", [])])}

## 2. Protocol Objectives & Endpoints
{chr(10).join(['- ' + ep for ep in extracted.get("endpoints", [])])}

## 3. Safety Stopping Boundaries
{chr(10).join(['- ' + s for s in extracted.get("safety_boundaries", [])])}
"""
                st.download_button(
                    label="⬇️ Download Audit Report (MD)",
                    data=md_output,
                    file_name="PharmacoAudit_Safety_Report.md",
                    mime="text/markdown",
                    key="pharmacoaudit_download_btn"
                )
        else:
            st.info("👋 Enter a clinical trial protocol text on the left to start parsing safety parameters.")
