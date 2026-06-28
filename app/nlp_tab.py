import streamlit as st
import os
import pandas as pd
from datetime import datetime
from core.nlp_engine import (
    analyze_and_deidentify,
    calculate_adr_similarity,
    analyze_and_deidentify_async,
    calculate_adr_similarity_async
)
import asyncio
from app.pdf_generator import generate_nlp_report_pdf

# Sample Patient Case Notes in English and French
SAMPLE_NARRATIVE_EN = """CLINICAL NOTE — CONFIDENTIAL
DATE: 2026-06-24
PATIENT: John Doe (DOB: 05/14/1959)
MRN: 948-204-118
CONTACT: 555-0199, john.doe@email.com

Patient is a 67-year-old male presenting with acute dyspnea and severe facial swelling. He was recently started on Lisinopril 20mg daily for hypertension. 
Other active home medications include Metformin 500mg BID and Aspirin 81mg daily. 
Upon physical examination, the patient exhibits marked lip and laryngeal swelling, consistent with severe drug-induced angioedema. 
Lisinopril was immediately discontinued. Epinephrine was administered subcutaneously, and his airway remains stable. 
Renal panels show serum creatinine is elevated at 1.8 mg/dL, raising concern for acute kidney injury.
"""

SAMPLE_NARRATIVE_FR = """NOTE CLINIQUE — CONFIDENTIEL
DATE: 24-06-2026
PATIENT: Jean Dupont (Né le: 14/05/1959)
DMP: 948-204-118
TEL: 01-45-78-90-12, jean.dupont@email.fr

Patient de 67 ans se présentant pour une dyspnée aiguë et un gonflement facial sévère. Il a récemment commencé du Lisinopril 20mg par jour pour l'hypertension.
Ses autres médicaments habituels comprennent de la Metformine 500mg deux fois par jour et de l'Aspirine 81mg par jour.
À l'examen physique, le patient présente un gonflement marqué des lèvres et du larynx, compatible avec un angioedème grave induit par le médicament.
Le Lisinopril a été immédiatement arrêté. De l'épinéphrine a été administrée par voie sous-cutanée, et les voies respiratoires restent stables.
Le bilan rénal montre une créatinine sérique élevée à 1,8 mg/dL, ce qui fait craindre une insuffisance rénale aiguë.
"""

def badge_html(text: str, label: str, bg_color: str) -> str:
    """Helper to generate HTML styling for entity badges."""
    return f"""
    <span style="
        background-color: {bg_color}; 
        color: white; 
        padding: 4px 10px; 
        margin: 4px; 
        border-radius: 12px; 
        font-size: 0.8rem; 
        font-weight: 600; 
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        {text} <span style="font-size: 0.65rem; opacity: 0.85; font-weight: 400;">({label})</span>
    </span>
    """

def render_nlp_tab() -> None:
    """
    Renders the Tab 10 UI for local clinical NLP, de-identification, and DoctoBERT/BioBERT similarity mapping.
    """
    st.subheader("🛡️ Clinical NLP & De-Identification")
    st.markdown("Local Privacy Redaction (HIPAA Compliant) & Deep Learning-based Cohort Adverse Event Similarity Mapping")

    # 1. Initialize session states
    if "nlp_narrative" not in st.session_state:
        st.session_state.nlp_narrative = ""
    if "nlp_results" not in st.session_state:
        st.session_state.nlp_results = None
    if "nlp_similarity" not in st.session_state:
        st.session_state.nlp_similarity = None

    # Side-by-side Layout: Left = input/controls, Right = output/analytics
    col_input, col_output = st.columns([2, 3])

    with col_input:
        st.markdown("### 📥 Narrative Input & Controls")
        
        # Load buttons
        load_en, load_fr = st.columns(2)
        with load_en:
            if st.button("🇬🇧 Load English Case Note", key="load_en_note"):
                st.session_state.nlp_narrative = SAMPLE_NARRATIVE_EN
        with load_fr:
            if st.button("🇫🇷 Charger Note Clinique (FR)", key="load_fr_note"):
                st.session_state.nlp_narrative = SAMPLE_NARRATIVE_FR

        # Case narrative text area
        narrative_input = st.text_area(
            "Paste Unstructured Case Narrative / Clinical Notes",
            value=st.session_state.nlp_narrative,
            placeholder="Paste clinical case notes here...",
            height=250,
            key="nlp_narrative_text_area"
        )
        st.session_state.nlp_narrative = narrative_input

        # Configuration options
        st.markdown("### ⚙️ NLP Configuration")
        
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            lang = st.selectbox(
                "Language",
                options=["en", "fr"],
                format_func=lambda x: "English (EN)" if x == "en" else "French (FR)",
                key="nlp_lang_select"
            )
            deid_method = st.selectbox(
                "De-identification Method",
                options=["mask", "replace", "hash"],
                format_func=lambda x: "Placeholder Masking ([NAME])" if x == "mask" else ("Surrogate Replacement" if x == "replace" else "Cryptographic Hashing"),
                key="nlp_deid_select"
            )
        with config_col2:
            model_id = st.selectbox(
                "Semantic Similarity Encoder",
                options=[
                    "doctolib-lab/doctomodernbert-fr-base",
                    "doctolib-lab/doctobert-fr-base",
                    "dmis-lab/biobert-v1.1"
                ],
                index=0 if lang == "fr" else 2,
                help="Select the transformer encoder to compute semantic ADR similarity embeddings.",
                key="nlp_model_select"
            )
            
        run_nlp = st.button("🚀 Analyze Clinical Text", use_container_width=True, type="primary")

    with col_output:
        st.markdown("### 🔍 Real-Time Clinical Analysis")
        
        if run_nlp and narrative_input.strip():
            async def run_nlp_async():
                task_deid = analyze_and_deidentify_async(narrative_input, method=deid_method, lang=lang)
                task_sim = calculate_adr_similarity_async(narrative_input, model_id=model_id, lang=lang)
                return await asyncio.gather(task_deid, task_sim)

            with st.spinner("Processing narrative (Asynchronous PII Redaction & Semantic Similarity)..."):
                try:
                    nlp_results, similarity_df = asyncio.run(run_nlp_async())
                    st.session_state.nlp_results = nlp_results
                    st.session_state.nlp_similarity = similarity_df
                except Exception as e:
                    st.error(f"Asynchronous clinical analysis failed: {e}")

        # If results exist, display them
        if st.session_state.nlp_results:
            results = st.session_state.nlp_results
            similarity = st.session_state.nlp_similarity
            
            # Sub-tabs for outputs
            o_tab1, o_tab2, o_tab3 = st.tabs([
                "🛡️ De-Identified Narrative", 
                "🧬 Extracted Entities", 
                "📊 Semantic ADR Similarity"
            ])
            
            with o_tab1:
                st.markdown("#### **Original vs. De-Identified Narrative**")
                
                # Side-by-side columns
                text_col1, text_col2 = st.columns(2)
                with text_col1:
                    st.caption("Original Clinical Note")
                    st.info(results["original_text"])
                with text_col2:
                    st.caption("Redacted / De-identified Note")
                    st.success(results["deidentified_text"])
                    
                st.caption(f"PII Redaction Summary: {len(results['pii_entities'])} privacy elements redacted using {deid_method.upper()} strategy.")
                
            with o_tab2:
                st.markdown("#### **Extracted Clinical Concepts**")
                if results["clinical_entities"]:
                    # Show badged tags
                    badge_html_str = ""
                    colors_map = {
                        "Drug": "#3B82F6",       # Blue
                        "Condition": "#EF4444",  # Red
                        "Anatomy": "#10B981"     # Green
                    }
                    for e in results["clinical_entities"]:
                        color = colors_map.get(e["label"], "#8B5CF6") # Default purple
                        badge_html_str += badge_html(e["text"], e["label"], color)
                        
                    st.markdown(badge_html_str, unsafe_allow_html=True)
                    st.markdown("---")
                    
                    # Also show details as a table
                    ent_df = pd.DataFrame(results["clinical_entities"])
                    # Format confidence as %
                    ent_df["confidence"] = ent_df["confidence"].apply(lambda x: f"{x:.2%}")
                    ent_df.columns = ["Clinical Term", "Category", "Confidence", "Start Position", "End Position"]
                    st.dataframe(ent_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No clinical concepts extracted.")
                    
            with o_tab3:
                st.markdown("#### **Deep Semantic Similarity Mapping**")
                if similarity is not None and not similarity.empty:
                    # Display similarity table
                    st.write(f"Embedding mapping using encoder model: `{model_id}`")
                    
                    # Highlight matches
                    def style_relevance(val):
                        if val >= 0.7:
                            return 'background-color: rgba(239, 68, 68, 0.2); color: #EF4444; font-weight: bold;'
                        elif val >= 0.4:
                            return 'background-color: rgba(245, 158, 11, 0.2); color: #F59E0B;'
                        elif val >= 0.2:
                            return 'background-color: rgba(16, 185, 129, 0.2); color: #10B981;'
                        return 'color: #94A3B8;'

                    styled_similarity = similarity.style.applymap(style_relevance, subset=["similarity"])
                    st.dataframe(styled_similarity, use_container_width=True, hide_index=True)
                    
                    # Show a bar chart
                    st.bar_chart(data=similarity, x="adr", y="similarity", use_container_width=True)
                else:
                    st.info("Calculate similarity results to view deep semantic mapping.")

            # Compile PDF Section
            st.markdown("---")
            st.markdown("### 📄 Export De-identified Case Report")
            if st.button("📁 Compile De-identified PDF Report", key="compile_nlp_pdf_btn"):
                os.makedirs("data/reports", exist_ok=True)
                report_path = f"data/reports/vigisignal_nlp_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                
                try:
                    generate_nlp_report_pdf(
                        output_path=report_path,
                        original_text=results["original_text"],
                        deidentified_text=results["deidentified_text"],
                        method=deid_method,
                        lang=lang,
                        pii_entities=results["pii_entities"],
                        clinical_entities=results["clinical_entities"],
                        similarity_df=similarity,
                        model_id=model_id
                    )
                    
                    with open(report_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.success("✅ Case Report PDF compiled successfully!")
                    st.download_button(
                        label="⬇️ Download De-identified Clinical PDF",
                        data=pdf_bytes,
                        file_name=f"VigiSignal_X_DeIdentified_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error compiling PDF: {e}")
        else:
            st.info("Paste clinical narrative and click 'Analyze Clinical Text' to view results.")
