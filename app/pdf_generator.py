import os
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from core.network_builder import map_adr_to_soc

def generate_clinical_pdf(
    output_path: str,
    signals_df: pd.DataFrame,
    target_drugs: list,
    total_cohort_reports: int = 10000,
    min_cases: int = 3,
    ror_threshold: float = 1.0
) -> str:
    """
    Generates a professional clinical PDF pharmacovigilance report.
    Printable width = 504 points (612 page width - 108 margin points).
    """
    # 1. Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54, # 0.75 inch
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    # 2. Set up styles
    styles = getSampleStyleSheet()
    
    # Custom colors
    PRIMARY_COLOR = colors.HexColor("#1A365D") # Navy Blue
    SECONDARY_COLOR = colors.HexColor("#2B6CB0") # Slate Blue
    NEUTRAL_DARK = colors.HexColor("#2D3748") # Dark Gray
    NEUTRAL_LIGHT = colors.HexColor("#EDF2F7") # Light Gray
    BORDER_COLOR = colors.HexColor("#CBD5E0") # Medium Gray
    
    # Custom text styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY_COLOR,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=12,
        textColor=SECONDARY_COLOR,
        spaceAfter=12
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=PRIMARY_COLOR,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=NEUTRAL_DARK,
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=NEUTRAL_DARK,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=NEUTRAL_DARK
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )
    
    story = []
    
    # 3. Add Header Section
    story.append(Paragraph("VigiSignal-X Pharmacovigilance Report", title_style))
    story.append(Paragraph(f"Computational Polypharmacy Signal Detection Audit &bull; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1, 5))
    
    # 4. Add Metadata Block
    drugs_str = ", ".join(target_drugs) if target_drugs else "ALL DRUGS IN COHORT"
    meta_html = f"""
    <b>Target Patient Cohort:</b> Geriatric (Age &ge; 60), Concomitant Medication Count &gt; 4<br/>
    <b>Total Analyzed Reports:</b> {total_cohort_reports:,} reports<br/>
    <b>Target Medications:</b> {drugs_str}<br/>
    <b>Filter Thresholds:</b> Minimum safety cases (a) &ge; {min_cases}, ROR threshold &gt; {ror_threshold:.1f}, 95% CI Lower Limit &gt; 1.0
    """
    
    meta_table_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), NEUTRAL_LIGHT),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LINEBELOW', (0,0), (-1,-1), 1.5, PRIMARY_COLOR),
        ('LINEABOVE', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LINELEFT', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LINERIGHT', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ])
    
    meta_p = Paragraph(meta_html, body_style)
    meta_table = Table([[meta_p]], colWidths=[504])
    meta_table.setStyle(meta_table_style)
    story.append(meta_table)
    story.append(Spacer(1, 10))
    
    # 5. Add Executive Summary
    story.append(Paragraph("Executive Summary", section_heading))
    sig_count = len(signals_df[signals_df["is_signal"] == True]) if not signals_df.empty else 0
    summary_text = (
        f"A disproportionality pharmacovigilance analysis was performed utilizing the openFDA Adverse Event Reporting System (FAERS). "
        f"Our analytical engine parsed geriatric polypharmacy profiles, calculating the Reporting Odds Ratio (ROR) and 95% Confidence Intervals. "
        f"The engine isolated <b>{sig_count} statistically significant safety signals</b> where the rate of the adverse drug reaction (ADR) "
        f"in patients exposed to the drug exceeded the background rate of other medications."
    )
    story.append(Paragraph(summary_text, body_style))
    
    # 6. Add Methodology Section
    story.append(Paragraph("Methodology & Analysis Framework", section_heading))
    method_text = (
        "Safety signals are detected using a 2x2 contingency matrix comparing target drug reports to all other reports in the cohort. "
        "The disproportionality is measured via the Reporting Odds Ratio (ROR) and 95% Confidence Interval. "
        "Haldane correction (+0.5 to all cells) is applied when any cell is zero to prevent mathematical exceptions. "
        "Signals are classified into three triage tiers to guide clinical safety reviews: "
        "<b>High Priority:</b> ROR &ge; 3.0, CI lower &gt; 1.5, and cases (a) &ge; 5; "
        "<b>Moderate Priority:</b> ROR &ge; 2.0, CI lower &gt; 1.0, and cases (a) &ge; 3; "
        "<b>Weak Priority:</b> ROR &gt; 1.0, CI lower &gt; 1.0, and cases (a) &ge; 3."
    )
    story.append(Paragraph(method_text, body_style))
    story.append(Spacer(1, 10))
    
    # 7. Add Safety Signals Table
    story.append(Paragraph("Statistically Significant Safety Signals (Triage Priority)", section_heading))
    
    # Build Table
    # Column Widths sum to 504
    col_widths = [75, 95, 105, 45, 45, 69, 70]
    
    table_data = [[
        Paragraph("Drug", table_header_style),
        Paragraph("Adverse Reaction", table_header_style),
        Paragraph("Organ System (SOC)", table_header_style),
        Paragraph("Cases (a)", table_header_style),
        Paragraph("ROR", table_header_style),
        Paragraph("95% CI (L-U)", table_header_style),
        Paragraph("Triage Priority", table_header_style)
    ]]
    
    if not signals_df.empty:
        # Get only active signals
        active_signals = signals_df[signals_df["is_signal"] == True].copy()
        
        if active_signals.empty:
            table_data.append([Paragraph("No statistically significant safety signals detected.", table_cell_style), "", "", "", "", "", ""])
        else:
            # Take top 25 signals to prevent infinite table pages
            top_signals = active_signals.head(25)
            for _, row in top_signals.iterrows():
                drug_p = Paragraph(f"<b>{row['drug']}</b>", table_cell_style)
                adr_p = Paragraph(row["adr"].lower().title(), table_cell_style)
                soc_p = Paragraph(map_adr_to_soc(row["adr"]), table_cell_style)
                cases_p = Paragraph(str(int(row["a"])), table_cell_style)
                ror_p = Paragraph(f"{row['ror']:.2f}", table_cell_style)
                ci_p = Paragraph(f"{row['ci_lower']:.2f} - {row['ci_upper']:.2f}", table_cell_style)
                
                # Format Priority
                triage = row.get("triage_tier", "Not Significant")
                if triage == "High Priority":
                    triage_html = "<font color='#C53030'><b>🔴 High</b></font>"
                elif triage == "Moderate Priority":
                    triage_html = "<font color='#DD6B20'><b>🟠 Moderate</b></font>"
                elif triage == "Weak Priority":
                    triage_html = "<font color='#D69E2E'><b>🟡 Weak</b></font>"
                else:
                    triage_html = "<font color='#4A5568'>⚪ None</font>"
                triage_p = Paragraph(triage_html, table_cell_style)
                
                table_data.append([drug_p, adr_p, soc_p, cases_p, ror_p, ci_p, triage_p])
    else:
        table_data.append([Paragraph("No data loaded for table mapping.", table_cell_style), "", "", "", "", "", ""])
        
    sig_table = Table(table_data, colWidths=col_widths)
    
    # Table Styling
    sig_table_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ])
    
    # Alternating row colors
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            sig_table_style.add('BACKGROUND', (0, i), (-1, i), NEUTRAL_LIGHT)
            
    sig_table.setStyle(sig_table_style)
    story.append(sig_table)
    story.append(Spacer(1, 10))
    
    # 8. Add Clinical Triage Action Plan (KeepTogether to prevent page splitting)
    action_plan = []
    action_plan.append(Paragraph("Clinical Triage & Pharmacovigilance Recommendations", section_heading))
    action_plan.append(Paragraph("For geriatric patients presenting with polypharmacy profiles, the following clinical triage actions are recommended:", body_style))
    
    action_plan.append(Paragraph("&bull; <b>Polypharmacy Audit:</b> Conduct a thorough medication reconciliation for patients taking >4 drugs. Deprescribe unnecessary medications to lower the risk of drug-drug interactions.", bullet_style))
    action_plan.append(Paragraph("&bull; <b>Targeted Monitoring:</b> For drugs with significant Cardiotoxicity signals (ROR > 1.5), regularly monitor blood pressure, ECG, and cardiovascular symptoms.", bullet_style))
    action_plan.append(Paragraph("&bull; <b>Renal Cleanses:</b> Adjust drug dosing for agents with high Renal/Urinary signal spikes (e.g., NSAIDs, ACE inhibitors) based on estimated glomerular filtration rate (eGFR).", bullet_style))
    action_plan.append(Paragraph("&bull; <b>Safety Reporting:</b> Report newly observed adverse reactions to the FDA MedWatch program to support ongoing post-marketing surveillance.", bullet_style))
    
    action_plan.append(Spacer(1, 10))
    action_plan.append(Paragraph("<i>Disclaimer: This report is generated programmatically from openFDA FAERS data for research and decision-support purposes. It does not replace professional clinical judgment or direct patient assessment.</i>", subtitle_style))
    
    story.append(KeepTogether(action_plan))
    
    # 9. Build document
    doc.build(story)
    return output_path

def generate_one_page_summary(
    output_path: str,
    signals_df: pd.DataFrame,
    target_drugs: list,
    total_cohort_reports: int = 10000,
    min_cases: int = 3,
    ror_threshold: float = 1.0
) -> str:
    """
    Generates a dense, professional one-page executive pharmacovigilance summary.
    Printable width = 540 points (612 page width - 72 margin points).
    Printable height = 720 points (792 page height - 72 margin points).
    """
    # 1. Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36, # 0.5 inch
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    # 2. Set up styles
    styles = getSampleStyleSheet()
    
    PRIMARY_COLOR = colors.HexColor("#1A365D")
    SECONDARY_COLOR = colors.HexColor("#2B6CB0")
    NEUTRAL_DARK = colors.HexColor("#2D3748")
    NEUTRAL_LIGHT = colors.HexColor("#EDF2F7")
    BORDER_COLOR = colors.HexColor("#CBD5E0")
    
    title_style = ParagraphStyle(
        'OneTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=PRIMARY_COLOR,
        spaceAfter=2
    )
    
    subtitle_style = ParagraphStyle(
        'OneSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=10,
        textColor=SECONDARY_COLOR,
        spaceAfter=8
    )
    
    section_heading = ParagraphStyle(
        'OneSectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=PRIMARY_COLOR,
        spaceBefore=6,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'OneBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=NEUTRAL_DARK,
        spaceAfter=4
    )
    
    bullet_style = ParagraphStyle(
        'OneBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=NEUTRAL_DARK,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2
    )
    
    table_cell_style = ParagraphStyle(
        'OneTableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9,
        textColor=NEUTRAL_DARK
    )
    
    table_header_style = ParagraphStyle(
        'OneTableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9,
        textColor=colors.white
    )
    
    story = []
    
    # 3. Add Header
    story.append(Paragraph("VigiSignal-X Executive Summary Report", title_style))
    story.append(Paragraph(f"Headless Computational Pharmacovigilance Brief &bull; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    
    # 4. Metadata and Statistics Matrix (2 columns)
    drugs_str = ", ".join(target_drugs) if target_drugs else "ALL DRUGS IN COHORT"
    sig_count = len(signals_df[signals_df["is_signal"] == True]) if not signals_df.empty else 0
    
    col_left_html = f"""
    <b>Target Patient Cohort:</b> Geriatric (Age &ge; 60), Concomitant Medications &gt; 4<br/>
    <b>Target Medications:</b> {drugs_str}<br/>
    <b>Total Cohort Size:</b> {total_cohort_reports:,} reports
    """
    
    col_right_html = f"""
    <b>Min Safety Cases (a):</b> &ge; {min_cases}<br/>
    <b>ROR threshold:</b> &gt; {ror_threshold:.1f}<br/>
    <b>Identified Safety Signals:</b> <b>{sig_count}</b>
    """
    
    meta_table_data = [[
        Paragraph(col_left_html, body_style),
        Paragraph(col_right_html, body_style)
    ]]
    
    meta_table = Table(meta_table_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), NEUTRAL_LIGHT),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,0), (-1,-1), 1.0, PRIMARY_COLOR),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR)
    ]))
    
    story.append(meta_table)
    story.append(Spacer(1, 4))
    
    # 5. Methodology & Context
    story.append(Paragraph("Methodology Abstract", section_heading))
    method_text = (
        "VigiSignal-X processes FAERS big data to calculate Reporting Odds Ratio (ROR) and 95% Confidence Intervals. "
        "Signals are categorized as: <b>High Priority</b> (ROR &ge; 3.0, CI lower &gt; 1.5, cases &ge; 5); "
        "<b>Moderate Priority</b> (ROR &ge; 2.0, CI lower &gt; 1.0, cases &ge; 3); "
        "<b>Weak Priority</b> (ROR &gt; 1.0, CI lower &gt; 1.0, cases &ge; 3). "
        "This serves as a clinical triage tool for polypharmacy safety auditing."
    )
    story.append(Paragraph(method_text, body_style))
    
    # 6. Safety Signals Table (Top 8 to fit page)
    story.append(Paragraph("Top Detected Safety Signals (Triage Priority)", section_heading))
    
    # widths sum to 540
    col_widths = [80, 100, 110, 45, 45, 80, 80]
    table_data = [[
        Paragraph("Drug", table_header_style),
        Paragraph("Adverse Reaction", table_header_style),
        Paragraph("Organ System (SOC)", table_header_style),
        Paragraph("Cases (a)", table_header_style),
        Paragraph("ROR", table_header_style),
        Paragraph("95% CI (L-U)", table_header_style),
        Paragraph("Priority Tier", table_header_style)
    ]]
    
    if not signals_df.empty:
        active_signals = signals_df[signals_df["is_signal"] == True].copy()
        if active_signals.empty:
            table_data.append([Paragraph("No statistically significant safety signals detected.", table_cell_style), "", "", "", "", "", ""])
        else:
            top_signals = active_signals.head(8) # Clamp to 8 rows max
            for _, row in top_signals.iterrows():
                drug_p = Paragraph(f"<b>{row['drug']}</b>", table_cell_style)
                adr_p = Paragraph(row["adr"].lower().title(), table_cell_style)
                soc_p = Paragraph(map_adr_to_soc(row["adr"]), table_cell_style)
                cases_p = Paragraph(str(int(row["a"])), table_cell_style)
                ror_p = Paragraph(f"{row['ror']:.2f}", table_cell_style)
                ci_p = Paragraph(f"{row['ci_lower']:.2f} - {row['ci_upper']:.2f}", table_cell_style)
                
                triage = row.get("triage_tier", "Not Significant")
                if triage == "High Priority":
                    triage_html = "<font color='#C53030'><b>🔴 High</b></font>"
                elif triage == "Moderate Priority":
                    triage_html = "<font color='#DD6B20'><b>🟠 Moderate</b></font>"
                elif triage == "Weak Priority":
                    triage_html = "<font color='#D69E2E'><b>🟡 Weak</b></font>"
                else:
                    triage_html = "<font color='#4A5568'>⚪ None</font>"
                triage_p = Paragraph(triage_html, table_cell_style)
                
                table_data.append([drug_p, adr_p, soc_p, cases_p, ror_p, ci_p, triage_p])
    else:
        table_data.append([Paragraph("No data loaded for table mapping.", table_cell_style), "", "", "", "", "", ""])
        
    sig_table = Table(table_data, colWidths=col_widths)
    sig_table_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ])
    
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            sig_table_style.add('BACKGROUND', (0, i), (-1, i), NEUTRAL_LIGHT)
            
    sig_table.setStyle(sig_table_style)
    story.append(sig_table)
    story.append(Spacer(1, 4))
    
    # 7. Clinical Triage Recommendations
    story.append(Paragraph("Clinical Triage Action Checklist", section_heading))
    story.append(Paragraph("&bull; <b>Polypharmacy Audit:</b> Reconcile medications for geriatric patients on >4 drugs; deprescribe unnecessary medications.", bullet_style))
    story.append(Paragraph("&bull; <b>Signal Monitoring:</b> Monitor cardiovascular metrics for cardiotoxicity signals (ROR > 1.5); adjust dosing for renal signals.", bullet_style))
    story.append(Paragraph("&bull; <b>Post-Market Surveillance:</b> Report newly observed adverse reactions to the FDA MedWatch program.", bullet_style))
    
    # 8. Disclaimer
    story.append(Spacer(1, 4))
    disclaimer_text = (
        "<i>Disclaimer: This report is generated programmatically from openFDA FAERS data for research and decision-support purposes. "
        "It does not replace professional clinical judgment or direct patient assessment.</i>"
    )
    story.append(Paragraph(disclaimer_text, subtitle_style))
    
    # 9. Build document
    doc.build(story)
    return output_path

def generate_nlp_report_pdf(
    output_path: str,
    original_text: str,
    deidentified_text: str,
    method: str,
    lang: str,
    pii_entities: list,
    clinical_entities: list,
    similarity_df: pd.DataFrame,
    model_id: str
) -> str:
    """
    Generates a professional PDF report containing the de-identified clinical narrative,
    extracted clinical entities, and semantic ADR similarity mapping results.
    Printable width = 504 points (612 page width - 108 margin points).
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom colors
    PRIMARY_COLOR = colors.HexColor("#1A365D")
    SECONDARY_COLOR = colors.HexColor("#2B6CB0")
    NEUTRAL_DARK = colors.HexColor("#2D3748")
    NEUTRAL_LIGHT = colors.HexColor("#EDF2F7")
    BORDER_COLOR = colors.HexColor("#CBD5E0")
    
    title_style = ParagraphStyle(
        'NLPTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=PRIMARY_COLOR,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'NLPSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=11,
        textColor=SECONDARY_COLOR,
        spaceAfter=10
    )
    
    section_heading = ParagraphStyle(
        'NLPSectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=PRIMARY_COLOR,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'NLPBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=NEUTRAL_DARK,
        spaceAfter=4
    )
    
    table_cell_style = ParagraphStyle(
        'NLPTableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=NEUTRAL_DARK
    )
    
    table_header_style = ParagraphStyle(
        'NLPTableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )
    
    story = []
    
    # Title
    story.append(Paragraph("🛡️ Clinical NLP & De-Identification Audit Report", title_style))
    story.append(Paragraph(f"Patient Privacy Redaction & Adverse Event Similarity Mapping &bull; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1, 4))
    
    # Metadata Block
    meta_html = f"""
    <b>Language:</b> {lang.upper()} (English/French)<br/>
    <b>De-Identification Strategy:</b> {method.upper()} (mask / replace / hash)<br/>
    <b>Extracted PII Entities Count:</b> {len(pii_entities)} detected<br/>
    <b>Extracted Clinical Concepts:</b> {len(clinical_entities)} extracted (Drugs, Conditions, Anatomy)<br/>
    <b>Semantic Similarity Encoder:</b> {model_id}
    """
    
    meta_table_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), NEUTRAL_LIGHT),
        ('PADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LINEBELOW', (0,0), (-1,-1), 1.5, PRIMARY_COLOR),
        ('LINEABOVE', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LINELEFT', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LINERIGHT', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ])
    
    meta_table = Table([[Paragraph(meta_html, body_style)]], colWidths=[504])
    meta_table.setStyle(meta_table_style)
    story.append(meta_table)
    story.append(Spacer(1, 8))
    
    # Narrative Comparison
    story.append(Paragraph("Case Narrative Comparison", section_heading))
    comparison_data = [
        [Paragraph("<b>Original Clinical Narrative</b>", table_header_style), Paragraph("<b>De-Identified Narrative</b>", table_header_style)],
        [Paragraph(original_text.replace('\n', '<br/>'), body_style), Paragraph(deidentified_text.replace('\n', '<br/>'), body_style)]
    ]
    comparison_table = Table(comparison_data, colWidths=[247, 247])
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY_COLOR),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(comparison_table)
    story.append(Spacer(1, 8))
    
    # Extracted Clinical Entities
    story.append(Paragraph("Extracted Clinical Concepts", section_heading))
    entities_data = [[
        Paragraph("<b>Concept/Term</b>", table_header_style),
        Paragraph("<b>Category</b>", table_header_style),
        Paragraph("<b>Confidence</b>", table_header_style),
        Paragraph("<b>Positions</b>", table_header_style)
    ]]
    
    if clinical_entities:
        for e in clinical_entities:
            conf_percent = f"{e['confidence']:.2%}" if isinstance(e['confidence'], float) else str(e['confidence'])
            entities_data.append([
                Paragraph(e['text'], table_cell_style),
                Paragraph(e['label'], table_cell_style),
                Paragraph(conf_percent, table_cell_style),
                Paragraph(f"{e['start']}-{e['end']}", table_cell_style)
            ])
    else:
        entities_data.append([Paragraph("No clinical entities extracted.", table_cell_style), "", "", ""])
        
    entities_table = Table(entities_data, colWidths=[180, 110, 110, 104])
    entities_table_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 4),
    ])
    for i in range(1, len(entities_data)):
        if i % 2 == 0:
            entities_table_style.add('BACKGROUND', (0, i), (-1, i), NEUTRAL_LIGHT)
    entities_table.setStyle(entities_table_style)
    story.append(entities_table)
    story.append(Spacer(1, 8))
    
    # Semantic ADR Mapping
    story.append(Paragraph("Semantic Adverse Drug Reaction (ADR) Similarity Mapping", section_heading))
    adr_data = [[
        Paragraph("<b>Standard Cohort ADR Term</b>", table_header_style),
        Paragraph("<b>Cosine Similarity Score</b>", table_header_style),
        Paragraph("<b>Relevance Strength</b>", table_header_style)
    ]]
    
    if not similarity_df.empty:
        for _, row in similarity_df.iterrows():
            score = row['similarity']
            if score >= 0.7:
                strength = "<font color='#C53030'><b>🔴 Strong Match</b></font>"
            elif score >= 0.4:
                strength = "<font color='#DD6B20'><b>🟠 Moderate Match</b></font>"
            elif score >= 0.2:
                strength = "<font color='#D69E2E'><b>🟡 Weak Match</b></font>"
            else:
                strength = "<font color='#718096'>⚪ No Match</font>"
                
            adr_data.append([
                Paragraph(row['adr'], table_cell_style),
                Paragraph(f"{score:.4f}", table_cell_style),
                Paragraph(strength, table_cell_style)
            ])
    else:
        adr_data.append([Paragraph("No similarity scores calculated.", table_cell_style), "", ""])
        
    adr_table = Table(adr_data, colWidths=[204, 150, 150])
    adr_table_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 4),
    ])
    for i in range(1, len(adr_data)):
        if i % 2 == 0:
            adr_table_style.add('BACKGROUND', (0, i), (-1, i), NEUTRAL_LIGHT)
    adr_table.setStyle(adr_table_style)
    story.append(adr_table)
    
    # Disclaimer
    story.append(Spacer(1, 8))
    disclaimer_text = (
        "<i>Disclaimer: This report is generated programmatically using clinical NLP models (OpenMed) and HuggingFace sentence encoders "
        "for research and cohort de-identification validation. It does not constitute direct patient medical advice.</i>"
    )
    story.append(Paragraph(disclaimer_text, subtitle_style))
    
    doc.build(story)
    return output_path

