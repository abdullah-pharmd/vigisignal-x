import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Comprehensive, normalized dictionary mapping reaction terms to System Organ Classes (SOCs)
SOC_MAP = {
    # Cardiotoxicity/Vascular
    "MYOCARDIAL INFARCTION": "Cardiotoxicity/Vascular",
    "CARDIAC INFARCTION": "Cardiotoxicity/Vascular",
    "HYPERTENSION": "Cardiotoxicity/Vascular",
    "CARDIAC FAILURE": "Cardiotoxicity/Vascular",
    "CONGESTIVE HEART FAILURE": "Cardiotoxicity/Vascular",
    "CARDIAC FAILURE CONGESTIVE": "Cardiotoxicity/Vascular",
    "ARRHYTHMIA": "Cardiotoxicity/Vascular",
    "CARDIAC ARREST": "Cardiotoxicity/Vascular",
    "THROMBOSIS": "Cardiotoxicity/Vascular",
    "ANGINA PECTORIS": "Cardiotoxicity/Vascular",
    "ATRIAL FIBRILLATION": "Cardiotoxicity/Vascular",
    "TACHYCARDIA": "Cardiotoxicity/Vascular",
    "BRADYCARDIA": "Cardiotoxicity/Vascular",
    "PALPITATIONS": "Cardiotoxicity/Vascular",
    "VASCULAR INFRACTION": "Cardiotoxicity/Vascular",
    "CARDIO-RESPIRATORY ARREST": "Cardiotoxicity/Vascular",
    "MYOCARDIAL MYOCARDITIS": "Cardiotoxicity/Vascular",
    "PERICARDITIS": "Cardiotoxicity/Vascular",
    "CARDIOGENIC SHOCK": "Cardiotoxicity/Vascular",
    
    # Gastrointestinal
    "VOMITING": "Gastrointestinal",
    "DIARRHOEA": "Gastrointestinal",
    "DIARRHEA": "Gastrointestinal",
    "GASTROINTESTINAL HAEMORRHAGE": "Gastrointestinal",
    "GASTROINTESTINAL HEMORRHAGE": "Gastrointestinal",
    "NAUSEA": "Gastrointestinal",
    "ABDOMINAL PAIN": "Gastrointestinal",
    "CONSTIPATION": "Gastrointestinal",
    "DYSPEPSIA": "Gastrointestinal",
    "GASTROINTESTINAL PAIN": "Gastrointestinal",
    "GASTRITIS": "Gastrointestinal",
    "COLITIS": "Gastrointestinal",
    "MELAENA": "Gastrointestinal",
    "HEMATEMESIS": "Gastrointestinal",
    "STOMACH ULCER": "Gastrointestinal",
    
    # Renal/Urinary
    "RENAL FAILURE": "Renal/Urinary",
    "ACUTE KIDNEY INJURY": "Renal/Urinary",
    "RENAL IMPAIRMENT": "Renal/Urinary",
    "RENAL INSUFFICIENCY": "Renal/Urinary",
    "DYSURIA": "Renal/Urinary",
    "HEMATURIA": "Renal/Urinary",
    "HAEMATURIA": "Renal/Urinary",
    "OLIGURIA": "Renal/Urinary",
    "NEPHROPATHY": "Renal/Urinary",
    "ANURIA": "Renal/Urinary",
    "NEPHROTIC SYNDROME": "Renal/Urinary",
    
    # Respiratory
    "DYSPNOEA": "Respiratory",
    "DYSPNEA": "Respiratory",
    "COUGH": "Respiratory",
    "PULMONARY OEDEMA": "Respiratory",
    "PULMONARY EDEMA": "Respiratory",
    "RESPIRATORY FAILURE": "Respiratory",
    "ASTHMA": "Respiratory",
    "PNEUMONIA": "Respiratory",
    "PULMONARY EMBOLISM": "Respiratory",
    "RESPIRATORY DISTRESS": "Respiratory",
    "RESPIRATORY ARREST": "Respiratory",
    "BRONCHOSPASM": "Respiratory",
    "COPD": "Respiratory",
    "CHRONIC OBSTRUCTIVE PULMONARY DISEASE": "Respiratory",
    
    # Nervous System
    "DIZZINESS": "Nervous System",
    "HEADACHE": "Nervous System",
    "TREMOR": "Nervous System",
    "SOMNOLENCE": "Nervous System",
    "CONVULSION": "Nervous System",
    "NEUROPATHY": "Nervous System",
    "SYNCOPE": "Nervous System",
    "PARAESTHESIA": "Nervous System",
    "PARESTHESIA": "Nervous System",
    "SEIZURE": "Nervous System",
    "CONFUSIONAL STATE": "Nervous System",
    "COMA": "Nervous System",
    "STUPOR": "Nervous System",
    "PERIPHERAL NEUROPATHY": "Nervous System",
}

def map_adr_to_soc(adr_term: str) -> str:
    """
    Maps an ADR term to a System Organ Class (SOC) using exact and keyword-based matching.
    """
    normalized = adr_term.strip().upper()
    if normalized in SOC_MAP:
        return SOC_MAP[normalized]
    
    # Keyword-based fallbacks
    if any(k in normalized for k in ["HEART", "CARDIAC", "MYOCARDIAL", "HYPERTEN", "INFARCT", "ARRHYTHM", "VASCULAR", "ATRIAL", "VENTRICULAR"]):
        return "Cardiotoxicity/Vascular"
    if any(k in normalized for k in ["GASTRO", "INTESTIN", "DIARRH", "VOMIT", "STOMACH", "NAUSEA", "ABDOMINAL", "ULCER", "GASTRIC"]):
        return "Gastrointestinal"
    if any(k in normalized for k in ["RENAL", "KIDNEY", "NEPHRO", "URINARY", "DYSURIA", "OLIGURIA"]):
        return "Renal/Urinary"
    if any(k in normalized for k in ["RESPIR", "PULMON", "LUNG", "DYSPN", "BRONCH", "COUGH", "ASTHMA", "PNEUMON"]):
        return "Respiratory"
    if any(k in normalized for k in ["NERVE", "BRAIN", "NEURO", "CONVULS", "HEADACHE", "DIZZY", "SYNCOPE", "SOMNOLENCE", "CONFUSION", "COMA", "SEIZURE"]):
        return "Nervous System"
        
    return "Other/Unclassified"

def build_safety_network(signals_df: pd.DataFrame) -> nx.Graph:
    """
    Constructs a NetworkX graph connecting target drugs to System Organ Classes (SOCs) 
    based on statistically significant disproportionality signals.
    """
    G = nx.Graph()
    
    if signals_df.empty:
        return G
        
    # Keep only significant signals
    sig_signals = signals_df[signals_df["is_signal"] == True].copy()
    if sig_signals.empty:
        return G
        
    # Map each ADR to a SOC
    sig_signals["soc"] = sig_signals["adr"].apply(map_adr_to_soc)
    
    # Aggregate signals by Drug and SOC
    # Group by drug and SOC to aggregate multiple ADRs into a single weighted edge
    grouped = sig_signals.groupby(["drug", "soc"])
    
    for (drug, soc), group in grouped:
        # Node attribute: Type
        G.add_node(drug, type="drug", label=drug)
        G.add_node(soc, type="soc", label=soc)
        
        # Calculate edge properties
        signals_count = len(group)
        max_ror = group["ror"].max()
        total_cases = group["a"].sum()
        
        # Determine dominant triage tier
        if "triage_tier" in group.columns:
            triage_values = group["triage_tier"].tolist()
            if "High Priority" in triage_values:
                dominant_triage = "High Priority"
            elif "Moderate Priority" in triage_values:
                dominant_triage = "Moderate Priority"
            elif "Weak Priority" in triage_values:
                dominant_triage = "Weak Priority"
            else:
                dominant_triage = "Not Significant"
        else:
            if max_ror >= 3.0:
                dominant_triage = "High Priority"
            elif max_ror >= 2.0:
                dominant_triage = "Moderate Priority"
            elif max_ror > 1.0:
                dominant_triage = "Weak Priority"
            else:
                dominant_triage = "Not Significant"
        
        # Detail formatting for tooltip card
        adr_details = []
        for _, row in group.sort_values(by="ror", ascending=False).iterrows():
            tier_str = f", Tier={row['triage_tier']}" if "triage_tier" in row else ""
            adr_details.append(f"{row['adr']} (Cases={row['a']}, ROR={row['ror']:.2f}, 95%CI=[{row['ci_lower']:.1f}-{row['ci_upper']:.1f}]{tier_str})")
            
        edge_label = f"Signals: {signals_count}<br>Max ROR: {max_ror:.2f}<br>Dominant Tier: {dominant_triage}<br>Total Cases: {total_cases}<br><br><b>Events:</b><br>" + "<br>".join(adr_details)
        
        G.add_edge(
            drug, 
            soc, 
            weight=signals_count, 
            max_ror=max_ror, 
            total_cases=total_cases,
            triage_tier=dominant_triage,
            label=edge_label
        )
        
    return G

def generate_network_plot(G: nx.Graph, layout_type: str = "bipartite") -> go.Figure:
    """
    Generates an interactive 2D Plotly Scatter/Line plot representing the NetworkX graph.
    Supports circular and bipartite (2-column) layouts.
    """
    fig = go.Figure()
    
    if len(G.nodes) == 0:
        fig.add_annotation(
            text="No significant safety signals found to map in the network.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        # Update layout with basic styles
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig
        
    # Calculate positions
    drugs = [n for n, attr in G.nodes(data=True) if attr.get("type") == "drug"]
    socs = [n for n, attr in G.nodes(data=True) if attr.get("type") == "soc"]
    
    pos = {}
    if layout_type == "bipartite":
        # Bipartite (Drugs on left column x=0, SOCs on right column x=1)
        # Space out Y coordinates evenly
        for i, drug in enumerate(sorted(drugs)):
            pos[drug] = (0.0, i / max(1, len(drugs) - 1))
        for i, soc in enumerate(sorted(socs)):
            pos[soc] = (1.0, i / max(1, len(socs) - 1))
    elif layout_type == "circular":
        # Circular Layout
        pos = nx.circular_layout(G)
    else:
        # Default to spring layout
        pos = nx.spring_layout(G, seed=42)
        
    # Extract edge coordinates and metrics for trace drawing
    edge_x = []
    edge_y = []
    
    # We will draw edges as individual scatter lines to allow different colors/widths
    for edge in G.edges(data=True):
        u, v, attr = edge
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        # Color by Triage Tier
        triage = attr.get("triage_tier", "Not Significant")
        max_ror = attr.get("max_ror", 1.0)
        alpha = min(1.0, 0.4 + max_ror/15.0)
        
        if triage == "High Priority":
            color = f"rgba(239, 68, 68, {alpha})"      # Red
        elif triage == "Moderate Priority":
            color = f"rgba(245, 158, 11, {alpha})"      # Orange
        elif triage == "Weak Priority":
            color = f"rgba(252, 211, 77, {alpha})"      # Yellow
        else:
            color = f"rgba(156, 163, 175, {alpha})"     # Gray
            
        width = min(8, 2 + attr.get("weight", 1) * 1.5)
        
        # Edge lines
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(
                    width=width, 
                    color=color
                ),
                hoverinfo="text",
                text=attr.get("label", ""),
                mode="lines",
                showlegend=False
            )
        )
        
    # Draw nodes
    node_drug_x = []
    node_drug_y = []
    node_drug_text = []
    node_drug_size = []
    
    node_soc_x = []
    node_soc_y = []
    node_soc_text = []
    node_soc_size = []
    
    for node, attr in G.nodes(data=True):
        x, y = pos[node]
        node_type = attr.get("type", "drug")
        degree = G.degree(node)
        
        # Size based on degree (how many connected nodes)
        size = 15 + degree * 5
        
        if node_type == "drug":
            node_drug_x.append(x)
            node_drug_y.append(y)
            node_drug_text.append(f"<b>Drug:</b> {node}<br>Connected SOCs: {degree}")
            node_drug_size.append(size)
        else:
            node_soc_x.append(x)
            node_soc_y.append(y)
            node_soc_text.append(f"<b>System Organ Class:</b> {node}<br>Connected Drugs: {degree}")
            node_soc_size.append(size)
            
    # Add Drug Nodes Trace (Blue/Cyan theme)
    if node_drug_x:
        fig.add_trace(
            go.Scatter(
                x=node_drug_x,
                y=node_drug_y,
                mode="markers+text",
                text=[n for n in sorted(drugs)] if layout_type == "bipartite" else [n for n in drugs],
                textposition="middle left",
                hoverinfo="text",
                hovertext=node_drug_text,
                marker=dict(
                    showscale=False,
                    color="#1F77B4",
                    size=node_drug_size,
                    line=dict(width=2, color="#0B0F19")
                ),
                name="Target Drugs"
            )
        )
        
    # Add SOC Nodes Trace (Amber/Orange theme)
    if node_soc_x:
        fig.add_trace(
            go.Scatter(
                x=node_soc_x,
                y=node_soc_y,
                mode="markers+text",
                text=[n for n in sorted(socs)] if layout_type == "bipartite" else [n for n in socs],
                textposition="middle right",
                hoverinfo="text",
                hovertext=node_soc_text,
                marker=dict(
                    showscale=False,
                    color="#FF7F0E",
                    size=node_soc_size,
                    line=dict(width=2, color="#0B0F19")
                ),
                name="Organ Systems (SOC)"
            )
        )
        
    # Customize Layout
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="Interactive Polypharmacy Safety Signal Network",
            x=0.5,
            y=0.98,
            xanchor="center",
            yanchor="top",
            font=dict(size=16, color="#F1F5F9")
        ),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(15, 23, 42, 0.6)", font=dict(color="#F1F5F9")),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_family="monospace", font_color="#E2E8F0"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=600
    )
    
    return fig
