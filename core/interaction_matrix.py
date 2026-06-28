"""
Drug-Drug Co-Prescription Interaction Matrix Builder

Analyzes polypharmacy co-prescription patterns from FAERS reports to identify
which drug combinations appear together most frequently and their associated
aggregate adverse event risk profiles.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
from typing import Dict, List, Tuple
from core.network_builder import map_adr_to_soc


def build_coprescription_matrix(df: pd.DataFrame, top_n_drugs: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds a symmetric co-prescription frequency matrix from the processed reports.
    
    Returns:
        co_matrix: DataFrame of shape (top_n_drugs, top_n_drugs) with co-prescription counts.
        pair_details: DataFrame with columns [drug_a, drug_b, co_count, avg_reactions, top_adrs].
    """
    print(f"[InteractionMatrix] Building co-prescription matrix for top {top_n_drugs} drugs...")
    
    if df.empty or "drugs" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
        
    # Step 1: Find top N most frequent drugs
    all_drugs = []
    for drug_str in df["drugs"].dropna():
        all_drugs.extend(str(drug_str).split(";"))
    
    drug_freq = pd.Series(all_drugs).value_counts()
    top_drugs = list(drug_freq.head(top_n_drugs).index)
    
    # Step 2: Count co-prescriptions for every pair
    pair_counts = {}
    pair_adrs = {}  # Track ADRs associated with each pair
    
    for _, row in df.iterrows():
        drugs_in_report = set(str(row["drugs"]).split(";"))
        reactions_in_report = set(str(row["reactions"]).split(";"))
        
        # Only consider drugs that are in our top_n list
        relevant_drugs = drugs_in_report.intersection(top_drugs)
        
        # Count all pairs of relevant drugs
        for drug_a, drug_b in combinations(sorted(relevant_drugs), 2):
            pair_key = (drug_a, drug_b)
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
            
            if pair_key not in pair_adrs:
                pair_adrs[pair_key] = []
            pair_adrs[pair_key].extend(reactions_in_report)
    
    # Step 3: Build symmetric matrix
    co_matrix = pd.DataFrame(0, index=top_drugs, columns=top_drugs, dtype=int)
    
    # Fill diagonal with individual drug frequencies (capped to top_n)
    for drug in top_drugs:
        co_matrix.loc[drug, drug] = int(drug_freq.get(drug, 0))
    
    # Fill off-diagonal with pair counts
    for (drug_a, drug_b), count in pair_counts.items():
        co_matrix.loc[drug_a, drug_b] = count
        co_matrix.loc[drug_b, drug_a] = count
    
    # Step 4: Build pair details DataFrame
    details_records = []
    for (drug_a, drug_b), count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
        adrs_list = pair_adrs[(drug_a, drug_b)]
        adr_freq = pd.Series(adrs_list).value_counts()
        top_adrs = list(adr_freq.head(5).index)
        avg_reactions = len(adrs_list) / max(count, 1)
        
        # Map top ADRs to SOCs
        soc_set = set(map_adr_to_soc(a) for a in top_adrs)
        
        details_records.append({
            "drug_a": drug_a,
            "drug_b": drug_b,
            "co_count": count,
            "avg_reactions_per_report": round(avg_reactions, 1),
            "top_adrs": ", ".join(top_adrs[:3]),
            "affected_systems": ", ".join(sorted(soc_set))
        })
    
    pair_details = pd.DataFrame(details_records)
    
    print(f"[InteractionMatrix] Matrix built. {len(pair_counts)} unique drug pairs found.")
    return co_matrix, pair_details


def generate_heatmap(co_matrix: pd.DataFrame) -> go.Figure:
    """
    Generates an interactive Plotly heatmap of co-prescription frequencies.
    Uses a logarithmic color scale with a clinical red-blue diverging palette.
    """
    # Apply log1p for better visual distribution (many small values, few large)
    log_values = np.log1p(co_matrix.values).tolist()
    
    # Build hover text matrix
    hover_text = []
    for i, row_drug in enumerate(co_matrix.index):
        row_texts = []
        for j, col_drug in enumerate(co_matrix.columns):
            count = co_matrix.iloc[i, j]
            if i == j:
                row_texts.append(f"<b>{row_drug}</b><br>Total Reports: {count}")
            elif count > 0:
                row_texts.append(
                    f"<b>{row_drug}</b> + <b>{col_drug}</b><br>"
                    f"Co-prescribed in: <b>{count}</b> reports"
                )
            else:
                row_texts.append(f"{row_drug} + {col_drug}<br>No co-prescriptions found")
        hover_text.append(row_texts)
    
    fig = go.Figure(data=go.Heatmap(
        z=log_values,
        x=co_matrix.columns.tolist(),
        y=co_matrix.index.tolist(),
        text=hover_text,
        hoverinfo="text",
        colorscale=[
            [0.0, "#0D1B2A"],       # Deep navy for zero
            [0.15, "#1B2838"],      # Dark slate
            [0.3, "#1E3A5F"],       # Navy
            [0.45, "#2563EB"],      # Blue
            [0.6, "#60A5FA"],       # Light blue
            [0.75, "#FBBF24"],      # Amber
            [0.9, "#F97316"],       # Orange
            [1.0, "#EF4444"],       # Red (hotspot)
        ],
        colorbar=dict(
            title=dict(text="Frequency<br>(log scale)", side="right", font=dict(color="#94A3B8")),
            thickness=15,
            len=0.7,
            tickfont=dict(size=10, color="#94A3B8"),
        ),
        xgap=2,
        ygap=2,
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="Drug Co-Prescription Interaction Heatmap",
            x=0.5,
            xanchor="center",
            font=dict(size=16, color="#F1F5F9")
        ),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=9, color="#94A3B8"),
            side="bottom"
        ),
        yaxis=dict(
            tickfont=dict(size=9, color="#94A3B8"),
            autorange="reversed"
        ),
        hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_color="#E2E8F0"),
        height=600,
        margin=dict(l=120, r=40, t=80, b=120),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig
