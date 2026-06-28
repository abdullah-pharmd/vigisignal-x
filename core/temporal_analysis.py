"""
Temporal Safety Signal Trend Analysis

Parses FAERS report dates, aggregates adverse event frequencies by time period
(quarterly), and generates interactive trend line visualizations to track how
safety signals emerge and evolve over time.
"""

import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
from core.network_builder import map_adr_to_soc


def parse_report_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses and adds a 'report_quarter' column derived from FAERS report dates.
    
    If date data is missing or corrupt, we drop the record from the timeline chart.
    We track the percentage of records excluded due to incomplete real-world documentation.
    """
    df_out = df.copy()
    total_records = len(df)
    
    if "report_date" not in df_out.columns:
        df_out = df_out.iloc[0:0].copy()
        df_out["report_quarter"] = pd.Series(dtype=str)
        df_out.attrs["omission_pct"] = 100.0
        logger.warning("No 'report_date' column found. 100.0% of records excluded.")
        return df_out
        
    # Convert and parse dates, drop NaT
    df_out["parsed_date"] = pd.to_datetime(df_out["report_date"], errors="coerce")
    
    # Drop rows where parsed_date is NaT
    valid_mask = df_out["parsed_date"].notna()
    num_valid = valid_mask.sum()
    num_dropped = total_records - num_valid
    omission_pct = (num_dropped / total_records) * 100.0 if total_records > 0 else 0.0
    
    df_out = df_out[valid_mask].copy()
    df_out["report_quarter"] = df_out["parsed_date"].dt.to_period("Q").astype(str)
    df_out = df_out.drop(columns=["parsed_date"])
    
    df_out.attrs["omission_pct"] = omission_pct
    logger.info(f"Report date parsing complete. Excluded {num_dropped} of {total_records} records ({omission_pct:.2f}% omission).")
    
    return df_out


def compute_quarterly_signal_counts(
    df: pd.DataFrame, 
    selected_drugs: List[str] = None,
    top_n_adrs: int = 8
) -> pd.DataFrame:
    """
    Computes the number of reports per quarter for the top ADRs.
    
    Returns a pivot table: rows=quarter, columns=ADR, values=count.
    """
    df_dated = parse_report_dates(df)
    omission_pct = df_dated.attrs.get("omission_pct", 0.0)
    
    # Filter for selected drugs if specified
    if selected_drugs:
        mask = df_dated["drugs"].apply(
            lambda x: any(sd in str(x).split(";") for sd in selected_drugs)
        )
        df_dated = df_dated[mask]
    
    if df_dated.empty:
        empty_df = pd.DataFrame()
        empty_df.attrs["omission_pct"] = omission_pct
        return empty_df
    
    # Explode reactions to get one row per report-reaction
    records = []
    for _, row in df_dated.iterrows():
        quarter = row["report_quarter"]
        reactions = str(row["reactions"]).split(";")
        for rxn in reactions:
            if rxn:
                records.append({"quarter": quarter, "adr": rxn, "soc": map_adr_to_soc(rxn)})
    
    if not records:
        empty_df = pd.DataFrame()
        empty_df.attrs["omission_pct"] = omission_pct
        return empty_df
    
    exploded = pd.DataFrame(records)
    
    # Find top N most frequent ADRs overall
    top_adrs = list(exploded["adr"].value_counts().head(top_n_adrs).index)
    exploded_top = exploded[exploded["adr"].isin(top_adrs)]
    
    # Pivot: rows=quarter, columns=ADR, values=count
    pivot = exploded_top.pivot_table(
        index="quarter", 
        columns="adr", 
        values="soc", 
        aggfunc="count", 
        fill_value=0
    )
    
    # Sort by quarter
    pivot = pivot.sort_index()
    pivot.attrs["omission_pct"] = omission_pct
    
    return pivot


def compute_quarterly_soc_counts(
    df: pd.DataFrame,
    selected_drugs: List[str] = None
) -> pd.DataFrame:
    """
    Computes the number of reports per quarter for each System Organ Class.
    
    Returns a pivot table: rows=quarter, columns=SOC, values=count.
    """
    df_dated = parse_report_dates(df)
    omission_pct = df_dated.attrs.get("omission_pct", 0.0)
    
    if selected_drugs:
        mask = df_dated["drugs"].apply(
            lambda x: any(sd in str(x).split(";") for sd in selected_drugs)
        )
        df_dated = df_dated[mask]
    
    if df_dated.empty:
        empty_df = pd.DataFrame()
        empty_df.attrs["omission_pct"] = omission_pct
        return empty_df
    
    records = []
    for _, row in df_dated.iterrows():
        quarter = row["report_quarter"]
        reactions = str(row["reactions"]).split(";")
        for rxn in reactions:
            if rxn:
                records.append({"quarter": quarter, "soc": map_adr_to_soc(rxn)})
    
    if not records:
        empty_df = pd.DataFrame()
        empty_df.attrs["omission_pct"] = omission_pct
        return empty_df
    
    exploded = pd.DataFrame(records)
    
    pivot = exploded.pivot_table(
        index="quarter",
        columns="soc",
        values="quarter",  # dummy; we use aggfunc="count"
        aggfunc="count",
        fill_value=0
    )
    
    pivot = pivot.sort_index()
    pivot.attrs["omission_pct"] = omission_pct
    return pivot


def generate_trend_line_chart(pivot: pd.DataFrame, title: str = "ADR Signal Trends Over Time") -> go.Figure:
    """
    Generates a multi-line Plotly chart showing temporal evolution of ADR/SOC counts.
    """
    if pivot.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No temporal data available for the selected filters.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="gray")
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    
    # Color palette
    colors = [
        "#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6",
        "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
        "#14B8A6", "#E11D48"
    ]
    
    fig = go.Figure()
    
    for i, col in enumerate(pivot.columns):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=pivot.index.tolist(),
            y=pivot[col].tolist(),
            mode="lines+markers",
            name=col,
            line=dict(width=2.5, color=color),
            marker=dict(size=7, color=color, line=dict(width=1, color="white")),
            hovertemplate=(
                f"<b>{col}</b><br>"
                "Quarter: %{x}<br>"
                "Reports: %{y}<br>"
                "<extra></extra>"
            )
        ))
    
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=16, color="#F1F5F9")
        ),
        xaxis=dict(
            title="Quarter",
            tickangle=-30,
            gridcolor="rgba(255,255,255,0.08)",
            tickfont=dict(size=10, color="#94A3B8"),
            title_font=dict(color="#F1F5F9")
        ),
        yaxis=dict(
            title="Number of Reports",
            gridcolor="rgba(255,255,255,0.08)",
            tickfont=dict(size=10, color="#94A3B8"),
            title_font=dict(color="#F1F5F9")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#94A3B8"),
            bgcolor="rgba(15, 23, 42, 0.6)"
        ),
        height=500,
        margin=dict(l=60, r=40, t=80, b=120),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_color="#E2E8F0"),
        hovermode="x unified"
    )
    
    return fig


def generate_stacked_area_chart(pivot: pd.DataFrame, title: str = "System Organ Class Burden Over Time") -> go.Figure:
    """
    Generates a stacked area chart showing the distribution of organ system 
    toxicity burden over time — a clinical impact visualization.
    """
    if pivot.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No temporal data available.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="gray")
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    
    soc_colors = {
        "Cardiotoxicity/Vascular": "#EF4444",
        "Gastrointestinal": "#F59E0B",
        "Renal/Urinary": "#3B82F6",
        "Respiratory": "#10B981",
        "Nervous System": "#8B5CF6",
        "Other/Unclassified": "#9CA3AF"
    }
    
    fig = go.Figure()
    
    for col in pivot.columns:
        color = soc_colors.get(col, "#6B7280")
        fig.add_trace(go.Scatter(
            x=pivot.index.tolist(),
            y=pivot[col].tolist(),
            mode="lines",
            name=col,
            stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.6)" if color.startswith("#") else color,
            hovertemplate=(
                f"<b>{col}</b><br>"
                "Quarter: %{x}<br>"
                "Reports: %{y}<br>"
                "<extra></extra>"
            )
        ))
    
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=16, color="#F1F5F9")
        ),
        xaxis=dict(
            title="Quarter",
            tickangle=-30,
            gridcolor="rgba(255,255,255,0.08)",
            tickfont=dict(size=10, color="#94A3B8"),
            title_font=dict(color="#F1F5F9")
        ),
        yaxis=dict(
            title="Cumulative Report Count",
            gridcolor="rgba(255,255,255,0.08)",
            tickfont=dict(size=10, color="#94A3B8"),
            title_font=dict(color="#F1F5F9")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#94A3B8"),
            bgcolor="rgba(15, 23, 42, 0.6)"
        ),
        height=500,
        margin=dict(l=60, r=40, t=80, b=120),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_color="#E2E8F0"),
        hovermode="x unified"
    )
    
    return fig
