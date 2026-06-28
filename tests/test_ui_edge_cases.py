import pytest
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from core.stats_engine import analyze_all_pairs
from core.network_builder import build_safety_network, generate_network_plot
from core.interaction_matrix import build_coprescription_matrix, generate_heatmap
from core.temporal_analysis import (
    compute_quarterly_signal_counts,
    compute_quarterly_soc_counts,
    generate_trend_line_chart,
    generate_stacked_area_chart
)

def test_empty_dataframe_stats():
    """Verify that stats engine handles empty dataframe cleanly."""
    df = pd.DataFrame(columns=["drugs", "reactions"])
    res = analyze_all_pairs(df)
    assert isinstance(res, pd.DataFrame)
    assert res.empty

def test_empty_dataframe_network():
    """Verify that network builder handles empty signals dataframe cleanly."""
    signals_df = pd.DataFrame()
    G = build_safety_network(signals_df)
    assert isinstance(G, nx.Graph)
    assert len(G.nodes) == 0
    
    fig = generate_network_plot(G)
    assert isinstance(fig, go.Figure)

def test_empty_dataframe_interaction():
    """Verify that interaction matrix handles empty dataframe cleanly."""
    df = pd.DataFrame()
    co_matrix, pair_details = build_coprescription_matrix(df)
    assert co_matrix.empty
    assert pair_details.empty
    
    # Test heatmap with empty matrix
    fig = generate_heatmap(co_matrix)
    assert isinstance(fig, go.Figure)

def test_empty_dataframe_temporal():
    """Verify that temporal analysis handles empty dataframe cleanly."""
    df = pd.DataFrame()
    
    # Should handle empty df and return empty df
    pivot_signals = compute_quarterly_signal_counts(df)
    assert pivot_signals.empty
    
    pivot_soc = compute_quarterly_soc_counts(df)
    assert pivot_soc.empty
    
    # Should handle empty pivot dfs and return go.Figure with empty annotation
    fig_trend = generate_trend_line_chart(pivot_signals)
    assert isinstance(fig_trend, go.Figure)
    
    fig_area = generate_stacked_area_chart(pivot_soc)
    assert isinstance(fig_area, go.Figure)

def test_minimal_dataframe_stats():
    """Verify that stats engine and charts handle minimal (1 row) dataframe cleanly."""
    # 1 report, 5 drugs, 2 reactions
    df = pd.DataFrame([{
        "drugs": "DRUG_A;DRUG_B;DRUG_C;DRUG_D;DRUG_E",
        "reactions": "REACTION_1;REACTION_2",
        "report_date": "20240115"
    }])
    
    # With min_cases=1, it should identify signals
    res = analyze_all_pairs(df, min_cases=1)
    assert not res.empty
    assert len(res) == 10  # 5 drugs * 2 reactions = 10 pairs
    
    # Force at least one signal to test network mapping and plotting code paths
    res.loc[0, "is_signal"] = True
    
    # Since all cells in contingency table (a, b, c, d) will have some zeros, Haldane must be applied
    assert res["haldane_applied"].all()
    
    # Test network with 1 signal
    G = build_safety_network(res)
    assert len(G.nodes) > 0
    fig_net = generate_network_plot(G)
    assert isinstance(fig_net, go.Figure)
    
    # Test interaction heatmap
    co_matrix, pair_details = build_coprescription_matrix(df, top_n_drugs=5)
    assert not co_matrix.empty
    assert co_matrix.shape == (5, 5)
    
    fig_heat = generate_heatmap(co_matrix)
    assert isinstance(fig_heat, go.Figure)
    
    # Test temporal analysis
    pivot_signals = compute_quarterly_signal_counts(df, top_n_adrs=2)
    assert not pivot_signals.empty
    fig_trend = generate_trend_line_chart(pivot_signals)
    assert isinstance(fig_trend, go.Figure)
    
    pivot_soc = compute_quarterly_soc_counts(df)
    assert not pivot_soc.empty
    fig_area = generate_stacked_area_chart(pivot_soc)
    assert isinstance(fig_area, go.Figure)
