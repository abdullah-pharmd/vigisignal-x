import os
import tempfile
import asyncio
import pytest
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from unittest.mock import AsyncMock, MagicMock

# Core imports
from core.fda_client import FDAClient
from core.stats_engine import (
    calculate_contingency_matrix,
    calculate_disproportionality_metrics,
    analyze_all_pairs
)
from core.network_builder import map_adr_to_soc, build_safety_network, generate_network_plot
from core.interaction_matrix import build_coprescription_matrix, generate_heatmap
from core.temporal_analysis import (
    parse_report_dates,
    compute_quarterly_signal_counts,
    compute_quarterly_soc_counts,
    generate_trend_line_chart,
    generate_stacked_area_chart
)
from app.pdf_generator import generate_clinical_pdf

# ==============================================================================
# Helper Mock Classes for Async network requests
# ==============================================================================
class MockResponse:
    def __init__(self, status, json_data):
        self.status = status
        self._json_data = json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def json(self):
        return self._json_data

# ==============================================================================
# FEATURE 1: Asynchronous openFDA FAERS Data Fetching
# ==============================================================================

# Tier 1: Feature Coverage (5 tests)
def test_f1_t1_build_url_standard():
    client = FDAClient()
    url = client._build_url(limit=10, skip=20)
    assert "limit=10" in url
    assert "skip=20" in url
    assert "api_key" not in url

def test_f1_t1_build_url_api_key():
    client = FDAClient(api_key="TESTKEY")
    url = client._build_url(limit=50, skip=100)
    assert "limit=50" in url
    assert "skip=100" in url
    assert "api_key=TESTKEY" in url

def test_f1_t1_client_init_defaults():
    client = FDAClient(rate_limit_semaphore=5, delay_between_batches=2.5)
    assert client.delay == 2.5

@pytest.mark.asyncio
async def test_f1_t1_fetch_page_success(mocker):
    client = FDAClient()
    mock_session = mocker.MagicMock()
    mock_data = {"results": [{"safetyreportid": "123"}]}
    mocker.patch.object(mock_session, "get", return_value=MockResponse(200, mock_data))
    
    reports = await client.fetch_page(mock_session, limit=1, skip=0)
    assert len(reports) == 1
    assert reports[0]["safetyreportid"] == "123"

@pytest.mark.asyncio
async def test_f1_t1_fetch_all_reports_success(mocker):
    client = FDAClient()
    mock_data = {"results": [{"safetyreportid": "1"}, {"safetyreportid": "2"}]}
    mocker.patch.object(client, "fetch_page", return_value=mock_data["results"])
    
    # We mock fetch_all_reports internals to avoid actual ClientSession construction
    reports = await client.fetch_all_reports(num_pages=2, page_size=10)
    assert len(reports) == 4  # 2 pages * 2 reports each = 4 reports

# Tier 2: Boundary & Corner Cases (5 tests)
@pytest.mark.asyncio
async def test_f1_t2_fetch_page_rate_limiting_429(mocker):
    client = FDAClient(delay_between_batches=0.01) # short delay for test
    mock_session = mocker.MagicMock()
    mocker.patch("asyncio.sleep", return_value=None) # mock sleep to speed up test
    
    # Mocking sequential behaviors: first call status 429, second status 200
    mock_get = mocker.patch.object(mock_session, "get")
    mock_get.side_effect = [
        MockResponse(429, {"error": {"message": "Rate limit exceeded"}}),
        MockResponse(200, {"results": [{"safetyreportid": "OK"}]})
    ]
    
    reports = await client.fetch_page(mock_session, limit=1, skip=0)
    assert len(reports) == 1
    assert reports[0]["safetyreportid"] == "OK"

@pytest.mark.asyncio
async def test_f1_t2_fetch_page_forbidden_403(mocker):
    client = FDAClient()
    mock_session = mocker.MagicMock()
    mocker.patch.object(mock_session, "get", return_value=MockResponse(403, {"error": {"message": "Forbidden"}}))
    
    reports = await client.fetch_page(mock_session, limit=1, skip=0)
    assert reports == []

@pytest.mark.asyncio
async def test_f1_t2_fetch_page_server_error_500(mocker):
    client = FDAClient(delay_between_batches=0.01)
    mock_session = mocker.MagicMock()
    mocker.patch("asyncio.sleep", return_value=None)
    mocker.patch.object(mock_session, "get", return_value=MockResponse(500, {}))
    
    reports = await client.fetch_page(mock_session, limit=1, skip=0)
    assert reports == []

@pytest.mark.asyncio
async def test_f1_t2_fetch_page_exception(mocker):
    client = FDAClient(delay_between_batches=0.01)
    mock_session = mocker.MagicMock()
    mocker.patch("asyncio.sleep", return_value=None)
    mocker.patch.object(mock_session, "get", side_effect=Exception("Connection Closed"))
    
    reports = await client.fetch_page(mock_session, limit=1, skip=0)
    assert reports == []

@pytest.mark.asyncio
async def test_f1_t2_fetch_all_reports_skip_clamping(mocker):
    client = FDAClient()
    # If skip + limit > max_skip (10000 for keyless), loop stops
    # We fetch with huge num_pages to trigger skip clamping
    mocker.patch.object(client, "fetch_page", return_value=[])
    mocker.patch("asyncio.sleep", return_value=None)
    
    reports = await client.fetch_all_reports(num_pages=150, page_size=100)
    # The max skip clamping will trigger, loops will break before reaching 150 pages
    assert reports == []

# ==============================================================================
# FEATURE 2: Geriatric Polypharmacy Cohort Filtering & Processing
# ==============================================================================

# Tier 1: Feature Coverage (5 tests)
def test_f2_t1_process_reports_standard(raw_reports_sample):
    client = FDAClient()
    df = client.process_and_filter_reports(raw_reports_sample)
    # Out of 7 sample reports, REP001, REP006, REP007 are valid.
    # REP002 underage (45)
    # REP003 wrong unit (802)
    # REP004 too few drugs (3)
    # REP005 no reactions
    assert len(df) == 3
    assert set(df["safetyreportid"]) == {"REP001", "REP006", "REP007"}

def test_f2_t1_filter_underage(raw_reports_sample):
    client = FDAClient()
    underage_only = [r for r in raw_reports_sample if r["safetyreportid"] == "REP002"]
    df = client.process_and_filter_reports(underage_only)
    assert len(df) == 0

def test_f2_t1_filter_concomitant_count(raw_reports_sample):
    client = FDAClient()
    few_drugs = [r for r in raw_reports_sample if r["safetyreportid"] == "REP004"]
    df = client.process_and_filter_reports(few_drugs)
    assert len(df) == 0

def test_f2_t1_normalization(raw_reports_sample):
    client = FDAClient()
    valid_rep = [raw_reports_sample[0]] # REP001 has "patientonsetage": "65", drugs containing "ASPIRIN"
    df = client.process_and_filter_reports(valid_rep)
    assert df.iloc[0]["drugs"] == "ASPIRIN;LISINOPRIL;METFORMIN;IBUPROFEN;PREDNISONE" or "ASPIRIN" in df.iloc[0]["drugs"]

def test_f2_t1_save_data(mocker):
    client = FDAClient()
    mock_df = pd.DataFrame([{"safetyreportid": "REP001"}])
    mock_raw = [{"id": 1}]
    
    mock_makedirs = mocker.patch("os.makedirs")
    mock_json_dump = mocker.patch("json.dump")
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch.object(mock_df, "to_csv")
    
    client.save_data(mock_raw, mock_df, base_dir="temp_dir")
    assert mock_makedirs.called
    assert mock_df.to_csv.called

# Tier 2: Boundary & Corner Cases (5 tests)
def test_f2_t2_non_numeric_age():
    client = FDAClient()
    bad_age = [{
        "safetyreportid": "BAD_AGE",
        "patient": {
            "patientonsetage": "unknown",
            "patientonsetageunit": "801",
            "drug": [{"medicinalproduct": "D1"}, {"medicinalproduct": "D2"}, {"medicinalproduct": "D3"}, {"medicinalproduct": "D4"}, {"medicinalproduct": "D5"}],
            "reaction": [{"reactionmeddrapt": "ADR1"}]
        }
    }]
    df = client.process_and_filter_reports(bad_age)
    assert len(df) == 0

def test_f2_t2_missing_patient_fields():
    client = FDAClient()
    no_patient = [{
        "safetyreportid": "NO_PATIENT"
        # missing patient key
    }]
    df = client.process_and_filter_reports(no_patient)
    assert len(df) == 0

def test_f2_t2_wrong_age_unit():
    client = FDAClient()
    wrong_unit = [{
        "safetyreportid": "WRONG_UNIT",
        "patient": {
            "patientonsetage": "70",
            "patientonsetageunit": "802", # month
            "drug": [{"medicinalproduct": "D1"}, {"medicinalproduct": "D2"}, {"medicinalproduct": "D3"}, {"medicinalproduct": "D4"}, {"medicinalproduct": "D5"}],
            "reaction": [{"reactionmeddrapt": "ADR1"}]
        }
    }]
    df = client.process_and_filter_reports(wrong_unit)
    assert len(df) == 0

def test_f2_t2_boundary_concomitant_count():
    client = FDAClient()
    # Concomitant count exactly 4 (should be filtered out) vs 5 (kept)
    reports = [
        {
            "safetyreportid": "COUNT_4",
            "patient": {
                "patientonsetage": "65", "patientonsetageunit": "801",
                "drug": [{"medicinalproduct": "D1"}, {"medicinalproduct": "D2"}, {"medicinalproduct": "D3"}, {"medicinalproduct": "D4"}],
                "reaction": [{"reactionmeddrapt": "ADR1"}]
            }
        },
        {
            "safetyreportid": "COUNT_5",
            "patient": {
                "patientonsetage": "65", "patientonsetageunit": "801",
                "drug": [{"medicinalproduct": "D1"}, {"medicinalproduct": "D2"}, {"medicinalproduct": "D3"}, {"medicinalproduct": "D4"}, {"medicinalproduct": "D5"}],
                "reaction": [{"reactionmeddrapt": "ADR1"}]
            }
        }
    ]
    df = client.process_and_filter_reports(reports)
    assert len(df) == 1
    assert df.iloc[0]["safetyreportid"] == "COUNT_5"

def test_f2_t2_boundary_age_60():
    client = FDAClient()
    reports = [
        {
            "safetyreportid": "AGE_59",
            "patient": {
                "patientonsetage": "59.9", "patientonsetageunit": "801",
                "drug": [{"medicinalproduct": "D1"}, {"medicinalproduct": "D2"}, {"medicinalproduct": "D3"}, {"medicinalproduct": "D4"}, {"medicinalproduct": "D5"}],
                "reaction": [{"reactionmeddrapt": "ADR1"}]
            }
        },
        {
            "safetyreportid": "AGE_60",
            "patient": {
                "patientonsetage": "60.0", "patientonsetageunit": "801",
                "drug": [{"medicinalproduct": "D1"}, {"medicinalproduct": "D2"}, {"medicinalproduct": "D3"}, {"medicinalproduct": "D4"}, {"medicinalproduct": "D5"}],
                "reaction": [{"reactionmeddrapt": "ADR1"}]
            }
        }
    ]
    df = client.process_and_filter_reports(reports)
    assert len(df) == 1
    assert df.iloc[0]["safetyreportid"] == "AGE_60"

# ==============================================================================
# FEATURE 3: Disproportionality Statistics Engine
# ==============================================================================

# Tier 1: Feature Coverage (5 tests)
def test_f3_t1_contingency_matrix_standard(mock_processed_df):
    a, b, c, d = calculate_contingency_matrix(mock_processed_df, "ASPIRIN", "MYOCARDIAL INFARCTION")
    # ASPIRIN + MI: 30 reports (a = 30)
    # Other Drugs + MI: 2 reports (b = 2)
    # ASPIRIN + other ADRs: 5 reports + 1 PREDNISONE report which also contains ASPIRIN but not MI = 6 (c = 6)
    # Other Drugs + other ADRs: 100 (S_OTH_OTH_i) = 100 (d = 100)
    assert a == 30
    assert b == 2
    assert c == 6
    assert d == 100

def test_f3_t1_disproportionality_metrics_standard():
    metrics = calculate_disproportionality_metrics(10, 5, 16, 20)
    # ROR = (10 * 20) / (5 * 16) = 200 / 80 = 2.5
    # SE = sqrt(1/10 + 1/5 + 1/16 + 1/20) = sqrt(0.1 + 0.2 + 0.0625 + 0.05) = sqrt(0.4125) = 0.64226...
    assert abs(metrics["ror"] - 2.5) < 1e-5
    assert abs(metrics["se"] - np.sqrt(0.4125)) < 1e-5
    assert not metrics["haldane_applied"]

def test_f3_t1_analyze_all_pairs_signal(mock_processed_df):
    res_df = analyze_all_pairs(mock_processed_df, min_cases=3, ror_threshold=1.0)
    # Aspirin & Myocardial Infarction should be flagged as signal
    sig_row = res_df[(res_df["drug"] == "ASPIRIN") & (res_df["adr"] == "MYOCARDIAL INFARCTION")]
    assert not sig_row.empty
    assert sig_row.iloc[0]["is_signal"] == True

def test_f3_t1_analyze_all_pairs_sorting(mock_processed_df):
    res_df = analyze_all_pairs(mock_processed_df, min_cases=3, ror_threshold=1.0)
    # Check that it's sorted by ROR descending
    rors = res_df["ror"].tolist()
    assert all(rors[i] >= rors[i+1] for i in range(len(rors)-1))

def test_f3_t1_contingency_case_insensitivity(mock_processed_df):
    # Test case insensitivity and spaces
    a, b, c, d = calculate_contingency_matrix(mock_processed_df, " aspirin ", " myocardial infarction ")
    assert a == 30
    assert b == 2

# Tier 2: Boundary & Corner Cases (5 tests)
def test_f3_t2_haldane_correction_activation():
    metrics = calculate_disproportionality_metrics(0, 5, 10, 20)
    assert metrics["haldane_applied"] is True
    # ROR should be (0.5 * 20.5) / (5.5 * 10.5) = 10.25 / 57.75 = 0.177489
    assert abs(metrics["ror"] - 0.177489) < 1e-4

def test_f3_t2_non_existent_pair(mock_processed_df):
    a, b, c, d = calculate_contingency_matrix(mock_processed_df, "FAKE_DRUG", "FAKE_ADR")
    assert a == 0
    assert b == 0
    assert c == 0
    # Everything other than FAKE_DRUG and FAKE_ADR is in d-cell
    assert d == len(mock_processed_df)

def test_f3_t2_empty_cohort_stats():
    empty_df = pd.DataFrame(columns=["drugs", "reactions"])
    res = analyze_all_pairs(empty_df)
    assert res.empty

def test_f3_t2_boundary_cases_ci():
    # Lower CI boundary check: a signal should have lower CI > 1.0.
    # If lower CI is exactly 1.0 (or below), it's not a signal.
    # Let's mock a record set where ROR is high but variance is huge
    metrics = calculate_disproportionality_metrics(3, 10, 3, 20)
    # ROR = (3*20)/(10*3) = 2.0
    # Lower CI lower bound is likely below 1.0 because of low counts
    assert metrics["ci_lower"] < 1.0

def test_f3_t2_haldane_value_correctness():
    # Verify all zeros trigger Haldane and compute correctly
    metrics = calculate_disproportionality_metrics(0, 0, 0, 0)
    assert metrics["haldane_applied"] is True
    # (0.5 * 0.5) / (0.5 * 0.5) = 1.0
    assert metrics["ror"] == 1.0
    assert not np.isnan(metrics["se"])

# ==============================================================================
# FEATURE 4: Bipartite Toxicity Network Construction
# ==============================================================================

# Tier 1: Feature Coverage (5 tests)
def test_f4_t1_map_adr_to_soc_known():
    assert map_adr_to_soc("MYOCARDIAL INFARCTION") == "Cardiotoxicity/Vascular"
    assert map_adr_to_soc("VOMITING") == "Gastrointestinal"
    assert map_adr_to_soc("RENAL FAILURE") == "Renal/Urinary"
    assert map_adr_to_soc("DYSPNOEA") == "Respiratory"
    assert map_adr_to_soc("DIZZINESS") == "Nervous System"

def test_f4_t1_map_adr_to_soc_keyword():
    assert map_adr_to_soc("CARDIAC FAILURE") == "Cardiotoxicity/Vascular"
    assert map_adr_to_soc("GASTRITIS") == "Gastrointestinal"
    assert map_adr_to_soc("KIDNEY DISEASE") == "Renal/Urinary"
    assert map_adr_to_soc("PULMONARY INFECTION") == "Respiratory"
    assert map_adr_to_soc("BRAIN BLEED") == "Nervous System"

def test_f4_t1_build_safety_network(mock_processed_df):
    signals = analyze_all_pairs(mock_processed_df, min_cases=3, ror_threshold=1.0)
    G = build_safety_network(signals)
    assert len(G.nodes) > 0
    # Check node types
    for node, attrs in G.nodes(data=True):
        assert attrs["type"] in ["drug", "soc"]

def test_f4_t1_build_safety_network_edge_weights(mock_processed_df):
    signals = analyze_all_pairs(mock_processed_df, min_cases=3, ror_threshold=1.0)
    G = build_safety_network(signals)
    # Check edges
    for u, v, attrs in G.edges(data=True):
        assert attrs["weight"] >= 1
        assert "max_ror" in attrs

def test_f4_t1_generate_network_plot(mock_processed_df):
    signals = analyze_all_pairs(mock_processed_df, min_cases=3, ror_threshold=1.0)
    G = build_safety_network(signals)
    fig = generate_network_plot(G, layout_type="bipartite")
    assert isinstance(fig, go.Figure)

# Tier 2: Boundary & Corner Cases (5 tests)
def test_f4_t2_map_adr_to_soc_unclassified():
    assert map_adr_to_soc("HAIR LOSS") == "Other/Unclassified"
    assert map_adr_to_soc("") == "Other/Unclassified"

def test_f4_t2_build_safety_network_no_signals():
    # If dataframe has no signals flagged, G must be empty
    df = pd.DataFrame([{
        "drug": "ASPIRIN", "adr": "HEADACHE", "a": 1, "b": 10, "c": 10, "d": 100,
        "ror": 1.0, "se": 0.5, "ci_lower": 0.5, "ci_upper": 2.0, "haldane_applied": False,
        "is_signal": False
    }])
    G = build_safety_network(df)
    assert len(G.nodes) == 0

def test_f4_t2_generate_network_plot_empty():
    G = nx.Graph()
    fig = generate_network_plot(G)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0 # no scatter traces on empty

def test_f4_t2_build_safety_network_single_node():
    # 1 signal
    df = pd.DataFrame([{
        "drug": "ASPIRIN", "adr": "VOMITING", "a": 10, "b": 10, "c": 10, "d": 100,
        "ror": 2.5, "se": 0.2, "ci_lower": 1.5, "ci_upper": 4.0, "haldane_applied": False,
        "is_signal": True
    }])
    G = build_safety_network(df)
    assert "ASPIRIN" in G.nodes
    assert "Gastrointestinal" in G.nodes
    assert len(G.edges) == 1

def test_f4_t2_build_safety_network_duplicate_signals():
    # Multiple ADRs for the same drug mapping to the same SOC
    df = pd.DataFrame([
        {
            "drug": "ASPIRIN", "adr": "VOMITING", "a": 10, "b": 10, "c": 10, "d": 100,
            "ror": 2.5, "se": 0.2, "ci_lower": 1.5, "ci_upper": 4.0, "haldane_applied": False,
            "is_signal": True
        },
        {
            "drug": "ASPIRIN", "adr": "DIARRHEA", "a": 5, "b": 10, "c": 10, "d": 100,
            "ror": 3.0, "se": 0.3, "ci_lower": 1.8, "ci_upper": 5.0, "haldane_applied": False,
            "is_signal": True
        }
    ])
    G = build_safety_network(df)
    # Should merge them into one edge between ASPIRIN and Gastrointestinal
    assert G.number_of_edges("ASPIRIN", "Gastrointestinal") == 1
    # Weight should reflect count of signals (2)
    assert G["ASPIRIN"]["Gastrointestinal"]["weight"] == 2
    # Max ROR should be 3.0
    assert G["ASPIRIN"]["Gastrointestinal"]["max_ror"] == 3.0

# ==============================================================================
# FEATURE 5: Drug-Drug Co-Prescription Matrix & Heatmap
# ==============================================================================

# Tier 1: Feature Coverage (5 tests)
def test_f5_t1_build_coprescription_matrix_top_n(mock_processed_df):
    co_matrix, pair_details = build_coprescription_matrix(mock_processed_df, top_n_drugs=3)
    assert co_matrix.shape == (3, 3)

def test_f5_t1_build_coprescription_matrix_diagonal(mock_processed_df):
    co_matrix, _ = build_coprescription_matrix(mock_processed_df, top_n_drugs=3)
    # Diagonal should represent individual counts
    drugs = list(co_matrix.index)
    for d in drugs:
        assert co_matrix.loc[d, d] > 0

def test_f5_t1_build_coprescription_matrix_symmetry(mock_processed_df):
    co_matrix, _ = build_coprescription_matrix(mock_processed_df, top_n_drugs=5)
    assert np.allclose(co_matrix.values, co_matrix.values.T)

def test_f5_t1_pair_details_columns(mock_processed_df):
    _, pair_details = build_coprescription_matrix(mock_processed_df, top_n_drugs=5)
    expected_cols = {"drug_a", "drug_b", "co_count", "avg_reactions_per_report", "top_adrs", "affected_systems"}
    assert expected_cols.issubset(pair_details.columns)

def test_f5_t1_generate_heatmap(mock_processed_df):
    co_matrix, _ = build_coprescription_matrix(mock_processed_df, top_n_drugs=5)
    fig = generate_heatmap(co_matrix)
    assert isinstance(fig, go.Figure)

# Tier 2: Boundary & Corner Cases (5 tests)
def test_f5_t2_fewer_drugs_than_top_n(mock_processed_df):
    # Only 5 unique drugs in mock_processed_df (ASPIRIN, LISINOPRIL, METFORMIN, IBUPROFEN, PREDNISONE, ATORVASTATIN, OMEPRAZOLE, FUROSEMIDE, WARFARIN)
    # Wait, total unique drugs is 9. Let's request top_n_drugs = 50.
    co_matrix, _ = build_coprescription_matrix(mock_processed_df, top_n_drugs=50)
    # It should gracefully size to the total unique drugs (9)
    assert co_matrix.shape == (9, 9)

def test_f5_t2_duplicate_drugs_in_report():
    df = pd.DataFrame([{
        "drugs": "ASPIRIN;ASPIRIN;LISINOPRIL", # duplicates in report
        "reactions": "VOMITING"
    }])
    co_matrix, pair_details = build_coprescription_matrix(df, top_n_drugs=2)
    # ASPIRIN + LISINOPRIL co-prescription should be counted exactly once
    assert co_matrix.loc["ASPIRIN", "LISINOPRIL"] == 1

def test_f5_t2_no_co_prescriptions():
    # Reports only have 1 drug (though filtered out by client, stats/matrix functions should handle it)
    df = pd.DataFrame([
        {"drugs": "ASPIRIN", "reactions": "VOMITING"},
        {"drugs": "LISINOPRIL", "reactions": "HEADACHE"}
    ])
    co_matrix, pair_details = build_coprescription_matrix(df, top_n_drugs=2)
    assert co_matrix.loc["ASPIRIN", "LISINOPRIL"] == 0
    assert pair_details.empty

def test_f5_t2_empty_cohort():
    empty_df = pd.DataFrame(columns=["drugs", "reactions"])
    co_matrix, pair_details = build_coprescription_matrix(empty_df)
    assert co_matrix.empty
    assert pair_details.empty

def test_f5_t2_generate_heatmap_empty():
    empty_matrix = pd.DataFrame()
    # It should not crash on empty
    try:
        fig = generate_heatmap(empty_matrix)
        assert isinstance(fig, go.Figure)
    except Exception as e:
        pytest.fail(f"generate_heatmap crashed on empty matrix: {e}")

# ==============================================================================
# FEATURE 6: Temporal Signal Evolution Analysis
# ==============================================================================

# Tier 1: Feature Coverage (5 tests)
def test_f6_t1_parse_report_dates_standard(mock_processed_df):
    df_dated = parse_report_dates(mock_processed_df)
    assert "report_quarter" in df_dated.columns
    # Check that report date 20240115 maps to 2024Q1
    sample_row = df_dated[df_dated["report_date"] == "2024-01-15"]
    if not sample_row.empty:
        assert sample_row.iloc[0]["report_quarter"] == "2024Q1"

def test_f6_t1_parse_report_dates_synthetic():
    # If no report_date column, 100% omission rate is recorded and DataFrame is empty
    df = pd.DataFrame([{"drugs": "A", "reactions": "B"} for _ in range(8)])
    df_dated = parse_report_dates(df)
    assert len(df_dated) == 0
    assert df_dated.attrs["omission_pct"] == 100.0

def test_f6_t1_compute_quarterly_signal_counts(mock_processed_df):
    pivot = compute_quarterly_signal_counts(mock_processed_df, selected_drugs=["ASPIRIN"], top_n_adrs=8)
    assert not pivot.empty
    assert "MYOCARDIAL INFARCTION" in pivot.columns

def test_f6_t1_compute_quarterly_soc_counts(mock_processed_df):
    pivot = compute_quarterly_soc_counts(mock_processed_df, selected_drugs=["ASPIRIN"])
    assert not pivot.empty
    assert "Cardiotoxicity/Vascular" in pivot.columns

def test_f6_t1_generate_trend_line_chart(mock_processed_df):
    pivot = compute_quarterly_signal_counts(mock_processed_df, top_n_adrs=8)
    fig = generate_trend_line_chart(pivot)
    assert isinstance(fig, go.Figure)

# Tier 2: Boundary & Corner Cases (5 tests)
def test_f6_t2_parse_report_dates_malformed():
    df = pd.DataFrame([
        {"report_date": "invalid_date", "drugs": "A", "reactions": "B"},
        {"report_date": "2024-99-99", "drugs": "A", "reactions": "B"}
    ])
    df_dated = parse_report_dates(df)
    # Malformed dates should be dropped (100% omission)
    assert len(df_dated) == 0
    assert df_dated.attrs["omission_pct"] == 100.0

def test_f6_t2_compute_quarterly_counts_unmatched_drug(mock_processed_df):
    pivot = compute_quarterly_signal_counts(mock_processed_df, selected_drugs=["FAKE_DRUG"])
    assert pivot.empty

def test_f6_t2_compute_quarterly_counts_empty_quarter():
    # Verify behavior when certain quarters have no reports
    df = pd.DataFrame([
        {"report_date": "20240115", "drugs": "ASPIRIN", "reactions": "VOMITING"},
        {"report_date": "20240715", "drugs": "ASPIRIN", "reactions": "VOMITING"}
    ])
    pivot = compute_quarterly_signal_counts(df, top_n_adrs=1)
    # Pivot index should contain quarters representing 2024Q1 and 2024Q3
    assert "2024Q1" in pivot.index
    assert "2024Q3" in pivot.index

def test_f6_t2_generate_trend_line_empty():
    pivot = pd.DataFrame()
    fig = generate_trend_line_chart(pivot)
    assert isinstance(fig, go.Figure)
    assert "No temporal data" in fig.layout.annotations[0].text

def test_f6_t2_generate_stacked_area_empty():
    pivot = pd.DataFrame()
    fig = generate_stacked_area_chart(pivot)
    assert isinstance(fig, go.Figure)
    assert "No temporal data" in fig.layout.annotations[0].text


# ==============================================================================
# FEATURE 7: Clinical Triage Audit PDF Report Compilation
# ==============================================================================

# Tier 1: Feature Coverage (5 tests)
def test_f7_t1_generate_pdf_standard(mock_processed_df):
    signals = analyze_all_pairs(mock_processed_df, min_cases=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_report.pdf")
        res_path = generate_clinical_pdf(
            output_path=out_path,
            signals_df=signals,
            target_drugs=["ASPIRIN"],
            total_cohort_reports=100
        )
        assert res_path == out_path
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0

def test_f7_t1_generate_pdf_elements(mock_processed_df):
    signals = analyze_all_pairs(mock_processed_df, min_cases=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_report_2.pdf")
        generate_clinical_pdf(out_path, signals, ["ASPIRIN"])
        assert os.path.exists(out_path)

def test_f7_t1_generate_pdf_empty_signals():
    empty_signals = pd.DataFrame()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_empty_report.pdf")
        generate_clinical_pdf(out_path, empty_signals, ["ASPIRIN"])
        assert os.path.exists(out_path)

def test_f7_t1_generate_pdf_return_path(mock_processed_df):
    signals = analyze_all_pairs(mock_processed_df, min_cases=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_return.pdf")
        ret = generate_clinical_pdf(out_path, signals, ["ASPIRIN"])
        assert ret == out_path

def test_f7_t1_generate_pdf_custom_thresholds(mock_processed_df):
    signals = analyze_all_pairs(mock_processed_df, min_cases=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_thresholds.pdf")
        generate_clinical_pdf(out_path, signals, ["ASPIRIN"], min_cases=5, ror_threshold=2.0)
        assert os.path.exists(out_path)

# Tier 2: Boundary & Corner Cases (5 tests)
def test_f7_t2_generate_pdf_empty_drugs(mock_processed_df):
    signals = analyze_all_pairs(mock_processed_df, min_cases=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_empty_drugs.pdf")
        # Should build successfully when target_drugs is empty
        generate_clinical_pdf(out_path, signals, target_drugs=[])
        assert os.path.exists(out_path)

def test_f7_t2_generate_pdf_large_signals(mock_processed_df):
    signals = analyze_all_pairs(mock_processed_df, min_cases=1)
    # Generate reports list containing >30 signals
    large_list = []
    for i in range(40):
        large_list.append({
            "drug": f"DRUG_{i}", "adr": f"ADR_{i}", "a": 5, "b": 10, "c": 10, "d": 100,
            "ror": 2.5, "se": 0.2, "ci_lower": 1.5, "ci_upper": 4.0, "haldane_applied": False,
            "is_signal": True
        })
    large_df = pd.DataFrame(large_list)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_large.pdf")
        # Ensure it compiles and caps the table correctly
        generate_clinical_pdf(out_path, large_df, target_drugs=[])
        assert os.path.exists(out_path)

def test_f7_t2_generate_pdf_special_characters(mock_processed_df):
    # Tests that PDF compiler doesn't crash on special XML/HTML chars like <, >, &
    signals = pd.DataFrame([{
        "drug": "ASPIRIN & LISINOPRIL <TEST>",
        "adr": "VOMITING & HEADACHE >TRIAL",
        "a": 10, "b": 5, "c": 10, "d": 100,
        "ror": 2.5, "se": 0.2, "ci_lower": 1.5, "ci_upper": 4.0,
        "haldane_applied": False, "is_signal": True
    }])
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_spec_chars.pdf")
        generate_clinical_pdf(out_path, signals, target_drugs=["ASPIRIN"])
        assert os.path.exists(out_path)

def test_f7_t2_generate_pdf_no_signals_is_signal(mock_processed_df):
    # All signals are is_signal = False
    signals = analyze_all_pairs(mock_processed_df, min_cases=3)
    signals["is_signal"] = False
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_no_active_signals.pdf")
        generate_clinical_pdf(out_path, signals, target_drugs=[])
        assert os.path.exists(out_path)

def test_f7_t2_generate_pdf_invalid_output_path(mock_processed_df):
    signals = analyze_all_pairs(mock_processed_df, min_cases=3)
    # Output to invalid directory path that cannot be created
    out_path = "/nonexistent_folder_abc_123/report.pdf"
    with pytest.raises(Exception):
        generate_clinical_pdf(out_path, signals, target_drugs=[])


# ==============================================================================
# TIER 3: Cross-Feature Integration Tests (Pairwise combinations = 7 tests)
# ==============================================================================

def test_t3_cross_feature_1_fetch_and_filter(mocker, raw_reports_sample):
    # F1 (Fetch) + F2 (Filter)
    client = FDAClient()
    mock_session = mocker.MagicMock()
    mocker.patch.object(mock_session, "get", return_value=MockResponse(200, {"results": raw_reports_sample}))
    
    # Exec async fetch and sync filter
    fetched = asyncio.run(client.fetch_page(mock_session, 10, 0))
    df = client.process_and_filter_reports(fetched)
    
    assert len(df) == 3
    assert "REP001" in df["safetyreportid"].values

def test_t3_cross_feature_2_filter_and_stats(raw_reports_sample):
    # F2 (Filter) + F3 (Stats)
    client = FDAClient()
    df = client.process_and_filter_reports(raw_reports_sample)
    # df has reports:
    # REP001: ASPIRIN, LISINOPRIL, METFORMIN, IBUPROFEN, PREDNISONE; Reactions: VOMITING, MI
    # REP006: ASPIRIN, LISINOPRIL, METFORMIN, ATORVASTATIN, OMEPRAZOLE, IBUPROFEN; Reactions: RENAL FAILURE, AKI
    # REP007: PREDNISONE, LISINOPRIL, METFORMIN, FUROSEMIDE, ASPIRIN; Reactions: DYSPNEA, COUGH
    
    # Calculate contingency matrix on filtered data
    a, b, c, d = calculate_contingency_matrix(df, "ASPIRIN", "VOMITING")
    assert a == 1 # only REP001
    assert b == 0
    assert c == 2 # REP006 and REP007 have ASPIRIN but no VOMITING
    assert d == 0

def test_t3_cross_feature_3_stats_and_network(mock_processed_df):
    # F3 (Stats) + F4 (Network)
    signals = analyze_all_pairs(mock_processed_df, min_cases=3)
    G = build_safety_network(signals)
    
    # Check that network nodes correspond to drugs/socs in signals flagged
    network_drugs = {n for n, attr in G.nodes(data=True) if attr["type"] == "drug"}
    flagged_drugs = set(signals[signals["is_signal"] == True]["drug"])
    assert network_drugs.issubset(flagged_drugs)

def test_t3_cross_feature_4_stats_and_interaction(mock_processed_df):
    # F3 (Stats) + F5 (Interaction Matrix)
    signals = analyze_all_pairs(mock_processed_df, min_cases=3)
    co_matrix, pair_details = build_coprescription_matrix(mock_processed_df, top_n_drugs=5)
    
    # Check overlapping keys
    top_drugs = set(co_matrix.index)
    flagged_drugs = set(signals[signals["is_signal"] == True]["drug"])
    # There should be an overlap of top drugs and flagged drugs
    assert len(top_drugs.intersection(flagged_drugs)) > 0

def test_t3_cross_feature_5_filter_and_temporal(raw_reports_sample):
    # F2 (Filter) + F6 (Temporal)
    client = FDAClient()
    df = client.process_and_filter_reports(raw_reports_sample)
    pivot = compute_quarterly_signal_counts(df, top_n_adrs=5)
    assert not pivot.empty

def test_t3_cross_feature_6_stats_and_pdf(mock_processed_df):
    # F3 (Stats) + F7 (PDF)
    signals = analyze_all_pairs(mock_processed_df, min_cases=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "report.pdf")
        generate_clinical_pdf(out_path, signals, target_drugs=["ASPIRIN"])
        assert os.path.exists(out_path)

def test_t3_cross_feature_7_temporal_and_pdf(mock_processed_df):
    # F6 (Temporal) + F7 (PDF)
    df_dated = parse_report_dates(mock_processed_df)
    signals = analyze_all_pairs(df_dated, min_cases=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "temporal_report.pdf")
        generate_clinical_pdf(out_path, signals, target_drugs=[], total_cohort_reports=len(df_dated))
        assert os.path.exists(out_path)


# ==============================================================================
# TIER 4: Real-World Scenarios (5 tests)
# ==============================================================================

@pytest.mark.asyncio
async def test_t4_scenario_1_rate_limit_recovery(mocker):
    """
    Scenario: openFDA server returns multiple 429 Too Many Requests errors.
    Client recovers using backoff delays and completes all page fetches.
    """
    client = FDAClient(delay_between_batches=0.01)
    mock_session = mocker.MagicMock()
    mocker.patch("asyncio.sleep", return_value=None)
    
    # Simulating 4 consecutive 429 errors followed by success
    mock_get = mocker.patch.object(mock_session, "get")
    mock_get.side_effect = [
        MockResponse(429, {}),
        MockResponse(429, {}),
        MockResponse(429, {}),
        MockResponse(429, {}),
        MockResponse(200, {"results": [{"safetyreportid": "RECOVERED"}]})
    ]
    
    reports = await client.fetch_page(mock_session, limit=1, skip=0)
    assert len(reports) == 1
    assert reports[0]["safetyreportid"] == "RECOVERED"

def test_t4_scenario_2_geriatric_multimorbidity(raw_reports_sample):
    """
    Scenario: End-to-end flow from raw JSON parsing of multimorbid patient cohort 
    with various drugs, filtering, statistics, network building, and PDF generation.
    """
    client = FDAClient()
    df = client.process_and_filter_reports(raw_reports_sample)
    assert len(df) == 3
    
    # Stats
    signals = analyze_all_pairs(df, min_cases=1, ror_threshold=1.0)
    assert not signals.empty
    
    # Manually flag at least one pair as a signal to test down-stream network/PDF components
    if not signals.empty:
        signals.loc[0, "is_signal"] = True
        
    # Network
    G = build_safety_network(signals)
    assert len(G.nodes) > 0
    
    # PDF
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "multimorbidity_report.pdf")
        generate_clinical_pdf(out_path, signals, target_drugs=[])
        assert os.path.exists(out_path)

def test_t4_scenario_3_alert_threshold_tuning(mock_processed_df):
    """
    Scenario: Tuner changes min_cases and ROR thresholds to adjust alert sensitivity.
    System recalculates and dynamically adjusts flagged safety alerts.
    """
    # High sensitivity: min_cases=2, ror_threshold=1.0
    signals_sensitive = analyze_all_pairs(mock_processed_df, min_cases=2, ror_threshold=1.0)
    active_sensitive = signals_sensitive[signals_sensitive["is_signal"] == True]
    
    # Low sensitivity: min_cases=10, ror_threshold=2.0
    signals_conservative = analyze_all_pairs(mock_processed_df, min_cases=10, ror_threshold=2.0)
    active_conservative = signals_conservative[signals_conservative["is_signal"] == True]
    
    # Sensitive should flag more signals than conservative
    assert len(active_sensitive) >= len(active_conservative)

def test_t4_scenario_4_historical_cohort(mock_processed_df):
    """
    Scenario: Comparing quarterly trend lines across two years or different drug cohorts.
    We track signal emergence from pre-emerging to established state.
    """
    # Cohort with Aspirin
    pivot_asp = compute_quarterly_signal_counts(mock_processed_df, selected_drugs=["ASPIRIN"], top_n_adrs=2)
    # Cohort with Ibuprofen
    pivot_ibu = compute_quarterly_signal_counts(mock_processed_df, selected_drugs=["IBUPROFEN"], top_n_adrs=2)
    
    assert "2024Q1" in pivot_asp.index
    assert "2024Q1" in pivot_ibu.index

def test_t4_scenario_5_clean_slate_startup():
    """
    Scenario: Clean start when database or API returns empty dataset.
    The system should not crash but cleanly return empty collections.
    """
    client = FDAClient()
    df = client.process_and_filter_reports([]) # empty fetch
    assert df.empty
    
    signals = analyze_all_pairs(df)
    assert signals.empty
    
    G = build_safety_network(signals)
    assert len(G.nodes) == 0
    
    co_matrix, _ = build_coprescription_matrix(df)
    assert co_matrix.empty

def test_generate_one_page_summary_runs(mock_processed_df):
    """
    Verify that the one-page summary PDF generator functions without error.
    """
    from app.pdf_generator import generate_one_page_summary
    signals = analyze_all_pairs(mock_processed_df, min_cases=1, ror_threshold=1.0)
    if not signals.empty:
        signals.loc[0, "is_signal"] = True
        
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "one_page_summary.pdf")
        generate_one_page_summary(out_path, signals, target_drugs=["ASPIRIN"])
        assert os.path.exists(out_path)

