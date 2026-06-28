import math
import pandas as pd
import pytest
from core.protocol_parser import parse_protocol
from core.simulation_engine import simulate_deprescribing_heor, simulate_sensitivity_analysis

def test_parse_protocol_empty():
    assert parse_protocol("") is None
    assert parse_protocol(None) is None

def test_parse_protocol_typical():
    sample_text = """
    CLINICAL TRIAL PROTOCOL
    
    PATIENTS MUST MEET THE FOLLOWING INCLUSION CRITERIA:
    - Male or female aged 60 years or older.
    - Documented essential hypertension.
    
    PATIENTS MEETING ANY OF THE FOLLOWING EXCLUSION CRITERIA WILL BE EXCLUDED:
    - History of angioedema.
    - Severe renal impairment.
    
    OBJECTIVES AND ENDPOINTS
    The primary objective is to evaluate efficacy.
    The secondary endpoint is hospitalizations.
    
    SAFETY MONITORING AND STOPPING RULES
    - Grade 3 or 4 Hyperkalemia will require treatment discontinuation.
    - Severe symptomatic angioedema requires withdrawal.
    """
    
    result = parse_protocol(sample_text)
    assert result is not None
    assert len(result["inclusion"]) >= 1
    assert any("aged 60" in x for x in result["inclusion"])
    
    assert len(result["exclusion"]) >= 1
    assert any("angioedema" in x for x in result["exclusion"])
    
    assert len(result["endpoints"]) >= 1
    assert any("objective" in x.lower() or "endpoint" in x.lower() for x in result["endpoints"])
    
    assert len(result["safety_boundaries"]) >= 1
    assert any("discontinuation" in x.lower() or "withdrawal" in x.lower() for x in result["safety_boundaries"])

def test_parse_protocol_unstructured_fallback():
    # Test fallback heuristics when headers or bullet structures are missing
    sample_unstructured = """
    This study evaluates hypertensive patients must meet.
    History of hyperkalemia excluded.
    We are monitoring safety and stopping rules.
    """
    result = parse_protocol(sample_unstructured)
    assert result is not None
    assert len(result["inclusion"]) >= 1
    assert len(result["exclusion"]) >= 1
    assert len(result["safety_boundaries"]) >= 1

def test_simulate_deprescribing_heor_typical():
    # Cohort Size (Patients): 1000
    # Deprescribing Success: 60%
    # Intervention Cost: $150
    # Prob ADE Control: 20%
    # Prob ADE Interv: 8%
    # Cost ADE: $12000
    # QALY Healthy: 0.9
    # QALY ADE: 0.5
    results = simulate_deprescribing_heor(
        cohort_size=1000,
        deprescribe_success=0.60,
        cost_intervention=150.0,
        prob_ade_control=0.20,
        prob_ade_interv=0.08,
        cost_ade=12000.0,
        qaly_healthy=0.9,
        qaly_ade=0.5
    )
    
    assert results["total_cost_control"] == 2400000.0
    assert pytest.approx(results["total_qaly_control"], abs=1e-5) == 820.0
    assert pytest.approx(results["effective_ade_prob"], abs=1e-5) == 0.128
    assert results["total_cost_interv"] == 1686000.0
    assert pytest.approx(results["total_qaly_interv"], abs=1e-5) == 848.8
    assert results["inc_cost"] == -714000.0
    assert pytest.approx(results["inc_qaly"], abs=1e-5) == 28.8
    assert pytest.approx(results["icer"], abs=1e-2) == -24791.67

def test_simulate_deprescribing_heor_zero_qaly_gain():
    # If inc_qaly is 0, ICER should be nan
    results = simulate_deprescribing_heor(
        cohort_size=1000,
        deprescribe_success=0.60,
        cost_intervention=150.0,
        prob_ade_control=0.20,
        prob_ade_interv=0.20, # same prob, so inc_qaly is 0
        cost_ade=12000.0,
        qaly_healthy=0.9,
        qaly_ade=0.9 # same QALY weight
    )
    assert math.isnan(results["icer"])

def test_simulate_sensitivity_analysis():
    df = simulate_sensitivity_analysis(
        inc_cost=-714000.0,
        inc_qaly=28.8,
        cost_ade=12000.0,
        cohort_size=1000,
        iterations=150
    )
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 150
    assert list(df.columns) == ["Iteration", "Incremental Cost ($)", "Incremental QALYs"]
