import numpy as np
import pandas as pd

def simulate_deprescribing_heor(
    cohort_size: int,
    deprescribe_success: float,
    cost_intervention: float,
    prob_ade_control: float,
    prob_ade_interv: float,
    cost_ade: float,
    qaly_healthy: float,
    qaly_ade: float
) -> dict:
    """
    Calculates HEOR decision tree model outcomes for control and intervention cohorts.
    
    Returns a dictionary of:
        total_cost_control, total_qaly_control,
        effective_ade_prob, total_cost_interv, total_qaly_interv,
        inc_cost, inc_qaly, icer.
    """
    try:
        cohort_size = int(cohort_size)
        deprescribe_success = float(deprescribe_success)
        cost_intervention = float(cost_intervention)
        prob_ade_control = float(prob_ade_control)
        prob_ade_interv = float(prob_ade_interv)
        cost_ade = float(cost_ade)
        qaly_healthy = float(qaly_healthy)
        qaly_ade = float(qaly_ade)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid input type in HEOR simulation parameters: {e}")

    # Control Group (Standard Care)

    total_cost_control = cohort_size * prob_ade_control * cost_ade
    total_qaly_control = cohort_size * ((1 - prob_ade_control) * qaly_healthy + prob_ade_control * qaly_ade)

    # Intervention Group
    # Success path: success * prob_ade_interv, failure path: (1-success) * prob_ade_control
    effective_ade_prob = (deprescribe_success * prob_ade_interv) + ((1 - deprescribe_success) * prob_ade_control)
    total_cost_interv = (cohort_size * cost_intervention) + (cohort_size * effective_ade_prob * cost_ade)
    total_qaly_interv = cohort_size * ((1 - effective_ade_prob) * qaly_healthy + effective_ade_prob * qaly_ade)

    # Incremental Metrics
    inc_cost = total_cost_interv - total_cost_control
    inc_qaly = total_qaly_interv - total_qaly_control

    # ICER calculation
    if inc_qaly == 0:
        icer = float('nan')
    else:
        icer = inc_cost / inc_qaly

    return {
        "total_cost_control": total_cost_control,
        "total_qaly_control": total_qaly_control,
        "effective_ade_prob": effective_ade_prob,
        "total_cost_interv": total_cost_interv,
        "total_qaly_interv": total_qaly_interv,
        "inc_cost": inc_cost,
        "inc_qaly": inc_qaly,
        "icer": icer
    }

def simulate_sensitivity_analysis(
    inc_cost: float,
    inc_qaly: float,
    cost_ade: float,
    cohort_size: int,
    iterations: int = 250,
    deprescribe_success: float = 0.60,
    cost_intervention: float = 150.0,
    prob_ade_control: float = 0.20,
    prob_ade_interv: float = 0.08,
    qaly_healthy: float = 0.9,
    qaly_ade: float = 0.5
) -> pd.DataFrame:
    """
    Runs a Monte Carlo Probabilistic Sensitivity Analysis (PSA) using Beta and Gamma distributions.
    
    Clinical probabilities and utilities are sampled from a Beta distribution.
    Costs and financial parameters are sampled from a Gamma distribution.
    """
    try:
        inc_cost = float(inc_cost)
        inc_qaly = float(inc_qaly)
        cost_ade = float(cost_ade)
        cohort_size = int(cohort_size)
        iterations = int(iterations)
        deprescribe_success = float(deprescribe_success)
        cost_intervention = float(cost_intervention)
        prob_ade_control = float(prob_ade_control)
        prob_ade_interv = float(prob_ade_interv)
        qaly_healthy = float(qaly_healthy)
        qaly_ade = float(qaly_ade)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid input type in sensitivity analysis parameters: {e}")

    np.random.seed(42)

    # Effective sample size for Beta distribution parameters
    n_eff = 100.0

    def sample_beta(mean_val: float, n_val: float = n_eff) -> float:
        # Clamp mean to avoid undefined beta parameters
        p = np.clip(mean_val, 1e-4, 1.0 - 1e-4)
        alpha = p * n_val
        beta = (1.0 - p) * n_val
        return np.random.beta(alpha, beta)

    def sample_gamma(mean_val: float, cv: float = 0.2) -> float:
        if mean_val <= 0:
            return 0.0
        # shape k = (1 / cv)^2, scale theta = mean / k
        shape = 1.0 / (cv ** 2)
        scale = mean_val / shape
        return np.random.gamma(shape, scale)

    sim_inc_costs = []
    sim_inc_qalys = []

    for _ in range(iterations):
        # Sample probabilities and utilities (Beta)
        s_deprescribe_success = sample_beta(deprescribe_success)
        s_prob_ade_control = sample_beta(prob_ade_control)
        s_prob_ade_interv = sample_beta(prob_ade_interv)
        s_qaly_healthy = sample_beta(qaly_healthy)
        s_qaly_ade = sample_beta(qaly_ade)
        
        # Sample costs (Gamma)
        s_cost_intervention = sample_gamma(cost_intervention)
        s_cost_ade = sample_gamma(cost_ade)

        # Run model for sampled parameters
        res = simulate_deprescribing_heor(
            cohort_size=cohort_size,
            deprescribe_success=s_deprescribe_success,
            cost_intervention=s_cost_intervention,
            prob_ade_control=s_prob_ade_control,
            prob_ade_interv=s_prob_ade_interv,
            cost_ade=s_cost_ade,
            qaly_healthy=s_qaly_healthy,
            qaly_ade=s_qaly_ade
        )
        
        sim_inc_costs.append(res["inc_cost"])
        sim_inc_qalys.append(res["inc_qaly"])

    return pd.DataFrame({
        "Iteration": range(1, iterations + 1),
        "Incremental Cost ($)": sim_inc_costs,
        "Incremental QALYs": sim_inc_qalys
    })
