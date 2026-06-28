import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

def calculate_contingency_matrix(
    df: pd.DataFrame, 
    target_drug: str, 
    target_adr: str, 
    drug_col: str = "drugs", 
    adr_col: str = "reactions"
) -> Tuple[int, int, int, int]:
    """
    Computes a 2x2 contingency matrix for a target drug and target ADR.
    
    a = Exposed with Event (Target Drug + Target ADR)
    b = Unexposed with Event (Other Drugs + Target ADR)
    c = Exposed without Event (Target Drug + Other ADRs)
    d = Unexposed without Event (Other Drugs + Other ADRs)
    """
    # Normalize inputs to uppercase
    target_drug = target_drug.strip().upper()
    target_adr = target_adr.strip().upper()
    
    # Check membership using vectorized or mapped string splitting
    # Splitting strings on ';' and searching
    has_drug = df[drug_col].apply(lambda x: target_drug in str(x).split(";"))
    has_adr = df[adr_col].apply(lambda x: target_adr in str(x).split(";"))
    
    a = int((has_drug & has_adr).sum())
    b = int((~has_drug & has_adr).sum())
    c = int((has_drug & ~has_adr).sum())
    d = int((~has_drug & ~has_adr).sum())
    
    return a, b, c, d

def calculate_disproportionality_metrics(
    a: int, b: int, c: int, d: int
) -> Dict[str, Any]:
    """
    Calculates ROR, SE of ln(ROR), 95% Confidence Interval, and Bayesian Shrinkage (Information Component, IC).
    Applies Haldane correction (adding 0.5 to all cells) if any cell in the 2x2 table is 0.
    """
    # Check if Haldane correction is needed
    haldane_applied = False
    if a == 0 or b == 0 or c == 0 or d == 0:
        a_c = a + 0.5
        b_c = b + 0.5
        c_c = c + 0.5
        d_c = d + 0.5
        haldane_applied = True
    else:
        a_c, b_c, c_c, d_c = float(a), float(b), float(c), float(d)
        
    # Calculate ROR
    try:
        ror = (a_c * d_c) / (b_c * c_c)
    except ZeroDivisionError:
        ror = np.nan
        
    # Calculate Standard Error of ln(ROR)
    try:
        se_ln_ror = np.sqrt((1.0 / a_c) + (1.0 / b_c) + (1.0 / c_c) + (1.0 / d_c))
    except (ZeroDivisionError, ValueError):
        se_ln_ror = np.nan
        
    # Calculate 95% Confidence Intervals
    if not np.isnan(ror) and not np.isnan(se_ln_ror):
        ln_ror = np.log(ror)
        ci_lower = np.exp(ln_ror - 1.96 * se_ln_ror)
        ci_upper = np.exp(ln_ror + 1.96 * se_ln_ror)
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    # --- Bayesian Shrinkage (BCPNN Information Component) ---
    n_total = float(a + b + c + d)
    n_drug = float(a + c)
    n_adr = float(a + b)
    
    if n_total == 0:
        expected = 0.0
    else:
        expected = (n_drug * n_adr) / n_total

    # Calculate IC with stabilization adjustment
    ic = np.log2((a + 0.5) / (expected + 0.5))
    
    # Calculate IC variance
    # V(IC) = (1 / ln(2))^2 * (N - a + 0.5) / ((a + 0.5) * (N + 1.0))
    inv_ln2_sq = 1.0 / (np.log(2) ** 2)
    ic_var = inv_ln2_sq * (n_total - a + 0.5) / ((a + 0.5) * (n_total + 1.0))
    ic_se = np.sqrt(ic_var)
    ic_lower = ic - 1.96 * ic_se
        
    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "ror": ror,
        "se": se_ln_ror,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "haldane_applied": haldane_applied,
        "ic": ic,
        "ic_lower": ic_lower
    }

def classify_triage_tier(ror: float, ci_lower: float, a: int, is_signal: bool) -> str:
    """
    Classifies a drug-ADR pair into clinical triage tiers:
    - High Priority: ROR >= 3.0, CI_lower > 1.5, and cases (a) >= 5 (and is a valid signal)
    - Moderate Priority: ROR >= 2.0, CI_lower > 1.0, and cases (a) >= 3 (and is a valid signal)
    - Weak Priority: ROR > 1.0, CI_lower > 1.0, and cases (a) >= 3 (and is a valid signal)
    - Not Significant: Otherwise
    """
    if not is_signal:
        return "Not Significant"
        
    if ror >= 3.0 and ci_lower > 1.5 and a >= 5:
        return "High Priority"
    elif ror >= 2.0 and ci_lower > 1.0 and a >= 3:
        return "Moderate Priority"
    elif ror > 1.0 and ci_lower > 1.0 and a >= 3:
        return "Weak Priority"
    else:
        return "Not Significant"

def analyze_all_pairs(df: pd.DataFrame, min_cases: int = 3, ror_threshold: float = 1.0) -> pd.DataFrame:
    """
    Finds all unique drug-ADR pairs, calculates metrics, and filters based on:
    - a >= min_cases
    - ror > ror_threshold
    - ci_lower > 1.0 (statistical significance signal)
    """
    print("[StatsEngine] Starting batch analysis of drug-ADR pairs...")
    
    # Step 1: Count co-occurrences of all drug-ADR pairs to find candidates with a >= min_cases
    pair_counts = {}
    
    for _, row in df.iterrows():
        drugs_list = str(row["drugs"]).split(";")
        adrs_list = str(row["reactions"]).split(";")
        
        for drug in drugs_list:
            if not drug:
                continue
            for adr in adrs_list:
                if not adr:
                    continue
                pair = (drug, adr)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                
    # Filter candidates by minimum cases
    candidates = [pair for pair, count in pair_counts.items() if count >= min_cases]
    print(f"[StatsEngine] Total unique co-occurring pairs: {len(pair_counts)}")
    print(f"[StatsEngine] Candidates with cases >= {min_cases}: {len(candidates)}")
    
    # Pre-calculate drug/adr masks for fast processing
    # Map drug/adr to series of booleans
    unique_drugs = list(set(p[0] for p in candidates))
    unique_adrs = list(set(p[1] for p in candidates))
    
    print("[StatsEngine] Building membership flags for fast lookup...")
    drug_flags = {d: df["drugs"].apply(lambda x: d in str(x).split(";")).values for d in unique_drugs}
    adr_flags = {a: df["reactions"].apply(lambda x: a in str(x).split(";")).values for a in unique_adrs}
    
    results = []
    
    for target_drug, target_adr in candidates:
        has_drug = drug_flags[target_drug]
        has_adr = adr_flags[target_adr]
        
        a = int(np.sum(has_drug & has_adr))
        b = int(np.sum((~has_drug) & has_adr))
        c = int(np.sum(has_drug & (~has_adr)))
        d = int(np.sum((~has_drug) & (~has_adr)))
        
        metrics = calculate_disproportionality_metrics(a, b, c, d)
        
        # Determine if it's a valid safety signal
        is_signal = (
            a >= min_cases and 
            metrics["ror"] > ror_threshold and 
            metrics["ci_lower"] > 1.0 and
            metrics["ic_lower"] > 0.0
        )
        
        triage = classify_triage_tier(metrics["ror"], metrics["ci_lower"], a, is_signal)
        
        results.append({
            "drug": target_drug,
            "adr": target_adr,
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "ror": metrics["ror"],
            "se": metrics["se"],
            "ci_lower": metrics["ci_lower"],
            "ci_upper": metrics["ci_upper"],
            "haldane_applied": metrics["haldane_applied"],
            "ic": metrics["ic"],
            "ic_lower": metrics["ic_lower"],
            "is_signal": is_signal,
            "triage_tier": triage
        })
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Sort by ROR descending
        res_df = res_df.sort_values(by="ror", ascending=False).reset_index(drop=True)
        
    print(f"[StatsEngine] Batch analysis complete. Signals found: {res_df['is_signal'].sum() if not res_df.empty else 0}")
    return res_df
