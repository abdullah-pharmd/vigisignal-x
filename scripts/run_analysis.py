import os
import sys
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.stats_engine import analyze_all_pairs

def main():
    print("=" * 60)
    print("VigiSignal-X Headless Pharmacovigilance Analyzer")
    print("=" * 60)
    
    processed_path = os.path.join(project_root, "data", "processed", "processed_reports.csv")
    if not os.path.exists(processed_path):
        print(f"Error: Processed reports file not found at {processed_path}")
        sys.exit(1)
        
    print(f"Loading cohort data from: {processed_path}...")
    df = pd.read_csv(processed_path)
    print(f"Loaded {len(df)} patient cohort reports.")
    
    print("\nRunning ROR statistical engine (min_cases=3, ror_threshold=1.0)...")
    signals = analyze_all_pairs(df, min_cases=3, ror_threshold=1.0)
    
    if signals.empty:
        print("No safety signals detected.")
        return
        
    active_signals = signals[signals["is_signal"] == True]
    print(f"Analysis complete. Isolated {len(active_signals)} statistically significant safety signals.")
    
    print("\nTop 15 Detected Adverse Drug Reaction (ADR) Signals:")
    print("-" * 105)
    header = f"{'Medication':<20} | {'Adverse Reaction (ADR)':<30} | {'Cases (a)':<10} | {'ROR':<8} | {'95% CI Lower':<12} | {'Priority Tier':<15}"
    print(header)
    print("-" * 105)
    
    for _, row in active_signals.head(15).iterrows():
        triage = row.get("triage_tier", "Not Significant")
        line = f"{row['drug']:<20} | {row['adr'][:30]:<30} | {int(row['a']):<10} | {row['ror']:<8.2f} | {row['ci_lower']:<12.2f} | {triage:<15}"
        print(line)
    print("-" * 105)

if __name__ == "__main__":
    main()
