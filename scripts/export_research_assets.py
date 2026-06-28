import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.stats_engine import analyze_all_pairs
from app.pdf_generator import generate_clinical_pdf, generate_one_page_summary

def main():
    print("=" * 60)
    print("VigiSignal-X Headless Research Assets Exporter")
    print("=" * 60)
    
    processed_path = os.path.join(project_root, "data", "processed", "processed_reports.csv")
    if not os.path.exists(processed_path):
        print(f"Error: Processed reports file not found at {processed_path}")
        sys.exit(1)
        
    print(f"Loading cohort data from: {processed_path}...")
    df = pd.read_csv(processed_path)
    
    print("\nRunning ROR statistical engine...")
    signals = analyze_all_pairs(df, min_cases=3, ror_threshold=1.0)
    active_signals = signals[signals["is_signal"] == True]
    
    # 1. Export CSV
    csv_output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(csv_output_dir, exist_ok=True)
    csv_path = os.path.join(csv_output_dir, "active_signals.csv")
    print(f"\n[1/3] Exporting safety signals table to CSV...")
    active_signals.to_csv(csv_path, index=False)
    print(f"Success: Exported {len(active_signals)} safety signals to {csv_path}")
    
    # 2. Export Full Clinical PDF Report
    pdf_output_dir = os.path.join(project_root, "data", "reports")
    os.makedirs(pdf_output_dir, exist_ok=True)
    
    full_report_path = os.path.join(pdf_output_dir, "vigisignal_clinical_report_headless.pdf")
    print(f"\n[2/3] Compiling Full Clinical Audit PDF Report...")
    generate_clinical_pdf(
        output_path=full_report_path,
        signals_df=signals,
        target_drugs=[],
        total_cohort_reports=len(df),
        min_cases=3,
        ror_threshold=1.0
    )
    print(f"Success: Compiled full clinical report to {full_report_path}")
    
    # 3. Export One-Page Executive Summary PDF
    summary_report_path = os.path.join(pdf_output_dir, "vigisignal_executive_summary_headless.pdf")
    print(f"\n[3/3] Compiling One-Page Executive Summary PDF...")
    generate_one_page_summary(
        output_path=summary_report_path,
        signals_df=signals,
        target_drugs=[],
        total_cohort_reports=len(df),
        min_cases=3,
        ror_threshold=1.0
    )
    print(f"Success: Compiled executive summary to {summary_report_path}")
    print("\n" + "=" * 60)
    print("All headless research assets exported successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
