import unittest
import numpy as np
import pandas as pd
from core.stats_engine import (
    calculate_disproportionality_metrics, 
    calculate_contingency_matrix, 
    analyze_all_pairs
)

class TestStatsEngine(unittest.TestCase):
    
    def test_standard_metrics_calculation(self):
        """
        Tests ROR, SE, and 95% Confidence Interval for standard inputs.
        For a = 10, b = 20, c = 30, d = 40:
        ROR = (10 * 40) / (20 * 30) = 400 / 600 = 0.66666...
        SE = sqrt(1/10 + 1/20 + 1/30 + 1/40) = sqrt(0.1 + 0.05 + 0.0333 + 0.025) = sqrt(0.208333...) = 0.456435...
        ln(ROR) = ln(0.66666...) = -0.405465...
        CI Lower = exp(-0.405465 - 1.96 * 0.456435) = exp(-1.300078) = 0.2725...
        CI Upper = exp(-0.405465 + 1.96 * 0.456435) = exp(0.489148) = 1.6309...
        """
        metrics = calculate_disproportionality_metrics(10, 20, 30, 40)
        
        self.assertAlmostEqual(metrics["ror"], 0.6666667, places=5)
        self.assertAlmostEqual(metrics["se"], 0.4564355, places=5)
        self.assertAlmostEqual(metrics["ci_lower"], 0.2725063, places=5)
        self.assertAlmostEqual(metrics["ci_upper"], 1.6309267, places=5)
        self.assertFalse(metrics["haldane_applied"])

    def test_haldane_correction_activation(self):
        """
        Tests that Haldane correction adds 0.5 to all cells when any cell is 0.
        For a = 0, b = 10, c = 5, d = 20:
        With Haldane: a_c=0.5, b_c=10.5, c_c=5.5, d_c=20.5
        ROR = (0.5 * 20.5) / (10.5 * 5.5) = 10.25 / 57.75 = 0.177489...
        SE = sqrt(1/0.5 + 1/10.5 + 1/5.5 + 1/20.5) = sqrt(2.0 + 0.095238 + 0.181818 + 0.048780) = sqrt(2.325837) = 1.525069...
        """
        metrics = calculate_disproportionality_metrics(0, 10, 5, 20)
        
        self.assertAlmostEqual(metrics["ror"], 0.177489, places=5)
        self.assertAlmostEqual(metrics["se"], 1.525069, places=5)
        self.assertTrue(metrics["haldane_applied"])

    def test_contingency_matrix_builder(self):
        """
        Tests the 2x2 contingency matrix builder from a synthetic pandas DataFrame.
        """
        # Create synthetic test dataset
        data = [
            {"safetyreportid": "1", "drugs": "ASPIRIN;LISINOPRIL", "reactions": "VOMITING;HEADACHE"},
            {"safetyreportid": "2", "drugs": "ASPIRIN;PREDNISONE", "reactions": "VOMITING"},
            {"safetyreportid": "3", "drugs": "LISINOPRIL;METFORMIN", "reactions": "HEADACHE"},
            {"safetyreportid": "4", "drugs": "ASPIRIN", "reactions": "HEADACHE"},
            {"safetyreportid": "5", "drugs": "IBUPROFEN", "reactions": "VOMITING"},
        ]
        df = pd.DataFrame(data)
        
        # Target: ASPIRIN and VOMITING
        # ASPIRIN + VOMITING: Reports 1, 2 (a = 2)
        # Other Drugs + VOMITING: Report 5 (b = 1)
        # ASPIRIN + Other ADRs: Report 4 (c = 1)
        # Other Drugs + Other ADRs: Report 3 (d = 1)
        a, b, c, d = calculate_contingency_matrix(df, "ASPIRIN", "VOMITING")
        
        self.assertEqual(a, 2)
        self.assertEqual(b, 1)
        self.assertEqual(c, 1)
        self.assertEqual(d, 1)

    def test_batch_signal_detection_logic(self):
        """
        Tests that batch signal analysis correctly identifies and filters signals.
        """
        # Mock cohort where ASPIRIN and GASTROINTESTINAL HEMORRHAGE is a strong signal
        # and Ibuprofen + Headache is not a signal (few cases)
        data = []
        # 10 cases of Aspirin + Gastrointestinal hemorrhage
        for i in range(10):
            data.append({"drugs": "ASPIRIN;LISINOPRIL", "reactions": "GASTROINTESTINAL HEMORRHAGE;NAUSEA"})
        # 10 cases of Aspirin without Gastrointestinal hemorrhage
        for i in range(10):
            data.append({"drugs": "ASPIRIN;METFORMIN", "reactions": "DIZZINESS"})
        # 2 cases of Ibuprofen + Headache
        for i in range(2):
            data.append({"drugs": "IBUPROFEN", "reactions": "HEADACHE"})
        # 40 background cases of other drugs + other reactions
        for i in range(40):
            data.append({"drugs": "LISINOPRIL;METFORMIN", "reactions": "DIZZINESS;NAUSEA"})
            
        df = pd.DataFrame(data)
        
        # Batch analyze with min_cases=3
        res = analyze_all_pairs(df, min_cases=3, ror_threshold=1.0)
        
        # Aspirin + Gastrointestinal Hemorrhage should be flagged as is_signal=True
        asp_gast = res[(res["drug"] == "ASPIRIN") & (res["adr"] == "GASTROINTESTINAL HEMORRHAGE")]
        self.assertFalse(asp_gast.empty)
        self.assertTrue(asp_gast.iloc[0]["is_signal"])
        
        # Ibuprofen + Headache should not exist in signals df because case count (a=2) is less than min_cases=3
        ibu_head = res[(res["drug"] == "IBUPROFEN") & (res["adr"] == "HEADACHE")]
        self.assertTrue(ibu_head.empty)

    def test_triage_tier_classification(self):
        """
        Tests the classify_triage_tier function rules:
        - High Priority: ROR >= 3.0, CI_lower > 1.5, a >= 5, is_signal = True
        - Moderate Priority: ROR >= 2.0, CI_lower > 1.0, a >= 3, is_signal = True
        - Weak Priority: ROR > 1.0, CI_lower > 1.0, a >= 3, is_signal = True
        - Not Significant: Otherwise or is_signal = False
        """
        from core.stats_engine import classify_triage_tier
        
        # High Priority
        self.assertEqual(classify_triage_tier(3.5, 1.6, 5, True), "High Priority")
        # Moderate Priority
        self.assertEqual(classify_triage_tier(2.5, 1.1, 3, True), "Moderate Priority")
        # Weak Priority
        self.assertEqual(classify_triage_tier(1.5, 1.1, 3, True), "Weak Priority")
        # Not Significant because not a signal
        self.assertEqual(classify_triage_tier(5.0, 2.0, 10, False), "Not Significant")
        # Not Significant due to low counts/thresholds
        self.assertEqual(classify_triage_tier(0.9, 0.5, 3, True), "Not Significant")

if __name__ == "__main__":
    unittest.main()
