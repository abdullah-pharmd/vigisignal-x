import pytest
import pandas as pd
from typing import Dict, Any, List

@pytest.fixture
def raw_reports_sample() -> List[Dict[str, Any]]:
    """
    Returns a list of raw openFDA safety reports.
    Contains both valid geriatric polypharmacy reports and invalid ones (for filtering tests).
    """
    return [
        # 1. Valid: Geriatric (65, 801), Polypharmacy (5 drugs), Reactions present
        {
            "safetyreportid": "REP001",
            "receivedate": "20240115",
            "patient": {
                "patientonsetage": "65",
                "patientonsetageunit": "801",
                "drug": [
                    {"medicinalproduct": "ASPIRIN"},
                    {"medicinalproduct": "LISINOPRIL"},
                    {"medicinalproduct": "METFORMIN"},
                    {"medicinalproduct": "IBUPROFEN"},
                    {"medicinalproduct": "PREDNISONE"}
                ],
                "reaction": [
                    {"reactionmeddrapt": "VOMITING"},
                    {"reactionmeddrapt": "MYOCARDIAL INFARCTION"}
                ]
            }
        },
        # 2. Invalid: Underage (45 years old)
        {
            "safetyreportid": "REP002",
            "receivedate": "20240120",
            "patient": {
                "patientonsetage": "45",
                "patientonsetageunit": "801",
                "drug": [
                    {"medicinalproduct": "ASPIRIN"},
                    {"medicinalproduct": "LISINOPRIL"},
                    {"medicinalproduct": "METFORMIN"},
                    {"medicinalproduct": "IBUPROFEN"},
                    {"medicinalproduct": "PREDNISONE"}
                ],
                "reaction": [
                    {"reactionmeddrapt": "VOMITING"}
                ]
            }
        },
        # 3. Invalid: Wrong age unit (70, but unit 802 which is months)
        {
            "safetyreportid": "REP003",
            "receivedate": "20240215",
            "patient": {
                "patientonsetage": "70",
                "patientonsetageunit": "802",
                "drug": [
                    {"medicinalproduct": "ASPIRIN"},
                    {"medicinalproduct": "LISINOPRIL"},
                    {"medicinalproduct": "METFORMIN"},
                    {"medicinalproduct": "IBUPROFEN"},
                    {"medicinalproduct": "PREDNISONE"}
                ],
                "reaction": [
                    {"reactionmeddrapt": "HEADACHE"}
                ]
            }
        },
        # 4. Invalid: Too few drugs (3 drugs)
        {
            "safetyreportid": "REP004",
            "receivedate": "20240310",
            "patient": {
                "patientonsetage": "80",
                "patientonsetageunit": "801",
                "drug": [
                    {"medicinalproduct": "ASPIRIN"},
                    {"medicinalproduct": "LISINOPRIL"},
                    {"medicinalproduct": "METFORMIN"}
                ],
                "reaction": [
                    {"reactionmeddrapt": "RENAL FAILURE"}
                ]
            }
        },
        # 5. Invalid: No reactions
        {
            "safetyreportid": "REP005",
            "receivedate": "20240315",
            "patient": {
                "patientonsetage": "75",
                "patientonsetageunit": "801",
                "drug": [
                    {"medicinalproduct": "ASPIRIN"},
                    {"medicinalproduct": "LISINOPRIL"},
                    {"medicinalproduct": "METFORMIN"},
                    {"medicinalproduct": "IBUPROFEN"},
                    {"medicinalproduct": "PREDNISONE"}
                ],
                "reaction": []
            }
        },
        # 6. Valid: Geriatric (72), Polypharmacy (6 drugs), Renal ADR
        {
            "safetyreportid": "REP006",
            "receivedate": "20240405",
            "patient": {
                "patientonsetage": "72",
                "patientonsetageunit": "801",
                "drug": [
                    {"medicinalproduct": "ASPIRIN"},
                    {"medicinalproduct": "LISINOPRIL"},
                    {"medicinalproduct": "METFORMIN"},
                    {"medicinalproduct": "ATORVASTATIN"},
                    {"medicinalproduct": "OMEPRAZOLE"},
                    {"medicinalproduct": "IBUPROFEN"}
                ],
                "reaction": [
                    {"reactionmeddrapt": "RENAL FAILURE"},
                    {"reactionmeddrapt": "ACUTE KIDNEY INJURY"}
                ]
            }
        },
        # 7. Valid: Geriatric (85), Polypharmacy (5 drugs), Respiratory ADR
        {
            "safetyreportid": "REP007",
            "receivedate": "20240620",
            "patient": {
                "patientonsetage": "85",
                "patientonsetageunit": "801",
                "drug": [
                    {"medicinalproduct": "PREDNISONE"},
                    {"medicinalproduct": "LISINOPRIL"},
                    {"medicinalproduct": "METFORMIN"},
                    {"medicinalproduct": "FUROSEMIDE"},
                    {"medicinalproduct": "ASPIRIN"}
                ],
                "reaction": [
                    {"reactionmeddrapt": "DYSPNEA"},
                    {"reactionmeddrapt": "COUGH"}
                ]
            }
        }
    ]

@pytest.fixture
def mock_processed_df() -> pd.DataFrame:
    """
    Returns a mock processed pandas DataFrame representing the cohort.
    It contains structured data of drugs (semicolon separated) and reactions.
    Designed to facilitate contingency matrix calculations and disproportionality tests.
    """
    data = []
    
    # 1. Add 30 reports with ASPIRIN and MYOCARDIAL INFARCTION (a-cell candidates)
    # Plus LISINOPRIL, METFORMIN, IBUPROFEN, PREDNISONE to pass concomitant count (>4)
    for i in range(30):
        quarter_month = "01" if i < 10 else ("04" if i < 20 else "07")
        data.append({
            "safetyreportid": f"S_ASP_MI_{i}",
            "report_date": f"2024{quarter_month}15",
            "age": 65 + (i % 20),
            "concomitant_count": 5,
            "drugs": "ASPIRIN;LISINOPRIL;METFORMIN;IBUPROFEN;PREDNISONE",
            "reactions": "MYOCARDIAL INFARCTION;VOMITING;NAUSEA"
        })
        
    # 2. Add 2 reports with other drugs (e.g. IBUPROFEN) and MYOCARDIAL INFARCTION (b-cell candidates)
    for i in range(2):
        data.append({
            "safetyreportid": f"S_OTH_MI_{i}",
            "report_date": "20240210",
            "age": 70,
            "concomitant_count": 5,
            "drugs": "IBUPROFEN;LISINOPRIL;METFORMIN;ATORVASTATIN;OMEPRAZOLE",
            "reactions": "MYOCARDIAL INFARCTION;HEADACHE"
        })
        
    # 3. Add 5 reports with ASPIRIN and other reactions (e.g. DIZZINESS) (c-cell candidates)
    for i in range(5):
        data.append({
            "safetyreportid": f"S_ASP_OTH_{i}",
            "report_date": "20240510",
            "age": 68,
            "concomitant_count": 5,
            "drugs": "ASPIRIN;LISINOPRIL;METFORMIN;FUROSEMIDE;WARFARIN",
            "reactions": "DIZZINESS;HEADACHE"
        })
        
    # 4. Add 100 reports with other drugs and other reactions (d-cell candidates)
    for i in range(100):
        quarter_month = "03" if i < 30 else ("06" if i < 60 else "09")
        data.append({
            "safetyreportid": f"S_OTH_OTH_{i}",
            "report_date": f"2024{quarter_month}20",
            "age": 72,
            "concomitant_count": 5,
            "drugs": "LISINOPRIL;METFORMIN;ATORVASTATIN;OMEPRAZOLE;FUROSEMIDE",
            "reactions": "DIZZINESS;PRURITUS"
        })
        
    # 5. Add 1 report with PREDNISONE and DYSPNEA
    data.append({
        "safetyreportid": "S_PRED_DYS",
        "report_date": "20241010",
        "age": 75,
        "concomitant_count": 5,
        "drugs": "PREDNISONE;LISINOPRIL;METFORMIN;OMEPRAZOLE;ASPIRIN",
        "reactions": "DYSPNEA;COUGH"
    })
    
    return pd.DataFrame(data)
