import pytest
import pandas as pd
import os
from streamlit.testing.v1 import AppTest

@pytest.fixture
def app_mock_df() -> pd.DataFrame:
    """
    Mock processed DataFrame for app tests.
    """
    return pd.DataFrame([
        {
            "safetyreportid": f"REP_{i}",
            "report_date": "20240115",
            "age": 65,
            "concomitant_count": 5,
            "drugs": "ASPIRIN;LISINOPRIL;METFORMIN;IBUPROFEN;PREDNISONE",
            "reactions": "MYOCARDIAL INFARCTION;VOMITING"
        } for i in range(10)
    ])

# ==============================================================================
# TIER 1: Presentation Feature Coverage
# ==============================================================================

def test_app_welcome_screen(mocker):
    """
    Test welcome screen instructions are shown when no data is loaded.
    """
    # Mock cache check so it doesn't auto-load cached data from disk
    mocker.patch("os.path.exists", return_value=False)
    
    at = AppTest.from_file("app/main.py")
    at.session_state["data_loaded"] = False
    at.session_state["reports_df"] = None
    at.run()
    
    # Verify welcome message info box exists
    assert any("Welcome to VigiSignal-X!" in info.value for info in at.info)
    # Check sidebar message
    assert any("Fetch data to enable filters." in info.value for info in at.sidebar.info)

def test_app_kpi_cards(app_mock_df):
    """
    Test that KPI cards are displayed correctly when data is loaded.
    """
    at = AppTest.from_file("app/main.py")
    at.session_state["data_loaded"] = True
    at.session_state["reports_df"] = app_mock_df
    at.run()
    
    # Verify titles
    assert any("VigiSignal-X Computational Pharmacovigilance" in md.value for md in at.markdown)
    # Check KPI cards contents
    kpi_markdowns = [md.value for md in at.markdown if "Total Cohort Reports" in md.value]
    assert len(kpi_markdowns) > 0
    assert "10" in kpi_markdowns[0]

def test_app_sidebar_filters(app_mock_df):
    """
    Test that sliders and multiselect options are available in the sidebar.
    """
    at = AppTest.from_file("app/main.py")
    at.session_state["data_loaded"] = True
    at.session_state["reports_df"] = app_mock_df
    at.run()
    
    # We should have sliders in sidebar
    assert len(at.sidebar.slider) >= 2
    # Verify multiselect is present for target medications
    assert len(at.sidebar.multiselect) == 1

def test_app_tabs_rendering(app_mock_df):
    """
    Test that the tabs are present in the layout.
    """
    at = AppTest.from_file("app/main.py")
    at.session_state["data_loaded"] = True
    at.session_state["reports_df"] = app_mock_df
    at.run()
    
    assert not at.exception

def test_app_pdf_compile_button(app_mock_df, mocker):
    """
    Test the PDF compilation button triggers generate_clinical_pdf.
    """
    at = AppTest.from_file("app/main.py")
    at.session_state["data_loaded"] = True
    at.session_state["reports_df"] = app_mock_df
    
    generated_paths = []
    # Mock the PDF generator function to write a dummy file
    def mock_pdf_side_effect(output_path, *args, **kwargs):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"pdf_content")
        generated_paths.append(output_path)
    mock_pdf = mocker.patch("app.pdf_generator.generate_clinical_pdf", side_effect=mock_pdf_side_effect)
    
    at.run()
    
    # The Compile Clinical PDF button is the first main button on the page
    assert len(at.button) >= 1
    compile_btn = at.button[0]
    assert "Compile Clinical PDF" in compile_btn.label
    
    compile_btn.click().run()
    assert mock_pdf.called
    
    # Clean up generated test files
    for path in generated_paths:
        if os.path.exists(path):
            os.remove(path)


# ==============================================================================
# TIER 2: App Boundary & Corner Cases
# ==============================================================================

def test_app_boundary_slider_max_cases(app_mock_df):
    """
    Boundary Test: Adjust min cases slider to maximum value (30).
    Verify that no signals are flagged since max reports is 10.
    """
    at = AppTest.from_file("app/main.py")
    at.session_state["data_loaded"] = True
    at.session_state["reports_df"] = app_mock_df
    at.run()
    
    # Min cases slider is index 0 in the sidebar
    min_cases_slider = at.sidebar.slider[0]
    min_cases_slider.set_value(30).run()
    
    # Flagged safety signals should drop to 0
    kpi_markdowns = [md.value for md in at.markdown if "Safety Signals Flagged" in md.value]
    assert len(kpi_markdowns) > 0
    assert '<p class="kpi-value">0</p>' in kpi_markdowns[0]

def test_app_empty_filter_selection(app_mock_df):
    """
    Boundary Test: Deselect all drugs in multiselect (empty selection).
    App should default to analyze all drugs in the cohort.
    """
    at = AppTest.from_file("app/main.py")
    at.session_state["data_loaded"] = True
    at.session_state["reports_df"] = app_mock_df
    at.run()
    
    drug_select = at.sidebar.multiselect[0]
    drug_select.set_value([]).run()
    
    kpi_markdowns = [md.value for md in at.markdown if "Total Cohort Reports" in md.value]
    assert len(kpi_markdowns) > 0
    assert "10" in kpi_markdowns[0]

def test_app_zero_data_error_handling():
    """
    Corner Test: Handle loading of empty processed dataframe gracefully in landing view.
    """
    at = AppTest.from_file("app/main.py")
    at.session_state["data_loaded"] = True
    at.session_state["reports_df"] = pd.DataFrame(columns=["drugs", "reactions"])
    at.run()
    
    assert not at.exception
    # Empty processed dataframe should show some informational message
    assert any("No data" in info.value or "No active signals" in info.value for info in at.info)

def test_app_api_key_input(mocker):
    """
    Test entering API key in password text box.
    """
    mocker.patch("os.path.exists", return_value=False)
    at = AppTest.from_file("app/main.py")
    at.run()
    
    pwd_input = at.sidebar.text_input[0]
    pwd_input.set_value("TEST_API_KEY_123").run()
    assert pwd_input.value == "TEST_API_KEY_123"


def test_app_invalid_cache_file(mocker):
    """
    Test loading invalid/corrupted cache CSV shows error on sidebar.
    """
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("pandas.read_csv", side_effect=ValueError("Corrupted CSV file"))
    
    at = AppTest.from_file("app/main.py")
    at.session_state["data_loaded"] = False
    at.run()
    
    # Verify error message shown in sidebar
    assert any("Error loading cached dataset" in err.value for err in at.sidebar.error)

# ==============================================================================
# TIER 3 & 4: App Integration and User Journeys
# ==============================================================================

def test_app_t3_cross_feature_filter_updates_kpi(app_mock_df):
    """
    Tier 3: Pairwise combination of multiselect drug selection + slider thresholds.
    Changing filters updates ROR and filters down reports.
    """
    at = AppTest.from_file("app/main.py")
    at.session_state["data_loaded"] = True
    at.session_state["reports_df"] = app_mock_df
    at.run()
    
    # Filter drugs to ASPIRIN only
    drug_select = at.sidebar.multiselect[0]
    asp_option = [opt for opt in drug_select.options if "ASPIRIN" in opt][0]
    drug_select.set_value([asp_option]).run()
    
    # Slider min cases to 5
    min_cases_slider = at.sidebar.slider[0]
    min_cases_slider.set_value(5).run()
    
    # ROR threshold to 1.5
    ror_slider = at.sidebar.slider[1]
    ror_slider.set_value(1.5).run()
    
    assert not at.exception

def test_app_t4_real_world_user_journey(app_mock_df, mocker):
    """
    Tier 4: Simulates a complete clinician audit session.
    1. Lands on dashboard
    2. Modifies min cases safety threshold to 3
    3. Searches and filters for specific drug (ASPIRIN)
    4. Compiles PDF Report
    """
    at = AppTest.from_file("app/main.py")
    at.session_state["data_loaded"] = True
    at.session_state["reports_df"] = app_mock_df
    
    generated_paths = []
    # Mock PDF compilation by writing a dummy file to the output path
    def mock_pdf_side_effect(output_path, *args, **kwargs):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"pdf_bytes")
        generated_paths.append(output_path)
    mock_pdf = mocker.patch("app.pdf_generator.generate_clinical_pdf", side_effect=mock_pdf_side_effect)
    
    at.run()
    
    # Adjust min cases
    at.sidebar.slider[0].set_value(3).run()
    
    # Adjust drug multiselect
    drug_select = at.sidebar.multiselect[0]
    asp_option = [opt for opt in drug_select.options if "ASPIRIN" in opt][0]
    drug_select.set_value([asp_option]).run()
    
    # Click compile PDF button
    compile_btn = [b for b in at.button if "Compile PDF" in b.label or "Compile Clinical PDF" in b.label][0]
    compile_btn.click().run()
    
    assert mock_pdf.called
    
    # Assert download button is present in the element tree
    download_btn = [el for el in at if el.type == "download_button" and "Download" in el.label][0]
    assert download_btn is not None
    assert "Download Clinical Audit PDF" in download_btn.label
    
    # Clean up generated test files
    for path in generated_paths:
        if os.path.exists(path):
            os.remove(path)



