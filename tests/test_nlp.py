from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import pandas as pd
import torch
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.processing.outputs import PredictionResult, EntityPrediction

from core.nlp_engine import analyze_and_deidentify, calculate_adr_similarity

@pytest.fixture
def mock_openmed():
    with patch("openmed.deidentify") as mock_deid, patch("openmed.analyze_text") as mock_analyze:
        # Mock deidentify dynamically based on the method argument
        def deid_side_effect(text, method="mask", **kwargs):
            mock_entity = PIIEntity(
                text="John Doe",
                label="PERSON",
                start=8,
                end=16,
                confidence=0.95
            )
            if method == "mask":
                deidentified_text = "Patient [NAME] is admitted."
            elif method == "replace":
                deidentified_text = "Patient Jane Smith is admitted."
            elif method == "hash":
                deidentified_text = "Patient PERSON_a1b2c3d4 is admitted."
            else:
                deidentified_text = text
                
            return DeidentificationResult(
                original_text=text,
                deidentified_text=deidentified_text,
                pii_entities=[mock_entity],
                method=method,
                timestamp=None,
                mapping={"[NAME]": "John Doe"}
            )
        
        mock_deid.side_effect = deid_side_effect

        # Mock analyze_text to return entities based on model_name
        def side_effect(text, model_name):
            if "disease" in model_name:
                return PredictionResult(
                    text=text,
                    entities=[EntityPrediction(text="headache", label="DISEASE", confidence=0.85, start=12, end=20)],
                    model_name=model_name,
                    timestamp="2026-06-25T10:00:00"
                )
            elif "pharma" in model_name:
                return PredictionResult(
                    text=text,
                    entities=[EntityPrediction(text="Aspirin", label="CHEM", confidence=0.95, start=0, end=7)],
                    model_name=model_name,
                    timestamp="2026-06-25T10:00:00"
                )
            elif "anatomy" in model_name:
                return PredictionResult(
                    text=text,
                    entities=[EntityPrediction(text="knee", label="ANATOMY", confidence=0.75, start=25, end=29)],
                    model_name=model_name,
                    timestamp="2026-06-25T10:00:00"
                )
            return PredictionResult(text=text, entities=[], model_name=model_name, timestamp="2026-06-25T10:00:00")

        mock_analyze.side_effect = side_effect
        yield mock_deid, mock_analyze

@pytest.fixture
def mock_transformers():
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_load, \
         patch("transformers.AutoModel.from_pretrained") as mock_model_load:
        
        # Mock Tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long)
        }
        mock_tok_load.return_value = mock_tokenizer
        
        # Mock Model & output states
        mock_model = MagicMock()
        
        def model_side_effect(**kwargs):
            model_side_effect.call_count += 1
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.ones((1, 5, 768)) * float(model_side_effect.call_count)
            return mock_output

        model_side_effect.call_count = 0
        mock_model.side_effect = model_side_effect
        
        mock_model_load.return_value = mock_model
        
        yield mock_tok_load, mock_model_load

def test_analyze_and_deidentify_empty():
    res = analyze_and_deidentify("")
    assert res["original_text"] == ""
    assert res["deidentified_text"] == ""
    assert len(res["pii_entities"]) == 0
    assert len(res["clinical_entities"]) == 0

def test_analyze_and_deidentify_success(mock_openmed):
    mock_deid, mock_analyze = mock_openmed
    
    text = "Aspirin for headache and knee pain."
    res = analyze_and_deidentify(text, method="mask", lang="en")
    
    assert res["original_text"] == text
    assert res["deidentified_text"] == "Patient [NAME] is admitted."
    assert len(res["pii_entities"]) == 1
    assert res["pii_entities"][0]["text"] == "John Doe"
    assert res["pii_entities"][0]["label"] == "PERSON"
    
    assert len(res["clinical_entities"]) == 3
    # Check clinical entities extraction & mapping
    categories = [e["label"] for e in res["clinical_entities"]]
    assert "Drug" in categories
    assert "Condition" in categories
    assert "Anatomy" in categories
    
    # Check sorting by start index
    starts = [e["start"] for e in res["clinical_entities"]]
    assert starts == sorted(starts)

def test_analyze_and_deidentify_different_methods(mock_openmed):
    mock_deid, mock_analyze = mock_openmed
    text = "Aspirin for headache."
    
    # Test mask
    res_mask = analyze_and_deidentify(text, method="mask", lang="en")
    assert res_mask["deidentified_text"] == "Patient [NAME] is admitted."
    
    # Test replace
    res_replace = analyze_and_deidentify(text, method="replace", lang="en")
    assert res_replace["deidentified_text"] == "Patient Jane Smith is admitted."
    
    # Test hash
    res_hash = analyze_and_deidentify(text, method="hash", lang="en")
    assert res_hash["deidentified_text"] == "Patient PERSON_a1b2c3d4 is admitted."

def test_calculate_adr_similarity_empty():
    df = calculate_adr_similarity("", "dummy-model")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

def test_calculate_adr_similarity_success(mock_transformers):
    text = "This is a clinical note."
    model_id = "doctolib-lab/doctobert-fr-base"
    
    df = calculate_adr_similarity(text, model_id, lang="fr")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4 # 4 French ADRs
    assert "adr" in df.columns
    assert "similarity" in df.columns
    
    # Check that they are sorted by similarity descending
    similarities = df["similarity"].tolist()
    assert similarities == sorted(similarities, reverse=True)
    
    for score in similarities:
        assert -1.0 <= score <= 1.0

def test_calculate_adr_similarity_different_models(mock_transformers):
    text = "Mild symptoms reported."
    
    # Test French doctomodernbert
    df_modern = calculate_adr_similarity(text, model_id="doctolib-lab/doctomodernbert-fr-base", lang="fr")
    assert len(df_modern) == 4
    
    # Test English biobert
    df_bio = calculate_adr_similarity(text, model_id="dmis-lab/biobert-v1.1", lang="en")
    assert len(df_bio) == 4
    assert df_bio["adr"].iloc[0] in ["Angioedema", "Acute Kidney Injury", "Gastrointestinal Haemorrhage", "Hyperkalemia"]

def test_calculate_adr_similarity_custom_adrs(mock_transformers):
    text = "Patient feels dizzy."
    custom_adrs = ["Dizziness", "Fatigue", "Dry Mouth"]
    model_id = "dmis-lab/biobert-v1.1"
    
    # Test the positional signature: calculate_adr_similarity(text, adrs, model_id)
    df = calculate_adr_similarity(text, custom_adrs, model_id)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df["adr"]) == set(custom_adrs)
    
    # Verify descending sort order
    similarities = df["similarity"].tolist()
    assert similarities == sorted(similarities, reverse=True)


def test_nlp_client_sync_success(mocker):
    # Mock requests.post
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "original_text": "Aspirin for John Doe.",
        "deidentified_text": "Aspirin for [NAME].",
        "pii_entities": [{"text": "John Doe", "label": "PERSON", "start": 12, "end": 20}],
        "clinical_entities": []
    }
    mocker.patch("requests.post", return_value=mock_response)

    res = analyze_and_deidentify("Aspirin for John Doe.", method="mask", lang="en")
    assert res["deidentified_text"] == "Aspirin for [NAME]."
    assert len(res["pii_entities"]) == 1


@pytest.mark.asyncio
async def test_nlp_client_async_success(mocker):
    # Mock httpx.AsyncClient.post
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "original_text": "Aspirin for John Doe.",
        "deidentified_text": "Aspirin for [NAME].",
        "pii_entities": [{"text": "John Doe", "label": "PERSON", "start": 12, "end": 20}],
        "clinical_entities": []
    }
    
    # Mock the async context manager of httpx.AsyncClient
    mock_client = mocker.MagicMock()
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    
    # httpx.AsyncClient() returns a context manager, so __aenter__ must return the mock_client
    mock_client_context = mocker.MagicMock()
    mock_client_context.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client_context.__aexit__ = mocker.AsyncMock(return_value=None)
    
    mocker.patch("httpx.AsyncClient", return_value=mock_client_context)

    from core.nlp_engine import analyze_and_deidentify_async
    res = await analyze_and_deidentify_async("Aspirin for John Doe.", method="mask", lang="en")
    assert res["deidentified_text"] == "Aspirin for [NAME]."
    assert len(res["pii_entities"]) == 1
