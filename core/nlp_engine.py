import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
import openmed
import httpx
import requests
import os
import asyncio

logger = logging.getLogger(__name__)

NLP_API_URL = os.getenv("NLP_API_URL", "http://localhost:8000")

# Cache for models and tokenizers to avoid reloading
_MODEL_CACHE: Dict[str, Any] = {}

# Target ADRs in English and French
ADR_TERMS = {
    "en": [
        "Angioedema",
        "Acute Kidney Injury",
        "Gastrointestinal Haemorrhage",
        "Hyperkalemia"
    ],
    "fr": [
        "Angioedème",
        "Insuffisance rénale aiguë",
        "Hémorragie gastro-intestinale",
        "Hyperkaliémie"
    ]
}

def get_device() -> torch.device:
    """Auto-detects device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_and_tokenizer(model_name: str) -> tuple:
    """Gets or loads model and tokenizer from cache, mapped to the detected device."""
    global _MODEL_CACHE
    if model_name not in _MODEL_CACHE:
        device = get_device()
        logger.info(f"Loading model {model_name} on device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        _MODEL_CACHE[model_name] = (model, tokenizer)
    return _MODEL_CACHE[model_name]

def compute_mean_pooled_embedding(text: str, model_name: str) -> np.ndarray:
    """Computes mean-pooled hidden state embedding for a text string using appropriate device."""
    model, tokenizer = get_model_and_tokenizer(model_name)
    device = next(model.parameters()).device
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # Move inputs to the model's device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.last_hidden_state is [batch_size, seq_len, hidden_dim]
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    embedding = (sum_embeddings / sum_mask)[0]
    return embedding.cpu().numpy()

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalizes a numpy array to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def _analyze_and_deidentify_local(text: str, method: str = "mask", lang: str = "en") -> Dict[str, Any]:
    """
    De-identifies the clinical narrative text using OpenMed and extracts clinical entities (Drugs, Conditions, Anatomy).
    
    Args:
        text: Unstructured clinical narrative.
        method: De-identification method ('mask', 'replace', 'hash').
        lang: Language ('en' or 'fr').
        
    Returns:
        A dictionary containing deidentified text, pii entities, and clinical entities.
    """
    if text is None or not isinstance(text, str):
        text = ""
    if not text.strip():
        return {
            "original_text": text,
            "deidentified_text": text,
            "pii_entities": [],
            "clinical_entities": []
        }


    # 1. PII De-identification
    # Ensure method matches DeidentificationMethod type (mask, replace, hash)
    valid_methods = ["mask", "replace", "hash"]
    if method not in valid_methods:
        method = "mask"
        
    try:
        # Note: openmed.deidentify returns a DeidentificationResult
        res = openmed.deidentify(text, method=method, lang=lang, keep_mapping=True, use_safety_sweep=True)
        # Handle if result is AuditReport instead of DeidentificationResult
        if hasattr(res, "deidentification_result"):
            res = res.deidentification_result
        deidentified_text = getattr(res, "deidentified_text", text)
        pii_entities = [
            {
                "text": e.text,
                "label": e.label,
                "start": e.start,
                "end": e.end,
                "confidence": getattr(e, "confidence", 0.9)
            }
            for e in getattr(res, "pii_entities", [])
        ]
    except Exception as e:
        logger.error(f"De-identification failed: {e}")
        deidentified_text = text
        pii_entities = []

    # 2. Clinical NER
    # We extract Drugs, Conditions, and Anatomy using corresponding openmed models.
    clinical_entities = []
    
    ner_models = {
        "Condition": "disease_detection_superclinical",
        "Drug": "pharma_detection_superclinical",
        "Anatomy": "anatomy_detection_electramed"
    }
    
    for category, model_name in ner_models.items():
        try:
            pred_result = openmed.analyze_text(text, model_name=model_name)
            # pred_result is a PredictionResult
            for e in getattr(pred_result, "entities", []):
                clinical_entities.append({
                    "text": e.text,
                    "label": category,
                    "confidence": getattr(e, "confidence", getattr(e, "score", 0.8)),
                    "start": e.start,
                    "end": e.end
                })
        except Exception as e:
            logger.error(f"Clinical NER for {category} failed: {e}")
            
    # Sort clinical entities by start index
    clinical_entities.sort(key=lambda x: x.get("start", 0) or 0)
    
    return {
        "original_text": text,
        "deidentified_text": deidentified_text,
        "pii_entities": pii_entities,
        "clinical_entities": clinical_entities
    }

def _calculate_adr_similarity_local(
    text: str,
    adrs: Any = None,
    model_id: Optional[str] = None,
    lang: str = "en"
) -> pd.DataFrame:
    """
    Calculates semantic similarity between a text narrative and safety cohort ADR terms.
    Supports signatures:
        calculate_adr_similarity(text, model_id, lang)
        calculate_adr_similarity(text, adrs, model_id)
        calculate_adr_similarity(text, model_id=model_id, lang=lang)
    
    Args:
        text: Clinical note narrative.
        adrs: Optional list of target ADR terms or string model_id (for legacy compatibility).
        model_id: HuggingFace model identifier.
        lang: Language ('en' or 'fr') to decide default target ADR terms if adrs not provided.
        
    Returns:
        DataFrame containing columns: 'adr' and 'similarity', sorted descending by similarity.
    """
    if text is None or not isinstance(text, str):
        text = ""
    if not text.strip():
        return pd.DataFrame(columns=["adr", "similarity"])


    # Handle flexible positional arguments:
    # If the second argument (adrs) is passed as a string, it represents the model_id
    if isinstance(adrs, str):
        model_id = adrs
        adrs = None

    # Determine terms to use based on language/provided list
    terms = adrs if adrs is not None else ADR_TERMS.get(lang, ADR_TERMS["en"])
    
    # Resolve default model_id if not provided
    if not model_id:
        if lang == "fr":
            model_id = "doctolib-lab/doctomodernbert-fr-base"
        else:
            model_id = "dmis-lab/biobert-v1.1"

    try:
        # 1. Compute embedding for narrative text
        text_embedding = compute_mean_pooled_embedding(text, model_id)
        text_emb_norm = normalize_vector(text_embedding)
        
        # 2. Compute embedding for each ADR term and calculate similarity
        scores = []
        for adr in terms:
            adr_embedding = compute_mean_pooled_embedding(adr, model_id)
            adr_emb_norm = normalize_vector(adr_embedding)
            
            # Cosine similarity on normalized vectors using scipy
            dist = cosine(text_emb_norm, adr_emb_norm)
            similarity = float(1.0 - dist)
            scores.append({
                "adr": adr,
                "similarity": round(similarity, 4)
            })
            
        df = pd.DataFrame(scores)
        df = df.sort_values(by="similarity", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Semantic similarity calculation failed: {e}")
        # Return fallback with 0 similarity
        scores = [{"adr": adr, "similarity": 0.0} for adr in terms]
        return pd.DataFrame(scores)


# --- Asynchronous Decoupled Client Functions ---

async def analyze_and_deidentify_async(text: str, method: str = "mask", lang: str = "en") -> Dict[str, Any]:
    """
    Asynchronously calls the FastAPI backend to de-identify text.
    Falls back to local processing if the API is unavailable.
    """
    url = f"{NLP_API_URL}/api/v1/deidentify"
    payload = {"text": text, "method": method, "lang": lang}
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"FastAPI NLP backend returned status code {response.status_code}. Falling back to local inference.")
    except Exception as e:
        logger.warning(f"Failed to connect to FastAPI NLP backend: {e}. Falling back to local inference.")
    
    return _analyze_and_deidentify_local(text, method, lang)


async def calculate_adr_similarity_async(
    text: str,
    adrs: Any = None,
    model_id: Optional[str] = None,
    lang: str = "en"
) -> pd.DataFrame:
    """
    Asynchronously calls the FastAPI backend to calculate ADR similarity.
    Falls back to local processing if the API is unavailable.
    """
    if isinstance(adrs, str):
        model_id = adrs
        adrs = None

    url = f"{NLP_API_URL}/api/v1/similarity"
    payload = {
        "text": text,
        "adrs": adrs,
        "model_id": model_id,
        "lang": lang
    }
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                records = response.json()
                return pd.DataFrame(records)
            else:
                logger.warning(f"FastAPI NLP backend similarity returned status code {response.status_code}. Falling back to local inference.")
    except Exception as e:
        logger.warning(f"Failed to connect to FastAPI NLP backend similarity: {e}. Falling back to local inference.")

    return _calculate_adr_similarity_local(text, adrs, model_id, lang)


# --- Synchronous Client Wrapper Functions ---

def analyze_and_deidentify(text: str, method: str = "mask", lang: str = "en") -> Dict[str, Any]:
    """
    De-identifies the clinical narrative text using the decoupled API (or local fallback).
    """
    url = f"{NLP_API_URL}/api/v1/deidentify"
    payload = {"text": text, "method": method, "lang": lang}
    try:
        response = requests.post(url, json=payload, timeout=1.0)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"FastAPI NLP backend returned status code {response.status_code}. Falling back to local inference.")
    except Exception as e:
        logger.warning(f"Failed to connect to FastAPI NLP backend: {e}. Falling back to local inference.")
    
    return _analyze_and_deidentify_local(text, method, lang)


def calculate_adr_similarity(
    text: str,
    adrs: Any = None,
    model_id: Optional[str] = None,
    lang: str = "en"
) -> pd.DataFrame:
    """
    Calculates semantic similarity between a text narrative and safety cohort ADR terms.
    """
    if isinstance(adrs, str):
        model_id = adrs
        adrs = None

    url = f"{NLP_API_URL}/api/v1/similarity"
    payload = {
        "text": text,
        "adrs": adrs,
        "model_id": model_id,
        "lang": lang
    }
    try:
        response = requests.post(url, json=payload, timeout=1.0)
        if response.status_code == 200:
            records = response.json()
            return pd.DataFrame(records)
        else:
            logger.warning(f"FastAPI NLP backend similarity returned status code {response.status_code}. Falling back to local inference.")
    except Exception as e:
        logger.warning(f"Failed to connect to FastAPI NLP backend similarity: {e}. Falling back to local inference.")

    return _calculate_adr_similarity_local(text, adrs, model_id, lang)

