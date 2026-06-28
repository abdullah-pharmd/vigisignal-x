import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from core.nlp_engine import (
    analyze_and_deidentify,
    calculate_adr_similarity
)

app = FastAPI(
    title="VigiSignal-X Clinical NLP API",
    description="Asynchronous deep learning inference backend for clinical note deidentification and ADR semantic similarity.",
    version="1.0.0"
)

class DeidentifyRequest(BaseModel):
    text: str
    method: str = "mask"
    lang: str = "en"

class DeidentifyResponse(BaseModel):
    original_text: str
    deidentified_text: str
    pii_entities: List[Dict[str, Any]]
    clinical_entities: List[Dict[str, Any]]

class SimilarityRequest(BaseModel):
    text: str
    adrs: Optional[List[str]] = None
    model_id: Optional[str] = None
    lang: str = "en"

class SimilarityItem(BaseModel):
    adr: str
    similarity: float

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "vigisignal-nlp-api"}

@app.post("/api/v1/deidentify", response_model=DeidentifyResponse)
async def deidentify(payload: DeidentifyRequest):
    try:
        # Run the CPU/GPU heavy openmed processing
        res = analyze_and_deidentify(
            text=payload.text,
            method=payload.method,
            lang=payload.lang
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PII deidentification failed: {str(e)}")

@app.post("/api/v1/similarity", response_model=List[SimilarityItem])
async def similarity(payload: SimilarityRequest):
    try:
        # Run similarity calculation
        df = calculate_adr_similarity(
            text=payload.text,
            adrs=payload.adrs,
            model_id=payload.model_id,
            lang=payload.lang
        )
        # Convert df to list of dicts
        if df.empty:
            return []
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ADR similarity computation failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("NLP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
