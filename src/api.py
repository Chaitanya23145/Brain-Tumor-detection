"""
Production REST API for Brain Tumor Detection
Built with FastAPI, Pydantic v2 schemas, and clinical response specifications.
"""
import os
import io
import time
import base64
import json
from contextlib import asynccontextmanager
from typing import List, Optional, Dict
from PIL import Image
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model import get_model, predict_mri, DEFAULT_WEIGHTS_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm model singleton on startup."""
    print("[API] Initializing and pre-warming model...")
    get_model()
    yield

# FastAPI App Instance
api_app = FastAPI(
    title="Clinical Neuro-Oncology Brain Tumor Detection API",
    description="Deep Transfer Learning (VGG19) REST microservice with morphological skull-stripping and Grad-CAM interpretability.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for cross-origin clinical integrations
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


class PredictionResponse(BaseModel):
    filename: str = Field(..., description="Original upload filename")
    label: str = Field(..., description="Diagnostic classification label")
    is_tumor: bool = Field(..., description="True if neoplastic mass detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    confidence_percentage: float = Field(..., description="Confidence represented as percentage")
    probabilities: Dict[str, float] = Field(..., description="Softmax posterior probabilities")
    gradcam_base64: Optional[str] = Field(None, description="Base64 encoded Grad-CAM overlay image (PNG)")
    processing_time_ms: float = Field(..., description="Inference latency in milliseconds")
    model_status: str = Field(..., description="Status of neural network weights checkpoint")


class ModelInfoResponse(BaseModel):
    architecture: str
    backbone: str
    input_resolution: str
    num_classes: int
    weights_path: str
    weights_loaded: bool
    benchmarks: Optional[Dict] = None





@api_app.get("/health", tags=["System"])
async def health_check():
    """Returns service health, model loading status, and system uptime."""
    _, weights_loaded = get_model()
    return {
        "status": "healthy",
        "service": "Brain Tumor Detection Engine",
        "version": "2.0.0",
        "weights_loaded": weights_loaded,
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }


@api_app.get("/api/info", response_model=ModelInfoResponse, tags=["Metadata"])
async def model_info():
    """Returns deep neural network specifications and holdout validation benchmarks."""
    _, weights_loaded = get_model()
    
    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model_weights",
        "metrics.json"
    )
    benchmarks = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                benchmarks = json.load(f)
        except Exception:
            pass

    return ModelInfoResponse(
        architecture="VGG19 Deep Transfer Learning with Custom Dense Classification Head",
        backbone="VGG19 (Simonyan & Zisserman)",
        input_resolution="120x120x3 (RGB)",
        num_classes=2,
        weights_path=DEFAULT_WEIGHTS_PATH,
        weights_loaded=weights_loaded,
        benchmarks=benchmarks
    )


@api_app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(
    file: UploadFile = File(..., description="Axial brain MRI image file (JPEG, PNG)"),
    include_gradcam: bool = Query(True, description="Whether to generate Grad-CAM interpretability overlay")
):
    """
    Classifies a cranial MRI scan and detects neoplastic lesions.
    Executes automated skull-stripping and outputs posterior probabilities + Grad-CAM heatmap.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file '{file.filename}' is not a valid image format. Received content-type: {file.content_type}"
        )

    t0 = time.time()
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

    try:
        result = predict_mri(image_np, include_gradcam=include_gradcam)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    latency_ms = round((time.time() - t0) * 1000, 2)

    gradcam_b64 = None
    if include_gradcam and result.get("gradcam_image") is not None:
        pil_overlay = Image.fromarray(result["gradcam_image"])
        buffered = io.BytesIO()
        pil_overlay.save(buffered, format="PNG")
        gradcam_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    return PredictionResponse(
        filename=file.filename,
        label=result["label"],
        is_tumor=result["is_tumor"],
        confidence=result["confidence"],
        confidence_percentage=result["confidence_percentage"],
        probabilities=result["probabilities"],
        gradcam_base64=gradcam_b64,
        processing_time_ms=latency_ms,
        model_status=result["status"]
    )


@api_app.post("/api/predict-batch", response_model=List[PredictionResponse], tags=["Inference"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Batch of MRI scans"),
    include_gradcam: bool = Query(False, description="Generate Grad-CAM heatmaps for batch")
):
    """Processes multiple MRI scans in batch mode."""
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images allowed per batch request.")

    responses = []
    for file in files:
        res = await predict_single(file, include_gradcam=include_gradcam)
        responses.append(res)
    return responses
