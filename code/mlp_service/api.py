#!/usr/bin/env python3
"""
MLP Coverage API — Primary Decision Service (Port 8001)
========================================================

FastAPI microservice for the MLP primary path.
Always-on, fast inference (~7ms), handles >95% of predictions.

Endpoints:
  POST /predict        → Single prediction
  POST /predict/batch  → Batch prediction
  GET  /health         → Service health
  GET  /metrics        → Model metrics

Author: NOK KO
"""

import torch
import numpy as np
import time
import json
import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Setup
BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent.parent  # Model_3_Coverage_Mamba3/
sys.path.insert(0, str(BASE_DIR))
from model import CoverageMLP

# ============================================================
# Load Model
# ============================================================
device = torch.device('cpu')  # CPU for lowest latency

model = CoverageMLP(input_dim=13)
ckpt_path = BASE_DIR / 'best_mlp_coverage.pth'

if ckpt_path.exists():
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    feature_stats = ckpt.get('feature_stats', None)
    model_metrics = ckpt.get('val_metrics', {})
    print(f"[MLP Service] Loaded checkpoint: R²={ckpt.get('r2', 'N/A')}")
else:
    feature_stats = None
    model_metrics = {}
    # Try to get feature stats from Mamba checkpoint
    mamba_ckpt_path = PARENT_DIR / 'training' / 'best_coverage.pth'
    if mamba_ckpt_path.exists():
        mamba_ckpt = torch.load(str(mamba_ckpt_path), weights_only=False, map_location='cpu')
        feature_stats = mamba_ckpt.get('feature_stats', None)
    print(f"[MLP Service] No checkpoint found — model untrained")

model.eval()

# ============================================================
# API
# ============================================================
app = FastAPI(
    title="MLP Coverage Service",
    description="Primary fast-path for 6G RSU coverage prediction",
    version="1.0"
)

class CoverageInput(BaseModel):
    rsu_x_position_m: float = 500.0
    rsu_y_position_m: float = 500.0
    tx_power_dbm: float = 33.0
    antenna_tilt_deg: float = 7.0
    antenna_azimuth_deg: float = 180.0
    distance_to_rx_m: float = 300.0
    angle_to_rx_deg: float = 90.0
    building_density: int = 1
    weather_condition: int = 0
    vehicle_density_per_km2: float = 25.0
    num_interferers: int = 0
    rx_height_m: float = 1.5
    frequency_ghz: float = 5.9

class BatchInput(BaseModel):
    samples: List[CoverageInput]

class PredictionResult(BaseModel):
    received_power_dbm: float
    sinr_db: float
    coverage_radius_m: float
    coverage_area_km2: float
    qos_score: float
    inference_time_ms: float
    model: str = "MLP"

class HealthResponse(BaseModel):
    status: str
    model: str
    parameters: int
    checkpoint_loaded: bool
    device: str


def features_from_input(inp: CoverageInput) -> tuple:
    """Extract raw + normalized features."""
    raw = torch.tensor([[
        inp.rsu_x_position_m, inp.rsu_y_position_m,
        inp.tx_power_dbm, inp.antenna_tilt_deg,
        inp.antenna_azimuth_deg, inp.distance_to_rx_m,
        inp.angle_to_rx_deg, inp.building_density,
        inp.weather_condition, inp.vehicle_density_per_km2,
        inp.num_interferers, inp.rx_height_m, inp.frequency_ghz
    ]], dtype=torch.float32)
    
    if feature_stats:
        mean = feature_stats['mean']
        std = feature_stats['std']
        if isinstance(mean, torch.Tensor):
            norm = (raw - mean) / (std + 1e-8)
        else:
            norm = (raw - torch.tensor(mean)) / (torch.tensor(std) + 1e-8)
    else:
        norm = raw  # fallback
    
    return raw, norm


def decode_prediction(pred: torch.Tensor) -> dict:
    """Reverse target scaling to physical units."""
    p = pred[0].cpu().tolist()
    return {
        'received_power_dbm': round(p[0] * 230 - 260, 2),
        'sinr_db': round(p[1] * 230 - 170, 2),
        'coverage_radius_m': round(p[2] * 90 + 150, 2),
        'coverage_area_km2': round(p[3] * 0.12 + 0.07, 4),
        'qos_score': round(p[4] * 100, 2),
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model="CoverageMLP",
        parameters=model.get_num_params(),
        checkpoint_loaded=ckpt_path.exists(),
        device=str(device)
    )


@app.post("/predict", response_model=PredictionResult)
def predict(inp: CoverageInput):
    raw, norm = features_from_input(inp)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(norm.to(device))
    elapsed = (time.perf_counter() - t0) * 1000
    
    result = decode_prediction(pred)
    return PredictionResult(
        **result,
        inference_time_ms=round(elapsed, 3),
        model="MLP"
    )


@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    results = []
    for inp in batch.samples:
        results.append(predict(inp))
    return {"predictions": results, "count": len(results)}


@app.get("/metrics")
def metrics():
    return {
        "model": "CoverageMLP",
        "parameters": model.get_num_params(),
        "val_metrics": {k: float(v) for k, v in model_metrics.items()} if model_metrics else {},
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else "not found"
    }


@app.post("/predict/raw")
def predict_raw(features: List[float]):
    """Direct prediction from normalized feature vector (for Selector Brain)."""
    t0 = time.perf_counter()
    x = torch.tensor([features], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x)
    elapsed = (time.perf_counter() - t0) * 1000
    
    result = decode_prediction(pred)
    result['inference_time_ms'] = round(elapsed, 3)
    result['raw_output'] = pred[0].cpu().tolist()
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
