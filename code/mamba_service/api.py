#!/usr/bin/env python3
"""
Mamba Coverage API — Physics-Aware Backup Service (Port 8002)
==============================================================

FastAPI microservice for the Mamba SSM backup path.
Always-on, continuously collecting data, activated when Selector
detects extreme physical scenarios.

Key Physics Learned by Mamba:
  - Distance-Power Decay: slope error Rural -1.4dB, Urban -0.5dB
  - Rain Attenuation: learned 6.97-9.73 dB (theory 8 dB)
  - Urban Density: correct obstruction path-loss scaling

Endpoints:
  POST /predict        → Single prediction
  POST /predict/batch  → Batch prediction
  GET  /health         → Service health
  GET  /physics        → Physics consistency report
  GET  /data/buffer    → Get collected data buffer status

Author: NOK KO
"""

import torch
import numpy as np
import time
import json
import sys
from pathlib import Path
from collections import deque
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Setup
BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent.parent  # Model_3_Coverage_Mamba3/
sys.path.insert(0, str(PARENT_DIR / 'models'))
from mamba3_coverage import CoverageMamba3

# ============================================================
# Load Model
# ============================================================
device = torch.device('cpu')

model = CoverageMamba3(input_dim=13)
ckpt_path = PARENT_DIR / 'training' / 'best_coverage.pth'

if ckpt_path.exists():
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    
    feature_stats = ckpt.get('feature_stats', None)
    mamba_r2 = ckpt.get('r2', 'N/A')
    val_metrics = ckpt.get('val_metrics', {})
    print(f"[Mamba Service] Loaded: epoch {ckpt.get('epoch')}, R²={mamba_r2}")
else:
    feature_stats = None
    mamba_r2 = 'N/A'
    val_metrics = {}
    print("[Mamba Service] WARNING: No checkpoint found")

model.eval()

# Data collection buffer (Mamba continuously observes)
data_buffer = deque(maxlen=10000)
activation_log = deque(maxlen=1000)

# ============================================================
# API
# ============================================================
app = FastAPI(
    title="Mamba Coverage Service",
    description="Physics-aware backup for 6G RSU coverage (SSM backbone)",
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
    model: str = "Mamba-3"
    physics_note: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model: str
    parameters: int
    checkpoint_loaded: bool
    best_r2: float
    data_buffer_size: int
    activation_count: int


def features_from_input(inp: CoverageInput) -> tuple:
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
        norm = raw
    
    return raw, norm


def decode_prediction(pred: torch.Tensor) -> dict:
    p = pred[0].cpu().tolist()
    return {
        'received_power_dbm': round(p[0] * 230 - 260, 2),
        'sinr_db': round(p[1] * 230 - 170, 2),
        'coverage_radius_m': round(p[2] * 90 + 150, 2),
        'coverage_area_km2': round(p[3] * 0.12 + 0.07, 4),
        'qos_score': round(p[4] * 100, 2),
    }


def detect_physics_note(inp: CoverageInput) -> Optional[str]:
    """Generate physics insight note for this prediction."""
    notes = []
    if inp.weather_condition >= 2:
        atten = {2: '~5dB', 3: '~8dB'}
        notes.append(f"Rain atten: {atten.get(inp.weather_condition, 'high')}")
    if inp.distance_to_rx_m > 500:
        notes.append(f"Long-range ({inp.distance_to_rx_m:.0f}m): path-loss dominant")
    if inp.building_density >= 2:
        density_name = {2: 'Urban', 3: 'Ultra-Dense'}
        notes.append(f"{density_name.get(inp.building_density, 'Dense')}: obstruction scaling")
    if inp.num_interferers >= 3:
        notes.append(f"High interference: {inp.num_interferers} co-channel")
    return " | ".join(notes) if notes else None


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model="CoverageMamba3",
        parameters=model.get_num_params(),
        checkpoint_loaded=ckpt_path.exists(),
        best_r2=float(mamba_r2) if isinstance(mamba_r2, (int, float)) else 0.0,
        data_buffer_size=len(data_buffer),
        activation_count=len(activation_log)
    )


@app.post("/predict", response_model=PredictionResult)
def predict(inp: CoverageInput):
    raw, norm = features_from_input(inp)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(norm.to(device))
    elapsed = (time.perf_counter() - t0) * 1000
    
    result = decode_prediction(pred)
    physics_note = detect_physics_note(inp)
    
    # Log to data buffer (Mamba continuously observes)
    data_buffer.append({
        'timestamp': time.time(),
        'input': inp.model_dump(),
        'output': result,
        'physics_note': physics_note,
    })
    
    return PredictionResult(
        **result,
        inference_time_ms=round(elapsed, 3),
        model="Mamba-3",
        physics_note=physics_note
    )


@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    results = []
    for inp in batch.samples:
        results.append(predict(inp))
    return {"predictions": results, "count": len(results)}


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


@app.post("/activate")
def activate(reason: str = "extreme_scenario"):
    """Log an activation event (called by Selector Brain)."""
    activation_log.append({
        'timestamp': time.time(),
        'reason': reason
    })
    return {"status": "activated", "total_activations": len(activation_log)}


@app.get("/data/buffer")
def get_buffer():
    return {
        "buffer_size": len(data_buffer),
        "max_size": data_buffer.maxlen,
        "recent": list(data_buffer)[-5:] if data_buffer else []
    }


@app.get("/metrics")
def metrics():
    return {
        "model": "CoverageMamba3 (Mamba-3 SSM)",
        "parameters": model.get_num_params(),
        "best_r2": float(mamba_r2) if isinstance(mamba_r2, (int, float)) else None,
        "val_metrics": {k: float(v) for k, v in val_metrics.items()} if val_metrics else {},
        "physics_capabilities": {
            "friis_path_loss": "slope error Rural -1.4dB, Urban -0.5dB",
            "rain_attenuation": "learned 6.97-9.73 dB (theory 8 dB)",
            "density_scaling": "correct obstruction propagation"
        },
        "data_buffer_size": len(data_buffer),
        "activation_count": len(activation_log)
    }


@app.get("/physics")
def physics_report():
    """Report on physics consistency capabilities."""
    return {
        "model": "CoverageMamba3",
        "physics_validation": {
            "exp1_distance_power": {
                "description": "Friis path-loss curve tracking",
                "rural_slope_error_dB": -1.4,
                "urban_slope_error_dB": -0.5,
                "theory": "PL(d) = L0 + 10*n*log10(d)"
            },
            "exp2_rain_attenuation": {
                "description": "ITU-R rain attenuation modeling",
                "learned_range_dB": "6.97 - 9.73",
                "theory_dB": 8.0,
                "note": "Mamba SSM state captures cumulative rain effect"
            },
            "exp3_density_gradient": {
                "description": "Urban obstruction path-loss scaling",
                "learned": "Correct density multiplier on path-loss exponent",
                "advantage": "SSM memory retains propagation context"
            }
        },
        "why_mamba_for_physics": (
            "Mamba's state-space parameters (A, B, C, D) naturally model "
            "physical system dynamics. The discretized state transition "
            "A_bar = exp(delta*A) mirrors electromagnetic propagation decay, "
            "making it inherently suited for learning correct path-loss "
            "exponents and attenuation coefficients from data."
        )
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
