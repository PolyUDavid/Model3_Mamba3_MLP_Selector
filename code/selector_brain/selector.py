#!/usr/bin/env python3
"""
Selector Brain — Intelligent Dual-Path Orchestrator (Port 8000)
================================================================

The "Smart Brain" that sits between MLP and Mamba services.

Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │                    Selector Brain (8000)                  │
  │                                                          │
  │  Input → PhysicsAnalyzer → Decision Logic → Output       │
  │              │                    │                       │
  │              ▼                    ▼                       │
  │     ┌──────────────┐   ┌──────────────────┐              │
  │     │ MLP Service  │   │ Mamba Service     │              │
  │     │   (8001)     │   │   (8002)          │              │
  │     │ Fast Primary │   │ Physics Backup    │              │
  │     └──────────────┘   └──────────────────┘              │
  └──────────────────────────────────────────────────────────┘

Decision Logic:
  1. ALWAYS call MLP for fast baseline prediction
  2. ALWAYS call Mamba (it's always running, collecting data)
  3. PhysicsAnalyzer detects extreme conditions:
     - Heavy rain (weather ≥ 2) → Mamba knows 6.97-9.73 dB atten
     - Long distance (>500m) → Mamba tracks correct path-loss slope
     - Ultra-dense urban (density ≥ 2) → Mamba captures obstruction
     - High interference (≥3) → Mamba models accumulation
     - Divergence detected: |MLP - Mamba| > threshold
  4. If extreme → USE Mamba prediction
     If normal  → USE MLP prediction (Mamba just observes)

Author: NOK KO
"""

import torch
import numpy as np
import time
import json
import httpx
import asyncio
from pathlib import Path
from collections import deque
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn

# ============================================================
# Configuration
# ============================================================
MLP_URL = "http://localhost:8001"
MAMBA_URL = "http://localhost:8002"

# Physics thresholds
DISTANCE_THRESHOLD = 500.0      # meters
WEATHER_THRESHOLD = 2           # moderate rain or worse
DENSITY_THRESHOLD = 2           # urban or denser
INTERFERENCE_THRESHOLD = 3      # co-channel count
DIVERGENCE_THRESHOLD_DB = 5.0   # dB difference between MLP/Mamba power

# ============================================================
# Decision History
# ============================================================
decision_history = deque(maxlen=10000)
stats = {
    'total_requests': 0,
    'mlp_decisions': 0,
    'mamba_decisions': 0,
    'mamba_activations': [],
    'avg_mlp_latency_ms': 0,
    'avg_mamba_latency_ms': 0,
}


# ============================================================
# Physics Analyzer
# ============================================================
class PhysicsAnalyzer:
    """
    Analyzes input conditions to determine if extreme physics
    scenario requires Mamba's physics-aware predictions.
    """
    
    @staticmethod
    def analyze(input_data: dict) -> dict:
        """
        Returns analysis with trigger score and reasons.
        
        trigger_score: 0.0 = normal, 1.0 = extreme
        
        Data-driven calibration (from Exp 4 results):
          Mamba beats MLP ONLY on extreme_distance (>500m) scenarios.
          For weather/density extremes, MLP remains superior.
          Strategy: Activate Mamba primarily for long-distance + 
          compound scenarios where physics extrapolation matters.
        """
        reasons = []
        score = 0.0
        
        distance = input_data.get('distance_to_rx_m', 0)
        weather = input_data.get('weather_condition', 0)
        density = input_data.get('building_density', 0)
        intf = input_data.get('num_interferers', 0)
        
        # PRIMARY trigger: Long distance (where Mamba genuinely outperforms MLP)
        if distance > 700:
            score += 0.35
            reasons.append(f"Very Long Range ({distance:.0f}m): Mamba tracks Friis slope better")
        elif distance > 500:
            score += 0.20
            reasons.append(f"Long Range ({distance:.0f}m): path-loss extrapolation zone")
        
        # SECONDARY trigger: Compound distance + weather (physics coupling)
        if distance > 500 and weather >= 2:
            score += 0.20
            reasons.append(f"Distance+Rain coupling: rain atten + path-loss compound effect")
        elif weather >= 3:
            score += 0.10
            reasons.append(f"Heavy Rain (weather={weather}): ITU-R rain attenuation ~8dB")
        
        # TERTIARY: Compound distance + density (propagation + obstruction)
        if distance > 500 and density >= 2:
            score += 0.15
            reasons.append(f"Distance+Urban coupling: multi-path propagation physics")
        
        # High interference at range
        if intf >= INTERFERENCE_THRESHOLD and distance > 400:
            score += 0.10
            reasons.append(f"Range+Interference: {intf} co-channel at {distance:.0f}m")
        
        # Triple compound (very rare, very extreme)
        n_risk_factors = sum([weather >= 2, distance > 500, density >= 2, intf >= 3])
        if n_risk_factors >= 3:
            score += 0.15
            reasons.append(f"Triple Compound: {n_risk_factors} extreme factors — physics critical")
        
        score = min(score, 1.0)
        
        return {
            'trigger_score': round(score, 3),
            'use_mamba': score >= 0.3,
            'reasons': reasons,
            'risk_level': 'EXTREME' if score >= 0.5 else 'ELEVATED' if score >= 0.3 else 'NORMAL',
            'n_risk_factors': n_risk_factors,
        }
    
    @staticmethod
    def check_divergence(mlp_result: dict, mamba_result: dict) -> dict:
        """Check if MLP and Mamba predictions diverge significantly."""
        power_diff = abs(mlp_result['received_power_dbm'] - mamba_result['received_power_dbm'])
        sinr_diff = abs(mlp_result['sinr_db'] - mamba_result['sinr_db'])
        radius_diff = abs(mlp_result['coverage_radius_m'] - mamba_result['coverage_radius_m'])
        
        divergent = power_diff > DIVERGENCE_THRESHOLD_DB or sinr_diff > DIVERGENCE_THRESHOLD_DB
        
        return {
            'power_diff_dB': round(power_diff, 2),
            'sinr_diff_dB': round(sinr_diff, 2),
            'radius_diff_m': round(radius_diff, 2),
            'divergent': divergent,
            'note': f"Power Δ={power_diff:.1f}dB" if divergent else "Models agree"
        }


# ============================================================
# FastAPI
# ============================================================
app = FastAPI(
    title="Selector Brain — Dual-Path Orchestrator",
    description="Intelligent routing between MLP (fast) and Mamba (physics-aware)",
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

class DualPredictionResult(BaseModel):
    # Final decision
    received_power_dbm: float
    sinr_db: float
    coverage_radius_m: float
    coverage_area_km2: float
    qos_score: float
    
    # Decision metadata
    selected_model: str
    trigger_score: float
    risk_level: str
    reasons: List[str]
    
    # Both model outputs (for transparency)
    mlp_prediction: dict
    mamba_prediction: dict
    divergence: dict
    
    # Timing
    total_inference_ms: float
    mlp_latency_ms: float
    mamba_latency_ms: float


async def call_mlp(input_data: dict) -> dict:
    """Call MLP service."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(f"{MLP_URL}/predict", json=input_data)
        return resp.json()


async def call_mamba(input_data: dict) -> dict:
    """Call Mamba service."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{MAMBA_URL}/predict", json=input_data)
        return resp.json()


@app.get("/health")
async def health():
    """Check health of all services."""
    results = {"selector": "healthy"}
    
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            r = await client.get(f"{MLP_URL}/health")
            results["mlp_service"] = r.json()
        except Exception as e:
            results["mlp_service"] = {"status": "unavailable", "error": str(e)}
        
        try:
            r = await client.get(f"{MAMBA_URL}/health")
            results["mamba_service"] = r.json()
        except Exception as e:
            results["mamba_service"] = {"status": "unavailable", "error": str(e)}
    
    results["stats"] = {
        'total_requests': stats['total_requests'],
        'mlp_decisions': stats['mlp_decisions'],
        'mamba_decisions': stats['mamba_decisions'],
        'mamba_activation_rate': f"{stats['mamba_decisions'] / max(stats['total_requests'], 1) * 100:.1f}%",
    }
    
    return results


@app.post("/predict", response_model=DualPredictionResult)
async def predict(inp: CoverageInput):
    """
    Main prediction endpoint.
    
    1. Analyze physics conditions
    2. Call BOTH MLP and Mamba in parallel
    3. Decide which output to use
    4. Return combined result with full transparency
    """
    t_start = time.perf_counter()
    input_data = inp.model_dump()
    
    # Step 1: Physics analysis
    analysis = PhysicsAnalyzer.analyze(input_data)
    
    # Step 2: Call both services in parallel (Mamba is always running)
    try:
        mlp_result, mamba_result = await asyncio.gather(
            call_mlp(input_data),
            call_mamba(input_data)
        )
    except Exception as e:
        # Fallback: if one service is down, use the other
        try:
            mlp_result = await call_mlp(input_data)
            mamba_result = {"received_power_dbm": 0, "sinr_db": 0,
                          "coverage_radius_m": 0, "coverage_area_km2": 0,
                          "qos_score": 0, "inference_time_ms": 0,
                          "model": "Mamba-3 (UNAVAILABLE)"}
        except:
            mamba_result = await call_mamba(input_data)
            mlp_result = {"received_power_dbm": 0, "sinr_db": 0,
                         "coverage_radius_m": 0, "coverage_area_km2": 0,
                         "qos_score": 0, "inference_time_ms": 0,
                         "model": "MLP (UNAVAILABLE)"}
    
    # Step 3: Check divergence
    divergence = PhysicsAnalyzer.check_divergence(mlp_result, mamba_result)
    
    # Step 4: Make decision
    # Only use divergence as tiebreaker when there's already some risk
    use_mamba = analysis['use_mamba'] or (divergence['divergent'] and analysis['trigger_score'] >= 0.15)
    
    if use_mamba:
        selected = mamba_result
        selected_model = "Mamba-3 (Physics Backup ACTIVATED)"
        stats['mamba_decisions'] += 1
        stats['mamba_activations'].append({
            'time': time.time(),
            'reasons': analysis['reasons'],
            'trigger_score': analysis['trigger_score']
        })
    else:
        selected = mlp_result
        selected_model = "MLP (Primary)"
        stats['mlp_decisions'] += 1
    
    stats['total_requests'] += 1
    
    t_total = (time.perf_counter() - t_start) * 1000
    
    # Record decision
    decision_history.append({
        'timestamp': time.time(),
        'input': input_data,
        'selected_model': 'mamba' if use_mamba else 'mlp',
        'trigger_score': analysis['trigger_score'],
        'risk_level': analysis['risk_level'],
        'reasons': analysis['reasons'],
        'divergence': divergence,
    })
    
    return DualPredictionResult(
        received_power_dbm=selected['received_power_dbm'],
        sinr_db=selected['sinr_db'],
        coverage_radius_m=selected['coverage_radius_m'],
        coverage_area_km2=selected['coverage_area_km2'],
        qos_score=selected['qos_score'],
        selected_model=selected_model,
        trigger_score=analysis['trigger_score'],
        risk_level=analysis['risk_level'],
        reasons=analysis['reasons'],
        mlp_prediction=mlp_result,
        mamba_prediction=mamba_result,
        divergence=divergence,
        total_inference_ms=round(t_total, 2),
        mlp_latency_ms=mlp_result.get('inference_time_ms', 0),
        mamba_latency_ms=mamba_result.get('inference_time_ms', 0),
    )


@app.post("/predict/batch")
async def predict_batch(samples: List[CoverageInput]):
    """Batch prediction with per-sample routing."""
    results = []
    for inp in samples:
        r = await predict(inp)
        results.append(r)
    
    mamba_count = sum(1 for r in results if 'Mamba' in r.selected_model)
    return {
        "predictions": results,
        "count": len(results),
        "mamba_activations": mamba_count,
        "mlp_decisions": len(results) - mamba_count,
    }


@app.get("/stats")
def get_stats():
    """Get decision statistics."""
    total = max(stats['total_requests'], 1)
    return {
        'total_requests': stats['total_requests'],
        'mlp_decisions': stats['mlp_decisions'],
        'mamba_decisions': stats['mamba_decisions'],
        'mamba_activation_rate': f"{stats['mamba_decisions'] / total * 100:.1f}%",
        'recent_activations': stats['mamba_activations'][-10:] if stats['mamba_activations'] else [],
    }


@app.get("/decisions/history")
def get_history(limit: int = 50):
    """Get recent decision history."""
    return {
        "decisions": list(decision_history)[-limit:],
        "total": len(decision_history)
    }


@app.get("/architecture")
def architecture():
    """Describe the Dual-Path architecture."""
    return {
        "name": "Physics-Aware Dual-Path Coverage Predictor (D²TL)",
        "design": "Microservice Orchestrator Pattern",
        "components": {
            "selector_brain": {
                "port": 8000,
                "role": "Intelligent orchestrator — routes requests based on physics analysis",
                "decision_logic": "PhysicsAnalyzer detects extreme scenarios → activates Mamba backup"
            },
            "mlp_service": {
                "port": 8001,
                "role": "Primary fast-path — handles >85% of predictions",
                "latency": "~7ms",
                "strength": "Fast, accurate for normal conditions (R²≈0.934)"
            },
            "mamba_service": {
                "port": 8002,
                "role": "Physics-aware backup — always running, collecting data",
                "latency": "~411ms",
                "strength": "Correct physics: path-loss slope, rain atten 6.97-9.73 dB (theory 8dB)"
            }
        },
        "trigger_conditions": {
            "heavy_rain": "weather ≥ 2 (≥5 dB attenuation)",
            "long_distance": ">500m (Friis extrapolation)",
            "dense_urban": "density ≥ 2 (obstruction physics)",
            "high_interference": "≥3 co-channel interferers",
            "model_divergence": f"|MLP - Mamba| > {DIVERGENCE_THRESHOLD_DB} dB"
        },
        "key_insight": (
            "Mamba's state-space parameters (A,B,C,D) inherently model physical "
            "system dynamics. When MLP extrapolates poorly under extreme conditions "
            "(heavy rain, long range), Mamba provides physically consistent predictions "
            "because it has learned the correct Friis path-loss exponents and "
            "ITU-R rain attenuation coefficients."
        )
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
