#!/usr/bin/env python3
"""
D²TL Pygame Simulation — Live Dual-Path Visualization
=======================================================

Real-time visualization of a vehicle driving through a city,
showing how the Selector Brain switches between MLP (blue) and
Mamba (red) based on physics conditions.

- By default uses local MLP + Mamba checkpoints (same weights as API).
- With --api or D2TL_USE_API=1, calls real Selector API at http://localhost:8000
  for predictions (requires services running).

Features:
  - English UI with real-time data (Power, QoS, Trigger, conditions, statistics)
  - Vehicle drives along a road with changing conditions
  - Weather / urban density zones; RSU coverage circles
  - Color-coded: Blue = MLP active, Red = Mamba active

Controls:
  SPACE: Pause/Resume   R: Reset   1-4: Weather   0: Auto   ESC: Quit

Author: NOK KO
"""

import pygame
import sys
import math
import time
import random
import numpy as np
import torch
from pathlib import Path
import os
import argparse

# Optional: use real API (Selector at 8000)
USE_REAL_API = os.environ.get('D2TL_USE_API', '').lower() in ('1', 'true', 'yes')
if '--api' in sys.argv:
    USE_REAL_API = True
    sys.argv.remove('--api')

# Setup model paths (for local mode)
BASE_DIR = Path(__file__).parent.parent.parent  # Model_3_Coverage_Mamba3/
sys.path.insert(0, str(BASE_DIR / 'models'))
sys.path.insert(0, str(BASE_DIR / 'd2tl' / 'mlp_service'))

from mamba3_coverage import CoverageMamba3
from model import CoverageMLP


# ============================================================
# Text rendering: English UI + real-time data (works when pygame.font fails)
# ============================================================
def _make_text_font(size, bold=False):
    """Return a font-like object with .render(text, antialias, color) -> Surface."""
    import pygame as _pg
    try:
        f = _pg.font.SysFont('Arial', size, bold=bold)
        return f
    except Exception:
        pass
    try:
        import pygame.freetype
        pygame.freetype.init()
        ft_font = pygame.freetype.SysFont('Arial', size)
        if bold and hasattr(ft_font, 'strong'):
            ft_font.strong = True

        class FreetypeWrapper:
            def render(self, text, antialias, color):
                c = (color[0], color[1], color[2], 255) if len(color) == 3 else color
                surf, _ = ft_font.render(text, c, size=size)
                return surf
        return FreetypeWrapper()
    except Exception:
        pass
    try:
        from PIL import Image, ImageDraw, ImageFont
        try:
            pil_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except Exception:
            try:
                pil_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except Exception:
                pil_font = ImageFont.load_default()

        class PILWrapper:
            def render(self, text, antialias, color):
                rgb = (color[0], color[1], color[2]) if len(color) >= 3 else (0, 0, 0)
                bbox = pil_font.getbbox(text)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                w, h = max(w, 1), max(h, 1)
                img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.text((0, 0), text, font=pil_font, fill=(*rgb, 255))
                data = img.tobytes()
                surf = _pg.image.frombuffer(data, (w, h), 'RGBA')
                return surf
        return PILWrapper()
    except Exception:
        class DummyFont:
            def render(self, text, antialias, color):
                w = min(len(text) * (size // 2) + 4, 800)
                s = _pg.Surface((w, size + 4))
                s.fill((255, 255, 255))
                return s
        return DummyFont()

# ============================================================
# Physics Constants
# ============================================================
CARRIER_FREQ_GHZ = 5.9
TX_POWER_DBM = 33.0
PATH_LOSS_EXPONENT = 3.5
REFERENCE_DISTANCE_M = 1.0
WAVELENGTH_M = 3e8 / (CARRIER_FREQ_GHZ * 1e9)
L0_DB = 20 * np.log10(4 * np.pi * REFERENCE_DISTANCE_M / WAVELENGTH_M)
WEATHER_ATTEN = {0: 0.0, 1: 2.0, 2: 5.0, 3: 8.0}
WEATHER_NAMES = {0: 'Clear', 1: 'Light Rain', 2: 'Moderate Rain', 3: 'Heavy Rain'}
DENSITY_OBSTACLE = {0: 1.0, 1: 1.2, 2: 1.5, 3: 2.0}
DENSITY_NAMES = {0: 'Rural', 1: 'Suburban', 2: 'Urban', 3: 'Ultra-Dense'}
ANTENNA_GAIN = 10.0 * 0.7

# ============================================================
# Colors
# ============================================================
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (80, 80, 80)
BLUE = (33, 150, 243)
RED = (244, 67, 54)
GREEN = (76, 175, 80)
ORANGE = (255, 152, 0)
YELLOW = (255, 235, 59)
LIGHT_BLUE = (187, 222, 251)
LIGHT_RED = (255, 205, 210)
DARK_BLUE = (21, 101, 192)
RAIN_BLUE = (100, 149, 237)

# ============================================================
# Window Layout
# ============================================================
WINDOW_W = 1400
WINDOW_H = 800
MAP_W = 900
MAP_H = 600
PANEL_X = MAP_W + 20
PANEL_W = WINDOW_W - MAP_W - 40
INFO_Y = MAP_H + 20


# ============================================================
# Load Models
# ============================================================
def load_models():
    mamba = CoverageMamba3(input_dim=13)
    ckpt_path = BASE_DIR / 'training' / 'best_coverage.pth'
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location='cpu')
    key = 'model_state_dict' if 'model_state_dict' in ckpt else 'model'
    mamba.load_state_dict(ckpt[key])
    mamba.eval()
    
    fstats = ckpt.get('feature_stats', None)
    
    mlp = CoverageMLP(input_dim=13)
    mlp_ckpt_path = BASE_DIR / 'd2tl' / 'mlp_service' / 'best_mlp_coverage.pth'
    if mlp_ckpt_path.exists():
        mlp_ckpt = torch.load(str(mlp_ckpt_path), weights_only=False, map_location='cpu')
        mlp.load_state_dict(mlp_ckpt['model_state_dict'])
    mlp.eval()
    
    return mamba, mlp, fstats


def normalize(raw, fstats):
    if fstats is None:
        return raw
    mean = fstats['mean']
    std = fstats['std']
    if isinstance(mean, torch.Tensor):
        return (raw - mean) / (std + 1e-8)
    return (raw - torch.tensor(mean)) / (torch.tensor(std) + 1e-8)


def selector_decision(weather, distance, density, n_intf):
    score = 0
    reasons = []
    
    # PRIMARY: Long distance
    if distance > 700:
        score += 0.35; reasons.append("Very Long Range")
    elif distance > 500:
        score += 0.20; reasons.append("Long Range")
    
    # SECONDARY: Compound distance + weather
    if distance > 500 and weather >= 2:
        score += 0.20; reasons.append("Distance+Rain Coupling")
    elif weather >= 3:
        score += 0.10; reasons.append("Heavy Rain")
    
    # TERTIARY: Compound distance + density
    if distance > 500 and density >= 2:
        score += 0.15; reasons.append("Distance+Urban Coupling")
    
    # High interference at range
    if n_intf >= 3 and distance > 400:
        score += 0.10; reasons.append("Range+Interference")
    
    # Triple compound
    n_factors = sum([weather >= 2, distance > 500, density >= 2, n_intf >= 3])
    if n_factors >= 3:
        score += 0.15; reasons.append("Triple Compound")
    
    return min(score, 1.0), score >= 0.3, reasons


def predict(mamba, mlp, fstats, distance, weather, density, n_intf=0):
    """Local inference (same weights as API)."""
    raw = torch.zeros(1, 13)
    raw[0, 0] = 500; raw[0, 1] = 500; raw[0, 2] = TX_POWER_DBM
    raw[0, 3] = 7; raw[0, 4] = 180; raw[0, 5] = distance
    raw[0, 6] = 90; raw[0, 7] = density; raw[0, 8] = weather
    raw[0, 9] = 25; raw[0, 10] = n_intf; raw[0, 11] = 1.5
    raw[0, 12] = CARRIER_FREQ_GHZ
    
    norm = normalize(raw, fstats)
    
    with torch.no_grad():
        y_mlp = mlp(norm)
        y_mamba = mamba(norm)
    
    score, use_mamba, reasons = selector_decision(weather, distance, density, n_intf)
    y_sel = y_mamba if use_mamba else y_mlp
    
    return {
        'mlp_power': y_mlp[0, 0].item() * 230 - 260,
        'mamba_power': y_mamba[0, 0].item() * 230 - 260,
        'selected_power': y_sel[0, 0].item() * 230 - 260,
        'mlp_qos': y_mlp[0, 4].item() * 100,
        'mamba_qos': y_mamba[0, 4].item() * 100,
        'selected_qos': y_sel[0, 4].item() * 100,
        'mlp_radius': y_mlp[0, 2].item() * 90 + 150,
        'mamba_radius': y_mamba[0, 2].item() * 90 + 150,
        'selected_radius': y_sel[0, 2].item() * 90 + 150,
        'trigger_score': score,
        'use_mamba': use_mamba,
        'reasons': reasons,
    }


SELECTOR_API_URL = "http://localhost:8000"


def predict_via_api(distance, weather, density, n_intf=0):
    """Call real Selector API (requires MLP 8001, Mamba 8002, Selector 8000 running)."""
    try:
        import urllib.request
        import json as _json
        body = {
            "rsu_x_position_m": 500.0, "rsu_y_position_m": 500.0,
            "tx_power_dbm": TX_POWER_DBM, "antenna_tilt_deg": 7.0, "antenna_azimuth_deg": 180.0,
            "distance_to_rx_m": float(distance), "angle_to_rx_deg": 90.0,
            "building_density": int(density), "weather_condition": int(weather),
            "vehicle_density_per_km2": 25.0, "num_interferers": int(n_intf),
            "rx_height_m": 1.5, "frequency_ghz": CARRIER_FREQ_GHZ,
        }
        req = urllib.request.Request(
            f"{SELECTOR_API_URL}/predict",
            data=_json.dumps(body).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            r = _json.loads(resp.read().decode())
        mlp_p = r.get("mlp_prediction", {})
        mamba_p = r.get("mamba_prediction", {})
        use_mamba = "Mamba" in r.get("selected_model", "")
        return {
            'mlp_power': mlp_p.get('received_power_dbm', 0),
            'mamba_power': mamba_p.get('received_power_dbm', 0),
            'selected_power': r.get('received_power_dbm', 0),
            'mlp_qos': mlp_p.get('qos_score', 0),
            'mamba_qos': mamba_p.get('qos_score', 0),
            'selected_qos': r.get('qos_score', 0),
            'mlp_radius': mlp_p.get('coverage_radius_m', 200),
            'mamba_radius': mamba_p.get('coverage_radius_m', 200),
            'selected_radius': r.get('coverage_radius_m', 200),
            'trigger_score': float(r.get('trigger_score', 0)),
            'use_mamba': use_mamba,
            'reasons': list(r.get('reasons', [])),
        }
    except Exception as e:
        return {
            'mlp_power': -90, 'mamba_power': -90, 'selected_power': -90,
            'mlp_qos': 0, 'mamba_qos': 0, 'selected_qos': 0,
            'mlp_radius': 150, 'mamba_radius': 150, 'selected_radius': 150,
            'trigger_score': 0, 'use_mamba': False,
            'reasons': [f"API error: {e}"],
        }


# ============================================================
# Main Simulation
# ============================================================
def main():
    mamba, mlp, fstats = None, None, None
    if USE_REAL_API:
        print("Using real Selector API at http://localhost:8000 (ensure services are running).")
    else:
        print("Loading local models...")
        mamba, mlp, fstats = load_models()
        print("Models loaded.")
    print("Starting simulation (English UI, real-time data)...")
    
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("D²TL: Physics-Aware Dual-Path Coverage Simulation")
    clock = pygame.time.Clock()
    
    # English UI: text fonts (fallback chain: pygame.font -> freetype -> PIL)
    font_title = _make_text_font(20, bold=True)
    font_label = _make_text_font(14, bold=True)
    font_small = _make_text_font(12)
    font_large = _make_text_font(28, bold=True)
    font_medium = _make_text_font(16)
    
    # RSU positions on map (18 points, spread across 900×600)
    rsus = [
        {'x': 80, 'y': 80, 'id': 'RSU-1'},   {'x': 280, 'y': 100, 'id': 'RSU-2'},  {'x': 480, 'y': 90, 'id': 'RSU-3'},
        {'x': 680, 'y': 120, 'id': 'RSU-4'}, {'x': 820, 'y': 80, 'id': 'RSU-5'},
        {'x': 120, 'y': 220, 'id': 'RSU-6'}, {'x': 320, 'y': 200, 'id': 'RSU-7'},  {'x': 520, 'y': 240, 'id': 'RSU-8'},
        {'x': 720, 'y': 220, 'id': 'RSU-9'}, {'x': 860, 'y': 260, 'id': 'RSU-10'},
        {'x': 100, 'y': 380, 'id': 'RSU-11'}, {'x': 300, 'y': 360, 'id': 'RSU-12'}, {'x': 500, 'y': 400, 'id': 'RSU-13'},
        {'x': 700, 'y': 380, 'id': 'RSU-14'}, {'x': 840, 'y': 420, 'id': 'RSU-15'},
        {'x': 260, 'y': 500, 'id': 'RSU-16'}, {'x': 540, 'y': 520, 'id': 'RSU-17'}, {'x': 760, 'y': 500, 'id': 'RSU-18'},
    ]
    
    # Vehicle state
    vehicle = {'x': 50.0, 'y': 350.0, 'speed': 2.0, 'angle': 0}
    
    # Path waypoints
    waypoints = [
        (50, 350), (200, 300), (350, 250), (500, 350),
        (650, 300), (800, 200), (850, 350), (700, 450),
        (500, 500), (300, 450), (150, 400), (50, 350)
    ]
    wp_idx = 0
    
    # Scenario timeline (distance-based)
    # [(start_x, weather, density, zone_name)]
    zones = [
        (0, 0, 0, "Rural Clear"),
        (200, 0, 1, "Suburban"),
        (400, 2, 2, "Urban Moderate Rain"),
        (550, 3, 3, "Ultra-Dense Heavy Rain"),
        (700, 1, 1, "Suburban Light Rain"),
        (850, 0, 0, "Rural Clear"),
    ]
    
    # History for plotting
    history = []
    decision_counts = {'mlp': 0, 'mamba': 0}
    
    paused = False
    running = True
    frame = 0
    override_weather = None
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    vehicle = {'x': 50.0, 'y': 350.0, 'speed': 2.0, 'angle': 0}
                    wp_idx = 0
                    history.clear()
                    decision_counts = {'mlp': 0, 'mamba': 0}
                    override_weather = None
                elif event.key == pygame.K_1:
                    override_weather = 0
                elif event.key == pygame.K_2:
                    override_weather = 1
                elif event.key == pygame.K_3:
                    override_weather = 2
                elif event.key == pygame.K_4:
                    override_weather = 3
                elif event.key == pygame.K_0:
                    override_weather = None
        
        if not paused:
            # Move vehicle along waypoints
            tx, ty = waypoints[(wp_idx + 1) % len(waypoints)]
            dx = tx - vehicle['x']
            dy = ty - vehicle['y']
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 5:
                wp_idx = (wp_idx + 1) % len(waypoints)
            else:
                vehicle['x'] += dx / dist * vehicle['speed']
                vehicle['y'] += dy / dist * vehicle['speed']
                vehicle['angle'] = math.degrees(math.atan2(dy, dx))
            
            # Determine zone
            vx = vehicle['x']
            current_zone = zones[0]
            for z in zones:
                if vx >= z[0]:
                    current_zone = z
            
            weather = override_weather if override_weather is not None else current_zone[1]
            density = current_zone[2]
            zone_name = current_zone[3]
            
            # Predict for nearest RSU
            nearest_rsu = min(rsus, key=lambda r: math.sqrt((r['x'] - vehicle['x'])**2 + (r['y'] - vehicle['y'])**2))
            dist_to_rsu = math.sqrt((nearest_rsu['x'] - vehicle['x'])**2 + (nearest_rsu['y'] - vehicle['y'])**2)
            # Scale to real-world distance (map pixels → meters, rough scale: 1px ≈ 2m)
            real_dist = max(dist_to_rsu * 2, 50)
            
            if USE_REAL_API:
                pred = predict_via_api(real_dist, weather, density)
            else:
                pred = predict(mamba, mlp, fstats, real_dist, weather, density)
            
            if pred['use_mamba']:
                decision_counts['mamba'] += 1
            else:
                decision_counts['mlp'] += 1
            
            history.append({
                'frame': frame,
                'x': vehicle['x'], 'y': vehicle['y'],
                'weather': weather, 'density': density,
                'zone': zone_name, 'pred': pred,
                'dist': real_dist,
            })
            if len(history) > 500:
                history.pop(0)
            
            frame += 1
        
        # ============ DRAW ============
        screen.fill(WHITE)
        
        # Map background
        pygame.draw.rect(screen, (240, 240, 240), (0, 0, MAP_W, MAP_H))
        
        # Draw zone backgrounds
        zone_colors = {
            0: (220, 245, 220),  # Rural green
            1: (230, 230, 240),  # Suburban
            2: (240, 220, 220),  # Urban
            3: (250, 200, 200),  # Ultra-dense
        }
        for i, z in enumerate(zones):
            x_start = z[0]
            x_end = zones[i+1][0] if i+1 < len(zones) else MAP_W
            color = zone_colors.get(z[2], (240, 240, 240))
            pygame.draw.rect(screen, color, (x_start, 0, x_end - x_start, MAP_H))
            # Zone label
            label = font_small.render(z[3], True, DARK_GRAY)
            screen.blit(label, (x_start + 5, 5))
        
        # Draw rain if applicable
        if history:
            cur = history[-1]
            w = cur['weather']
            if w >= 1:
                for _ in range(w * 30):
                    rx = random.randint(0, MAP_W)
                    ry = random.randint(0, MAP_H)
                    length = w * 3 + 2
                    alpha_color = (100, 149, 237) if w < 3 else (50, 80, 180)
                    pygame.draw.line(screen, alpha_color, (rx, ry), (rx + 1, ry + length), 1)
        
        # Draw RSUs
        for rsu in rsus:
            # Coverage circle
            if history:
                pred = history[-1]['pred']
                radius_px = int(pred['selected_radius'] / 2)  # scale
                color = RED if pred['use_mamba'] else BLUE
                surf = pygame.Surface((radius_px*2, radius_px*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*color, 30), (radius_px, radius_px), radius_px)
                screen.blit(surf, (rsu['x'] - radius_px, rsu['y'] - radius_px))
            
            pygame.draw.circle(screen, DARK_GRAY, (rsu['x'], rsu['y']), 12)
            pygame.draw.circle(screen, GREEN, (rsu['x'], rsu['y']), 8)
            label = font_small.render(rsu['id'], True, BLACK)
            screen.blit(label, (rsu['x'] - 15, rsu['y'] - 25))
        
        # Draw path trail
        if len(history) > 1:
            for i in range(1, len(history)):
                h = history[i]
                color = RED if h['pred']['use_mamba'] else BLUE
                pygame.draw.line(screen, color,
                               (int(history[i-1]['x']), int(history[i-1]['y'])),
                               (int(h['x']), int(h['y'])), 2)
        
        # Draw vehicle
        vx, vy = int(vehicle['x']), int(vehicle['y'])
        if history and history[-1]['pred']['use_mamba']:
            v_color = RED
        else:
            v_color = BLUE
        
        pygame.draw.circle(screen, v_color, (vx, vy), 8)
        pygame.draw.circle(screen, WHITE, (vx, vy), 5)
        pygame.draw.circle(screen, v_color, (vx, vy), 3)
        
        # Map border
        pygame.draw.rect(screen, BLACK, (0, 0, MAP_W, MAP_H), 2)
        
        # ============ RIGHT PANEL ============
        py = 10
        
        # Title
        title = font_title.render("D²TL Selector Brain", True, BLACK)
        screen.blit(title, (PANEL_X, py)); py += 30
        
        if history:
            cur = history[-1]
            pred = cur['pred']
            
            # Active model indicator
            if pred['use_mamba']:
                pygame.draw.rect(screen, LIGHT_RED, (PANEL_X, py, PANEL_W, 35))
                model_text = font_label.render("MAMBA-3 ACTIVE", True, RED)
            else:
                pygame.draw.rect(screen, LIGHT_BLUE, (PANEL_X, py, PANEL_W, 35))
                model_text = font_label.render("MLP PRIMARY", True, BLUE)
            screen.blit(model_text, (PANEL_X + 10, py + 8))
            
            trigger_text = font_small.render(f"Trigger: {pred['trigger_score']:.2f}", True, DARK_GRAY)
            screen.blit(trigger_text, (PANEL_X + PANEL_W - 90, py + 10))
            py += 45
            
            # Trigger bar
            bar_w = PANEL_W - 20
            bar_h = 12
            pygame.draw.rect(screen, GRAY, (PANEL_X + 10, py, bar_w, bar_h))
            fill_w = int(bar_w * pred['trigger_score'])
            bar_color = RED if pred['trigger_score'] >= 0.3 else ORANGE if pred['trigger_score'] >= 0.15 else GREEN
            pygame.draw.rect(screen, bar_color, (PANEL_X + 10, py, fill_w, bar_h))
            # Threshold line
            thresh_x = PANEL_X + 10 + int(bar_w * 0.3)
            pygame.draw.line(screen, BLACK, (thresh_x, py - 2), (thresh_x, py + bar_h + 2), 2)
            screen.blit(font_small.render("0.3", True, BLACK), (thresh_x - 8, py + bar_h + 2))
            py += 25
            
            # Reasons
            if pred['reasons']:
                screen.blit(font_label.render("Trigger Reasons:", True, DARK_GRAY), (PANEL_X, py))
                py += 18
                for reason in pred['reasons'][:4]:
                    screen.blit(font_small.render(f"  • {reason}", True, RED), (PANEL_X, py))
                    py += 16
            else:
                screen.blit(font_small.render("  Normal conditions", True, GREEN), (PANEL_X, py))
                py += 16
            py += 10
            
            # Conditions
            pygame.draw.line(screen, GRAY, (PANEL_X, py), (PANEL_X + PANEL_W, py), 1)
            py += 5
            screen.blit(font_label.render("Current Conditions:", True, BLACK), (PANEL_X, py))
            py += 20
            
            w_name = WEATHER_NAMES.get(cur['weather'], str(cur['weather']))
            d_name = DENSITY_NAMES.get(cur['density'], str(cur['density']))
            screen.blit(font_small.render(f"  Weather: {w_name}", True, RED if cur['weather'] >= 2 else BLACK), (PANEL_X, py)); py += 16
            screen.blit(font_small.render(f"  Density: {d_name}", True, RED if cur['density'] >= 2 else BLACK), (PANEL_X, py)); py += 16
            screen.blit(font_small.render(f"  Distance: {cur['dist']:.0f}m", True, RED if cur['dist'] > 500 else BLACK), (PANEL_X, py)); py += 16
            screen.blit(font_small.render(f"  Zone: {cur['zone']}", True, DARK_GRAY), (PANEL_X, py)); py += 25
            
            # Model outputs
            pygame.draw.line(screen, GRAY, (PANEL_X, py), (PANEL_X + PANEL_W, py), 1)
            py += 5
            screen.blit(font_label.render("Model Outputs:", True, BLACK), (PANEL_X, py)); py += 20
            
            screen.blit(font_small.render(f"  MLP Power:   {pred['mlp_power']:>+7.1f} dBm", True, BLUE), (PANEL_X, py)); py += 16
            screen.blit(font_small.render(f"  Mamba Power: {pred['mamba_power']:>+7.1f} dBm", True, RED), (PANEL_X, py)); py += 16
            screen.blit(font_small.render(f"  Selected:    {pred['selected_power']:>+7.1f} dBm", True, GREEN), (PANEL_X, py)); py += 20
            
            screen.blit(font_small.render(f"  MLP QoS:     {pred['mlp_qos']:>6.1f}", True, BLUE), (PANEL_X, py)); py += 16
            screen.blit(font_small.render(f"  Mamba QoS:   {pred['mamba_qos']:>6.1f}", True, RED), (PANEL_X, py)); py += 16
            screen.blit(font_small.render(f"  Selected:    {pred['selected_qos']:>6.1f}", True, GREEN), (PANEL_X, py)); py += 25
            
            # Statistics
            pygame.draw.line(screen, GRAY, (PANEL_X, py), (PANEL_X + PANEL_W, py), 1)
            py += 5
            screen.blit(font_label.render("Decision Statistics:", True, BLACK), (PANEL_X, py)); py += 20
            
            total_d = decision_counts['mlp'] + decision_counts['mamba']
            mlp_pct = decision_counts['mlp'] / max(total_d, 1) * 100
            mamba_pct = decision_counts['mamba'] / max(total_d, 1) * 100
            
            screen.blit(font_small.render(f"  MLP decisions:   {decision_counts['mlp']:4d} ({mlp_pct:.1f}%)", True, BLUE), (PANEL_X, py)); py += 16
            screen.blit(font_small.render(f"  Mamba decisions: {decision_counts['mamba']:4d} ({mamba_pct:.1f}%)", True, RED), (PANEL_X, py)); py += 16
            screen.blit(font_small.render(f"  Total:           {total_d:4d}", True, BLACK), (PANEL_X, py)); py += 25
            
            # Mini decision bar
            if total_d > 0:
                bar_w = PANEL_W - 20
                mlp_w = int(bar_w * mlp_pct / 100)
                pygame.draw.rect(screen, BLUE, (PANEL_X + 10, py, mlp_w, 15))
                pygame.draw.rect(screen, RED, (PANEL_X + 10 + mlp_w, py, bar_w - mlp_w, 15))
                py += 25
        
        # ============ BOTTOM INFO BAR ============
        pygame.draw.rect(screen, (245, 245, 245), (0, MAP_H, WINDOW_W, WINDOW_H - MAP_H))
        pygame.draw.line(screen, DARK_GRAY, (0, MAP_H), (WINDOW_W, MAP_H), 2)
        
        info_y = MAP_H + 10
        screen.blit(font_label.render("D²TL: Physics-Aware Dual-Path Coverage Predictor", True, BLACK), (10, info_y))
        screen.blit(font_small.render("MLP (fast, primary) + Mamba (physics-aware backup) + Selector Brain", True, DARK_GRAY), (10, info_y + 20))
        
        # Legend
        lx = 10
        ly = info_y + 45
        pygame.draw.circle(screen, BLUE, (lx + 5, ly + 5), 5)
        screen.blit(font_small.render("MLP Active", True, BLUE), (lx + 15, ly))
        pygame.draw.circle(screen, RED, (lx + 105, ly + 5), 5)
        screen.blit(font_small.render("Mamba Active", True, RED), (lx + 115, ly))
        pygame.draw.circle(screen, GREEN, (lx + 225, ly + 5), 5)
        screen.blit(font_small.render("RSU", True, GREEN), (lx + 235, ly))
        
        # Controls
        screen.blit(font_small.render("Controls: SPACE=Pause  R=Reset  1-4=Weather  0=Auto  ESC=Quit", True, DARK_GRAY), (10, ly + 25))
        # Mode: Local vs Live API
        mode_str = "Mode: Live API (Selector 8000)" if USE_REAL_API else "Mode: Local (same weights as API)"
        screen.blit(font_small.render(mode_str, True, DARK_BLUE if USE_REAL_API else DARK_GRAY), (10, ly + 42))
        
        # Current frame
        screen.blit(font_small.render(f"Frame: {frame}", True, DARK_GRAY), (MAP_W - 100, info_y))
        
        if paused:
            pause_text = font_large.render("PAUSED", True, ORANGE)
            screen.blit(pause_text, (MAP_W // 2 - 60, MAP_H // 2))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()
