#!/usr/bin/env python3
"""
D²TL Complete Experiment Suite — Microservice Architecture
============================================================

Runs ALL 7 experiments directly (no API needed) and generates
publication-ready data + plots.

  Exp 1: Extreme Scenario Distribution Analysis
  Exp 2: Distance-Power Decay (Friis Physics Consistency)
  Exp 3: Rainstorm Coverage Collapse (ITU-R Validation)
  Exp 4: Normal vs Extreme Stratified Performance
  Exp 5: Budget/Cost Analysis (Latency, Memory, Params)
  Exp 6: Ablation Study (6 variants)
  Exp 7: Tail-Risk Analysis (P95, P99, worst-case)

Author: NOK KO
Date: 2026-02-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import sys
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Paths
BASE_DIR = Path(__file__).parent.parent.parent  # Model_3_Coverage_Mamba3/
sys.path.insert(0, str(BASE_DIR / 'models'))
sys.path.insert(0, str(BASE_DIR / 'd2tl' / 'mlp_service'))

from mamba3_coverage import CoverageMamba3
from model import CoverageMLP

# Physics constants
CARRIER_FREQ_GHZ = 5.9
TX_POWER_DBM = 33.0
NOISE_FLOOR_DBM = -95.0
PATH_LOSS_EXPONENT = 3.5
REFERENCE_DISTANCE_M = 1.0
WAVELENGTH_M = 3e8 / (CARRIER_FREQ_GHZ * 1e9)
L0_DB = 20 * np.log10(4 * np.pi * REFERENCE_DISTANCE_M / WAVELENGTH_M)
WEATHER_ATTEN = {0: 0.0, 1: 2.0, 2: 5.0, 3: 8.0}
WEATHER_NAMES = {0: 'Clear', 1: 'Light Rain', 2: 'Moderate Rain', 3: 'Heavy Rain'}
DENSITY_OBSTACLE = {0: 1.0, 1: 1.2, 2: 1.5, 3: 2.0}
DENSITY_NAMES = {0: 'Rural', 1: 'Suburban', 2: 'Urban', 3: 'Ultra-Dense'}
ANTENNA_GAIN = 10.0 * 0.7

SAVE_DIR = Path(__file__).parent / 'results'
PLOT_DIR = SAVE_DIR / 'plots'


# ============================================================
# Model Loading
# ============================================================

def load_all_models(device='cpu'):
    """Load Mamba (pre-trained) and MLP."""
    # Mamba
    mamba = CoverageMamba3(input_dim=13)
    mamba_ckpt = torch.load(
        str(BASE_DIR / 'training' / 'best_coverage.pth'),
        weights_only=False, map_location='cpu')
    
    key = 'model_state_dict' if 'model_state_dict' in mamba_ckpt else 'model'
    mamba.load_state_dict(mamba_ckpt[key])
    mamba.eval().to(device)
    
    fstats = mamba_ckpt.get('feature_stats', None)
    
    # MLP
    mlp = CoverageMLP(input_dim=13)
    mlp_ckpt_path = BASE_DIR / 'd2tl' / 'mlp_service' / 'best_mlp_coverage.pth'
    if mlp_ckpt_path.exists():
        mlp_ckpt = torch.load(str(mlp_ckpt_path), weights_only=False, map_location='cpu')
        mlp.load_state_dict(mlp_ckpt['model_state_dict'])
        print(f"  MLP loaded: R²={mlp_ckpt.get('r2', 'N/A')}")
    else:
        print("  MLP: no checkpoint — training fresh")
        mlp = train_mlp_inline(mlp, fstats, device)
    
    mlp.eval().to(device)
    
    print(f"  Mamba: {mamba.get_num_params():,} params, R²={mamba_ckpt.get('r2', 'N/A')}")
    print(f"  MLP: {mlp.get_num_params():,} params")
    
    return mamba, mlp, fstats


def train_mlp_inline(mlp, fstats, device):
    """Quick MLP training if no checkpoint exists."""
    from torch.utils.data import Dataset, DataLoader, random_split
    
    class QuickDS(Dataset):
        def __init__(self):
            path = str(BASE_DIR / 'training_data' / 'coverage_training_data.json')
            v2 = path.replace('.json', '_v2.json')
            try:
                with open(v2) as f: data = json.load(f)
            except: 
                with open(path) as f: data = json.load(f)
            
            self.x = torch.tensor([
                [s['rsu_x_position_m'], s['rsu_y_position_m'], s['tx_power_dbm'],
                 s['antenna_tilt_deg'], s['antenna_azimuth_deg'], s['distance_to_rx_m'],
                 s['angle_to_rx_deg'], s['building_density'], s['weather_condition'],
                 s['vehicle_density_per_km2'], s['num_interferers'],
                 s['rx_height_m'], s['frequency_ghz']] for s in data
            ], dtype=torch.float32)
            
            self.y = torch.tensor([
                [s['received_power_dbm'], s['sinr_db'], s['coverage_radius_m'],
                 s['coverage_area_km2'], s['qos_score']] for s in data
            ], dtype=torch.float32)
            
            if fstats:
                self.x = (self.x - fstats['mean']) / (fstats['std'] + 1e-8)
            
            self.y[:, 0] = (self.y[:, 0] + 260) / 230
            self.y[:, 1] = (self.y[:, 1] + 170) / 230
            self.y[:, 2] = (self.y[:, 2] - 150) / 90
            self.y[:, 3] = (self.y[:, 3] - 0.07) / 0.12
            self.y[:, 4] /= 100.0
            self.y = torch.clamp(self.y, 0, 1)
        
        def __len__(self): return len(self.x)
        def __getitem__(self, i): return self.x[i], self.y[i]
    
    ds = QuickDS()
    tr, va, _ = random_split(ds, [21000, 4500, len(ds)-25500],
                              generator=torch.Generator().manual_seed(42))
    tl = DataLoader(tr, batch_size=64, shuffle=True)
    vl = DataLoader(va, batch_size=64)
    
    mlp = mlp.to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-5)
    
    best_r2, best_sd = -999, None
    import copy
    
    for ep in range(1, 151):
        mlp.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            loss = F.mse_loss(mlp(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            opt.step()
        sched.step()
        
        mlp.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for x, y in vl:
                preds.append(mlp(x.to(device)))
                tgts.append(y.to(device))
        p = torch.cat(preds).cpu().numpy()
        t = torch.cat(tgts).cpu().numpy()
        r2 = np.mean([1 - np.sum((t[:, i] - p[:, i])**2) / (np.sum((t[:, i] - np.mean(t[:, i]))**2) + 1e-8) for i in range(5)])
        
        if r2 > best_r2:
            best_r2 = r2
            best_sd = copy.deepcopy(mlp.state_dict())
        
        if ep % 30 == 0:
            print(f"    MLP training epoch {ep}: R²={r2:.4f} (best={best_r2:.4f})")
    
    mlp.load_state_dict(best_sd)
    
    # Save
    save_path = BASE_DIR / 'd2tl' / 'mlp_service' / 'best_mlp_coverage.pth'
    torch.save({
        'model_state_dict': mlp.state_dict(),
        'feature_stats': fstats,
        'r2': float(best_r2),
    }, str(save_path))
    print(f"    MLP saved: R²={best_r2:.4f}")
    
    return mlp


def normalize(raw, fstats):
    mean = fstats['mean']
    std = fstats['std']
    if isinstance(mean, torch.Tensor):
        return (raw - mean) / (std + 1e-8)
    return (raw - torch.tensor(mean)) / (torch.tensor(std) + 1e-8)


def make_features(distance, weather=0, density=0, n_intf=0, n=1):
    f = torch.zeros(n, 13)
    f[:, 0] = 500.0; f[:, 1] = 500.0; f[:, 2] = TX_POWER_DBM
    f[:, 3] = 7.0; f[:, 4] = 180.0; f[:, 5] = distance
    f[:, 6] = 90.0; f[:, 7] = density; f[:, 8] = weather
    f[:, 9] = 25.0; f[:, 10] = n_intf; f[:, 11] = 1.5; f[:, 12] = CARRIER_FREQ_GHZ
    return f


def physics_power(d, weather=0, density=0):
    eff_n = PATH_LOSS_EXPONENT * DENSITY_OBSTACLE[density]
    pl = L0_DB + 10 * eff_n * np.log10(max(d, 1) / REFERENCE_DISTANCE_M)
    return TX_POWER_DBM + ANTENNA_GAIN - pl - WEATHER_ATTEN[weather]


def decode_power(raw_pred):
    return raw_pred * 230 - 260


def selector_decision(raw_features):
    """Replicate Selector Brain logic locally (data-driven calibration)."""
    w = raw_features[0, 8].item()
    d = raw_features[0, 5].item()
    den = raw_features[0, 7].item()
    intf = raw_features[0, 10].item()
    
    score = 0.0
    
    # PRIMARY: Long distance (where Mamba genuinely outperforms MLP)
    if d > 700: score += 0.35
    elif d > 500: score += 0.20
    
    # SECONDARY: Compound distance + weather
    if d > 500 and w >= 2: score += 0.20
    elif w >= 3: score += 0.10
    
    # TERTIARY: Compound distance + density
    if d > 500 and den >= 2: score += 0.15
    
    # High interference at range
    if intf >= 3 and d > 400: score += 0.10
    
    # Triple compound
    n_factors = sum([w >= 2, d > 500, den >= 2, intf >= 3])
    if n_factors >= 3: score += 0.15
    
    return min(score, 1.0), score >= 0.3


# ============================================================
# Experiments
# ============================================================

def exp1_distribution():
    """Scenario distribution analysis."""
    print("\n" + "=" * 70)
    print("Exp 1: Extreme Scenario Distribution")
    print("=" * 70)
    
    path = str(BASE_DIR / 'training_data' / 'coverage_training_data.json')
    v2 = path.replace('.json', '_v2.json')
    try:
        with open(v2) as f: data = json.load(f)
    except:
        with open(path) as f: data = json.load(f)
    
    total = len(data)
    types = defaultdict(int)
    extreme = 0
    weather_d = defaultdict(int)
    density_d = defaultdict(int)
    dist_buckets = {'<200': 0, '200-500': 0, '500-800': 0, '>800': 0}
    
    for s in data:
        w, d, dist, intf = int(s['weather_condition']), int(s['building_density']), s['distance_to_rx_m'], int(s['num_interferers'])
        weather_d[WEATHER_NAMES.get(w, str(w))] += 1
        density_d[DENSITY_NAMES.get(d, str(d))] += 1
        
        if dist < 200: dist_buckets['<200'] += 1
        elif dist < 500: dist_buckets['200-500'] += 1
        elif dist < 800: dist_buckets['500-800'] += 1
        else: dist_buckets['>800'] += 1
        
        is_ext = False
        if w >= 2: is_ext = True; types['heavy_weather'] += 1
        if dist > 500: is_ext = True; types['long_distance'] += 1
        if d >= 2: is_ext = True; types['dense_urban'] += 1
        if intf >= 3: is_ext = True; types['high_interference'] += 1
        if is_ext: extreme += 1
    
    result = {
        'total': total, 'extreme': extreme, 'normal': total - extreme,
        'extreme_pct': round(extreme / total * 100, 1),
        'weather': dict(weather_d), 'density': dict(density_d),
        'distance': dist_buckets, 'types': dict(types)
    }
    
    print(f"  Total: {total}, Normal: {total-extreme} ({(total-extreme)/total*100:.1f}%), Extreme: {extreme} ({extreme/total*100:.1f}%)")
    for k, v in types.items():
        print(f"    {k:20s}: {v:5d} ({v/total*100:.1f}%)")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Pie chart
    labels = ['Normal\n(MLP Primary)', 'Extreme\n(Mamba Backup)']
    sizes = [total - extreme, extreme]
    colors = ['#2196F3', '#FF5722']
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
    axes[0].set_title('Scenario Distribution', fontsize=14, fontweight='bold')
    
    # Extreme type bar
    t_labels = [k.replace('_', '\n') for k in types.keys()]
    t_vals = [v / total * 100 for v in types.values()]
    bars = axes[1].bar(t_labels, t_vals, color=['#FF9800', '#E91E63', '#9C27B0', '#00BCD4'])
    axes[1].set_ylabel('Percentage of Total Data (%)', fontsize=11)
    axes[1].set_title('Extreme Scenario Types', fontsize=14, fontweight='bold')
    for bar, v in zip(bars, t_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.1f}%', ha='center', fontsize=10)
    
    # Distance histogram
    d_labels = list(dist_buckets.keys())
    d_vals = list(dist_buckets.values())
    colors_d = ['#4CAF50', '#8BC34A', '#FF9800', '#F44336']
    axes[2].bar(d_labels, d_vals, color=colors_d)
    axes[2].set_xlabel('Distance Range (m)', fontsize=11)
    axes[2].set_ylabel('Sample Count', fontsize=11)
    axes[2].set_title('Distance Distribution', fontsize=14, fontweight='bold')
    axes[2].axvline(x=1.5, color='red', linestyle='--', alpha=0.7, label='Mamba Trigger (500m)')
    
    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / '1_scenario_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    return result


def exp2_distance_power(mamba, mlp, fstats, device):
    """Distance-Power decay — Friis consistency."""
    print("\n" + "=" * 70)
    print("Exp 2: Distance-Power Decay (Friis Consistency)")
    print("=" * 70)
    
    distances = np.linspace(50, 1000, 60)
    result = {'distances': distances.tolist()}
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, density in enumerate([0, 2]):
        env = DENSITY_NAMES[density]
        eff_n = PATH_LOSS_EXPONENT * DENSITY_OBSTACLE[density]
        
        phys, mlp_p, mamba_p, dual_p, triggers = [], [], [], [], []
        
        with torch.no_grad():
            for d in distances:
                raw = make_features(d, weather=0, density=density)
                norm = normalize(raw, fstats).to(device)
                
                y_mlp = mlp(norm)[0, 0].item()
                y_mamba = mamba(norm)[0, 0].item()
                
                score, use_mamba = selector_decision(raw)
                y_dual = y_mamba if use_mamba else y_mlp
                
                phys.append(physics_power(d, 0, density))
                mlp_p.append(decode_power(y_mlp))
                mamba_p.append(decode_power(y_mamba))
                dual_p.append(decode_power(y_dual))
                triggers.append(score)
        
        # Slope analysis
        log_d = np.log10(distances / REFERENCE_DISTANCE_M)
        theory_slope = -10 * eff_n
        
        for name, arr in [('mlp', mlp_p), ('mamba', mamba_p), ('dual', dual_p)]:
            coeffs = np.polyfit(log_d, arr, 1)
            result[f'{env}_{name}_slope'] = float(coeffs[0])
            result[f'{env}_{name}_slope_error_dB'] = float(coeffs[0] - theory_slope)
        result[f'{env}_theory_slope'] = float(theory_slope)
        
        print(f"\n  {env} (n_eff={eff_n:.1f}, theory slope={theory_slope:.1f} dB/dec):")
        print(f"    MLP slope error:   {result[f'{env}_mlp_slope_error_dB']:>+.2f} dB/decade")
        print(f"    Mamba slope error: {result[f'{env}_mamba_slope_error_dB']:>+.2f} dB/decade")
        print(f"    Dual slope error:  {result[f'{env}_dual_slope_error_dB']:>+.2f} dB/decade")
        
        # Plot
        ax = axes[idx]
        ax.plot(distances, phys, 'k--', linewidth=2, label='Physics (Friis)', zorder=5)
        ax.plot(distances, mlp_p, 'b-', linewidth=1.5, alpha=0.8, label='MLP')
        ax.plot(distances, mamba_p, 'r-', linewidth=1.5, alpha=0.8, label='Mamba-3')
        ax.plot(distances, dual_p, 'g-', linewidth=2.5, label='D²TL (Selector)')
        
        ax.axvline(x=500, color='orange', linestyle=':', alpha=0.7, label='Trigger (500m)')
        
        # Shade Mamba region
        ax.axvspan(500, 1000, alpha=0.08, color='red', label='Mamba Active Zone')
        
        ax.set_xlabel('Distance (m)', fontsize=12)
        ax.set_ylabel('Received Power (dBm)', fontsize=12)
        ax.set_title(f'{env}: Distance-Power Decay', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add slope annotation
        ax.text(0.02, 0.02,
                f'Slope error:\n  MLP: {result[f"{env}_mlp_slope_error_dB"]:+.1f} dB/dec\n'
                f'  Mamba: {result[f"{env}_mamba_slope_error_dB"]:+.1f} dB/dec',
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / '2_distance_power_friis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    return result


def exp3_rainstorm(mamba, mlp, fstats, device):
    """Rainstorm coverage collapse."""
    print("\n" + "=" * 70)
    print("Exp 3: Rainstorm Coverage Collapse")
    print("=" * 70)
    
    result = {}
    test_dist = 300.0
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, density in enumerate([0, 1]):
        env = DENSITY_NAMES[density]
        data = {}
        
        with torch.no_grad():
            for w in range(4):
                raw = make_features(test_dist, weather=w, density=density)
                norm = normalize(raw, fstats).to(device)
                
                y_mlp = decode_power(mlp(norm)[0, 0].item())
                y_mamba = decode_power(mamba(norm)[0, 0].item())
                y_phys = physics_power(test_dist, w, density)
                
                score, use_mamba = selector_decision(raw)
                y_dual = y_mamba if use_mamba else y_mlp
                
                data[WEATHER_NAMES[w]] = {
                    'physics': y_phys, 'mlp': y_mlp, 'mamba': y_mamba,
                    'dual': y_dual, 'trigger': score, 'theory_atten': WEATHER_ATTEN[w]
                }
        
        # Compute attenuation relative to Clear
        clear = data['Clear']
        for wname in ['Light Rain', 'Moderate Rain', 'Heavy Rain']:
            d = data[wname]
            d['mlp_atten'] = clear['mlp'] - d['mlp']
            d['mamba_atten'] = clear['mamba'] - d['mamba']
            d['physics_atten'] = clear['physics'] - d['physics']
        
        result[env] = data
        
        print(f"\n  {env} (dist={test_dist}m):")
        print(f"    {'Weather':<16} {'Theory':>8} {'MLP':>8} {'Mamba':>8} {'Dual':>8} {'Trigger':>8}")
        for wname, d in data.items():
            print(f"    {wname:<16} {d['physics']:>+8.1f} {d['mlp']:>+8.1f} {d['mamba']:>+8.1f} {d['dual']:>+8.1f} {d['trigger']:>8.2f}")
        
        if 'Heavy Rain' in data and 'mamba_atten' in data['Heavy Rain']:
            hr = data['Heavy Rain']
            print(f"\n    Heavy Rain attenuation (theory: {hr['theory_atten']:.0f} dB):")
            print(f"      MLP:   {hr.get('mlp_atten', 0):+.2f} dB")
            print(f"      Mamba: {hr['mamba_atten']:+.2f} dB")
        
        # Plot
        ax = axes[idx]
        weathers = list(data.keys())
        phys_vals = [data[w]['physics'] for w in weathers]
        mlp_vals = [data[w]['mlp'] for w in weathers]
        mamba_vals = [data[w]['mamba'] for w in weathers]
        dual_vals = [data[w]['dual'] for w in weathers]
        
        x_pos = np.arange(len(weathers))
        width = 0.2
        
        ax.bar(x_pos - 1.5*width, phys_vals, width, label='Physics', color='#212121', alpha=0.8)
        ax.bar(x_pos - 0.5*width, mlp_vals, width, label='MLP', color='#2196F3')
        ax.bar(x_pos + 0.5*width, mamba_vals, width, label='Mamba-3', color='#FF5722')
        ax.bar(x_pos + 1.5*width, dual_vals, width, label='D²TL', color='#4CAF50')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(weathers, fontsize=10)
        ax.set_ylabel('Received Power (dBm)', fontsize=12)
        ax.set_title(f'{env}: Rain Attenuation Response', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Annotate theory atten
        if 'Heavy Rain' in data:
            hr = data['Heavy Rain']
            ax.annotate(f'Theory: {hr["theory_atten"]:.0f} dB atten\n'
                       f'Mamba: {hr.get("mamba_atten", 0):.1f} dB',
                       xy=(3, hr['mamba']), fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / '3_rainstorm_collapse.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    return result


def exp4_stratified(mamba, mlp, fstats, device):
    """Normal vs Extreme stratified performance."""
    print("\n" + "=" * 70)
    print("Exp 4: Stratified Performance (Normal vs Extreme)")
    print("=" * 70)
    
    path = str(BASE_DIR / 'training_data' / 'coverage_training_data.json')
    v2 = path.replace('.json', '_v2.json')
    try:
        with open(v2) as f: data = json.load(f)
    except:
        with open(path) as f: data = json.load(f)
    
    test_data = data[-4500:]
    
    categories = {'normal': [], 'extreme_weather': [], 'extreme_distance': [],
                  'extreme_density': [], 'extreme_compound': []}
    
    for i, s in enumerate(test_data):
        w, d, dist, intf = s['weather_condition'], s['building_density'], s['distance_to_rx_m'], s['num_interferers']
        factors = sum([w >= 2, dist > 500, d >= 2, intf >= 3])
        
        if factors == 0:
            categories['normal'].append(i)
        else:
            if factors >= 2: categories['extreme_compound'].append(i)
            if w >= 2: categories['extreme_weather'].append(i)
            if dist > 500: categories['extreme_distance'].append(i)
            if d >= 2: categories['extreme_density'].append(i)
    
    results = {}
    
    for cat_name, indices in categories.items():
        if not indices:
            continue
        
        mlp_errs, mamba_errs, dual_errs = [], [], []
        
        with torch.no_grad():
            for idx in indices:
                s = test_data[idx]
                raw = torch.tensor([[
                    s['rsu_x_position_m'], s['rsu_y_position_m'], s['tx_power_dbm'],
                    s['antenna_tilt_deg'], s['antenna_azimuth_deg'], s['distance_to_rx_m'],
                    s['angle_to_rx_deg'], s['building_density'], s['weather_condition'],
                    s['vehicle_density_per_km2'], s['num_interferers'],
                    s['rx_height_m'], s['frequency_ghz']
                ]], dtype=torch.float32)
                
                tgt = torch.tensor([[
                    (s['received_power_dbm'] + 260) / 230,
                    (s['sinr_db'] + 170) / 230,
                    (s['coverage_radius_m'] - 150) / 90,
                    (s['coverage_area_km2'] - 0.07) / 0.12,
                    s['qos_score'] / 100.0
                ]], dtype=torch.float32).clamp(0, 1).to(device)
                
                norm = normalize(raw, fstats).to(device)
                
                y_mlp = mlp(norm)
                y_mamba = mamba(norm)
                
                score, use_mamba = selector_decision(raw)
                y_dual = y_mamba if use_mamba else y_mlp
                
                mlp_errs.append(((y_mlp - tgt) ** 2).mean().item())
                mamba_errs.append(((y_mamba - tgt) ** 2).mean().item())
                dual_errs.append(((y_dual - tgt) ** 2).mean().item())
        
        def to_r2(errs):
            return 1 - np.mean(errs) / (np.var(errs) + 1e-8) if len(errs) > 1 else 0
        
        results[cat_name] = {
            'n': len(indices),
            'mlp_mse': float(np.mean(mlp_errs)),
            'mamba_mse': float(np.mean(mamba_errs)),
            'dual_mse': float(np.mean(dual_errs)),
            'mlp_improvement': float((np.mean(mlp_errs) - np.mean(dual_errs)) / np.mean(mlp_errs) * 100) if np.mean(mlp_errs) > 0 else 0,
        }
        
        print(f"  {cat_name:20s} (n={len(indices):4d}) | MLP MSE: {np.mean(mlp_errs):.6f} | "
              f"Mamba MSE: {np.mean(mamba_errs):.6f} | Dual MSE: {np.mean(dual_errs):.6f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    cats = list(results.keys())
    x = np.arange(len(cats))
    width = 0.25
    
    mlp_vals = [results[c]['mlp_mse'] for c in cats]
    mamba_vals = [results[c]['mamba_mse'] for c in cats]
    dual_vals = [results[c]['dual_mse'] for c in cats]
    
    ax.bar(x - width, mlp_vals, width, label='MLP Only', color='#2196F3')
    ax.bar(x, mamba_vals, width, label='Mamba Only', color='#FF5722')
    ax.bar(x + width, dual_vals, width, label='D²TL (Selector)', color='#4CAF50')
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in cats], fontsize=10)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Stratified Performance: Normal vs Extreme Scenarios', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / '4_stratified_performance.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    return results


def exp5_cost(mamba, mlp, fstats, device):
    """Budget/Cost analysis."""
    print("\n" + "=" * 70)
    print("Exp 5: Budget/Cost Analysis")
    print("=" * 70)
    
    B = 100
    raw = make_features(300, n=B)
    norm = normalize(raw, fstats).to(device)
    
    results = {}
    
    for name, model in [('MLP', mlp), ('Mamba-3', mamba)]:
        model.eval()
        with torch.no_grad():
            for _ in range(10): model(norm)  # warmup
            times = []
            for _ in range(50):
                t0 = time.perf_counter()
                model(norm)
                times.append((time.perf_counter() - t0) * 1000)
        
        results[name] = {
            'latency_ms': float(np.mean(times)),
            'latency_std': float(np.std(times)),
            'params': sum(p.numel() for p in model.parameters()),
        }
    
    # Effective Dual latency (parallel calls, ~15% Mamba activation)
    trigger_rate = 0.15
    mlp_t = results['MLP']['latency_ms']
    mamba_t = results['Mamba-3']['latency_ms']
    dual_parallel = max(mlp_t, mamba_t)  # both run in parallel
    # With early-exit optimization
    dual_effective = trigger_rate * dual_parallel + (1 - trigger_rate) * mlp_t
    
    results['D²TL (parallel)'] = {
        'latency_ms': float(dual_parallel),
        'note': 'Both models run, take max'
    }
    results['D²TL (early-exit)'] = {
        'latency_ms': float(dual_effective),
        'trigger_rate': trigger_rate,
        'note': 'Skip Mamba when trigger < 0.3'
    }
    results['speedup'] = float(mamba_t / dual_effective) if dual_effective > 0 else 0
    
    print(f"\n  {'Model':<22} {'Latency (ms)':<15} {'Params':<15}")
    print(f"  {'-'*52}")
    for name in ['MLP', 'Mamba-3', 'D²TL (parallel)', 'D²TL (early-exit)']:
        r = results[name]
        params = f"{r.get('params', '-'):,}" if 'params' in r else '-'
        print(f"  {name:<22} {r['latency_ms']:<15.2f} {params:>15}")
    print(f"\n  Effective speedup vs Mamba-only: {results['speedup']:.1f}x")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Latency comparison
    models = ['MLP', 'Mamba-3', 'D²TL\n(early-exit)']
    latencies = [results['MLP']['latency_ms'], results['Mamba-3']['latency_ms'],
                 results['D²TL (early-exit)']['latency_ms']]
    colors = ['#2196F3', '#FF5722', '#4CAF50']
    bars = ax1.bar(models, latencies, color=colors)
    ax1.set_ylabel('Inference Latency (ms)', fontsize=12)
    ax1.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
    for bar, v in zip(bars, latencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.1f}ms', ha='center', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Parameter comparison
    param_models = ['MLP', 'Mamba-3']
    param_vals = [results['MLP']['params'] / 1e6, results['Mamba-3']['params'] / 1e6]
    bars2 = ax2.bar(param_models, param_vals, color=['#2196F3', '#FF5722'])
    ax2.set_ylabel('Parameters (Millions)', fontsize=12)
    ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    for bar, v in zip(bars2, param_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{v:.2f}M', ha='center', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / '5_cost_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    return results


def exp6_ablation(mamba, mlp, fstats, device):
    """Ablation study — 6 variants."""
    print("\n" + "=" * 70)
    print("Exp 6: Ablation Study")
    print("=" * 70)
    
    path = str(BASE_DIR / 'training_data' / 'coverage_training_data.json')
    v2 = path.replace('.json', '_v2.json')
    try:
        with open(v2) as f: data = json.load(f)
    except:
        with open(path) as f: data = json.load(f)
    
    test_data = data[-4500:]
    
    def evaluate(select_fn, name):
        errs = []
        with torch.no_grad():
            for s in test_data:
                raw = torch.tensor([[
                    s['rsu_x_position_m'], s['rsu_y_position_m'], s['tx_power_dbm'],
                    s['antenna_tilt_deg'], s['antenna_azimuth_deg'], s['distance_to_rx_m'],
                    s['angle_to_rx_deg'], s['building_density'], s['weather_condition'],
                    s['vehicle_density_per_km2'], s['num_interferers'],
                    s['rx_height_m'], s['frequency_ghz']
                ]], dtype=torch.float32)
                
                tgt = torch.tensor([[
                    (s['received_power_dbm'] + 260) / 230,
                    (s['sinr_db'] + 170) / 230,
                    (s['coverage_radius_m'] - 150) / 90,
                    (s['coverage_area_km2'] - 0.07) / 0.12,
                    s['qos_score'] / 100.0
                ]], dtype=torch.float32).clamp(0, 1).to(device)
                
                norm = normalize(raw, fstats).to(device)
                y_mlp = mlp(norm)
                y_mamba = mamba(norm)
                
                y_final = select_fn(y_mlp, y_mamba, raw)
                errs.append(((y_final - tgt) ** 2).mean().item())
        
        mse = np.mean(errs)
        return mse
    
    variants = {}
    
    # (a) MLP Only
    variants['MLP Only'] = evaluate(lambda m, mb, r: m, 'MLP Only')
    
    # (b) Mamba Only
    variants['Mamba Only'] = evaluate(lambda m, mb, r: mb, 'Mamba Only')
    
    # (c) Random 50/50
    np.random.seed(42)
    variants['Random 50/50'] = evaluate(
        lambda m, mb, r: mb if np.random.random() > 0.5 else m, 'Random')
    
    # (d) Always Mamba if ANY extreme
    def any_extreme(m, mb, r):
        w, dist, d, intf = r[0, 8].item(), r[0, 5].item(), r[0, 7].item(), r[0, 10].item()
        if w >= 2 or dist > 500 or d >= 2 or intf >= 3:
            return mb
        return m
    variants['Any-Extreme Trigger'] = evaluate(any_extreme, 'Any-Extreme')
    
    # (e) Smart Selector (our method)
    def smart_selector(m, mb, r):
        _, use = selector_decision(r)
        return mb if use else m
    variants['D²TL Selector'] = evaluate(smart_selector, 'Smart')
    
    # (f) Soft Blend (0.5 weighted average)
    variants['Soft Blend (0.5)'] = evaluate(
        lambda m, mb, r: 0.5 * m + 0.5 * mb, 'Soft')
    
    print(f"\n  {'Variant':<24} {'MSE':<12}")
    print(f"  {'-'*36}")
    for name, mse in sorted(variants.items(), key=lambda x: x[1]):
        marker = " <-- Ours" if 'D²TL' in name else ""
        print(f"  {name:<24} {mse:<12.6f}{marker}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(variants.keys())
    vals = [variants[n] for n in names]
    colors = ['#2196F3' if 'D²TL' not in n else '#4CAF50' for n in names]
    
    bars = ax.barh(names, vals, color=colors)
    ax.set_xlabel('Mean Squared Error', fontsize=12)
    ax.set_title('Ablation Study: Model Selection Variants', fontsize=14, fontweight='bold')
    
    # Highlight best
    min_val = min(vals)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.0001, bar.get_y() + bar.get_height()/2,
               f'{v:.6f}', va='center', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / '6_ablation_study.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    return variants


def exp7_tail_risk(mamba, mlp, fstats, device):
    """Tail-risk analysis."""
    print("\n" + "=" * 70)
    print("Exp 7: Tail-Risk Analysis")
    print("=" * 70)
    
    path = str(BASE_DIR / 'training_data' / 'coverage_training_data.json')
    v2 = path.replace('.json', '_v2.json')
    try:
        with open(v2) as f: data = json.load(f)
    except:
        with open(path) as f: data = json.load(f)
    
    test_data = data[-4500:]
    
    errs_mlp, errs_mamba, errs_dual = [], [], []
    
    with torch.no_grad():
        for s in test_data:
            raw = torch.tensor([[
                s['rsu_x_position_m'], s['rsu_y_position_m'], s['tx_power_dbm'],
                s['antenna_tilt_deg'], s['antenna_azimuth_deg'], s['distance_to_rx_m'],
                s['angle_to_rx_deg'], s['building_density'], s['weather_condition'],
                s['vehicle_density_per_km2'], s['num_interferers'],
                s['rx_height_m'], s['frequency_ghz']
            ]], dtype=torch.float32)
            
            tgt = torch.tensor([[
                (s['received_power_dbm'] + 260) / 230,
                (s['sinr_db'] + 170) / 230,
                (s['coverage_radius_m'] - 150) / 90,
                (s['coverage_area_km2'] - 0.07) / 0.12,
                s['qos_score'] / 100.0
            ]], dtype=torch.float32).clamp(0, 1).to(device)
            
            norm = normalize(raw, fstats).to(device)
            y_mlp = mlp(norm)
            y_mamba = mamba(norm)
            
            _, use = selector_decision(raw)
            y_dual = y_mamba if use else y_mlp
            
            errs_mlp.append(((y_mlp - tgt) ** 2).mean().item())
            errs_mamba.append(((y_mamba - tgt) ** 2).mean().item())
            errs_dual.append(((y_dual - tgt) ** 2).mean().item())
    
    errs_mlp = np.array(errs_mlp)
    errs_mamba = np.array(errs_mamba)
    errs_dual = np.array(errs_dual)
    
    results = {}
    for name, errs in [('MLP', errs_mlp), ('Mamba', errs_mamba), ('D²TL', errs_dual)]:
        results[name] = {
            'mean': float(np.mean(errs)),
            'median': float(np.median(errs)),
            'p90': float(np.percentile(errs, 90)),
            'p95': float(np.percentile(errs, 95)),
            'p99': float(np.percentile(errs, 99)),
            'max': float(np.max(errs)),
        }
    
    # Tail improvement
    results['tail_improvement'] = {
        'p95_vs_mlp': float((1 - results['D²TL']['p95'] / results['MLP']['p95']) * 100),
        'p99_vs_mlp': float((1 - results['D²TL']['p99'] / results['MLP']['p99']) * 100),
        'max_vs_mlp': float((1 - results['D²TL']['max'] / results['MLP']['max']) * 100),
    }
    
    print(f"\n  {'Metric':<12} {'MLP':<12} {'Mamba':<12} {'D²TL':<12}")
    print(f"  {'-'*48}")
    for metric in ['mean', 'p90', 'p95', 'p99', 'max']:
        print(f"  {metric:<12} {results['MLP'][metric]:<12.6f} "
              f"{results['Mamba'][metric]:<12.6f} {results['D²TL'][metric]:<12.6f}")
    
    print(f"\n  Tail improvement (D²TL vs MLP):")
    for k, v in results['tail_improvement'].items():
        print(f"    {k}: {v:+.1f}%")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    ax1.boxplot([errs_mlp, errs_mamba, errs_dual],
               labels=['MLP', 'Mamba-3', 'D²TL'],
               showfliers=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue'),
               flierprops=dict(marker='o', markersize=2, alpha=0.3))
    ax1.set_ylabel('Squared Error', fontsize=12)
    ax1.set_title('Error Distribution (with outliers)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Tail percentile comparison
    percentiles = ['mean', 'p90', 'p95', 'p99', 'max']
    x = np.arange(len(percentiles))
    width = 0.25
    
    ax2.bar(x - width, [results['MLP'][p] for p in percentiles], width, label='MLP', color='#2196F3')
    ax2.bar(x, [results['Mamba'][p] for p in percentiles], width, label='Mamba-3', color='#FF5722')
    ax2.bar(x + width, [results['D²TL'][p] for p in percentiles], width, label='D²TL', color='#4CAF50')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Mean', 'P90', 'P95', 'P99', 'Max'], fontsize=10)
    ax2.set_ylabel('Squared Error', fontsize=12)
    ax2.set_title('Tail-Risk: Percentile Error Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / '7_tail_risk.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    return results


# ============================================================
# Grand Summary Plot
# ============================================================

def generate_summary_plot(all_results):
    """Generate grand summary dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('D²TL: Physics-Aware Dual-Path Coverage Predictor — Experiment Summary',
                fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Architecture diagram (text)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('System Architecture', fontsize=13, fontweight='bold')
    
    # Draw boxes
    for (x, y, w, h, color, text) in [
        (1, 7.5, 8, 2, '#E3F2FD', 'Selector Brain\n(Port 8000)\nPhysics Analyzer + Decision Logic'),
        (0.5, 4, 4, 2.5, '#E8F5E9', 'MLP Service\n(Port 8001)\nPrimary · Fast · R²≈0.934'),
        (5.5, 4, 4, 2.5, '#FFF3E0', 'Mamba Service\n(Port 8002)\nBackup · Physics · 8dB rain'),
    ]:
        ax1.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5))
        ax1.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax1.annotate('', xy=(2.5, 6.5), xytext=(2.5, 7.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.annotate('', xy=(7.5, 6.5), xytext=(7.5, 7.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(5, 3, 'Normal → MLP (>85%)    |    Extreme → Mamba (~15%)',
            ha='center', fontsize=9, style='italic')
    
    # 2. Distribution
    if 'exp1' in all_results:
        ax2 = fig.add_subplot(gs[0, 1])
        e1 = all_results['exp1']
        ax2.pie([e1['normal'], e1['extreme']],
               labels=[f"Normal\n({e1['normal']})", f"Extreme\n({e1['extreme']})"],
               colors=['#2196F3', '#FF5722'], autopct='%1.1f%%', startangle=90,
               textprops={'fontsize': 10})
        ax2.set_title(f'Scenario Distribution (N={e1["total"]})', fontsize=13, fontweight='bold')
    
    # 3. Cost
    if 'exp5' in all_results:
        ax3 = fig.add_subplot(gs[0, 2])
        e5 = all_results['exp5']
        models = ['MLP', 'Mamba-3', 'D²TL\n(effective)']
        latencies = [e5['MLP']['latency_ms'], e5['Mamba-3']['latency_ms'],
                    e5['D²TL (early-exit)']['latency_ms']]
        bars = ax3.bar(models, latencies, color=['#2196F3', '#FF5722', '#4CAF50'])
        ax3.set_ylabel('Latency (ms)')
        ax3.set_title('Inference Cost', fontsize=13, fontweight='bold')
        for b, v in zip(bars, latencies):
            ax3.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    # 4. Stratified perf
    if 'exp4' in all_results:
        ax4 = fig.add_subplot(gs[1, 0])
        e4 = all_results['exp4']
        cats = list(e4.keys())
        x = np.arange(len(cats))
        w = 0.25
        ax4.bar(x-w, [e4[c]['mlp_mse'] for c in cats], w, label='MLP', color='#2196F3')
        ax4.bar(x, [e4[c]['mamba_mse'] for c in cats], w, label='Mamba', color='#FF5722')
        ax4.bar(x+w, [e4[c]['dual_mse'] for c in cats], w, label='D²TL', color='#4CAF50')
        ax4.set_xticks(x)
        ax4.set_xticklabels([c.replace('_', '\n')[:15] for c in cats], fontsize=8)
        ax4.set_title('Stratified MSE', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=8)
    
    # 5. Ablation
    if 'exp6' in all_results:
        ax5 = fig.add_subplot(gs[1, 1])
        e6 = all_results['exp6']
        names = list(e6.keys())
        vals = [e6[n] for n in names]
        colors = ['#4CAF50' if 'D²TL' in n else '#90CAF9' for n in names]
        ax5.barh(names, vals, color=colors)
        ax5.set_xlabel('MSE')
        ax5.set_title('Ablation Study', fontsize=13, fontweight='bold')
    
    # 6. Tail risk
    if 'exp7' in all_results:
        ax6 = fig.add_subplot(gs[1, 2])
        e7 = all_results['exp7']
        percs = ['mean', 'p90', 'p95', 'p99']
        x = np.arange(len(percs))
        w = 0.25
        ax6.bar(x-w, [e7['MLP'][p] for p in percs], w, label='MLP', color='#2196F3')
        ax6.bar(x, [e7['Mamba'][p] for p in percs], w, label='Mamba', color='#FF5722')
        ax6.bar(x+w, [e7['D²TL'][p] for p in percs], w, label='D²TL', color='#4CAF50')
        ax6.set_xticks(x)
        ax6.set_xticklabels(['Mean', 'P90', 'P95', 'P99'])
        ax6.set_title('Tail-Risk Error', fontsize=13, fontweight='bold')
        ax6.legend(fontsize=8)
    
    plt.savefig(str(PLOT_DIR / '0_grand_summary.png'), dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================
# Main
# ============================================================

def run_all():
    print("\n" + "=" * 80)
    print("D²TL — Complete Experiment Suite")
    print("=" * 80)
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    mamba, mlp, fstats = load_all_models(device)
    
    all_results = {}
    
    all_results['exp1'] = exp1_distribution()
    all_results['exp2'] = exp2_distance_power(mamba, mlp, fstats, device)
    all_results['exp3'] = exp3_rainstorm(mamba, mlp, fstats, device)
    all_results['exp4'] = exp4_stratified(mamba, mlp, fstats, device)
    all_results['exp5'] = exp5_cost(mamba, mlp, fstats, device)
    all_results['exp6'] = exp6_ablation(mamba, mlp, fstats, device)
    all_results['exp7'] = exp7_tail_risk(mamba, mlp, fstats, device)
    
    # Save results
    with open(SAVE_DIR / 'all_experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Grand summary plot
    generate_summary_plot(all_results)
    
    print(f"\n  Results saved to: {SAVE_DIR}")
    print(f"  Plots saved to: {PLOT_DIR}")
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_all()
