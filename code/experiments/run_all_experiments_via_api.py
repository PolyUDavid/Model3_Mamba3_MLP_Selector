#!/usr/bin/env python3
"""
D²TL All Experiments — Via Real API
===================================

Runs all 7 experiments by calling the real Selector API (8000) and MLP (8001) / Mamba (8002).
Requires: services running (bash d2tl/start_all_services.sh).

Output: same structure as all_experiment_results.json, saved to
  d2tl/experiments/results/all_experiment_results_via_api.json
  and copied to paper_package/02_experiment_data/ and GITHUB/data/experiment/

Usage:
  python d2tl/experiments/run_all_experiments_via_api.py
  python d2tl/experiments/run_all_experiments_via_api.py --url http://192.168.1.10:8000

Author: NOK KO
"""

import json
import time
import urllib.request
import urllib.error
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Model_3_Coverage_Mamba3
SELECTOR_URL = "http://localhost:8000"
MLP_URL = "http://localhost:8001"
MAMBA_URL = "http://localhost:8002"

WEATHER_NAMES = {0: 'Clear', 1: 'Light Rain', 2: 'Moderate Rain', 3: 'Heavy Rain'}
DENSITY_NAMES = {0: 'Rural', 1: 'Suburban', 2: 'Urban', 3: 'Ultra-Dense'}
REFERENCE_DISTANCE_M = 1.0
PATH_LOSS_EXPONENT = 3.5
DENSITY_OBSTACLE = {0: 1.0, 1: 1.2, 2: 1.5, 3: 2.0}
WEATHER_ATTEN = {0: 0.0, 1: 2.0, 2: 5.0, 3: 8.0}
CARRIER_FREQ_GHZ = 5.9
TX_POWER_DBM = 33.0
WAVELENGTH_M = 3e8 / (CARRIER_FREQ_GHZ * 1e9)
L0_DB = 20 * np.log10(4 * np.pi * REFERENCE_DISTANCE_M / WAVELENGTH_M)
ANTENNA_GAIN = 10.0 * 0.7


def physics_power(d_m, weather=0, density=0):
    """Theory: Friis + density + weather attenuation."""
    eff_n = PATH_LOSS_EXPONENT * DENSITY_OBSTACLE[density]
    pl = L0_DB + 10 * eff_n * np.log10(max(d_m, 1) / REFERENCE_DISTANCE_M)
    return TX_POWER_DBM + ANTENNA_GAIN - pl - WEATHER_ATTEN[weather]

SAVE_DIR = Path(__file__).parent / 'results'
BATCH_SIZE = 80  # requests per batch for exp4/6/7


def load_training_data():
    """Load coverage data (try v2 then v1, multiple paths)."""
    for base in [BASE_DIR / 'training_data', BASE_DIR / 'Model_3_Complete_Package' / 'training_data_package']:
        for name in ['coverage_training_data_v2.json', 'coverage_training_data.json']:
            path = base / name
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                print(f"  Loaded: {path} ({len(data)} samples)")
                return data
    raise FileNotFoundError("No coverage_training_data*.json found under training_data/ or Model_3_Complete_Package/")


def sample_to_body(s):
    """Convert a data sample to API request body (single prediction)."""
    return {
        "rsu_x_position_m": float(s.get('rsu_x_position_m', 500)),
        "rsu_y_position_m": float(s.get('rsu_y_position_m', 500)),
        "tx_power_dbm": float(s.get('tx_power_dbm', TX_POWER_DBM)),
        "antenna_tilt_deg": float(s.get('antenna_tilt_deg', 7)),
        "antenna_azimuth_deg": float(s.get('antenna_azimuth_deg', 180)),
        "distance_to_rx_m": float(s['distance_to_rx_m']),
        "angle_to_rx_deg": float(s.get('angle_to_rx_deg', 90)),
        "building_density": int(s['building_density']),
        "weather_condition": int(s['weather_condition']),
        "vehicle_density_per_km2": float(s.get('vehicle_density_per_km2', 25)),
        "num_interferers": int(s.get('num_interferers', 0)),
        "rx_height_m": float(s.get('rx_height_m', 1.5)),
        "frequency_ghz": float(s.get('frequency_ghz', CARRIER_FREQ_GHZ)),
    }


def post_predict(url, body, timeout=15):
    """POST single prediction; return parsed JSON."""
    req = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps(body).encode('utf-8'),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def post_batch(url, bodies, timeout=300):
    """POST batch to Selector; returns list of predictions in same order."""
    # Selector expects body = list of CoverageInput
    req = urllib.request.Request(
        f"{url}/predict/batch",
        data=json.dumps(bodies).encode('utf-8'),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        r = json.loads(resp.read().decode())
    return r.get("predictions", [])


def check_services():
    """Ensure Selector, MLP, Mamba are up."""
    for name, url in [("Selector", SELECTOR_URL), ("MLP", MLP_URL), ("Mamba", MAMBA_URL)]:
        try:
            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as _:
                print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAIL — {e}")
            print("  Run: bash d2tl/start_all_services.sh")
            sys.exit(1)


# ---------- Exp 1: Distribution (no API) ----------
def exp1_distribution():
    print("\n[Exp1] Scenario distribution (from data, no API)")
    data = load_training_data()
    total = len(data)
    types = defaultdict(int)
    extreme = 0
    weather_d = defaultdict(int)
    density_d = defaultdict(int)
    dist_buckets = {'<200': 0, '200-500': 0, '500-800': 0, '>800': 0}
    for s in data:
        w = int(s['weather_condition'])
        d = int(s['building_density'])
        dist = s['distance_to_rx_m']
        intf = int(s.get('num_interferers', 0))
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
    return {
        'total': total, 'extreme': extreme, 'normal': total - extreme,
        'extreme_pct': round(extreme / total * 100, 1),
        'weather': dict(weather_d), 'density': dict(density_d),
        'distance': dist_buckets, 'types': dict(types),
    }


# ---------- Exp 2: Distance–Power (API) ----------
def exp2_distance_power():
    print("\n[Exp2] Distance–power decay (API)")
    distances = np.linspace(50, 1000, 60)
    result = {'distances': distances.tolist()}
    for density, env in [(0, 'Rural'), (2, 'Urban')]:
        eff_n = PATH_LOSS_EXPONENT * DENSITY_OBSTACLE[density]
        theory_slope = -10 * eff_n
        mlp_p, mamba_p, dual_p = [], [], []
        for j, d in enumerate(distances):
            if j % 20 == 0:
                print(f"    Exp2 {env} {j}/{len(distances)}")
            body = sample_to_body({
                'distance_to_rx_m': d, 'weather_condition': 0, 'building_density': density,
                'rsu_x_position_m': 500, 'rsu_y_position_m': 500, 'tx_power_dbm': TX_POWER_DBM,
                'antenna_tilt_deg': 7, 'antenna_azimuth_deg': 180, 'angle_to_rx_deg': 90,
                'vehicle_density_per_km2': 25, 'num_interferers': 0, 'rx_height_m': 1.5, 'frequency_ghz': CARRIER_FREQ_GHZ,
            })
            r = post_predict(SELECTOR_URL, body)
            mlp_p.append(r['mlp_prediction']['received_power_dbm'])
            mamba_p.append(r['mamba_prediction']['received_power_dbm'])
            dual_p.append(r['received_power_dbm'])
        log_d = np.log10(distances / REFERENCE_DISTANCE_M)
        for name, arr in [('mlp', mlp_p), ('mamba', mamba_p), ('dual', dual_p)]:
            coeffs = np.polyfit(log_d, arr, 1)
            result[f'{env}_{name}_slope'] = float(coeffs[0])
            result[f'{env}_{name}_slope_error_dB'] = float(coeffs[0] - theory_slope)
        result[f'{env}_theory_slope'] = float(theory_slope)
    return result


# ---------- Exp 3: Rainstorm (API) ----------
def exp3_rainstorm():
    print("\n[Exp3] Rainstorm coverage (API)")
    result = {}
    test_dist = 300.0
    for density, env in [(0, 'Rural'), (1, 'Suburban')]:
        data = {}
        for w in range(4):
            body = sample_to_body({
                'distance_to_rx_m': test_dist, 'weather_condition': w, 'building_density': density,
                'rsu_x_position_m': 500, 'rsu_y_position_m': 500, 'tx_power_dbm': TX_POWER_DBM,
                'antenna_tilt_deg': 7, 'antenna_azimuth_deg': 180, 'angle_to_rx_deg': 90,
                'vehicle_density_per_km2': 25, 'num_interferers': 0, 'rx_height_m': 1.5, 'frequency_ghz': CARRIER_FREQ_GHZ,
            })
            r = post_predict(SELECTOR_URL, body)
            data[WEATHER_NAMES[w]] = {
                'physics': float(physics_power(test_dist, w, density)),
                'mlp': r['mlp_prediction']['received_power_dbm'],
                'mamba': r['mamba_prediction']['received_power_dbm'],
                'dual': r['received_power_dbm'],
                'trigger': r['trigger_score'],
                'theory_atten': {0: 0, 1: 2, 2: 5, 3: 8}[w],
            }
        result[env] = data
    return result


# ---------- Exp 4, 6, 7: shared test run ----------
def _get_test_responses(test_data, progress=True):
    """Call API in batches for test set; return list of responses (same order as test_data)."""
    total_batches = (len(test_data) + BATCH_SIZE - 1) // BATCH_SIZE
    all_responses = []
    for i in range(0, len(test_data), BATCH_SIZE):
        batch = test_data[i:i + BATCH_SIZE]
        bodies = [sample_to_body(s) for s in batch]
        preds = post_batch(SELECTOR_URL, bodies)
        all_responses.extend(preds)
        batch_no = i // BATCH_SIZE + 1
        if progress:
            print(f"    batch {batch_no}/{total_batches} ({len(all_responses)}/{len(test_data)})")
    return all_responses


def _mse_one(pred, true):
    """MSE over 5 outputs (power, sinr, radius, area, qos)."""
    p = (pred.get('received_power_dbm', 0), pred.get('sinr_db', 0), pred.get('coverage_radius_m', 0),
         pred.get('coverage_area_km2', 0), pred.get('qos_score', 0))
    t = (true.get('received_power_dbm', 0), true.get('sinr_db', 0), true.get('coverage_radius_m', 0),
         true.get('coverage_area_km2', 0), true.get('qos_score', 0))
    return np.mean([(a - b) ** 2 for a, b in zip(p, t)])


def exp4_stratified(test_data, responses):
    print("\n[Exp4] Stratified performance (using shared batch responses)")
    categories = {'normal': [], 'extreme_weather': [], 'extreme_distance': [], 'extreme_density': [], 'extreme_compound': []}
    for i, s in enumerate(test_data):
        w, d, dist, intf = s['weather_condition'], s['building_density'], s['distance_to_rx_m'], int(s.get('num_interferers', 0))
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
        for idx in indices:
            s = test_data[idx]
            r = responses[idx]
            true = {'received_power_dbm': s['received_power_dbm'], 'sinr_db': s['sinr_db'],
                    'coverage_radius_m': s['coverage_radius_m'], 'coverage_area_km2': s['coverage_area_km2'], 'qos_score': s['qos_score']}
            mlp_errs.append(_mse_one(r['mlp_prediction'], true))
            mamba_errs.append(_mse_one(r['mamba_prediction'], true))
            dual_errs.append(_mse_one(r, true))
        mlp_mse = float(np.mean(mlp_errs))
        dual_mse = float(np.mean(dual_errs))
        results[cat_name] = {
            'n': len(indices),
            'mlp_mse': mlp_mse,
            'mamba_mse': float(np.mean(mamba_errs)),
            'dual_mse': dual_mse,
            'mlp_improvement': float((mlp_mse - dual_mse) / mlp_mse * 100) if mlp_mse > 0 else 0,
        }
    return results


def exp5_cost():
    print("\n[Exp5] Latency & params (API)")
    body = sample_to_body({'distance_to_rx_m': 300, 'weather_condition': 0, 'building_density': 0,
                           'rsu_x_position_m': 500, 'rsu_y_position_m': 500, 'tx_power_dbm': TX_POWER_DBM,
                           'antenna_tilt_deg': 7, 'antenna_azimuth_deg': 180, 'angle_to_rx_deg': 90,
                           'vehicle_density_per_km2': 25, 'num_interferers': 0, 'rx_height_m': 1.5, 'frequency_ghz': CARRIER_FREQ_GHZ})
    results = {}
    for name, url in [('MLP', MLP_URL), ('Mamba-3', MAMBA_URL)]:
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            post_predict(url, body, timeout=5)
            times.append((time.perf_counter() - t0) * 1000)
        try:
            h = urllib.request.urlopen(urllib.request.Request(f"{url}/health", method="GET"), timeout=2)
            health = json.loads(h.read().decode())
            params = health.get('parameters', 0)
        except Exception:
            params = 0
        results[name] = {'latency_ms': float(np.mean(times)), 'latency_std': float(np.std(times)), 'params': params}
    # Selector (full dual path)
    times_sel = []
    for _ in range(50):
        t0 = time.perf_counter()
        post_predict(SELECTOR_URL, body, timeout=10)
        times_sel.append((time.perf_counter() - t0) * 1000)
    mlp_t = results['MLP']['latency_ms']
    mamba_t = results['Mamba-3']['latency_ms']
    dual_parallel = max(mlp_t, mamba_t)
    trigger_rate = 0.15
    dual_effective = trigger_rate * dual_parallel + (1 - trigger_rate) * mlp_t
    results['D²TL (parallel)'] = {'latency_ms': float(np.mean(times_sel)), 'note': 'Both models run, take max'}
    results['D²TL (early-exit)'] = {'latency_ms': float(dual_effective), 'trigger_rate': trigger_rate, 'note': 'Skip Mamba when trigger < 0.3'}
    results['speedup'] = float(mamba_t / dual_effective) if dual_effective > 0 else 0
    return results


def exp6_ablation(test_data, responses):
    print("\n[Exp6] Ablation (using shared batch responses)")
    np.random.seed(42)
    variants = {'MLP Only': [], 'Mamba Only': [], 'Random 50/50': [], 'Any-Extreme Trigger': [], 'D²TL Selector': [], 'Soft Blend (0.5)': []}
    for idx, s in enumerate(test_data):
        r = responses[idx]
        true = {'received_power_dbm': s['received_power_dbm'], 'sinr_db': s['sinr_db'],
                'coverage_radius_m': s['coverage_radius_m'], 'coverage_area_km2': s['coverage_area_km2'], 'qos_score': s['qos_score']}
        mlp_p, mamba_p = r['mlp_prediction'], r['mamba_prediction']
        dist, w, den, intf = s['distance_to_rx_m'], s['weather_condition'], s['building_density'], int(s.get('num_interferers', 0))
        any_ext = (w >= 2 or dist > 500 or den >= 2 or intf >= 3)
        soft = {k: 0.5 * mlp_p.get(k, 0) + 0.5 * mamba_p.get(k, 0) for k in ['received_power_dbm', 'sinr_db', 'coverage_radius_m', 'coverage_area_km2', 'qos_score']}
        variants['MLP Only'].append(_mse_one(mlp_p, true))
        variants['Mamba Only'].append(_mse_one(mamba_p, true))
        variants['Random 50/50'].append(_mse_one(mamba_p if np.random.random() > 0.5 else mlp_p, true))
        variants['Any-Extreme Trigger'].append(_mse_one(mamba_p if any_ext else mlp_p, true))
        variants['D²TL Selector'].append(_mse_one(r, true))
        variants['Soft Blend (0.5)'].append(_mse_one(soft, true))
    return {k: float(np.mean(v)) for k, v in variants.items()}


def exp7_tail_risk(test_data, responses):
    print("\n[Exp7] Tail risk (using shared batch responses)")
    errs_mlp, errs_mamba, errs_dual = [], [], []
    for idx, s in enumerate(test_data):
        r = responses[idx]
        true = {'received_power_dbm': s['received_power_dbm'], 'sinr_db': s['sinr_db'],
                'coverage_radius_m': s['coverage_radius_m'], 'coverage_area_km2': s['coverage_area_km2'], 'qos_score': s['qos_score']}
        errs_mlp.append(_mse_one(r['mlp_prediction'], true))
        errs_mamba.append(_mse_one(r['mamba_prediction'], true))
        errs_dual.append(_mse_one(r, true))
    errs_mlp, errs_mamba, errs_dual = np.array(errs_mlp), np.array(errs_mamba), np.array(errs_dual)
    results = {}
    for name, errs in [('MLP', errs_mlp), ('Mamba', errs_mamba), ('D²TL', errs_dual)]:
        results[name] = {
            'mean': float(np.mean(errs)), 'median': float(np.median(errs)),
            'p90': float(np.percentile(errs, 90)), 'p95': float(np.percentile(errs, 95)),
            'p99': float(np.percentile(errs, 99)), 'max': float(np.max(errs)),
        }
    results['tail_improvement'] = {
        'p95_vs_mlp': float((1 - results['D²TL']['p95'] / results['MLP']['p95']) * 100) if results['MLP']['p95'] > 0 else 0,
        'p99_vs_mlp': float((1 - results['D²TL']['p99'] / results['MLP']['p99']) * 100) if results['MLP']['p99'] > 0 else 0,
        'max_vs_mlp': float((1 - results['D²TL']['max'] / results['MLP']['max']) * 100) if results['MLP']['max'] > 0 else 0,
    }
    return results


def main():
    global SELECTOR_URL, MLP_URL, MAMBA_URL
    if '--url' in sys.argv:
        i = sys.argv.index('--url')
        if i + 1 < len(sys.argv):
            base = sys.argv[i + 1].rstrip('/')
            SELECTOR_URL = base
            MLP_URL = base.replace(':8000', ':8001')
            MAMBA_URL = base.replace(':8000', ':8002')
            sys.argv.pop(i); sys.argv.pop(i)
    print("D²TL — All Experiments Via Real API")
    print("=" * 60)
    print(f"  Selector: {SELECTOR_URL}")
    check_services()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_results['exp1'] = exp1_distribution()
    all_results['exp2'] = exp2_distance_power()
    all_results['exp3'] = exp3_rainstorm()
    all_results['exp5'] = exp5_cost()

    # Load test set once and run batch API once for Exp4/6/7 (saves ~2/3 of API time)
    print("\n[Batch] Loading test set and calling Selector batch API (used by Exp4, Exp6, Exp7)...")
    data = load_training_data()
    test_data = data[-4500:]
    responses = _get_test_responses(test_data)

    all_results['exp4'] = exp4_stratified(test_data, responses)
    all_results['exp6'] = exp6_ablation(test_data, responses)
    all_results['exp7'] = exp7_tail_risk(test_data, responses)

    out_path = SAVE_DIR / 'all_experiment_results_via_api.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    # Copy to paper_package and GITHUB/data
    for dest in [BASE_DIR / 'paper_package' / '02_experiment_data' / 'all_experiment_results_via_api.json',
                 BASE_DIR / 'GITHUB' / 'data' / 'experiment' / 'all_experiment_results_via_api.json']:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Copied: {dest}")

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS VIA API COMPLETE")
    return all_results


if __name__ == "__main__":
    main()
