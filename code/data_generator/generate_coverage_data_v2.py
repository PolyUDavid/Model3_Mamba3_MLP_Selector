"""
Coverage Optimization Data Generator V2 - FIXED Physics-Based Simulation
=========================================================================

Model 3: RSU Coverage & Signal Strength Prediction
Architecture: Mamba-3

FIXES:
1. Smooth Radius calculation (no hard thresholds)
2. Multi-factor Radius dependency (Power, SINR, Distance, Environment)
3. More realistic propagation modeling

Author: NOK KO
Date: 2026-01-28
Version: 2.0 (FIXED)
"""

import numpy as np
import json
from typing import Dict, List, Tuple
import random
import math


class CoverageDataGeneratorV2:
    """Fixed physics-based wireless coverage simulator"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        # System parameters
        self.CARRIER_FREQ = 5.9  # GHz
        self.TX_POWER = 33.0  # dBm
        self.NOISE_FLOOR = -95.0  # dBm
        self.BANDWIDTH = 10.0  # MHz
        
        # Propagation parameters
        self.PATH_LOSS_EXPONENT = 3.5
        self.REFERENCE_DISTANCE = 1.0
        self.SHADOWING_STD = 8.0
        
        # Antenna parameters
        self.ANTENNA_HEIGHT = 8.0
        self.ANTENNA_GAIN_MAX = 10.0
        self.ANTENNA_BEAMWIDTH = 120.0
        
        # Coverage thresholds
        self.MIN_RX_POWER = -90.0  # dBm (sensitivity)
        self.MIN_SINR = -5.0  # dB
        
    def generate_dataset(self, num_samples: int = 30000) -> List[Dict]:
        print(f"Generating {num_samples} coverage samples (V2 - FIXED)...")
        print("=" * 60)
        
        dataset = []
        
        for i in range(num_samples):
            if (i + 1) % 5000 == 0:
                print(f"Progress: {i+1}/{num_samples} ({(i+1)/num_samples*100:.1f}%)")
            
            sample = self._generate_single_sample()
            dataset.append(sample)
        
        print(f"\n✓ Generated {len(dataset)} samples")
        self._validate_dataset(dataset)
        
        return dataset
    
    def _generate_single_sample(self) -> Dict:
        """Generate a single sample with FIXED radius calculation"""
        
        # ===== RSU CONFIGURATION =====
        rsu_x = np.random.uniform(0, 2000)
        rsu_y = np.random.uniform(0, 2000)
        tx_power_dbm = self.TX_POWER + np.random.uniform(-3, 3)
        antenna_tilt = np.random.uniform(0, 15)
        antenna_azimuth = np.random.choice([0, 60, 120, 180, 240, 300])
        
        # ===== ENVIRONMENT =====
        building_density = np.random.choice([0, 1, 2, 3])
        obstacle_factor = {0: 1.0, 1: 1.2, 2: 1.5, 3: 2.0}[building_density]
        
        weather = np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.15, 0.05])
        weather_attenuation = {0: 0.0, 1: 2.0, 2: 5.0, 3: 8.0}[weather]
        
        vehicle_density = np.random.lognormal(mean=np.log(50), sigma=0.6)
        vehicle_density = np.clip(vehicle_density, 5, 200)
        
        # ===== TEST POINT =====
        distance_m = np.random.uniform(10, 1000)
        angle_deg = np.random.uniform(0, 360)
        rx_height = 1.5
        
        # ===== SIGNAL PROPAGATION =====
        path_loss_db = self._calculate_path_loss(
            distance_m, 
            self.PATH_LOSS_EXPONENT * obstacle_factor,
            self.SHADOWING_STD
        )
        
        antenna_gain_db = self._calculate_antenna_gain(
            angle_deg, antenna_azimuth, 
            self.ANTENNA_BEAMWIDTH, self.ANTENNA_GAIN_MAX
        )
        
        received_power_dbm = (
            tx_power_dbm +
            antenna_gain_db -
            path_loss_db -
            weather_attenuation
        )
        
        # Interference
        num_interferers = max(0, int(vehicle_density / 50) - 1)
        interference_power_dbm = self._calculate_interference(
            received_power_dbm, num_interferers, distance_m
        )
        
        # SINR
        sinr_db = self._calculate_sinr(
            received_power_dbm,
            interference_power_dbm,
            self.NOISE_FLOOR
        )
        
        # ===== COVERAGE RADIUS - FIXED CALCULATION =====
        # Use reverse path loss to find max distance for connectivity
        coverage_radius_m = self._calculate_coverage_radius(
            tx_power_dbm=tx_power_dbm,
            antenna_gain_db=self.ANTENNA_GAIN_MAX * 0.7,  # Average gain
            path_loss_exponent=self.PATH_LOSS_EXPONENT * obstacle_factor,
            weather_attenuation=weather_attenuation,
            num_interferers=num_interferers,
            min_rx_power=self.MIN_RX_POWER,
            min_sinr=self.MIN_SINR
        )
        
        # Add smooth variations based on local conditions
        # SINR influence: better SINR → larger coverage
        sinr_normalized = (sinr_db + 50) / 100.0  # Normalize to 0-1 range
        sinr_factor = 0.8 + sinr_normalized * 0.4  # 0.8-1.2
        sinr_factor = np.clip(sinr_factor, 0.8, 1.2)
        
        # Environment factor: denser areas reduce coverage
        environment_factor = 1.0 / np.sqrt(obstacle_factor)
        
        coverage_radius_m *= (sinr_factor * environment_factor)
        
        # Realistic range: 150-1000m (not 50m)
        coverage_radius_m = np.clip(coverage_radius_m, 150, 1000)
        
        # Area calculation
        coverage_area_km2 = np.pi * (coverage_radius_m / 1000) ** 2
        
        # ===== OTHER METRICS =====
        coverage_prob = self._sinr_to_coverage_prob(sinr_db)
        qos_score = self._calculate_qos(
            sinr_db, received_power_dbm, 
            distance_m, coverage_prob
        )
        throughput_mbps = self._estimate_throughput(sinr_db, self.BANDWIDTH)
        
        # ===== CONSTRUCT SAMPLE =====
        sample = {
            'rsu_x_position_m': float(rsu_x),
            'rsu_y_position_m': float(rsu_y),
            'tx_power_dbm': float(tx_power_dbm),
            'antenna_tilt_deg': float(antenna_tilt),
            'antenna_azimuth_deg': float(antenna_azimuth),
            'distance_to_rx_m': float(distance_m),
            'angle_to_rx_deg': float(angle_deg),
            'building_density': int(building_density),
            'weather_condition': int(weather),
            'vehicle_density_per_km2': float(vehicle_density),
            'num_interferers': int(num_interferers),
            'rx_height_m': float(rx_height),
            'frequency_ghz': float(self.CARRIER_FREQ),
            
            'received_power_dbm': float(received_power_dbm),
            'sinr_db': float(sinr_db),
            'coverage_radius_m': float(coverage_radius_m),
            'coverage_area_km2': float(coverage_area_km2),
            'qos_score': float(qos_score),
            
            'path_loss_db': float(path_loss_db),
            'antenna_gain_db': float(antenna_gain_db),
            'interference_power_dbm': float(interference_power_dbm),
            'coverage_probability': float(coverage_prob),
            'throughput_mbps': float(throughput_mbps),
            'is_covered': int(sinr_db >= self.MIN_SINR),
            'is_good_quality': int(sinr_db >= 10.0),
        }
        
        return sample
    
    def _calculate_coverage_radius(self, tx_power_dbm, antenna_gain_db,
                                   path_loss_exponent, weather_attenuation,
                                   num_interferers, min_rx_power, min_sinr):
        """
        Calculate coverage radius using reverse path loss equation
        
        FIXED: Use proper sensitivity threshold and realistic calculation
        """
        # Free space path loss at 1m
        wavelength_m = 3e8 / (self.CARRIER_FREQ * 1e9)
        L0 = 20 * np.log10(4 * np.pi * self.REFERENCE_DISTANCE / wavelength_m)
        
        # Maximum allowable path loss for connectivity
        # Use more realistic sensitivity (-85 dBm for good link, not -90)
        effective_min_rx = -85.0 + np.random.uniform(-5, 5)  # Add variation
        
        # Interference margin (lighter penalty)
        interference_margin = num_interferers * 1.5  # dB (reduced from 2.0)
        
        max_path_loss = (
            tx_power_dbm +
            antenna_gain_db -
            effective_min_rx -
            weather_attenuation -
            interference_margin
        )
        
        # Solve for distance
        if max_path_loss <= L0:
            return 100.0  # Minimum coverage
        
        log_distance = (max_path_loss - L0) / (10 * path_loss_exponent)
        distance_m = 10 ** log_distance
        
        # Shadowing margin (less aggressive reduction)
        # Use 0.5 std instead of 1.0 std
        shadowing_margin = 0.5 * self.SHADOWING_STD
        distance_m *= 10 ** (-shadowing_margin / (10 * path_loss_exponent))
        
        # Realistic RSU coverage: 200-1000m
        return np.clip(distance_m, 200, 1000)
    
    def _calculate_path_loss(self, distance_m, exponent, shadowing_std):
        wavelength_m = 3e8 / (self.CARRIER_FREQ * 1e9)
        L0 = 20 * np.log10(4 * np.pi * self.REFERENCE_DISTANCE / wavelength_m)
        
        if distance_m < self.REFERENCE_DISTANCE:
            distance_m = self.REFERENCE_DISTANCE
        
        path_loss = L0 + 10 * exponent * np.log10(distance_m / self.REFERENCE_DISTANCE)
        shadowing = np.random.normal(0, shadowing_std)
        path_loss += shadowing
        
        return max(path_loss, 40.0)
    
    def _calculate_antenna_gain(self, angle_deg, azimuth_deg, 
                                beamwidth_deg, max_gain_dbi):
        angle_diff = abs(((angle_deg - azimuth_deg + 180) % 360) - 180)
        
        if angle_diff <= beamwidth_deg / 2:
            gain = max_gain_dbi * np.cos(angle_diff / (beamwidth_deg / 2) * np.pi / 2) ** 2
        else:
            gain = max_gain_dbi * 0.01
        
        return gain
    
    def _calculate_interference(self, signal_power_dbm, num_interferers, distance_m):
        if num_interferers == 0:
            return -np.inf
        
        interferer_powers = []
        for _ in range(num_interferers):
            interferer_distance_factor = np.random.uniform(1.5, 3.0)
            path_loss_diff = 10 * self.PATH_LOSS_EXPONENT * np.log10(interferer_distance_factor)
            interferer_power = signal_power_dbm - path_loss_diff - np.random.uniform(5, 15)
            interferer_powers.append(10 ** (interferer_power / 10))
        
        total_interference_linear = sum(interferer_powers)
        return 10 * np.log10(total_interference_linear)
    
    def _calculate_sinr(self, signal_dbm, interference_dbm, noise_dbm):
        signal_linear = 10 ** (signal_dbm / 10)
        interference_linear = 10 ** (interference_dbm / 10) if not np.isinf(interference_dbm) else 0
        noise_linear = 10 ** (noise_dbm / 10)
        
        sinr_linear = signal_linear / (interference_linear + noise_linear)
        return 10 * np.log10(sinr_linear)
    
    def _sinr_to_coverage_prob(self, sinr_db):
        threshold = 0.0
        scale = 5.0
        return 1.0 / (1.0 + np.exp(-(sinr_db - threshold) / scale))
    
    def _calculate_qos(self, sinr_db, power_dbm, distance_m, coverage_prob):
        sinr_score = np.clip((sinr_db + 10) / 30 * 40, 0, 40)
        power_score = np.clip((power_dbm + 80) / 50 * 20, 0, 20)
        distance_score = np.clip((1000 - distance_m) / 1000 * 20, 0, 20)
        coverage_score = coverage_prob * 20
        
        qos = sinr_score + power_score + distance_score + coverage_score
        return np.clip(qos, 0, 100)
    
    def _estimate_throughput(self, sinr_db, bandwidth_mhz):
        sinr_linear = 10 ** (sinr_db / 10)
        capacity_mbps = bandwidth_mhz * np.log2(1 + sinr_linear)
        return max(capacity_mbps * 0.7, 0.1)
    
    def _validate_dataset(self, dataset):
        print("\n" + "=" * 60)
        print("Dataset Validation (V2)")
        print("=" * 60)
        
        sinr = [s['sinr_db'] for s in dataset]
        power = [s['received_power_dbm'] for s in dataset]
        radius = [s['coverage_radius_m'] for s in dataset]
        area = [s['coverage_area_km2'] for s in dataset]
        qos = [s['qos_score'] for s in dataset]
        
        print(f"\n✓ SINR: {min(sinr):.1f} - {max(sinr):.1f} dB (mean: {np.mean(sinr):.1f})")
        print(f"✓ Power: {min(power):.1f} - {max(power):.1f} dBm (mean: {np.mean(power):.1f})")
        print(f"✓ Radius: {min(radius):.1f} - {max(radius):.1f} m (mean: {np.mean(radius):.1f})")
        print(f"✓ Area: {min(area):.3f} - {max(area):.3f} km² (mean: {np.mean(area):.3f})")
        print(f"✓ QoS: {min(qos):.1f} - {max(qos):.1f} (mean: {np.mean(qos):.1f})")
        
        # Check correlations
        power_sinr_corr = np.corrcoef(power, sinr)[0, 1]
        sinr_radius_corr = np.corrcoef(sinr, radius)[0, 1]
        radius_area_corr = np.corrcoef(radius, area)[0, 1]
        
        print(f"\n📈 Correlations:")
        print(f"   Power ↔ SINR:   {power_sinr_corr:.3f} (expect: ~0.95-0.99)")
        print(f"   SINR  ↔ Radius: {sinr_radius_corr:.3f} (expect: ~0.60-0.80)")
        print(f"   Radius↔ Area:   {radius_area_corr:.3f} (expect: ~0.99)")
        
        # Check smoothness
        radius_cv = np.std(radius) / np.mean(radius)
        print(f"\n✓ Radius variation: CV = {radius_cv:.3f} (lower is smoother)")
        
        covered = sum(1 for s in dataset if s['is_covered'])
        good = sum(1 for s in dataset if s['is_good_quality'])
        
        print(f"\n📊 Statistics:")
        print(f"   Covered: {covered} ({covered/len(dataset)*100:.1f}%)")
        print(f"   Good quality: {good} ({good/len(dataset)*100:.1f}%)")
        print(f"\n✅ V2 Validation complete!")


def main():
    print("=" * 60)
    print("Coverage Data Generator V2 (FIXED)")
    print("=" * 60)
    
    generator = CoverageDataGeneratorV2(seed=42)
    dataset = generator.generate_dataset(num_samples=30000)
    
    output_path = "../training_data/coverage_training_data_v2.json"
    print(f"\n💾 Saving to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Saved {len(dataset)} samples")
    
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"📦 File size: {size_mb:.1f} MB")
    print("\n" + "=" * 60)
    print("✅ V2 Data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
