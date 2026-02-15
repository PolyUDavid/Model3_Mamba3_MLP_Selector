"""
Coverage Optimization Data Generator - Physics-Based Simulation
================================================================

Model 3: RSU Coverage & Signal Strength Prediction
Architecture: Mamba-3 (Selective State Space Model)

This generator creates realistic wireless coverage data based on:
1. Signal propagation models (path loss, shadowing, fading)
2. Antenna patterns and configurations
3. Interference modeling
4. SINR calculations
5. QoS metrics

Author: NOK KO
Date: 2026-01-28
Version: 1.0
"""

import numpy as np
import json
from typing import Dict, List, Tuple
import random
import math

class CoverageDataGenerator:
    """
    Physics-based wireless coverage simulator
    
    Key Physical Models:
    1. Path Loss: Log-distance model with shadowing
    2. Antenna Gain: Directional patterns
    3. Interference: Co-channel + adjacent channel
    4. SINR: Signal-to-Interference-plus-Noise Ratio
    """
    
    def __init__(self, seed: int = 42):
        """Initialize generator with physical constants"""
        np.random.seed(seed)
        random.seed(seed)
        
        # System parameters (typical V2I/DSRC)
        self.CARRIER_FREQ = 5.9  # GHz (DSRC/C-V2X)
        self.TX_POWER = 33.0  # dBm (typical RSU)
        self.NOISE_FLOOR = -95.0  # dBm
        self.BANDWIDTH = 10.0  # MHz
        
        # Propagation parameters (3GPP Urban Micro)
        self.PATH_LOSS_EXPONENT = 3.5  # Higher in urban
        self.REFERENCE_DISTANCE = 1.0  # meters
        self.SHADOWING_STD = 8.0  # dB (log-normal)
        
        # Antenna parameters
        self.ANTENNA_HEIGHT = 8.0  # meters (typical pole)
        self.ANTENNA_GAIN_MAX = 10.0  # dBi (directional)
        self.ANTENNA_BEAMWIDTH = 120.0  # degrees
        
        # Coverage targets
        self.MIN_SINR = -5.0  # dB (minimum for connectivity)
        self.TARGET_SINR = 10.0  # dB (good quality)
        
    def generate_dataset(self, num_samples: int = 30000) -> List[Dict]:
        """
        Generate complete training dataset
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of samples with features and targets
        """
        print(f"Generating {num_samples} coverage samples...")
        print("=" * 60)
        
        dataset = []
        
        for i in range(num_samples):
            if (i + 1) % 5000 == 0:
                print(f"Progress: {i+1}/{num_samples} samples ({(i+1)/num_samples*100:.1f}%)")
            
            sample = self._generate_single_sample()
            dataset.append(sample)
        
        print(f"\n✓ Generated {len(dataset)} samples")
        
        # Validation
        self._validate_dataset(dataset)
        
        return dataset
    
    def _generate_single_sample(self) -> Dict:
        """Generate a single realistic coverage sample"""
        
        # ===== RSU CONFIGURATION =====
        
        # RSU position (grid deployment)
        rsu_x = np.random.uniform(0, 2000)  # meters
        rsu_y = np.random.uniform(0, 2000)  # meters
        
        # Transmit power (configurable)
        tx_power_dbm = self.TX_POWER + np.random.uniform(-3, 3)  # Small variation
        
        # Antenna configuration
        antenna_tilt = np.random.uniform(0, 15)  # degrees (downward)
        antenna_azimuth = np.random.choice([0, 60, 120, 180, 240, 300])  # Sectored
        
        # ===== ENVIRONMENT =====
        
        # Building density (affects shadowing)
        building_density = np.random.choice([0, 1, 2, 3])  # 0=open, 1=suburban, 2=urban, 3=dense
        density_labels = {0: 'open', 1: 'suburban', 2: 'urban', 3: 'dense_urban'}
        
        # Obstacle density multiplier
        obstacle_factor = {0: 1.0, 1: 1.2, 2: 1.5, 3: 2.0}[building_density]
        
        # Weather condition
        weather = np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.15, 0.05])  
        # 0=clear, 1=rain, 2=fog, 3=heavy_rain
        weather_attenuation = {0: 0.0, 1: 2.0, 2: 5.0, 3: 8.0}[weather]  # dB
        
        # Vehicle density (creates interference)
        vehicle_density = np.random.lognormal(mean=np.log(50), sigma=0.6)
        vehicle_density = np.clip(vehicle_density, 5, 200)  # vehicles/km²
        
        # ===== TEST POINT (Vehicle location) =====
        
        # Distance from RSU
        distance_m = np.random.uniform(10, 1000)  # 10m to 1km
        
        # Angle relative to antenna boresight
        angle_deg = np.random.uniform(0, 360)
        
        # Height (vehicle antenna)
        rx_height = 1.5  # meters (typical vehicle)
        
        # ===== SIGNAL PROPAGATION =====
        
        # 1. Path Loss (log-distance + shadowing)
        path_loss_db = self._calculate_path_loss(
            distance_m, 
            self.PATH_LOSS_EXPONENT * obstacle_factor,
            self.SHADOWING_STD
        )
        
        # 2. Antenna Gain (directional)
        antenna_gain_db = self._calculate_antenna_gain(
            angle_deg,
            antenna_azimuth,
            self.ANTENNA_BEAMWIDTH,
            self.ANTENNA_GAIN_MAX
        )
        
        # 3. Received Signal Strength
        received_power_dbm = (
            tx_power_dbm +
            antenna_gain_db -
            path_loss_db -
            weather_attenuation
        )
        
        # 4. Interference
        # Co-channel interference from nearby RSUs
        num_interferers = max(0, int(vehicle_density / 50) - 1)
        interference_power_dbm = self._calculate_interference(
            received_power_dbm,
            num_interferers,
            distance_m
        )
        
        # 5. SINR Calculation
        sinr_db = self._calculate_sinr(
            received_power_dbm,
            interference_power_dbm,
            self.NOISE_FLOOR
        )
        
        # ===== COVERAGE METRICS =====
        
        # Coverage probability (based on SINR)
        coverage_prob = self._sinr_to_coverage_prob(sinr_db)
        
        # Effective coverage area (estimated)
        # Higher SINR → larger effective coverage radius
        if sinr_db >= self.TARGET_SINR:
            coverage_radius_m = 800 + np.random.uniform(-100, 100)
        elif sinr_db >= self.MIN_SINR:
            coverage_radius_m = 400 + (sinr_db - self.MIN_SINR) / (self.TARGET_SINR - self.MIN_SINR) * 400
        else:
            coverage_radius_m = 100 + max(0, 100 + sinr_db * 20)
        
        coverage_radius_m = np.clip(coverage_radius_m, 50, 1000)
        
        coverage_area_km2 = np.pi * (coverage_radius_m / 1000) ** 2
        
        # QoS score (0-100)
        qos_score = self._calculate_qos(
            sinr_db,
            received_power_dbm,
            distance_m,
            coverage_prob
        )
        
        # Throughput estimate (Mbps) - Shannon capacity
        throughput_mbps = self._estimate_throughput(sinr_db, self.BANDWIDTH)
        
        # ===== CONSTRUCT SAMPLE =====
        
        sample = {
            # Input features (13 features)
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
            
            # Target predictions (5 targets)
            'received_power_dbm': float(received_power_dbm),
            'sinr_db': float(sinr_db),
            'coverage_radius_m': float(coverage_radius_m),
            'coverage_area_km2': float(coverage_area_km2),
            'qos_score': float(qos_score),
            
            # Derived metrics (for analysis)
            'path_loss_db': float(path_loss_db),
            'antenna_gain_db': float(antenna_gain_db),
            'interference_power_dbm': float(interference_power_dbm),
            'coverage_probability': float(coverage_prob),
            'throughput_mbps': float(throughput_mbps),
            'is_covered': int(sinr_db >= self.MIN_SINR),
            'is_good_quality': int(sinr_db >= self.TARGET_SINR),
        }
        
        return sample
    
    def _calculate_path_loss(self, distance_m: float, exponent: float, shadowing_std: float) -> float:
        """
        Calculate path loss using log-distance model with shadowing
        
        L = L₀ + 10·n·log₁₀(d/d₀) + Xσ
        
        Where:
        - L₀: Free space path loss at reference distance
        - n: Path loss exponent
        - d: Distance
        - d₀: Reference distance (1m)
        - Xσ: Log-normal shadowing
        """
        # Free space path loss at 1m
        wavelength_m = 3e8 / (self.CARRIER_FREQ * 1e9)
        L0 = 20 * np.log10(4 * np.pi * self.REFERENCE_DISTANCE / wavelength_m)
        
        # Log-distance component
        if distance_m < self.REFERENCE_DISTANCE:
            distance_m = self.REFERENCE_DISTANCE
        
        path_loss = L0 + 10 * exponent * np.log10(distance_m / self.REFERENCE_DISTANCE)
        
        # Add shadowing (log-normal)
        shadowing = np.random.normal(0, shadowing_std)
        path_loss += shadowing
        
        return max(path_loss, 40.0)  # Minimum path loss
    
    def _calculate_antenna_gain(self, angle_deg: float, azimuth_deg: float, 
                                beamwidth_deg: float, max_gain_dbi: float) -> float:
        """
        Calculate antenna gain based on angle from boresight
        
        Sectored antenna pattern (cosine approximation)
        """
        # Angle difference from boresight
        angle_diff = abs(((angle_deg - azimuth_deg + 180) % 360) - 180)
        
        # Gain pattern (cosine with beamwidth)
        if angle_diff <= beamwidth_deg / 2:
            gain = max_gain_dbi * np.cos(angle_diff / (beamwidth_deg / 2) * np.pi / 2) ** 2
        else:
            # Side lobe
            gain = max_gain_dbi * 0.01  # -20dB from main lobe
        
        return gain
    
    def _calculate_interference(self, signal_power_dbm: float, num_interferers: int, distance_m: float) -> float:
        """
        Calculate total interference power from nearby RSUs
        
        Assumes interferers are at similar distances but with variations
        """
        if num_interferers == 0:
            return -np.inf  # No interference
        
        # Each interferer is weaker (further away on average)
        interferer_powers = []
        for _ in range(num_interferers):
            # Interferers are typically 1.5-3x further
            interferer_distance_factor = np.random.uniform(1.5, 3.0)
            # Approximate path loss difference
            path_loss_diff = 10 * self.PATH_LOSS_EXPONENT * np.log10(interferer_distance_factor)
            interferer_power = signal_power_dbm - path_loss_diff - np.random.uniform(5, 15)
            interferer_powers.append(10 ** (interferer_power / 10))  # Convert to linear
        
        # Sum interference power (linear domain)
        total_interference_linear = sum(interferer_powers)
        total_interference_dbm = 10 * np.log10(total_interference_linear)
        
        return total_interference_dbm
    
    def _calculate_sinr(self, signal_dbm: float, interference_dbm: float, noise_dbm: float) -> float:
        """
        Calculate SINR
        
        SINR = S / (I + N)
        """
        # Convert to linear scale
        signal_linear = 10 ** (signal_dbm / 10)
        interference_linear = 10 ** (interference_dbm / 10) if not np.isinf(interference_dbm) else 0
        noise_linear = 10 ** (noise_dbm / 10)
        
        # SINR
        sinr_linear = signal_linear / (interference_linear + noise_linear)
        sinr_db = 10 * np.log10(sinr_linear)
        
        return sinr_db
    
    def _sinr_to_coverage_prob(self, sinr_db: float) -> float:
        """
        Convert SINR to coverage probability using sigmoid
        
        P_cov = 1 / (1 + exp(-(SINR - threshold)/scale))
        """
        threshold = 0.0  # dB (50% coverage point)
        scale = 5.0  # dB (steepness)
        
        prob = 1.0 / (1.0 + np.exp(-(sinr_db - threshold) / scale))
        return prob
    
    def _calculate_qos(self, sinr_db: float, power_dbm: float, distance_m: float, coverage_prob: float) -> float:
        """
        Calculate Quality of Service score (0-100)
        
        Combines SINR, power, distance, and coverage probability
        """
        # SINR contribution (0-40 points)
        sinr_score = np.clip((sinr_db + 10) / 30 * 40, 0, 40)
        
        # Power contribution (0-20 points)
        power_score = np.clip((power_dbm + 80) / 50 * 20, 0, 20)
        
        # Distance contribution (0-20 points, closer is better)
        distance_score = np.clip((1000 - distance_m) / 1000 * 20, 0, 20)
        
        # Coverage probability (0-20 points)
        coverage_score = coverage_prob * 20
        
        qos = sinr_score + power_score + distance_score + coverage_score
        
        return np.clip(qos, 0, 100)
    
    def _estimate_throughput(self, sinr_db: float, bandwidth_mhz: float) -> float:
        """
        Estimate throughput using Shannon capacity
        
        C = B · log₂(1 + SINR)
        """
        sinr_linear = 10 ** (sinr_db / 10)
        capacity_mbps = bandwidth_mhz * np.log2(1 + sinr_linear)
        
        # Practical efficiency factor (0.7)
        throughput = capacity_mbps * 0.7
        
        return max(throughput, 0.1)  # Minimum 0.1 Mbps
    
    def _validate_dataset(self, dataset: List[Dict]):
        """Validate generated dataset for physical correctness"""
        print("\n" + "=" * 60)
        print("Dataset Validation")
        print("=" * 60)
        
        sinr = [s['sinr_db'] for s in dataset]
        power = [s['received_power_dbm'] for s in dataset]
        coverage_area = [s['coverage_area_km2'] for s in dataset]
        qos = [s['qos_score'] for s in dataset]
        
        print(f"\n✓ SINR Range: {min(sinr):.1f} - {max(sinr):.1f} dB")
        print(f"  Mean SINR: {np.mean(sinr):.1f} dB")
        print(f"  Std SINR: {np.std(sinr):.1f} dB")
        
        print(f"\n✓ Received Power: {min(power):.1f} - {max(power):.1f} dBm")
        print(f"  Mean Power: {np.mean(power):.1f} dBm")
        
        print(f"\n✓ Coverage Area: {min(coverage_area):.2f} - {max(coverage_area):.2f} km²")
        print(f"  Mean Coverage: {np.mean(coverage_area):.2f} km²")
        
        print(f"\n✓ QoS Score: {min(qos):.1f} - {max(qos):.1f}")
        print(f"  Mean QoS: {np.mean(qos):.1f}")
        
        # Check for violations
        violations = 0
        for s in dataset:
            if s['coverage_radius_m'] < 0:
                violations += 1
            if s['qos_score'] < 0 or s['qos_score'] > 100:
                violations += 1
        
        if violations == 0:
            print(f"\n✅ All {len(dataset)} samples pass physical constraints!")
        else:
            print(f"\n⚠️  {violations} constraint violations detected")
        
        # Statistics
        covered = sum(1 for s in dataset if s['is_covered'])
        good_quality = sum(1 for s in dataset if s['is_good_quality'])
        
        print(f"\n📊 Statistics:")
        print(f"  Covered points (SINR > {self.MIN_SINR}dB): {covered} ({covered/len(dataset)*100:.1f}%)")
        print(f"  Good quality (SINR > {self.TARGET_SINR}dB): {good_quality} ({good_quality/len(dataset)*100:.1f}%)")
        print(f"  Mean throughput: {np.mean([s['throughput_mbps'] for s in dataset]):.2f} Mbps")


def main():
    """Main execution"""
    print("=" * 60)
    print("Coverage Optimization Data Generator")
    print("Model 3: Mamba-3 Architecture")
    print("=" * 60)
    
    # Initialize generator
    generator = CoverageDataGenerator(seed=42)
    
    # Generate dataset
    dataset = generator.generate_dataset(num_samples=30000)
    
    # Save to JSON
    output_path = "../training_data/coverage_training_data.json"
    print(f"\n💾 Saving to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Saved {len(dataset)} samples")
    
    # File size
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"📦 File size: {size_mb:.1f} MB")
    
    print("\n" + "=" * 60)
    print("✅ Data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
