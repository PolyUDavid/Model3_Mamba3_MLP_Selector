"""
Coverage Mamba-3 Training Script - With Real-time Terminal Output
==================================================================

Model 3: Coverage Optimization
Architecture: Mamba-3 (Selective State Space Model)

Features:
- Real-time training progress display
- Live metrics (Loss, R², MAE)
- 3-stage learning rate schedule
- Early stopping
- Multi-task loss balancing

Author: NOK KO
Date: 2026-01-28
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import json
import numpy as np
from pathlib import Path
import sys
import time

# Add models path
sys.path.append(str(Path(__file__).parent.parent / 'models'))
from mamba3_coverage import CoverageMamba3


class CoverageDataset(Dataset):
    """Coverage dataset with normalization (V2 - improved)"""
    
    def __init__(self, json_path: str, feature_stats=None):
        # Try V2 first, fallback to V1
        v2_path = json_path.replace('.json', '_v2.json')
        try:
            with open(v2_path, 'r') as f:
                data = json.load(f)
                print(f"   ✓ Loaded V2 data: {v2_path}")
        except FileNotFoundError:
            with open(json_path, 'r') as f:
                data = json.load(f)
                print(f"   ✓ Loaded V1 data: {json_path}")
        
        # Extract features (13 features)
        features = torch.tensor([
            [
                sample['rsu_x_position_m'],
                sample['rsu_y_position_m'],
                sample['tx_power_dbm'],
                sample['antenna_tilt_deg'],
                sample['antenna_azimuth_deg'],
                sample['distance_to_rx_m'],
                sample['angle_to_rx_deg'],
                sample['building_density'],
                sample['weather_condition'],
                sample['vehicle_density_per_km2'],
                sample['num_interferers'],
                sample['rx_height_m'],
                sample['frequency_ghz']
            ]
            for sample in data
        ], dtype=torch.float32)
        
        # Extract targets (5 targets)
        targets = torch.tensor([
            [
                sample['received_power_dbm'],
                sample['sinr_db'],
                sample['coverage_radius_m'],
                sample['coverage_area_km2'],
                sample['qos_score']
            ]
            for sample in data
        ], dtype=torch.float32)
        
        # Feature normalization
        if feature_stats is None:
            self.feature_mean = features.mean(dim=0)
            self.feature_std = features.std(dim=0) + 1e-8
        else:
            self.feature_mean = feature_stats['mean']
            self.feature_std = feature_stats['std']
        
        self.features = (features - self.feature_mean) / self.feature_std
        
        # Target scaling (FIXED for V2 data)
        # V2 ranges: Power[-251,-31], SINR[-156,64], Radius[150,240], Area[0.07,0.18], QoS[0,100]
        self.targets = targets.clone()
        self.targets[:, 0] = (targets[:, 0] + 260) / 230  # Power: [-251,-31] → [0.04, 0.99]
        self.targets[:, 1] = (targets[:, 1] + 170) / 230  # SINR: [-156,64] → [0.06, 1.02]
        self.targets[:, 2] = (targets[:, 2] - 150) / 90   # Radius: [150,240] → [0, 1]
        self.targets[:, 3] = (targets[:, 3] - 0.07) / 0.12  # Area: [0.07,0.18] → [0, 0.92]
        self.targets[:, 4] /= 100.0                       # QoS: [0,100] → [0, 1]
        
        # Clip to [0, 1] to ensure positivity
        self.targets = torch.clamp(self.targets, 0.0, 1.0)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
    def get_feature_stats(self):
        return {'mean': self.feature_mean, 'std': self.feature_std}


def compute_metrics(predictions, targets):
    """Compute R² and MAE for each target (FIXED for V2 scaling)"""
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    metrics = {}
    target_names = ['Power', 'SINR', 'Radius', 'Area', 'QoS']
    
    # Reverse scaling (V2)
    # Power: x * 230 - 260
    # SINR:  x * 230 - 170
    # Radius: x * 90 + 150
    # Area: x * 0.12 + 0.07
    # QoS: x * 100
    
    for i, name in enumerate(target_names):
        if i == 0:  # Power
            pred = predictions[:, i] * 230 - 260
            true = targets[:, i] * 230 - 260
        elif i == 1:  # SINR
            pred = predictions[:, i] * 230 - 170
            true = targets[:, i] * 230 - 170
        elif i == 2:  # Radius
            pred = predictions[:, i] * 90 + 150
            true = targets[:, i] * 90 + 150
        elif i == 3:  # Area
            pred = predictions[:, i] * 0.12 + 0.07
            true = targets[:, i] * 0.12 + 0.07
        else:  # QoS
            pred = predictions[:, i] * 100
            true = targets[:, i] * 100
        
        # R²
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # MAE
        mae = np.mean(np.abs(true - pred))
        
        metrics[f'r2_{name.lower()}'] = r2
        metrics[f'mae_{name.lower()}'] = mae
    
    metrics['r2_overall'] = np.mean([metrics[f'r2_{name.lower()}'] for name in target_names])
    metrics['mae_overall'] = np.mean([metrics[f'mae_{name.lower()}'] for name in target_names])
    
    return metrics


def multi_task_loss(predictions, targets):
    """
    Multi-task loss with focused weighting on Radius/Area
    
    Power/SINR are easy (linear), Radius/Area are hard (non-linear)
    """
    # MSE loss per task
    loss_power = F.mse_loss(predictions[:, 0], targets[:, 0])
    loss_sinr = F.mse_loss(predictions[:, 1], targets[:, 1])
    loss_radius = F.mse_loss(predictions[:, 2], targets[:, 2])
    loss_area = F.mse_loss(predictions[:, 3], targets[:, 3])
    loss_qos = F.mse_loss(predictions[:, 4], targets[:, 4])
    
    # Weighted sum (focus on Radius/Area - the hard tasks)
    total_loss = (
        loss_power * 0.15 +      # Power: easy
        loss_sinr * 0.15 +       # SINR: easy
        loss_radius * 0.30 +     # Radius: hard ⭐
        loss_area * 0.30 +       # Area: hard ⭐
        loss_qos * 0.10          # QoS: medium
    )
    
    # Physics penalty (soft constraints)
    penalty = 0.0
    penalty += torch.sum(F.relu(-predictions[:, 2])) * 0.05  # Radius >= 0
    penalty += torch.sum(F.relu(-predictions[:, 3])) * 0.05  # Area >= 0
    penalty += torch.sum(F.relu(-predictions[:, 4])) * 0.02  # QoS >= 0
    penalty += torch.sum(F.relu(predictions[:, 4] - 1.0)) * 0.02  # QoS <= 1
    
    return total_loss + penalty


def get_lr_schedule(epoch, total_epochs=150):
    """3-stage learning rate schedule"""
    warmup_epochs = 10
    stable_epochs = 90
    
    base_lr = 1e-4
    min_lr = 1e-6
    
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    elif epoch < warmup_epochs + stable_epochs:
        return base_lr
    else:
        progress = (epoch - warmup_epochs - stable_epochs) / (total_epochs - warmup_epochs - stable_epochs)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


def print_progress_bar(current, total, metrics, prefix='', length=40):
    """Print progress bar with metrics"""
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    
    metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    print(f'\r{prefix} [{bar}] {percent*100:.1f}% | {metrics_str}', end='', flush=True)


def train_model():
    """Main training loop"""
    
    print("\n" + "=" * 80)
    print("Coverage Mamba-3 Training - Model 3")
    print("=" * 80)
    
    # Device - Use MPS (Apple Silicon GPU) if available
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\n🖥️  Device: MPS (Apple Silicon GPU) ✅")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n🖥️  Device: CUDA GPU ✅")
    else:
        device = torch.device('cpu')
        print(f"\n🖥️  Device: CPU ⚠️")
    
    # Load data
    print("\n📂 Loading data...")
    data_path = Path(__file__).parent.parent / 'training_data' / 'coverage_training_data.json'
    
    full_dataset = CoverageDataset(str(data_path))
    feature_stats = full_dataset.get_feature_stats()
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = 21000
    val_size = 4500
    test_size = 4500
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"   ✓ Train: {len(train_dataset)} samples")
    print(f"   ✓ Val:   {len(val_dataset)} samples")
    print(f"   ✓ Test:  {len(test_dataset)} samples")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Model
    print("\n🧠 Creating model...")
    model = CoverageMamba3(
        input_dim=13,
        d_model=256,
        n_layers=8,
        d_state=16,
        d_conv=4,
        expand=2,
        output_dim=5,
        dropout=0.1
    ).to(device)
    
    print(f"   ✓ Parameters: {model.get_num_params():,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training config
    num_epochs = 150
    patience = 25
    best_val_r2 = -float('inf')
    patience_counter = 0
    
    # Training history
    history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'val_mae': [],
        'lr': []
    }
    
    print("\n" + "=" * 80)
    print("🚀 Starting Training")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Update learning rate
        lr = get_lr_schedule(epoch - 1, num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Determine stage
        if epoch <= 10:
            stage = "WARMUP"
        elif epoch <= 100:
            stage = "STABLE"
        else:
            stage = "DECAY"
        
        # ===== TRAINING =====
        model.train()
        train_losses = []
        
        print(f"\n📊 Epoch {epoch}/{num_epochs} [{stage}] | LR: {lr:.2e}")
        print("-" * 80)
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            predictions = model(features)
            loss = multi_task_loss(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Real-time progress
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = np.mean(train_losses[-50:])
                metrics = {'Loss': avg_loss}
                print_progress_bar(
                    batch_idx + 1,
                    len(train_loader),
                    metrics,
                    prefix=f"  Training  ",
                    length=50
                )
        
        print()
        
        epoch_train_loss = np.mean(train_losses)
        
        # ===== VALIDATION =====
        model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(val_loader):
                features, targets = features.to(device), targets.to(device)
                
                predictions = model(features)
                loss = multi_task_loss(predictions, targets)
                
                val_losses.append(loss.item())
                all_predictions.append(predictions)
                all_targets.append(targets)
                
                # Real-time progress
                if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(val_loader):
                    avg_loss = np.mean(val_losses)
                    metrics = {'Loss': avg_loss}
                    print_progress_bar(
                        batch_idx + 1,
                        len(val_loader),
                        metrics,
                        prefix=f"  Validation",
                        length=50
                    )
        
        print()
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        val_metrics = compute_metrics(all_predictions, all_targets)
        epoch_val_loss = np.mean(val_losses)
        
        # Print epoch summary
        print("\n  📈 Epoch Summary:")
        print(f"     Train Loss: {epoch_train_loss:.6f}")
        print(f"     Val Loss:   {epoch_val_loss:.6f}")
        print(f"     Val R²:     {val_metrics['r2_overall']:.4f} (Power: {val_metrics['r2_power']:.4f}, SINR: {val_metrics['r2_sinr']:.4f})")
        print(f"     Val MAE:    {val_metrics['mae_overall']:.4f}")
        
        epoch_time = time.time() - epoch_start
        print(f"     Time:       {epoch_time:.1f}s")
        
        # Save history
        history['epochs'].append(epoch)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_r2'].append(val_metrics['r2_overall'])
        history['val_mae'].append(val_metrics['mae_overall'])
        history['lr'].append(lr)
        
        # Check for best model
        if val_metrics['r2_overall'] > best_val_r2:
            best_val_r2 = val_metrics['r2_overall']
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'r2': val_metrics['r2_overall'],
                'mae': val_metrics['mae_overall'],
                'val_metrics': val_metrics,
                'feature_stats': feature_stats
            }
            
            checkpoint_path = Path(__file__).parent / 'best_coverage.pth'
            torch.save(checkpoint, checkpoint_path)
            
            print(f"     ⭐ NEW BEST! R² = {best_val_r2:.4f} (saved)")
        else:
            patience_counter += 1
            print(f"     Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch} (patience {patience} reached)")
            break
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best Val R²: {best_val_r2:.4f}")
    
    # Save training history
    history_path = Path(__file__).parent / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert to regular Python types
        history_clean = {
            'epochs': [int(e) for e in history['epochs']],
            'train_loss': [float(l) for l in history['train_loss']],
            'val_loss': [float(l) for l in history['val_loss']],
            'val_r2': [float(r) for r in history['val_r2']],
            'val_mae': [float(m) for m in history['val_mae']],
            'lr': [float(l) for l in history['lr']]
        }
        json.dump(history_clean, f, indent=2)
    print(f"\n💾 Training history saved to: {history_path}")
    
    # ===== FINAL TEST =====
    print("\n" + "=" * 80)
    print("🧪 Final Test Evaluation")
    print("=" * 80)
    
    # Load best model
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)
            test_predictions.append(predictions)
            test_targets.append(targets)
    
    test_predictions = torch.cat(test_predictions, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    test_metrics = compute_metrics(test_predictions, test_targets)
    
    print(f"\n📊 Test Results:")
    print(f"   Overall R²:  {test_metrics['r2_overall']:.4f} {'✅' if test_metrics['r2_overall'] > 0.93 else '⚠️'}")
    print(f"   Overall MAE: {test_metrics['mae_overall']:.4f}")
    print(f"\n   Per-target R²:")
    print(f"     Power:  {test_metrics['r2_power']:.4f} (MAE: {test_metrics['mae_power']:.2f} dBm)")
    print(f"     SINR:   {test_metrics['r2_sinr']:.4f} (MAE: {test_metrics['mae_sinr']:.2f} dB)")
    print(f"     Radius: {test_metrics['r2_radius']:.4f} (MAE: {test_metrics['mae_radius']:.2f} m)")
    print(f"     Area:   {test_metrics['r2_area']:.4f} (MAE: {test_metrics['mae_area']:.4f} km²)")
    print(f"     QoS:    {test_metrics['r2_qos']:.4f} (MAE: {test_metrics['mae_qos']:.2f})")
    
    print(f"\n🎯 Target Achievement:")
    print(f"   R² > 0.93:  {'✅ PASS' if test_metrics['r2_overall'] > 0.93 else '❌ FAIL'}")
    
    print("\n" + "=" * 80)
    print("🎉 Model 3 Training Complete!")
    print("=" * 80)
    
    return test_metrics


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train model
    test_metrics = train_model()
