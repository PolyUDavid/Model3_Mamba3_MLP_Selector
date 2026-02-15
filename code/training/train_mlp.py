#!/usr/bin/env python3
"""
MLP Training Script — Primary Path Model
==========================================

Trains CoverageMLP using the SAME dataset and pipeline as Mamba-3
to ensure fair comparison. Uses the Mamba checkpoint's feature_stats
for consistent normalization.

Author: NOK KO
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
import copy

BASE_DIR = Path(__file__).parent.parent.parent  # Model_3_Coverage_Mamba3/
sys.path.insert(0, str(BASE_DIR / 'd2tl' / 'mlp_service'))
from model import CoverageMLP


class CoverageDataset(Dataset):
    """Coverage dataset with normalization (same as Mamba training)."""
    
    def __init__(self, json_path: str, feature_stats=None):
        v2_path = json_path.replace('.json', '_v2.json')
        try:
            with open(v2_path, 'r') as f:
                data = json.load(f)
                print(f"   Loaded V2 data: {v2_path}")
        except FileNotFoundError:
            with open(json_path, 'r') as f:
                data = json.load(f)
                print(f"   Loaded V1 data: {json_path}")
        
        features = torch.tensor([
            [s['rsu_x_position_m'], s['rsu_y_position_m'], s['tx_power_dbm'],
             s['antenna_tilt_deg'], s['antenna_azimuth_deg'], s['distance_to_rx_m'],
             s['angle_to_rx_deg'], s['building_density'], s['weather_condition'],
             s['vehicle_density_per_km2'], s['num_interferers'],
             s['rx_height_m'], s['frequency_ghz']]
            for s in data
        ], dtype=torch.float32)
        
        targets = torch.tensor([
            [s['received_power_dbm'], s['sinr_db'], s['coverage_radius_m'],
             s['coverage_area_km2'], s['qos_score']]
            for s in data
        ], dtype=torch.float32)
        
        if feature_stats is None:
            self.feature_mean = features.mean(dim=0)
            self.feature_std = features.std(dim=0) + 1e-8
        else:
            self.feature_mean = feature_stats['mean']
            self.feature_std = feature_stats['std']
        
        self.features = (features - self.feature_mean) / self.feature_std
        
        # Target scaling (same as Mamba training)
        self.targets = targets.clone()
        self.targets[:, 0] = (targets[:, 0] + 260) / 230
        self.targets[:, 1] = (targets[:, 1] + 170) / 230
        self.targets[:, 2] = (targets[:, 2] - 150) / 90
        self.targets[:, 3] = (targets[:, 3] - 0.07) / 0.12
        self.targets[:, 4] /= 100.0
        self.targets = torch.clamp(self.targets, 0.0, 1.0)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
    def get_feature_stats(self):
        return {'mean': self.feature_mean, 'std': self.feature_std}


def multi_task_loss(predictions, targets):
    """Same loss as Mamba training for fair comparison."""
    loss_power = F.mse_loss(predictions[:, 0], targets[:, 0])
    loss_sinr = F.mse_loss(predictions[:, 1], targets[:, 1])
    loss_radius = F.mse_loss(predictions[:, 2], targets[:, 2])
    loss_area = F.mse_loss(predictions[:, 3], targets[:, 3])
    loss_qos = F.mse_loss(predictions[:, 4], targets[:, 4])
    
    total = (loss_power * 0.15 + loss_sinr * 0.15 +
             loss_radius * 0.30 + loss_area * 0.30 + loss_qos * 0.10)
    
    penalty = 0.0
    penalty += torch.sum(F.relu(-predictions[:, 2])) * 0.05
    penalty += torch.sum(F.relu(-predictions[:, 3])) * 0.05
    penalty += torch.sum(F.relu(-predictions[:, 4])) * 0.02
    penalty += torch.sum(F.relu(predictions[:, 4] - 1.0)) * 0.02
    
    return total + penalty


def compute_metrics(predictions, targets):
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    metrics = {}
    names = ['Power', 'SINR', 'Radius', 'Area', 'QoS']
    scales = [(230, -260), (230, -170), (90, 150), (0.12, 0.07), (100, 0)]
    
    for i, (name, (s, o)) in enumerate(zip(names, scales)):
        pred = predictions[:, i] * s + o if i < 4 else predictions[:, i] * s
        true = targets[:, i] * s + o if i < 4 else targets[:, i] * s
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        metrics[f'r2_{name.lower()}'] = 1 - ss_res / (ss_tot + 1e-8)
        metrics[f'mae_{name.lower()}'] = np.mean(np.abs(true - pred))
    
    metrics['r2_overall'] = np.mean([metrics[f'r2_{n.lower()}'] for n in names])
    metrics['mae_overall'] = np.mean([metrics[f'mae_{n.lower()}'] for n in names])
    return metrics


def train():
    print("\n" + "=" * 80)
    print("CoverageMLP Training — Primary Path Model")
    print("=" * 80)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"\n  Device: {device}")
    
    # Load feature stats from Mamba checkpoint (for consistent normalization)
    mamba_ckpt = torch.load(
        str(BASE_DIR / 'training' / 'best_coverage.pth'),
        weights_only=False, map_location='cpu')
    feature_stats = mamba_ckpt.get('feature_stats', None)
    print(f"  Feature stats loaded from Mamba checkpoint")
    
    # Dataset
    data_path = str(BASE_DIR / 'training_data' / 'coverage_training_data.json')
    dataset = CoverageDataset(data_path, feature_stats)
    
    train_size = 21000
    val_size = 4500
    test_size = len(dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Model
    model = CoverageMLP(input_dim=13).to(device)
    print(f"  MLP Parameters: {model.get_num_params():,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)
    
    best_r2 = -999
    best_state = None
    patience_counter = 0
    history = {'epochs': [], 'train_loss': [], 'val_loss': [], 'val_r2': [], 'val_mae': [], 'lr': []}
    
    print("\n" + "=" * 80)
    print("Training...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(1, 151):
        model.train()
        losses = []
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            pred = model(features)
            loss = multi_task_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_tgts, val_losses = [], [], []
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                pred = model(features)
                val_losses.append(multi_task_loss(pred, targets).item())
                val_preds.append(pred)
                val_tgts.append(targets)
        
        val_preds = torch.cat(val_preds)
        val_tgts = torch.cat(val_tgts)
        metrics = compute_metrics(val_preds, val_tgts)
        
        history['epochs'].append(epoch)
        history['train_loss'].append(float(np.mean(losses)))
        history['val_loss'].append(float(np.mean(val_losses)))
        history['val_r2'].append(float(metrics['r2_overall']))
        history['val_mae'].append(float(metrics['mae_overall']))
        history['lr'].append(float(scheduler.get_last_lr()[0]))
        
        if metrics['r2_overall'] > best_r2:
            best_r2 = metrics['r2_overall']
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == 1 or patience_counter == 0:
            marker = " *BEST*" if patience_counter == 0 else ""
            print(f"  Epoch {epoch:3d}/150 | Loss: {np.mean(losses):.6f} | "
                  f"Val R²: {metrics['r2_overall']:.4f} | Best: {best_r2:.4f}{marker}")
        
        if patience_counter >= 30:
            print(f"\n  Early stopping at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    model.load_state_dict(best_state)
    
    # Test evaluation
    model.eval()
    test_preds, test_tgts = [], []
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            test_preds.append(model(features))
            test_tgts.append(targets)
    
    test_preds = torch.cat(test_preds)
    test_tgts = torch.cat(test_tgts)
    test_metrics = compute_metrics(test_preds, test_tgts)
    
    print(f"\n  Test Results:")
    print(f"    Overall R²:  {test_metrics['r2_overall']:.4f}")
    print(f"    Overall MAE: {test_metrics['mae_overall']:.4f}")
    for name in ['Power', 'SINR', 'Radius', 'Area', 'QoS']:
        print(f"    {name:8s}: R²={test_metrics[f'r2_{name.lower()}']:.4f}  MAE={test_metrics[f'mae_{name.lower()}']:.4f}")
    
    # Save
    save_dir = BASE_DIR / 'd2tl' / 'mlp_service'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'feature_stats': feature_stats if feature_stats else dataset.get_feature_stats(),
        'r2': float(test_metrics['r2_overall']),
        'mae': float(test_metrics['mae_overall']),
        'val_metrics': {k: float(v) for k, v in test_metrics.items()},
        'epoch': len(history['epochs']),
        'training_time_min': total_time / 60,
    }
    torch.save(checkpoint, save_dir / 'best_mlp_coverage.pth')
    print(f"\n  Saved: {save_dir / 'best_mlp_coverage.pth'}")
    
    # Save history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Also save to training dir
    train_dir = BASE_DIR / 'd2tl' / 'training'
    train_dir.mkdir(parents=True, exist_ok=True)
    with open(train_dir / 'mlp_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n  Training time: {total_time/60:.1f} min")
    print(f"  Best R²: {best_r2:.4f}")
    
    return model, history


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train()
