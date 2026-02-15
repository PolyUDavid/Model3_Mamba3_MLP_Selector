"""
Generate D²TL full architecture: Selector Brain + MLP + Mamba3.
Saves to paper_package/08_backbones/plots/D2TL_full_architecture.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')

color_input = '#E8F4F8'
color_selector = '#E8E0F0'   # light purple
color_physics = '#D4C8E8'
color_mlp = '#B8E6F0'       # light blue
color_mamba = '#FFE5CC'     # light orange
color_output = '#C8E6C9'     # light green

def box(ax, x, y, w, h, text, color, fs=9, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(p)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, weight=weight, wrap=True)

def arrow(ax, x1, y1, x2, y2, color='#333'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', color=color, lw=2, mutation_scale=15))

# Title
ax.text(7, 11.4, 'D²TL Full Architecture — Selector + MLP + Mamba3', ha='center', fontsize=14, weight='bold')
ax.text(7, 10.9, 'Dual-Path with Physics-Aware Trigger Logic | Ports 8000, 8001, 8002', ha='center', fontsize=9, style='italic', color='gray')

# Input
box(ax, 4.5, 9.8, 5, 0.5, 'Input: 13 features (RSU, env, distance, weather, density, interference)', color_input, 9, bold=True)
arrow(ax, 7, 9.8, 7, 9.2)

# Selector Brain (center top)
box(ax, 3.5, 8.2, 7, 0.9, 'Selector Brain (Port 8000)\nPhysicsAnalyzer · Trigger Score · Divergence Check · Decision', color_selector, 9, bold=True)
arrow(ax, 7, 8.2, 5.5, 7.6)
arrow(ax, 7, 8.2, 8.5, 7.6)

# Two branches: MLP left, Mamba right
box(ax, 1.2, 5.8, 3.8, 1.4,
    'MLP Service (8001)\nPrimary Path\n8-layer MLP, 256D\n~469K params · 1–7 ms', color_mlp, 9, bold=True)
box(ax, 9.0, 5.8, 3.8, 1.4,
    'Mamba Service (8002)\nBackup Path\n8× MambaBlock, 256D\n~13.7M params · physics', color_mamba, 9, bold=True)

arrow(ax, 5.5, 8.2, 3.1, 7.2)
arrow(ax, 8.5, 8.2, 10.9, 7.2)

# Decision logic note
ax.text(5.5, 7.0, 's ≥ 0.3 → Mamba\nelse → MLP', ha='center', fontsize=8, style='italic', color='#555')
ax.text(8.5, 7.0, 'divergent ∨\nextreme', ha='center', fontsize=8, style='italic', color='#555')

# Outputs from backbones
box(ax, 1.2, 4.2, 3.8, 0.5, '5 outputs (Power, SINR, Radius, Area, QoS)', color_mlp, 8)
box(ax, 9.0, 4.2, 3.8, 0.5, '5 outputs (Power, SINR, Radius, Area, QoS)', color_mamba, 8)
arrow(ax, 3.1, 5.8, 3.1, 4.7)
arrow(ax, 10.9, 5.8, 10.9, 4.7)

# Merge to final output
arrow(ax, 3.1, 4.45, 6, 4.0)
arrow(ax, 10.9, 4.45, 8, 4.0)
box(ax, 5.5, 3.5, 3, 0.6, 'Chosen prediction (MLP or Mamba)', color_output, 10, bold=True)
box(ax, 5.2, 2.6, 3.6, 0.5, 'Output: 5 scalars + metadata (trigger, risk, reasons)', color_output, 9)
arrow(ax, 7, 4.0, 7, 4.1)
arrow(ax, 7, 3.5, 7, 3.1)

# Physics conditions (small legend)
ax.text(7, 1.8, 'Trigger: d>500m, weather≥2, density≥2, N_int≥3, divergence>5dB', ha='center', fontsize=8, style='italic', color='gray')
ax.text(7, 1.2, 'Model 3 D²TL — Selector + MLP + Mamba3 | Author: NOK KO', ha='center', fontsize=8, style='italic', color='gray')

plt.tight_layout()
out = OUT_DIR / "D2TL_full_architecture.png"
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")
