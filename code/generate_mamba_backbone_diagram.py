"""
Generate Mamba-3 backbone architecture diagram.
Saves to paper_package/08_backbones/plots/Mamba3_backbone_architecture.png
Adapted from repo root generate_backbone_diagram.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 22)
ax.axis('off')

color_input = '#E8F4F8'
color_mamba = '#B8E6F0'
color_ssm = '#7ECEE0'
color_head = '#FFE5CC'
color_output = '#FFD4B8'

def draw_box(ax, x, y, width, height, text, color, fontsize=9, bold=False):
    p = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(p)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, weight=weight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, width=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', color='#555', linewidth=width, mutation_scale=15))

ax.text(5, 21.2, 'CoverageMamba3 — Physics Backup Backbone', ha='center', fontsize=14, weight='bold')
ax.text(5, 20.6, 'Selective SSM, 8 blocks, d_model=256, d_state=16 | ~13.7M params', ha='center', fontsize=9, style='italic', color='gray')

y = 19.5
draw_box(ax, 1.5, y - 0.5, 7, 0.5, 'Input: 13 features (RSU + env + interference)', color_input, 10, bold=True)
draw_arrow(ax, 5, y - 0.5, 5, 18.5)

draw_box(ax, 2, 18, 6, 0.45, 'Input projection: Linear(13→256) · LayerNorm · GELU · Dropout(0.1)', color_input, 8)
draw_arrow(ax, 5, 18, 5, 17.2)

ax.text(5, 16.8, '8 × MambaBlock', ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#D4E6F1', edgecolor='black'))
draw_box(ax, 2.5, 15.2, 5, 0.5, 'Each: LayerNorm → InProj → SelectiveSSM → Gate → OutProj + residual', color_ssm, 8)
draw_arrow(ax, 5, 15.2, 5, 14.2)

draw_box(ax, 2.5, 13.5, 5, 0.45, 'SelectiveSSM: Conv1d(4) → Δ,B,C → stable SSM → skip(D) → OutProj', color_ssm, 8)
draw_arrow(ax, 5, 13.5, 5, 12.5)

draw_box(ax, 2.5, 11.8, 5, 0.45, 'Final LayerNorm(256) · mean(seq) → 256D', color_mamba, 8)
draw_arrow(ax, 5, 11.8, 5, 10.8)

ax.text(5, 10.3, '5 × Linear(256→1)', ha='center', fontsize=10, weight='bold')
heads = ['Power', 'SINR', 'Radius', 'Area', 'QoS']
for i, h in enumerate(heads):
    x = 1.2 + i * 1.92
    draw_box(ax, x, 9.2, 1.5, 0.5, h, color_head, 8, bold=True)
    draw_arrow(ax, 5, 10.8, x + 0.75, 9.7)
draw_box(ax, 1.5, 8.4, 7, 0.45, 'Output: 5 scalars (Power, SINR, Radius, Area, QoS)', color_output, 9, bold=True)

ax.text(5, 7.6, 'Physics: path-loss slope, rain atten 6.97–9.73 dB, density scaling', ha='center', fontsize=8, style='italic', color='gray')
ax.text(5, 7.0, 'Model 3 D²TL — Mamba-3 backbone | Author: NOK KO', ha='center', fontsize=8, style='italic', color='gray')
plt.tight_layout()
out = OUT_DIR / "Mamba3_backbone_architecture.png"
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")
