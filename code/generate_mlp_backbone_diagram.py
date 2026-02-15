"""Generate MLP backbone architecture diagram. Saves to 08_backbones/plots/."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

def box(ax, x, y, w, h, text, color='#E8F4F8', fs=9):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', color='#333', lw=2, mutation_scale=15))

# Title
ax.text(5, 13.3, 'CoverageMLP — Primary Path Backbone', ha='center', fontsize=14, weight='bold')
ax.text(5, 12.8, '8-layer MLP, 256 hidden, 5 heads | ~469K params', ha='center', fontsize=9, style='italic', color='gray')

y = 12
box(ax, 2, y - 0.5, 6, 0.5, 'Input: 13 features (RSU + env + interference)', '#E8F4F8', 9)
arrow(ax, 5, y - 0.5, 5, 11.2)

y = 11
box(ax, 2.5, y - 0.4, 5, 0.4, 'Linear(13 → 256) · LayerNorm · GELU', '#B8E6F0', 8)
arrow(ax, 5, y - 0.4, 5, 10.4)

for i in range(7):
    y = 10 - i * 1.1
    box(ax, 2.5, y - 0.4, 5, 0.4, f'Linear(256→256) · LayerNorm · GELU · Dropout(0.1)  [Layer {i+2}]', '#B8E6F0', 7)
    arrow(ax, 5, y - 0.4, 5, y - 0.95)

y = 2.2
box(ax, 2.5, y - 0.35, 5, 0.35, 'Hidden 256D', '#D4E6F1', 8)
arrow(ax, 5, y - 0.35, 5, 1.5)

ax.text(5, 1.25, '5 × Linear(256 → 1)', ha='center', fontsize=9, weight='bold')
heads = ['Power', 'SINR', 'Radius', 'Area', 'QoS']
for i, h in enumerate(heads):
    x = 1.2 + i * 1.9
    box(ax, x, 0.3, 1.5, 0.5, h, '#FFE5CC', 8)
    arrow(ax, 5, 1.5, x + 0.75, 0.8)
box(ax, 1.5, -0.35, 7, 0.35, 'Output: 5 scalars (dBm, dB, m, km², %)', '#FFD4B8', 9)

ax.text(5, -0.9, 'Model 3 D²TL — MLP backbone | Author: NOK KO', ha='center', fontsize=8, style='italic', color='gray')
plt.tight_layout()
out = OUT_DIR / "MLP_backbone_architecture.png"
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")
