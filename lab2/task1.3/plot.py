import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors


def cosine_matrix(E: torch.Tensor) -> np.ndarray:
    E = torch.nn.functional.normalize(E, p=2, dim=1)
    return (E @ E.T).numpy()

def plot_2d(ax, S, title, labels=None):
    n = S.shape[0]
    labels = labels or [f"S{i+1}" for i in range(n)]

    im = ax.imshow(S, cmap="viridis", vmin=-1, vmax=1)    # same scale on all models

    # number in each square
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if S[i, j] < 0.6 else "black")   # readable on dark and light cells

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    return ax

def plot_3d(ax, S, title, labels=None):
    n = S.shape[0]
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    x, y, z = x.ravel(), y.ravel(), S.ravel()

    norm = colors.Normalize(vmin=-1, vmax=1)            # same scale on every plot
    bar_colors = cm.viridis(norm(z))

    ax.bar3d(x - 0.4, y - 0.4, np.zeros_like(z),          # centre bars on integer ticks
             0.8, 0.8, z,
             color=bar_colors, shade=True,                # shade=True gives light/dark faces
             edgecolor="white", linewidth=0.3,            # thin edges separate the bars
             alpha=0.95)

    ax.set_title(title, pad=12)
    ax.set_zlim(-0.2, 1)
    ax.view_init(elev=28, azim=-55)
    ax.set_box_aspect((1, 1, 0.6))                        # lower the height so tall bars don't dominate

    ticks = range(n)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([f"S{i+1}" for i in ticks], fontsize=8)
    ax.set_yticklabels([f"S{i+1}" for i in ticks], fontsize=8)
    ax.set_zlabel("cosine similarity")

    ax.xaxis.pane.set_alpha(0.05); ax.yaxis.pane.set_alpha(0.05); ax.zaxis.pane.set_alpha(0.05)
    ax.grid(alpha=0.3)

    sm = cm.ScalarMappable(norm=norm, cmap="viridis")      # colorbar legend
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label="similarity")