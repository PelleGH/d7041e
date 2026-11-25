import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results.csv")
df["acc"] = df["acc"] / 100  # percent -> decimal

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["d"], df["acc"], marker="o", markersize=6, linewidth=2,
        color="#2a6fdb", label="Accuracy")
ax.plot(df["d"], df["f1"], marker="s", markersize=6, linewidth=2,
        color="#e8731a", label="F1 (macro)", linestyle="--")

ax.set_title("Accuracy and F1 (macro) depending on dimension count")
ax.set_xlabel("Number of dimensions")
ax.set_ylabel("Score")
ax.set_ylim(0.6, 1)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")

fig.tight_layout()
fig.savefig("results.png", dpi=200)
plt.show()