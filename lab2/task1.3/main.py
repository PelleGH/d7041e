import matplotlib.pyplot as plt
import torch
from dataset import get_sentences
from infer import infer
from models import get_models
from plot import cosine_matrix, plot_2d, plot_3d
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plt_3d(S, name):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    plot_3d(ax, S, name)
    fig.tight_layout()
    fig.savefig(f"img/{name}_3d.png", dpi=200)


def plt_2d(S, name):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot()
    plot_2d(ax, S, name)
    fig.tight_layout()
    fig.savefig(f"img/{name}_2d.png", dpi=200)

def main():
    sentences = get_sentences()
    models = get_models()

    device = torch.device("cpu")

    for name, modelTuple in models:
        emb = infer(modelTuple, sentences, device)
        S = cosine_matrix(emb)

        plt_2d(S, name)
        plt_3d(S, name)

    plt.show()

if __name__ == "__main__":
    main()