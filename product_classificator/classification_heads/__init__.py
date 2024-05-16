import torch
import pickle


def load_head(name):
    return torch.load(f'./classification_heads/mlp_{name}.pt')