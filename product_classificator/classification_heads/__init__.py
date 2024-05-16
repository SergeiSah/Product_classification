import os
import torch
import pickle


def load_head(name):
    return torch.load(os.path.join(os.path.dirname(__file__), f'mlp_{name}.pt'))


def load_id_to_label():
    with open(os.path.join(os.path.dirname(__file__), 'id_to_label.pkl'), 'rb') as f:
        return pickle.load(f)


def load_pca():
    with open(os.path.join(os.path.dirname(__file__), 'pca_all.pkl'), 'rb') as f:
        return pickle.load(f)
