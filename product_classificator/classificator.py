import os
import pickle
import torch
import numpy as np
import ruclip
from PIL import Image


class Classificator:
    path_to_heads = os.path.dirname(__file__) + '/classification_heads/'

    def __init__(self, device='cpu', quiet=True, classes=None, ruclip_model='ruclip-vit-base-patch16-384'):
        if classes is None:
            classes = ['category', 'sub_category', 'isadult']

        self.clip, self.processor = ruclip.load(ruclip_model)
        self.clip_predictor = ruclip.Predictor(self.clip, self.processor, device, quiet=quiet)

        self.heads = {}
        for cl in classes:
            self.load_head(cl)

        with open(f'{self.path_to_heads}id_to_label.pkl', 'rb') as f:
            self.id_to_label = pickle.load(f)

        with open(f'{self.path_to_heads}pca_all.pkl', 'rb') as f:
            self.reducer = pickle.load(f)

    def classify_products(self, texts: list[str], images: list[Image.Image], characteristics: list[str] = None):
        if characteristics is None:
            characteristics = self.heads.keys()

        text_vec = self.clip_predictor.get_text_latents(texts).detach().cpu().numpy()
        image_vec = self.clip_predictor.get_image_latents(images).detach().cpu().numpy()
        concat_vec = np.concatenate([text_vec, image_vec], axis=1)
        reduced_vec = self.reducer.transform(concat_vec)

        results = {}
        for name in characteristics:
            indexes = self.heads[name](torch.tensor(reduced_vec)).argmax(dim=1).detach().cpu().numpy()
            results[name] = [self.id_to_label[name][index] for index in indexes]

        return results

    def load_head(self, name):
        self.heads[name] = torch.load(f'{self.path_to_heads}mlp_{name}.pt')
        self.heads[name].eval()
