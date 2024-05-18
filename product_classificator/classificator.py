import os
import pickle
import torch
import numpy as np
from PIL import Image

from .ruclip_model import CLIP
from .processor import RuCLIPProcessor
from .predictor import Predictor


class Classificator:
    path_to_heads = os.path.join(os.path.dirname(__file__), '/')
    available_heads = ['category', 'sub_category', 'isadult', 'sex', 'season', 'age_restrictions', 'fragility']

    def __init__(self,
                 device='cpu',
                 quiet=True,
                 heads=None,
                 model_name='ruclip-vit-base-patch16-384',
                 cache_dir='/tmp/ruclip/'):

        if heads is None:
            heads = ['category', 'sub_category', 'isadult']
        else:
            for head in heads:
                if head not in self.available_heads:
                    raise ValueError(f'Unknown head: {head}, available heads: {self.available_heads}')

        self.clip_predictor = Predictor(
            CLIP.from_pretrained(cache_dir + model_name).eval().to(device),
            RuCLIPProcessor.from_pretrained(cache_dir + model_name),
            device,
            quiet=quiet
        )

        self.heads = {}
        for cl in heads:
            self.load_head(cl)

        with open(f'{self.path_to_heads}id_to_label.pkl', 'rb') as f:
            self.id_to_label = pickle.load(f)

        with open(f'{self.path_to_heads}pca_all.pkl', 'rb') as f:
            self.reducer = pickle.load(f)

    def classify_products(self, texts: list[str], images: list[Image.Image], characteristics: list[str] = None):
        """
        Классифицирует товары на основе их текстовых описаний и изображений.

        Arguments:
            texts (list[str]): Список описаний товаров.
            images (list[Image.Image]): Список изображений товаров.
            characteristics (list[str], optional): Список характеристик для классификации товаров. По умолчанию включает
                                                   'category', 'sub_category', 'isadult'.

        Returns:
            dict: Словарь с результатами классификации для каждой характеристики. Ключи - названия характеристик,
                  значения - списки соответствующих меток характеристик для каждого продукта.
        """
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
        """
        Загружает MLP классификатор, обученный для предсказания заданной характеристики товара.

        Arguments:
            name (str): Название характеристики для которой будет загружен классификатор.
        """
        self.heads[name] = torch.load(f'{self.path_to_heads}{name}.pt')
        self.heads[name].eval()

