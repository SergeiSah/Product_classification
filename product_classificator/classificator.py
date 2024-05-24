import os
import pickle
import torch
import numpy as np
from PIL import Image

from .ruclip.ruclip_model import CLIP
from .ruclip.processor import RuCLIPProcessor
from .ruclip.predictor import Predictor


class Classificator:
    path_to_heads = os.path.join(os.path.dirname(__file__), 'heads')
    available_heads = ['category', 'sub_category', 'isadult', 'sex', 'season', 'age_restrictions', 'fragility']

    def __init__(self,
                 device='cpu',
                 heads=None,
                 model_name='ruclip-vit-base-patch16-384',
                 heads_ver='wb-6_cats-pca',
                 cache_dir='/tmp/ruclip/'):

        self.device = device

        if heads is None:
            heads = ['category', 'sub_category', 'isadult']
        else:
            for head in heads:
                if head not in self.available_heads:
                    raise ValueError(f'Unknown head: {head}, available heads: {self.available_heads}')

        self.heads_ver = heads_ver

        self.clip_predictor = Predictor(
            CLIP.from_pretrained(cache_dir + model_name).eval().to(device),
            RuCLIPProcessor.from_pretrained(cache_dir + model_name),
            device,
        )

        self.heads = {}
        for cl in heads:
            self.load_head(cl)

        with open(f'{self.path_to_heads}/{self.heads_ver}/label_to_char.pkl', 'rb') as f:
            self.label_to_char = pickle.load(f)

        with open(f'{self.path_to_heads}/{self.heads_ver}/pca_all.pkl', 'rb') as f:
            self.reducer = pickle.load(f)

    def classify_products(self,
                          texts: list[str] | str,
                          images: list[Image.Image] | Image.Image,
                          characteristics: list[str] = None):
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

        with torch.no_grad():
            text_vec = self.clip_predictor.get_text_latents(texts).detach().cpu().numpy()
            image_vec = self.clip_predictor.get_image_latents(images).detach().cpu().numpy()

        concat_vec = np.concatenate([text_vec, image_vec], axis=1)

        if self.reducer is not None:
            concat_vec = self.reducer.transform(concat_vec)

        results = {}
        for name in characteristics:
            indexes = self.heads[name](torch.tensor(concat_vec)).argmax(dim=1).detach().cpu().numpy()
            results[name] = [self.label_to_char[name][index] for index in indexes]

        return results

    def load_head(self, name):
        """
        Загружает MLP классификатор, обученный для предсказания заданной характеристики товара.

        Arguments:
            name (str): Название характеристики для которой будет загружен классификатор.
        """
        self.heads[name] = torch.load(f'{self.path_to_heads}/{self.heads_ver}/{name}.pt')
        self.heads[name].eval()
