import os
import pickle
import torch
from PIL import Image

from .utils import export_onnx
from .ruclip.model import CLIP
from .ruclip.processor import RuCLIPProcessor
from .ruclip.predictor import Predictor
from .ruclip.onnx_model import ONNXCLIP


class Classificator:
    path_to_heads = os.path.join(os.path.dirname(__file__), 'heads')
    base_head_version = 'wb-6_cats-pca'

    def __init__(self,
                 device='cpu',
                 heads=None,
                 model_name='ruclip-vit-base-patch16-384',
                 heads_ver: str = 'wb-6_cats-pca',
                 cache_dir: str = '/tmp/ruclip/'):

        self.is_onnx = False
        self.model_name = model_name
        self.device = device
        self.heads_ver = heads_ver
        self.cache_dir = cache_dir
        self.available_heads = [head.split('.')[0] for head in os.listdir(os.path.join(self.path_to_heads, heads_ver))
                                if head.endswith('.pt')]

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

        concat_vec = self.get_products_embeddings(texts, images).detach().cpu().numpy()

        if self.reducer is not None:
            concat_vec = self.reducer.transform(concat_vec)

        results = {}
        for name in characteristics:
            indexes = self.heads[name](torch.tensor(concat_vec)).argmax(dim=1).detach().cpu().numpy()
            results[name] = [self.label_to_char[name][index] for index in indexes]

        return results

    def get_products_embeddings(self,
                                texts: str | list[str],
                                images: Image.Image | list[Image.Image]) -> torch.Tensor:

        with torch.inference_mode():
            text_vec = self.clip_predictor.get_text_latents(texts)
            image_vec = self.clip_predictor.get_image_latents(images)

            concat_vec = torch.cat((image_vec, text_vec), dim=1)

        return concat_vec

    def load_head(self, name):
        """
        Загружает MLP классификатор, обученный для предсказания заданной характеристики товара.

        Arguments:
            name (str): Название характеристики для которой будет загружен классификатор.
        """
        self.heads[name] = torch.load(f'{self.path_to_heads}/{self.heads_ver}/{name}.pt')
        self.heads[name].eval()

    def get_clip(self):
        return self.clip_predictor.clip_model

    def get_processor(self):
        return self.clip_predictor.clip_processor

    def export_model_to_onnx(self, text, image, params: dict = None):
        params = params or dict(
            input_names=['input'], output_names=['output'],
            export_params=True, verbose=False, opset_version=17,
            do_constant_folding=True,
            dynamic_axes={
                "input":  {0: "batch_size",
                           1: "sequence_len"},
                "output": {0: "batch_size"}
            })

        export_onnx(self, text, image, params)

    def to_onnx_clip(self):
        if not self.is_onnx_created():
            raise FileNotFoundError('.onnx files not found. Run `export_model_to_onnx` method first')

        clip = self.clip_predictor.clip_model

        self.clip_predictor.clip_model = ONNXCLIP(clip.clip if self.is_onnx else clip,
                                                  self.device,
                                                  os.path.join(self.cache_dir, self.model_name))
        self.is_onnx = True

    def to_clip(self):
        self.clip_predictor.clip_model = CLIP.from_pretrained(self.cache_dir + self.model_name).eval().to(self.device)
        self.is_onnx = False

    def is_onnx_created(self):
        path_to_model = os.path.join(self.cache_dir, self.model_name)
        is_visual = os.path.exists(os.path.join(path_to_model, "clip_visual.onnx"))
        is_transformer = os.path.exists(os.path.join(path_to_model, "clip_transformer.onnx"))
        return is_visual and is_transformer

    def to(self, device):
        if self.device == device:
            return self

        self.device = device
        self.clip_predictor.device = device

        if self.is_onnx:
            self.to_onnx_clip()
        else:
            self.clip_predictor.clip_model = self.clip_predictor.clip_model.to(device)

        return self

    def __repr__(self):
        return f'''Classificator: 
                    model_name={self.model_name}, 
                    heads_ver={self.heads_ver}, 
                    heads={self.heads.keys()}, 
                    device={self.device}'''
