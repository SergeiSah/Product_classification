import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from stocaching import SharedCache
from .image_loader import get_images, get_images_from_zip


class ImageDescriptionDataset(Dataset):
    __cache = None
    __idx_to_image = None
    __image_to_idx = None

    @classmethod
    def set_cache(cls, idx_to_image):
        """
        Устанавливает кэш для класса ImageDescriptionDataset. Кэш хранит конкатенированные эмбеддинги изображений
        и текстов. Доступ к элементам кэша осуществляется по индексу товара. Для корректной работы необходимо
        объединить все характеристики в один датасет, обновить индексы и передать функции словарь
        {индекс: название_изображения}

        Параметры:
            idx_to_image (dict): Словарь, который сопоставляет индексы в датасете товаров с именами изображений.

        Возвращает:
            None
        """
        cls.__idx_to_image = idx_to_image
        cls.__image_to_idx = {v: k for k, v in cls.__idx_to_image.items()}

        cls.__cache = SharedCache(
            size_limit_gib=32,
            dataset_len=len(idx_to_image),
            data_dims=(1024,),
            dtype=torch.float32
        )

    @classmethod
    def clear_cache(cls):
        """
        Очищает кэш для ImageDescriptionDataset.
        """
        cls.__cache = None
        cls.__idx_to_image = None
        cls.__image_to_idx = None

    def __init__(self,
                 image_names: list[str],
                 descriptions: list[str],
                 chars: list[str],
                 predictor,
                 path_to_images: str,
                 files_in_zip: bool = True):

        self.image_names = image_names
        self.descriptions = descriptions
        self.chars = chars
        self.files_in_zip = files_in_zip

        self.char_to_label = {char: idx for idx, char in enumerate(sorted(set(chars)))}
        self.label_to_char = {idx: char for char, idx in self.char_to_label.items()}

        self.predictor = predictor
        self.path_to_images = path_to_images

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        concat_all = torch.Tensor()
        chars_all = torch.LongTensor()

        not_cached_idx = []  # индексы некэшированных элементов

        # случай, когда берется срез из датасета
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            for i in range(start, stop, step):
                if self.__cache is not None:
                    x = self.__cache.get_slot(self.__image_to_idx[self.image_names[i]])
                else:
                    x = None

                if x is None:
                    not_cached_idx.append(i)
                else:
                    concat_all = torch.cat([concat_all, x.view(1, -1)])
                    chars_all = torch.cat([chars_all, torch.LongTensor([self.char_to_label[self.chars[i]]])])

        # случай, когда берется один элемент из датасета
        else:
            if self.__cache is not None:
                x = self.__cache.get_slot(self.__image_to_idx[self.image_names[idx]])
            else:
                x = None

            if x is None:
                not_cached_idx = np.array([idx])
            else:
                concat_all = torch.cat([concat_all, x])
                chars_all = torch.cat([chars_all, torch.LongTensor([self.char_to_label[self.chars[idx]]])])

        # если есть некэшированные элементы, рассчитать эмбеддинги
        if len(not_cached_idx) > 0:
            image_names = []
            descriptions = []
            chars = []

            for idx in not_cached_idx:
                image_names.append(self.image_names[idx])
                descriptions.append(self.descriptions[idx])
                chars.append(self.chars[idx])

            if self.files_in_zip:
                images = get_images_from_zip(image_names, self.path_to_images)
            else:
                images = get_images(image_names, self.path_to_images)

            img_vecs = self.predictor.get_image_latents(images).detach().cpu()
            text_vecs = self.predictor.get_text_latents(descriptions).detach().cpu()

            concat = torch.cat([img_vecs, text_vecs], dim=1)
            chars = torch.LongTensor([self.char_to_label[char] for char in chars])

            # сохранение эмбеддингов в кэш
            if self.__cache is not None:
                for i, idx in enumerate(not_cached_idx):
                    self.__cache.set_slot(self.__image_to_idx[self.image_names[idx]], concat[i])

            concat_all = torch.cat([concat_all, concat]).squeeze()
            chars_all = torch.cat([chars_all, chars])

        return concat_all, chars_all


def get_dataloaders(df: pd.DataFrame,
                    chars: list,
                    predictor,
                    path_to_images: str,
                    batch_size: int = 256) -> dict:
    """
    Принимает DataFrame, список характеристик, объект predictor и необязательный размер пакета,
    и возвращает словарь даталоадеров для каждой характеристики.

    Arguments:
        df (pd.DataFrame): данные с указанием номера картинки товара (nm), его описания (description) и
                           характеристик.
        chars (list): Список характеристик, по которым нужно разделить данные.
        predictor: Объект predictor, используемый для получения эмбеддингов картинок и текста.
        path_to_images (str): Путь к директории с изображениями или путь к zip-архиву с изображениями.
        batch_size (int, optional): Размер каждого пакета. По умолчанию 256.

    Returns:
        dict: Словарь даталоадеров для каждой характеристики. Для каждой характеристики имеется
              два даталоадера: train и valid.
    """
    for p in ['nm', 'description', *chars]:
        if p not in df.columns:
            raise ValueError(f'Column {p} is not in df')

    if os.path.isdir(path_to_images):
        files_in_zip = False
    elif os.path.isfile(path_to_images):
        if path_to_images.endswith('.zip'):
            files_in_zip = True
        else:
            raise ValueError('Path to images must be a directory or a zip file')

    idx_to_image = {idx: image for idx, image in enumerate(df['nm'].apply(lambda x: str(x) + '.jpg').values)}
    ImageDescriptionDataset.set_cache(idx_to_image)
    dataloaders = {}

    for char in chars:
        # выбираем данные, для которых указана характеристика
        char_dataset = df[df[char].notna()]
        image_names = char_dataset['nm'].apply(lambda x: str(x) + '.jpg').values
        descriptions = char_dataset['description'].values

        # разбиваем датасет на трейн и валидационную части
        idx_train, idx_valid = train_test_split(np.arange(len(image_names)), test_size=0.2, random_state=42,
                                                stratify=char_dataset[char].values)

        # создаем даталоадеры
        dataloaders[char] = {}
        for idx, name in zip([idx_train, idx_valid], ['train', 'valid']):
            ds = ImageDescriptionDataset(image_names[idx], descriptions[idx], char_dataset[char].values[idx],
                                         predictor, path_to_images, files_in_zip)
            dataloaders[char][name] = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloaders
