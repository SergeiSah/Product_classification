import os
import gzip
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from stocaching import SharedCache
from .image_loader import get_images, get_images_from_zip


class Cache:

    def __init__(self, path_to_cache_dir: str):
        self.path_to_cache_dir = path_to_cache_dir
        self.check_dir()

    def check_dir(self):
        if not os.path.exists(self.path_to_cache_dir):
            os.makedirs(self.path_to_cache_dir)

    def get(self, image_name: str, key: str) -> torch.Tensor | torch.LongTensor | None:
        try:
            image_name = image_name.split('.')[0]
            if key == 'txt':
                np_value = np.load(os.path.join(self.path_to_cache_dir, f'{image_name}_{key}.npy'))
            else:
                with gzip.open(os.path.join(self.path_to_cache_dir, f'{image_name}_{key}.gz'), 'rb') as f:
                    np_value = np.load(f)

            return torch.from_numpy(np_value)
        except FileNotFoundError:
            return None

    def add(self, image_name: str, value: torch.Tensor | torch.LongTensor, key: str) -> None:
        image_name = image_name.split('.')[0]
        file_path = os.path.join(self.path_to_cache_dir, f'{image_name}_{key}')
        if key == 'txt':
            np.save(file_path + '.npy', value.numpy())
        else:
            with gzip.open(file_path + '.gz', 'wb') as f:
                np.save(f, value.numpy())

    def clear_cache(self):
        self.check_dir()
        for file in os.listdir(self.path_to_cache_dir):
            os.remove(os.path.join(self.path_to_cache_dir, file))

        os.removedirs(self.path_to_cache_dir)


class DatasetForProcessor(Dataset):
    __cache = None
    __idx_to_image = None
    __image_to_idx = None

    @classmethod
    def set_cache(cls, idx_to_image: dict[int, str], path_to_cache_dir: str):
        cls.__idx_to_image = idx_to_image
        cls.__image_to_idx = {v: k for k, v in cls.__idx_to_image.items()}

        cls.__cache = Cache(path_to_cache_dir)

    @classmethod
    def clear_cache(cls):
        del cls.__cache

    def __init__(self,
                 image_names: list[str],
                 descriptions: list[str],
                 processor,
                 path_to_images: str,
                 img_size: int,
                 tokens_num: int):

        self.img_size = img_size
        self.tokens_num = tokens_num
        self.image_names = image_names
        self.descriptions = descriptions
        self.files_in_zip = is_files_in_zip(path_to_images)

        self.processor = processor
        self.path_to_images = path_to_images

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_tensors = torch.Tensor()
        txt_tokens = torch.LongTensor()

        not_cached_idx = []  # индексы некэшированных элементов

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            idxs = range(start, stop, step)
        else:
            idxs = [idx]

        for i in idxs:
            if self.__cache is not None:
                x_img = self.__cache.get(self.image_names[i], 'img')
                x_txt = self.__cache.get(self.image_names[i], 'txt')
            else:
                x_img, x_txt = None, None

            if x_img is None:
                not_cached_idx.append(i)
            else:
                img_tensors = torch.cat([img_tensors, x_img.view(1, 3, self.img_size, self.img_size)])
                txt_tokens = torch.cat([txt_tokens, x_txt.view(1, self.tokens_num)])

        # если есть некэшированные элементы, рассчитать эмбеддинги
        if len(not_cached_idx) > 0:
            image_names = []
            descriptions = []

            for idx in not_cached_idx:
                image_names.append(self.image_names[idx])
                descriptions.append(self.descriptions[idx])

            if self.files_in_zip:
                images = get_images_from_zip(image_names, self.path_to_images)
            else:
                images = get_images(image_names, self.path_to_images)

            res = self.processor(images=images, text=descriptions, return_tensors='pt')
            tensors = res['pixel_values']
            tokens = res['input_ids']

            # сохранение преобразованных картинок и текстов в кэш
            if self.__cache is not None:
                for i, idx in enumerate(not_cached_idx):
                    self.__cache.add(self.image_names[idx], tensors[i], 'img')
                    self.__cache.add(self.image_names[idx], tokens[i], 'txt')

            img_tensors = torch.cat([img_tensors, tensors]).view(-1, 3, self.img_size, self.img_size)
            txt_tokens = torch.cat([txt_tokens, tokens]).view(-1, self.tokens_num)

        return idx, img_tensors, txt_tokens


class DatasetForPredictor(Dataset):
    __cache = None
    __cache_embed = None
    __idx_to_image = None
    __image_to_idx = None

    @classmethod
    def set_cache(cls, idx_to_image: dict[int, str], path_to_cache_dir: str):
        cls.__idx_to_image = idx_to_image
        cls.__image_to_idx = {v: k for k, v in cls.__idx_to_image.items()}

        cls.__cache_embed = SharedCache(
            size_limit_gib=32,
            dataset_len=len(idx_to_image),
            data_dims=(1024,),
            dtype=torch.float32
        )

        cls.__cache = Cache(path_to_cache_dir)

    @classmethod
    def get_all_cached_embeds(cls):
        all_embeds = {}
        for idx, img in cls.__idx_to_image.items():
            embed = cls.__cache_embed.get_slot(idx)
            if embed is not None:
                all_embeds[img] = embed

        return all_embeds

    @classmethod
    def clear_cache(cls):
        cls.__cache_embed = None
        cls.__idx_to_image = None
        cls.__image_to_idx = None

    @classmethod
    def add_cache(cls, embeddings, image_names: list[str] = None):
        if image_names is None:
            for i in range(len(embeddings)):
                cls.__cache_embed.set_slot(i, embeddings[i])
        else:
            for i, image_name in enumerate(image_names):
                cls.__cache_embed.set_slot(cls.__image_to_idx[image_name], embeddings[i])

    def __init__(self,
                 image_names: list[str],
                 descriptions: list[str],
                 chars: list[str],
                 predictor,
                 path_to_images: str,
                 img_size: int,
                 tokens_num: int):

        self.image_names = image_names
        self.descriptions = descriptions
        self.chars = chars
        self.files_in_zip = is_files_in_zip(path_to_images)

        self.char_to_label = {char: idx for idx, char in enumerate(sorted(set(chars)))}
        self.label_to_char = {idx: char for char, idx in self.char_to_label.items()}

        self.predictor = predictor
        self.path_to_images = path_to_images

        self.img_size = img_size
        self.tokens_num = tokens_num

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        concat_all = torch.Tensor()
        chars_all = torch.LongTensor()

        not_cached_idx = []  # индексы некэшированных элементов

        # случай, когда берется срез из датасета
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            indexes = list(range(start, stop, step))
        else:
            indexes = [idx]

        for i in indexes:
            if self.__cache_embed is not None:
                x = self.__cache_embed.get_slot(self.__image_to_idx[self.image_names[i]])
            else:
                x = None

            if x is None:
                not_cached_idx.append(i)
            else:
                concat_all = torch.cat([concat_all, x.view(-1, 512 * 2)])
                chars_all = torch.cat([chars_all, torch.LongTensor([self.char_to_label[self.chars[i]]])])

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

            txt_tokens = []
            img_tensors = []
            for i, image_name in enumerate(image_names):
                if self.__cache is not None:
                    img = self.__cache.get(image_name, 'img')
                    txt = self.__cache.get(image_name, 'txt')
                else:
                    img, txt = None, None

                if img is None:
                    img = self.predictor.clip_processor(images=images[i:i+1])['pixel_values']
                    if self.__cache is not None:
                        self.__cache.add(image_name, img, 'img')

                if txt is None:
                    txt = self.predictor.clip_processor(text=descriptions[i:i+1])['input_ids']
                    if self.__cache is not None:
                        self.__cache.add(image_name, txt, 'txt')

                img_tensors.append(img.view(1, 3, self.img_size, self.img_size))
                txt_tokens.append(txt.view(1, self.tokens_num))

            img_tensors = torch.cat(img_tensors)
            txt_tokens = torch.cat(txt_tokens)

            img_vecs = self.predictor.get_image_latents_(img_tensors.to(self.predictor.device)).detach().cpu()
            text_vecs = self.predictor.get_text_latents_(txt_tokens.to(self.predictor.device)).detach().cpu()

            concat = torch.cat([img_vecs, text_vecs], dim=1).view(-1, 512 * 2)
            chars = torch.LongTensor([self.char_to_label[char] for char in chars])

            # сохранение эмбеддингов в кэш
            if self.__cache_embed is not None:
                for i, idx in enumerate(not_cached_idx):
                    self.__cache_embed.set_slot(self.__image_to_idx[self.image_names[idx]], concat[i])

            concat_all = torch.cat([concat_all, concat])
            chars_all = torch.cat([chars_all, chars])

        return concat_all, chars_all


def check_df(df: pd.DataFrame, chars: list = None) -> None:
    cols = ['nm', 'description']
    if chars is not None:
        cols += chars

    for col in cols:
        if col not in df.columns:
            raise ValueError(f'Column {col} is not in the dataframe')


def is_files_in_zip(path_to_images: str) -> bool:
    if os.path.isdir(path_to_images):
        files_in_zip = False
    elif os.path.isfile(path_to_images):
        if path_to_images.endswith('.zip'):
            files_in_zip = True
    else:
        raise ValueError('Path to images must be a directory or a zip file')

    return files_in_zip


def get_char_dataloaders(df: pd.DataFrame,
                         chars: list,
                         predictor,
                         path_to_images: str,
                         path_to_cache_dir: str,
                         img_size: int,
                         tokens_num: int,
                         embeddings: torch.Tensor = None,
                         batch_size: int = 1024) -> dict:

    check_df(df, chars)

    idx_to_image = {idx: image for idx, image in enumerate(df['nm'].values)}
    DatasetForPredictor.set_cache(idx_to_image, path_to_cache_dir)

    if embeddings is not None:
        DatasetForPredictor.add_cache(embeddings)

    dataloaders = {}

    for char in chars:
        # выбираем данные, для которых указана характеристика
        char_dataset = df[df[char].notna()]
        image_names = char_dataset['nm'].values
        descriptions = char_dataset['description'].values

        # разбиваем датасет на трейн и валидационную части
        idx_train, idx_valid = train_test_split(np.arange(len(image_names)), test_size=0.2, random_state=42,
                                                stratify=char_dataset[char].values)

        # создаем даталоадеры
        dataloaders[char] = {}
        for idx, name in zip([idx_train, idx_valid], ['train', 'valid']):
            ds = DatasetForPredictor(image_names[idx], descriptions[idx], char_dataset[char].values[idx],
                                     predictor, path_to_images, img_size, tokens_num)
            dataloaders[char][name] = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloaders


def get_ruclip_dataloader(df: pd.DataFrame,
                          processor,
                          path_to_images: str,
                          path_to_cache_dir: str,
                          img_size: int,
                          tokens_num: int,
                          batch_size: int = 2048) -> DataLoader:

    check_df(df)

    image_names = df['nm'].values
    descriptions = df['description'].values
    idx_to_image = {idx: image for idx, image in enumerate(image_names)}

    DatasetForProcessor.set_cache(idx_to_image, path_to_cache_dir)

    return DataLoader(DatasetForProcessor(image_names, descriptions, processor, path_to_images, img_size, tokens_num),
                      batch_size=batch_size, shuffle=True)

