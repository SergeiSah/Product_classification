import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from stocaching import SharedCache
from product_classificator.training.utils import get_images, get_images_from_zip, is_files_in_zip


class EmbedCharDataset(Dataset):
    __cache_embed = None
    __idx_to_image = None
    __image_to_idx = None

    @classmethod
    def set_cache_embed(cls, idx_to_image: dict[int, str], path_to_cache_dir: str):
        cls.__idx_to_image = idx_to_image
        cls.__image_to_idx = {v: k for k, v in cls.__idx_to_image.items()}

        cls.__cache_embed = SharedCache(
            size_limit_gib=32,
            dataset_len=len(idx_to_image),
            data_dims=(1024,),
            dtype=torch.float32
        )

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
        concat_all = torch.Tensor().to(self.predictor.device)
        chars_all = torch.LongTensor().to(self.predictor.device)

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
                concat_all = torch.cat([concat_all, x.view(-1, 512 * 2).to(self.predictor.device)])
                chars_all = torch.cat([chars_all, torch.LongTensor([self.char_to_label[self.chars[i]]])
                                      .to(self.predictor.device)])

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
                with torch.no_grad():
                    img = self.predictor.clip_processor(images=images[i:i+1])['pixel_values']
                    txt = self.predictor.clip_processor(text=descriptions[i:i+1])['input_ids']

                img_tensors.append(img.view(1, 3, self.img_size, self.img_size))
                txt_tokens.append(txt.view(1, self.tokens_num))

            img_tensors = torch.cat(img_tensors)
            txt_tokens = torch.cat(txt_tokens)

            img_vecs = self.predictor.get_image_latents(img_tensors.to(self.predictor.device))
            text_vecs = self.predictor.get_text_latents(txt_tokens.to(self.predictor.device))

            concat = torch.cat([img_vecs, text_vecs], dim=1).view(-1, 512 * 2)
            chars = torch.LongTensor([self.char_to_label[char] for char in chars]).to(self.predictor.device)

            # сохранение эмбеддингов в кэш
            if self.__cache_embed is not None:
                for i, idx in enumerate(not_cached_idx):
                    self.__cache_embed.set_slot(self.__image_to_idx[self.image_names[idx]], concat[i].detach().cpu())

            concat_all = torch.cat([concat_all, concat])
            chars_all = torch.cat([chars_all, chars])

        return concat_all, chars_all


class TextImageDataset(Dataset):

    def __init__(self, processor, df: pd.DataFrame, path_to_images: str):
        self.processor = processor
        self.descriptions = df['description'].values
        self.image_names = df['nm'].values
        self.path_to_images = path_to_images
        self.files_in_zip = is_files_in_zip(path_to_images)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if not isinstance(idx, slice):
            idx = [idx]

        if self.files_in_zip:
            images = get_images_from_zip(self.image_names[idx], self.path_to_images)
        else:
            images = get_images(self.image_names[idx], self.path_to_images)

        img_tensors = self.processor(images=images, return_tensors="pt", padding=True)['pixel_values']
        txt_tokens = self.processor(text=self.descriptions[idx], return_tensors="pt", padding=True)['input_ids']

        return img_tensors, txt_tokens


def check_df(df: pd.DataFrame, chars: list = None) -> None:
    cols = ['nm', 'description']
    if chars is not None:
        cols += chars

    for col in cols:
        if col not in df.columns:
            raise ValueError(f'Column {col} is not in the dataframe')


def get_char_dataloaders(df: pd.DataFrame,
                         chars: list,
                         predictor,
                         path_to_images: str,
                         path_to_cache_dir: str,
                         img_size: int,
                         tokens_num: int,
                         cache_embed: bool,
                         batch_size: int = 1024) -> dict:

    check_df(df, chars)

    if cache_embed:
        idx_to_image = {idx: image for idx, image in enumerate(df['nm'].values)}
        EmbedCharDataset.set_cache_embed(idx_to_image, path_to_cache_dir)

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
            ds = EmbedCharDataset(image_names[idx], descriptions[idx], char_dataset[char].values[idx],
                                  predictor, path_to_images, img_size, tokens_num)
            dataloaders[char][name] = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloaders


def get_ruclip_dataloader(df: pd.DataFrame,
                          processor,
                          path_to_images: str,
                          batch_size: int) -> DataLoader:

    dataset = TextImageDataset(processor, df, path_to_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


