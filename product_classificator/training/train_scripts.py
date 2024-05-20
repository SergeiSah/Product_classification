"""
Скрипт для обучения только MLP классификаторов, без дообучения трансформеров (text и vision) ruCLIP
"""
import os.path
import pickle
import pandas as pd
import numpy as np

import torch
from torchvision.ops import MLP

from clearml import Task

from product_classificator import load
from ..ruclip.predictor import Predictor
from ..ruclip.processor import RuCLIPProcessor
from ..ruclip.ruclip_model import CLIP
from .modules.cleaner import clean_dataset, TextCleaner
from .modules.char_processor import CharExtractor, CharReducer
from .modules.dataset import get_dataloaders
from .modules.train_procedure import train_mlp_classifier
from .modules.visualisation import plot_history


class Timer:

    def __init__(self):
        self.start = None
        self.end = None

        self.last_period = None

    def __enter__(self):
        self.start = pd.Timestamp.now()
        return self

    def __exit__(self, *args):
        self.end_timer()

    def start_timer(self):
        self.start = pd.Timestamp.now()

    def end_timer(self):
        self.end = pd.Timestamp.now()
        self.last_period = str(self.end - self.start)


class Trainer:

    def __init__(self,
                 path_to_images: str,
                 path_to_texts: str,
                 characteristics: list[str] = None,
                 task: Task = None,
                 main_params: dict = None,
                 mlp_params: dict = None,
                 optim_params: dict = None,
                 min_products_in_sub_cat: int = 10,
                 ruclip_model: str = 'ruclip-vit-base-patch16-384',
                 cache_dir: str = '/tmp/ruclip/',
                 heads_dir: str = 'heads/wb-6_cats',
                 save_dir: str = '.'):

        self.task = task

        if isinstance(task, Task):
            self.logger = task.get_logger()

        self.ruclip_model = ruclip_model

        self.path_to_images = path_to_images
        self.path_to_dfs = path_to_texts
        self.cache_dir = cache_dir
        self.heads_dir = heads_dir
        self.save_dir = save_dir

        self.main_params = main_params or {
            'criterion': torch.nn.CrossEntropyLoss,
            'optimizer': torch.optim.Adam,
            'classificator': MLP,
            'epochs': 15,
            'batch_size': 1024}

        self.mlp_params = mlp_params or {
            'in_channels': 1024,
            'hidden_channels': [1024],
            'dropout': 0.2,
            'activation': torch.nn.ReLU
        }

        self.optim_params = optim_params or {}

        if min_products_in_sub_cat <= 1:
            raise ValueError('min_products_in_sub_cat must be > 1')
        self.min_products_in_sub_cat = min_products_in_sub_cat
        self.characteristics = characteristics or [
            'category', 'sub_category', 'isadult', 'sex', 'season', 'age_restrictions', 'fragility']

        self.wb_train_df = 'wb_school_train.parquet'
        self.wb_test_df = 'wb_school_test.parquet'

        self.timer = Timer()

    def _show_info(self, text: str):
        if self.task is not None:
            self.logger.report_text(text)
        else:
            print(text)

    def _log_history(self, char: str, history: dict[str, list[float]]):
        for i in range(len(history['train_loss'])):
            self.logger.report_scalar(title=f'{char}: CrossEntropyLoss', series=f'train',
                                      value=history['train_loss'][i],
                                      iteration=i + 1)
            self.logger.report_scalar(title=f'{char}: CrossEntropyLoss', series=f'valid',
                                      value=history['valid_loss'][i],
                                      iteration=i + 1)

            self.logger.report_scalar(title=f'{char}: F1-macro', series=f'train', value=history['train_f1'][i],
                                      iteration=i + 1)
            self.logger.report_scalar(title=f'{char}: F1-macro', series=f'valid', value=history['valid_f1'][i],
                                      iteration=i + 1)

    def loading_texts(self) -> pd.DataFrame:
        train = pd.read_parquet(self.path_to_dfs + self.wb_train_df)
        train = clean_dataset(train)

        test = pd.read_parquet(self.path_to_dfs + self.wb_test_df)
        test = clean_dataset(test)

        return pd.concat([train, test], ignore_index=True).reset_index(drop=True)

    def preprocessing_texts(self, df: pd.DataFrame) -> pd.DataFrame:
        train = df[df.characteristics.notna()]
        test = df[df.characteristics.isna()]

        # извлечение характеристик в train датасете
        char_extractor = CharExtractor()
        char_reducer = CharReducer()

        self._show_info('Extracting characteristics from train dataset')
        with self.timer:
            train = char_extractor.fit_transform(train)
            train = char_reducer.fit_transform(train)
        self._show_info('End. Extraction time: ' + str(self.timer.last_period))

        train = train.drop('characteristics', axis=1)

        # предобработка характеристик в test датасете
        sub_cat_freqs = test.sub_category.value_counts()
        test['sub_category'] = test.sub_category.apply(
            lambda x: np.nan if sub_cat_freqs[x] < self.min_products_in_sub_cat else x)

        condition = (test.category.isin(['Товары для взрослых', 'Товары для курения']) & (~test.isadult))
        test.loc[condition, 'isadult'] = True

        # объединение датасетов
        train_test = pd.concat([train, test], ignore_index=True).drop('title', axis=1)

        # оставляем только ту часть датасета, что будет использоваться при обучении
        train_test = train_test[train_test[self.characteristics].notna().any(axis=1)].reset_index(drop=True)

        cleaner = TextCleaner()
        self._show_info('Cleaning texts')
        with self.timer:
            train_test['description'] = cleaner.fit_transform(train_test['description'])
            if self.task is not None:
                self.task.register_artifact('Prepared data', train_test)
        self._show_info('End. Cleaning time: ' + str(self.timer.last_period))

        return train_test

    def run(self) -> None:

        timer = Timer()

        self._show_info('Start text preprocessing')
        with timer:
            train_test = self.loading_texts()
            train_test = self.preprocessing_texts(train_test)
        self._show_info('End. Preprocessing time: ' + str(timer.last_period))

        self._show_info('Start ruCLIP loading')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        load(self.ruclip_model, cache_dir=self.cache_dir)
        clip = CLIP.from_pretrained(self.cache_dir + self.ruclip_model).eval().to(device)
        processor = RuCLIPProcessor.from_pretrained(self.cache_dir + self.ruclip_model)
        predictor = Predictor(clip, processor, device, quiet=True)

        self._show_info('Preparing Data Loaders')
        dataloaders = get_dataloaders(train_test, self.characteristics, predictor,
                                      path_to_images=self.path_to_images, batch_size=self.main_params['batch_size'])

        self._show_info('Saving a dict for transformation labels to characteristic values')
        label_to_char = {}
        for char in self.characteristics:
            label_to_char[char] = dataloaders[char]['train'].dataset.label_to_char

        if self.task is not None:
            for char in label_to_char:
                df = pd.DataFrame(
                    {'label': list(label_to_char[char].keys()), char: list(label_to_char[char].values())}
                )
                self.task.register_artifact(char, df)

        with open(os.path.join(self.save_dir, 'label_to_char.pkl'), 'wb') as f:
            pickle.dump(label_to_char, f)

        char_uniq = train_test[self.characteristics].nunique()

        if self.task is not None:
            self.task.register_artifact('Num of unique characteristics', char_uniq.to_frame())
            self.task.connect({k: str(v) for k, v in self.main_params.items()},
                              name='Heads training parameters')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(os.path.join(self.save_dir, self.heads_dir)):
            os.makedirs(os.path.join(self.save_dir, self.heads_dir))

        self._show_info('Start training MLP classifiers')
        for char in self.characteristics:
            mlp = MLP(in_channels=self.mlp_params['in_channels'],
                      hidden_channels=[*self.mlp_params['hidden_channels'], char_uniq[char]],
                      dropout=self.mlp_params['dropout'],
                      activation_layer=self.mlp_params['activation'])

            criterion = self.main_params['criterion']()
            optimizer = self.main_params['optimizer'](mlp.parameters(), **self.optim_params)

            with timer:
                history, best_params = train_mlp_classifier(mlp, dataloaders[char], criterion, optimizer, char,
                                                            epochs=self.main_params['epochs'])
            plot_history(history, char_name=char)

            self._show_info(f'End training MLP classifier {char}. Time: ' + str(timer.last_period))
            if self.task is not None:
                self._log_history(char, history)

            torch.save(best_params, os.path.join(self.save_dir, self.heads_dir, f'{char}.pt'))

        if self.task is not None:
            self.task.close()

