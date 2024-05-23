"""
Скрипт для обучения только MLP классификаторов, без дообучения трансформеров (text и vision) ruCLIP
"""
import os.path
import shutil
import pickle
import pandas as pd
import numpy as np

import torch
from torchvision.ops import MLP

from tqdm import tqdm
from clearml import Task

from product_classificator import load
from ..ruclip.predictor import Predictor
from ..ruclip.processor import RuCLIPProcessor
from ..ruclip.ruclip_model import CLIP
from .modules.cleaner import clean_dataset, TextCleaner
from .modules.char_processor import CharExtractor, CharReducer
from .modules.dataset import get_char_dataloaders, get_ruclip_dataloader
from .modules.train_procedure import train_mlp_classifier, train_ruclip_one_epoch
from .modules.visualisation import *
from .modules.clusterization import *
from .modules.timer import Timer


class Trainer:

    def __init__(self,
                 path_to_images: str,
                 path_to_texts: str,
                 characteristics: list[str] = None,
                 ruclip_train_params: dict = None,
                 heads_train_params: dict = None,
                 clusterization_params: dict = None,
                 min_products_in_sub_cat: int = 10,
                 ruclip_model: str = 'ruclip-vit-base-patch16-384',
                 cache_dir: str = '/tmp',
                 save_dir: str = '/tmp/experiments'):

        self.task = None
        self.logger = None

        self.ruclip_model = ruclip_model

        self.path_to_images = path_to_images
        self.path_to_dfs = path_to_texts
        self.cache_dir = cache_dir
        self.save_dir = save_dir

        self.clean_text = True
        self.set_cache = True

        self.ruclip_train_params = ruclip_train_params or {
            'img_criterion': torch.nn.CrossEntropyLoss,
            'txt_criterion': torch.nn.CrossEntropyLoss,
            'epochs': 5,
            'batch_size': 256,
            'optimizer': torch.optim.Adam,
            'optimizer_params': {
                'lr': 5e-5,
                'betas': (0.9, 0.98),
                'eps': 1e-6,
                'weight_decay': 0.2
            },
            'num_last_resblocks_to_train': 1
        }

        self.heads_train_params = heads_train_params or {
            'criterion': torch.nn.CrossEntropyLoss,
            'optimizer': torch.optim.Adam,
            'optimizer_params': {},
            'classificator': MLP,
            'classificator_params': {
                'in_channels': 1024,
                'hidden_channels': [1024],
                'dropout': 0.2,
                'activation': torch.nn.ReLU
            },
            'epochs': 15,
            'batch_size': 1024}

        self.clusterization_params = clusterization_params or {
            'n_clusters': 1000,
        }

        self.end_of_train = {
            'del_img_and_txt_tensors': False,
            'del_loaded_ruclip_model': False
        }

        if min_products_in_sub_cat <= 1:
            raise ValueError('min_products_in_sub_cat must be > 1')
        self.min_products_in_sub_cat = min_products_in_sub_cat
        self.characteristics = characteristics or [
            'category', 'sub_category', 'isadult', 'sex', 'season', 'age_restrictions', 'fragility']

        self.wb_train_df = 'wb_school_train.parquet'
        self.wb_test_df = 'wb_school_test.parquet'

        self.img_size = int(self.ruclip_model.split('-')[-1])
        self.tokens = 77

        self.timer_ = Timer()

        self._create_dirs_for_saving_results_and_cache()

    def _create_dirs_for_saving_results_and_cache(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _create_experiment_dir(self, name: str = None):
        if name is None:
            dirs = [int(d.split('_')[-1]) for d in os.listdir(self.save_dir) if 'experiment' in d]
            if not dirs:
                dir_name = 'experiment_0'
            else:
                dir_name = f'experiment_{max(dirs) + 1}'
        else:
            if os.path.exists(os.path.join(self.save_dir, name)):
                raise FileExistsError(f'Experiment "{name}" already exists')
            dir_name = name

        self.experiment_dir = os.path.join(self.save_dir, dir_name)
        self.heads_dir = os.path.join(self.experiment_dir, 'heads')
        os.makedirs(self.heads_dir)

    def _show_info(self, text: str):
        if self.task is not None:
            self.logger.report_text(text)
        else:
            print(text)

    def _log_history(self, char: str, history: dict[str, list[float]], add_info: str = ''):
        for i in range(len(history['train_loss'])):
            self.logger.report_scalar(title=f'{char}{add_info}: CrossEntropyLoss', series=f'train',
                                      value=history['train_loss'][i],
                                      iteration=i + 1)
            self.logger.report_scalar(title=f'{char}{add_info}: CrossEntropyLoss', series=f'valid',
                                      value=history['valid_loss'][i],
                                      iteration=i + 1)

            self.logger.report_scalar(title=f'{char}{add_info}: F1-macro', series=f'train',
                                      value=history['train_f1'][i],
                                      iteration=i + 1)
            self.logger.report_scalar(title=f'{char}{add_info}: F1-macro', series=f'valid',
                                      value=history['valid_f1'][i],
                                      iteration=i + 1)

    def _log_ruclip_loss(self, loss: float, iteration: int):
        if self.task is not None:
            self.logger.report_scalar(title='ruCLIP Cross Entropy Loss', series='CrossEntropyLoss',
                                      value=loss, iteration=iteration)

    def check_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.heads_dir):
            os.makedirs(self.heads_dir)

    def _loading_texts(self) -> pd.DataFrame:
        train = pd.read_parquet(os.path.join(self.path_to_dfs, self.wb_train_df))
        train = clean_dataset(train)

        test = pd.read_parquet(os.path.join(self.path_to_dfs, self.wb_test_df))
        test = clean_dataset(test)

        return pd.concat([train, test], ignore_index=True).reset_index(drop=True)

    def _preprocessing_texts(self, df: pd.DataFrame, clean: bool) -> pd.DataFrame:
        train = df[df.characteristics.notna()]
        test = df[df.characteristics.isna()]

        # извлечение характеристик в train датасете
        char_extractor = CharExtractor()
        char_reducer = CharReducer()

        self._show_info('Extracting characteristics from train dataset')
        with self.timer_:
            train = char_extractor.fit_transform(train)
            train = char_reducer.fit_transform(train)
        self._show_info('End. Extraction time: ' + str(self.timer_.last_period))

        train = train.drop('characteristics', axis=1)

        # предобработка характеристик в test датасете
        sub_cat_freqs = test.sub_category.value_counts()
        test['sub_category'] = test.sub_category.apply(
            lambda x: np.nan if sub_cat_freqs[x] < self.min_products_in_sub_cat else x)

        condition = (test.category.isin(['Товары для взрослых', 'Товары для курения']) & (~test.isadult))
        test.loc[condition, 'isadult'] = True

        # объединение датасетов
        train_test = pd.concat([train, test], ignore_index=True).drop('title', axis=1)

        if clean:
            cleaner = TextCleaner()
            self._show_info('Cleaning texts')
            with self.timer_:
                train_test['description'] = cleaner.fit_transform(train_test['description'])
                if self.task is not None:
                    self.task.register_artifact('Prepared data', train_test)
            self._show_info('End. Cleaning time: ' + str(self.timer_.last_period))

        train_test['nm'] = train_test['nm'].apply(lambda x: str(x) + '.jpg')

        return train_test

    def _first_part(self, timer, experiment_name: str):
        self._show_info('Start text preprocessing')
        with timer:
            train_test = self._loading_texts()
            train_test = self._preprocessing_texts(train_test, self.clean_text)
        self._show_info('End. Preprocessing time: ' + str(timer.last_period))

        self._show_info('Start ruCLIP loading')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ruclip_path = os.path.join(self.cache_dir, 'ruclip')
        load(self.ruclip_model, cache_dir=ruclip_path)
        clip = CLIP.from_pretrained(os.path.join(ruclip_path, self.ruclip_model)).eval().to(device)
        processor = RuCLIPProcessor.from_pretrained(os.path.join(ruclip_path, self.ruclip_model))
        predictor = Predictor(clip, processor, device, quiet=True)

        return train_test, clip, processor, predictor, device

    def _prepare_dataloaders_for_heads(self, train_test, predictor, embeddings=None):
        self._show_info('Preparing Data Loaders')
        dataloaders = get_char_dataloaders(train_test, self.characteristics, predictor,
                                           path_to_images=self.path_to_images,
                                           path_to_cache_dir=os.path.join(self.cache_dir, 'cache'),
                                           img_size=self.img_size,
                                           tokens_num=self.tokens,
                                           embeddings=embeddings,
                                           batch_size=self.heads_train_params['batch_size'])

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

        with open(os.path.join(self.heads_dir, 'label_to_char.pkl'), 'wb') as f:
            pickle.dump(label_to_char, f)

        char_uniq = train_test[self.characteristics].nunique()

        return dataloaders, char_uniq

    def _train_heads_part(self, char_uniq, dataloaders, timer, add_info: str = '', save=True):
        mlp_history = {char: {'train_loss': [], 'valid_loss': [], 'train_f1': [], 'valid_f1': []}
                       for char in self.characteristics}

        self._show_info('Start training MLP classifiers')
        for char in self.characteristics:
            mlp = MLP(in_channels=self.heads_train_params['classificator_params']['in_channels'],
                      hidden_channels=[*self.heads_train_params['classificator_params']['hidden_channels'],
                                       char_uniq[char]],
                      dropout=self.heads_train_params['classificator_params']['dropout'],
                      activation_layer=self.heads_train_params['classificator_params']['activation'])

            if self.task is not None:
                self.task.connect({x.split(': ')[0].strip(): x.split(': ')[1] for x in str(mlp).split('\n')[1:-1]},
                                  f'{char} MLP{add_info}')

            criterion = self.heads_train_params['criterion']()
            optimizer = self.heads_train_params['optimizer'](mlp.parameters(),
                                                             **self.heads_train_params['optimizer_params'])

            with timer:
                history, best_params = train_mlp_classifier(mlp, dataloaders[char], criterion, optimizer, char,
                                                            epochs=self.heads_train_params['epochs'])
            fig = plot_history(history, char_name=char + add_info)
            fig.savefig(os.path.join(self.experiment_dir, f'{char}_history{add_info}.png'))

            for key in mlp_history[char]:
                mlp_history[char][key].append(max(history[key]))

            self._show_info(f'End training MLP classifier "{char}". Time: ' + str(timer.last_period))
            if self.task is not None:
                s = '' if add_info == '' else '/' + add_info.strip()
                self._log_history(char, history, add_info=s)

            if save:
                torch.save(best_params, os.path.join(self.heads_dir, f'{char}.pt'))

        return mlp_history

    def _clusterization(self, embeddings: np.ndarray, df, params, save: bool, add_info: str = ''):
        reduced_embed = get_reduced_embeds(embeddings)
        clusterizer = get_clusterizer(reduced_embed,
                                      min(self.clusterization_params['n_clusters'], df.shape[0]))
        df['clusters'] = clusterizer.predict(reduced_embed)

        fig = plot_clusters(df['clusters'], reduced_embed)
        if save:
            fig.savefig(os.path.join(self.experiment_dir, f'clusters{add_info}.png'))

        clust_metrics = get_clusterization_metrics(df, params, clusterizer, reduced_embed)
        clust_metrics.to_csv(os.path.join(self.experiment_dir, f'clust_metrics{add_info}.csv'))

        if self.task is not None:
            self.task.register_artifact('Clusterization metrics' + add_info, clust_metrics)

    def train_heads_only(self, task: Task = None, experiment_name: str = None) -> None:
        self._start_experiment(task, experiment_name)

        timer = Timer()
        train_test, clip, processor, predictor, device = self._first_part(timer, experiment_name)

        # оставляем только ту часть датасета, что будет использоваться при обучении
        train_test = train_test[train_test[self.characteristics].notna().any(axis=1)].reset_index(drop=True)
        dataloaders, char_uniq = self._prepare_dataloaders_for_heads(train_test, predictor)

        if self.task is not None:
            self.task.register_artifact('Num of unique characteristics', char_uniq.to_frame())

        self.check_save_dir()
        self._train_heads_part(char_uniq, dataloaders, timer)

        self._show_info('Start clusterization')
        cache = dataloaders[self.characteristics[0]]['train'].dataset.get_all_cached_embeds()
        embeddings = np.concatenate([x.numpy().reshape(1, -1) for x in cache.values()])
        df = train_test.set_index('nm').loc[cache.keys()].reset_index()
        with timer:
            self._clusterization(embeddings, df, self.characteristics, save=True)
        self._show_info(f'End clusterization. Time: ' + str(timer.last_period))

        self._end_experiment()

    def train_ruclip(self, task: Task = None, experiment_name: str = None) -> None:
        self._start_experiment(task, experiment_name)

        timer = Timer()
        train_test, clip, processor, predictor, device = self._first_part(timer, experiment_name)

        self._show_info('Preparing ruclip dataloader')
        dataloader = get_ruclip_dataloader(train_test, processor, self.path_to_images,
                                           os.path.join(self.cache_dir, 'cache'),
                                           self.img_size, self.tokens,
                                           self.ruclip_train_params['batch_size'],
                                           self.set_cache)

        self._freeze_parameters(clip, self.ruclip_train_params['num_last_resblocks_to_train'])

        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(clip.parameters(), **self.ruclip_train_params['optimizer_params'])

        self.check_save_dir()
        ruclip_loss = []
        mlp_history = {char: {'train_loss': [], 'valid_loss': [], 'train_f1': [], 'valid_f1': []}
                       for char in self.characteristics}

        self._show_info('Start training ruclip')
        epochs = self.ruclip_train_params['epochs']
        best_params = clip.state_dict()

        for epoch in tqdm(range(1, epochs + 1),
                          desc='Train ruclip. Epochs'):
            with timer:
                embeddings, loss = train_ruclip_one_epoch(clip, dataloader, loss_img, loss_txt, optimizer, device)
            self._show_info(f'Epoch {epoch}. Training time: ' + str(timer.last_period))
            ruclip_loss.append(loss)
            self._log_ruclip_loss(loss, epoch)

            if epoch > 1 and ruclip_loss[-1] > ruclip_loss[-2]:
                best_params = clip.state_dict()

            dataloaders, char_uniq = self._prepare_dataloaders_for_heads(train_test, predictor, embeddings)
            history = self._train_heads_part(char_uniq, dataloaders, timer, add_info=f' (Epoch {epoch})', save=True)

            self._show_info('Start clusterization')
            cache = dataloaders[self.characteristics[0]]['train'].dataset.get_all_cached_embeds()
            embeddings = np.concatenate([x.numpy().reshape(1, -1) for x in cache.values()])
            df = train_test.set_index('nm').loc[cache.keys()].reset_index()
            with timer:
                self._clusterization(embeddings, df, self.characteristics, save=True, add_info=f' (Epoch {epoch})')
            self._show_info(f'End clusterization. Time: ' + str(timer.last_period))

            for char in mlp_history:
                for key in mlp_history[char]:
                    mlp_history[char][key].append(max(history[char][key]))

        for char in mlp_history:
            fig = plot_history(mlp_history[char], f'{char}')
            fig.savefig(os.path.join(self.experiment_dir,  f'{char}_history.png'))
            self._log_history(char, mlp_history[char], add_info=f' (ruclip train)')

        torch.save(best_params, os.path.join(self.experiment_dir,  f'trained_{self.ruclip_model}.pt'))
        self._end_experiment()

    @staticmethod
    def _freeze_parameters(clip: CLIP, num_last_resblocks: int) -> None:
        for param in clip.parameters():
            param.requires_grad = False

        clip.visual.transformer.resblocks[-num_last_resblocks:].requires_grad_(True)
        clip.visual.ln_post.requires_grad_(True)
        clip.transformer.resblocks[-num_last_resblocks:].requires_grad_(True)

    def _start_experiment(self, task: Task, experiment_name: str):
        if task is not None:
            self.task = task
            self.logger = task.get_logger()
        self._create_experiment_dir(experiment_name)
        self._save_config()

    def _save_config(self):
        config = {
            'ruclip_train_params': self.ruclip_train_params,
            'heads_train_params': self.heads_train_params
        }

        with open(os.path.join(self.experiment_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)

        if self.task is not None:
            self.task.connect(config, 'Experiment configuration')

    def _end_experiment(self) -> None:
        self._show_info('End of experiment')
        if self.task is not None:
            self.task.close()
            self.task = None
            self.logger = None

        if self.end_of_train['del_img_and_txt_tensors']:
            path = os.path.join(self.cache_dir, 'cache')
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))

            self._show_info('Image and text tensors deleted')

        if self.end_of_train['del_loaded_ruclip_model']:
            path = os.path.join(self.cache_dir, 'ruclip')
            shutil.rmtree(path)

            self._show_info('Ruclip model deleted')
