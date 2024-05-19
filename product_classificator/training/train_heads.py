"""
Скрипт для обучения только MLP классификаторов, без дообучения трансформеров (text и vision) ruCLIP
"""
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
from .modules.cleaner import clean_dataset
from .modules.char_processor import CharExtractor, CharReducer
from .modules.dataset import get_dataloaders
from .modules.train_procedure import train_mlp_classifier, plot_history


def wb_preprocessing(path_to_parquets,
                     characteristics: list[str],
                     min_products_in_sub_cat: int = 10):

    if min_products_in_sub_cat <= 1:
        raise ValueError('min_products_in_sub_cat must be > 1')

    # загрузка данных, первичная очистка пропусков и "плохих" записей
    train = pd.read_parquet(path_to_parquets + 'wb_school_train.parquet')
    train = clean_dataset(train)

    test = pd.read_parquet(path_to_parquets + 'wb_school_test.parquet')
    test = clean_dataset(test)

    # извлечение характеристик в train датасете
    char_extractor = CharExtractor()
    char_reducer = CharReducer()

    train = char_extractor.fit_transform(train)
    train = char_reducer.fit_transform(train)
    train = train.drop('characteristics', axis=1)

    # предобработка характеристик в test датасете
    sub_cat_freqs = test.sub_category.value_counts()
    test['sub_category'] = test.sub_category.apply(
        lambda x: np.nan if sub_cat_freqs[x] < min_products_in_sub_cat else x)

    condition = (test.category.isin(['Товары для взрослых', 'Товары для курения']) & (~test.isadult))
    test.loc[condition, 'isadult'] = True

    # объединение датасетов
    train_test = pd.concat([train, test], ignore_index=True).drop('title', axis=1)

    # оставляем только ту часть датасета, что будет использоваться при обучении
    train_test = train_test[train_test[characteristics].notna().any(axis=1)].reset_index(drop=True)

    return train_test


def run(path_to_images: str,
        path_to_dfs: str,
        task: Task,
        ruclip_model_name='ruclip-vit-base-patch16-384',
        save_dir='',
        main_params: dict = None):

    if main_params is None:
        main_params = {
            'criterion': torch.nn.CrossEntropyLoss,
            'optimizer': torch.optim.Adam,
            'classificator': MLP,
        }

    # task = Task.init(project_name='WBTECH: HorizontalML',
    #                  task_name=f'{experiment_name}')
    logger = task.get_logger()

    # предобработка данных
    characteristics = ['category', 'sub_category', 'isadult', 'sex', 'season', 'age_restrictions', 'fragility']
    train_test = wb_preprocessing(path_to_dfs, characteristics)

    # загружаем модель ruCLIP
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load(ruclip_model_name)
    clip = CLIP.from_pretrained(ruclip_model_name).eval().to(device)
    processor = RuCLIPProcessor.from_pretrained(ruclip_model_name)
    predictor = Predictor(clip, processor, device, quiet=True)

    dataloaders = get_dataloaders(train_test, characteristics, predictor,
                                  path_to_images=path_to_images, batch_size=1024)

    # сохраняем словари для перевода меток в названия характеристик
    label_to_char = {}
    for char in characteristics:
        label_to_char[char] = dataloaders['train'].dataset.label_to_char

    with open(save_dir + 'label_to_char.pkl', 'wb') as f:
        pickle.dump(label_to_char, f)

    # число уникальных элементов в каждой характеристике
    char_uniq = train_test[characteristics].nunique()

    # запись параметров обучения и MLP классификаторов
    task.connect({k: str(v) for k, v in main_params.items()}, name='Main parameters')
    task.connect({
        'in_channels': 1024,
        'hidden_channels': [1024, 'unique chars num'],
        'dropout': 0.2,
        'activation_layer': 'ReLU'
    }, name='MLP parameters')

    for char in characteristics:
        mlp = MLP(in_channels=1024, hidden_channels=[1024, char_uniq[char]], dropout=0.2,
                  activation_layer=torch.nn.ReLU)

        criterion = main_params['criterion']()
        optimizer = main_params['optimizer'](mlp.parameters())

        history, best_params = train_mlp_classifier(mlp, dataloaders[char], criterion, optimizer, char, epochs=10)
        plot_history(history, char_name=char)

        for i in range(len(history['train_loss'])):
            logger.report_scalar(title=f'{char}: CrossEntropyLoss', series=f'train', value=history['train_loss'][i],
                                 iteration=i + 1)
            logger.report_scalar(title=f'{char}: CrossEntropyLoss', series=f'valid', value=history['valid_loss'][i],
                                 iteration=i + 1)

            logger.report_scalar(title=f'{char}: F1-macro', series=f'train', value=history['train_f1'][i],
                                 iteration=i + 1)
            logger.report_scalar(title=f'{char}: F1-macro', series=f'valid', value=history['valid_f1'][i],
                                 iteration=i + 1)

        torch.save(best_params, f'classificator_{char}.pt')

    task.close()
