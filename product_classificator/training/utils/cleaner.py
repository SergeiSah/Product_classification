import re
import string
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


EMPTY_NMS = ['100131619', '181127370', '146641342', '181258675', '182997563', '14292958',
             '57268154', '92037880', '186633414', '169609171', '194133030', '160308940',
             '11539678', '154204044', '124835529', '104948635', '4845057', '146641585',
             '94544207', '167737624', '137610465', '176704297', '151253921', '154549519',
             '159361146', '177873216']


def clean_dataset(df: pd.DataFrame, cols_to_drop=None):
    if cols_to_drop is None:
        cols_to_drop = ['brand', 'price']

    df = df.copy()

    df.loc[df.title == '', 'title'] = np.nan
    df.loc[df.description == '', 'description'] = np.nan

    # удаляем товары, где нет названия и описания
    df = df[df[['title', 'description']].notna().all(axis=1)]

    df = df.copy()

    # заполняем пропуски в описаниях заголовком, где это возможно
    df['description'] = df.description.fillna(df.title)

    # заполням пропуски в заголовках первыми 10 словами из описания
    df['title'] = df['title'].fillna(df['description'].str.split().str[:10].str.join(' '))

    # исключаем товары с пустыми картинками
    df = df[~df.nm.isin(EMPTY_NMS)]

    # удалим товары, заголовок которых состоит менее чем из 2х символов
    df = df[df['title'].str.len() > 2]

    # удаляем колонки, которые не будут использованы
    for col in df.columns:
        if col in cols_to_drop:
            df = df.drop(col, axis=1)

    return df


class TextCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, max_word_len=15, text_size=200):
        self.text_size = text_size
        self.max_word_len = max_word_len

        units = ['мг', 'г', 'гр', 'кг', 'мл', 'л', 'мм', 'см', 'м', 'км', 'шт', 'штук']
        stop_words = stopwords.words('russian')

        # часть из стоп-слов оставим
        stop_words.remove('не')
        stop_words.remove('для')

        self.text_exclude_list = list(string.punctuation) + stop_words + units

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        if isinstance(texts, pd.Series):
            return texts.progress_apply(self._process_text)

        return [self._process_text(text) for text in texts]

    def _process_text(self, text: str) -> str:
        # удаляем сочетания `<число> x <число>` и `<число><единица измерения>`
        text = re.sub(r'\d+[хХxX]\d+|\d+[кмглршт]+|\d+ [xXхХ] \d+', '', text)

        return ' '.join([word for word in word_tokenize(text.lower()) if word not in self.text_exclude_list and
                         not word.isdigit() and
                         len(word) <= self.max_word_len][:self.text_size])


__all__ = ['clean_dataset', 'TextCleaner']
