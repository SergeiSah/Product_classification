import re
import string
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class DFPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 min_chars_in_desc: int = 2,
                 cols_to_drop: list[str] = None):

        self.cols_to_drop = cols_to_drop
        self.min_chars_in_desc = min_chars_in_desc

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame):
        df.loc[df.description == '', 'description'] = df.loc[df.description == '', 'title']
        df.loc[df.title == '', 'title'] = df.loc[df.title == '', 'description']

        df['description'] = df['description'].fillna(df['title'])
        df['title'] = df['title'].fillna(df['description'].str.split().str[:15].str.join(' '))

        df = df[df[['title', 'description']].notna().all(axis=1)]
        df = df[(df['title'] != '') & (df['description'] != '')]

        df = df[df['description'].str.len() > self.min_chars_in_desc]

        if self.cols_to_drop is not None:
            df = df.drop(self.cols_to_drop, axis=1)

        df['nm'] = df['nm'].apply(lambda x: str(x) + '.jpg')

        return df


class TextCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, max_word_len=15, text_size=100):
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
            return texts.progress_apply(self._clean_text)

        return [self._clean_text(text) for text in texts]

    def _clean_text(self, text: str) -> str:
        # удаляем сочетания `<число> x <число>` и `<число><единица измерения>`
        text = re.sub(r'\d+[хХxX]\d+|\d+[кмглршт]+|\d+ [xXхХ] \d+', '', text)

        return ' '.join([word for word in word_tokenize(text.lower()) if word not in self.text_exclude_list and
                         not word.isdigit() and
                         len(word) <= self.max_word_len][:self.text_size])


__all__ = ['DFPreprocessor', 'TextCleaner']
