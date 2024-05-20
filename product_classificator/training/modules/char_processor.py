import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CharExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, char_names: list = None, col_names: list = None):
        if char_names is None and col_names is None:
            self.char_names = ['Пол', 'Возрастные ограничения', 'Сезон', 'Хрупкость']
            self.col_names = ['sex', 'age_restrictions', 'season', 'fragility']

        elif char_names is None or col_names is None:
            raise ValueError('Both char_names and col_names must be provided')

        else:
            self.char_names = char_names
            self.col_names = col_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        chars = X['characteristics'].apply(lambda x: self._get_characteristic(self._process_characteristics(x)))
        return pd.concat([X, pd.DataFrame(chars, columns=self.col_names, index=X.index)], axis=1)

    def _get_characteristic(self, char_list: list[dict]) -> float | list[float]:
        res = [np.nan] * len(self.char_names)

        for char in char_list:

            if char['charcName'] in self.char_names:

                if 'value' in char and char['value'] is not None:
                    if isinstance(char['value'], str):
                        char['value'] = float(char['value'].replace(',', '.'))

                    res[self.char_names.index(char['charcName'])] = char['value']

                if 'charcValues' in char and char['charcValues'] is not None:
                    if len(char['charcValues']) == 1:
                        res[self.char_names.index(char['charcName'])] = char['charcValues'][0]
                    else:
                        res[self.char_names.index(char['charcName'])] = list(char['charcValues'])

        return res

    @staticmethod
    def _process_characteristics(text: str) -> list[dict]:
        text = text.decode('utf-8')

        # первый паттерн
        if '\'b\\\'' in text:
            return eval(re.sub(r'\\+', r'\\', text)[5:-3].replace('true', 'True').replace('false', 'False'))

        # второй паттерн
        text = text.replace("}\n", "},")
        text = re.sub(r'\n\s+', ' ', text)
        text = re.sub(r'array', 'np.array', text)

        return eval(text)


class CharReducer(BaseEstimator, TransformerMixin):

    def __init__(self, char_cols: list = None):
        if char_cols is None:
            self.char_cols = ['sex', 'age_restrictions', 'season', 'fragility']
        else:
            self.char_cols = char_cols

        if 'season' in self.char_cols:
            self.season_seq = ['лето', 'зима', 'демисезон', 'круглогодичный']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        for char_col in self.char_cols:
            X[char_col] = X[char_col].apply(getattr(self, f'_get_{char_col}'))

        return X

    @staticmethod
    def _get_sex(x):
        if x is np.nan:
            return x
        x = x.lower()

        if x == 'детский':
            return 'для детей'

        if x in ['женский', 'women']:
            return 'для женщин'

        if x in ['мужской', 'men']:
            return 'для мужчин'

        if x in ['мальчики', 'boy']:
            return 'для мальчиков'

        if x in ['девочки', 'girl']:
            return 'для девочек'

        return np.nan

    def _get_season(self, x: list[str] | str) -> str:
        """
        Обработка характеристик, где встречается несколько значения списком.
        """

        if isinstance(x, list):
            for param in self.season_seq:
                if param in x:
                    return param
                return x[0]

        return x

    @staticmethod
    def _process_fragility(x):
        if re.match(r'не х[рупкое]+|нет|прочн|над[её]ж', x) is not None:
            return 'не хрупкий'
        elif re.match(r'х[рупкое]+|да|не брос', x) is not None:
            return 'хрупкий'
        else:
            return 'не хрупкий'

    def _get_fragility(self, x):
        if not isinstance(x, str):
            return np.nan

        if isinstance(x, list):
            x = set([self._process_fragility(i) for i in x])
            if len(x) == 1:
                return x[0]
            else:
                return np.nan

        return self._process_fragility(x)

    @staticmethod
    def _get_age_restrictions(ar):
        if ar is np.nan:
            return ar

        if isinstance(ar, list):
            ar = ' '.join(ar)

        if ar in ['0+', '0 +']:
            return np.nan

        ar = ar.lower()

        if re.findall(r'без огран|нет огран|для всех|любо[йг]', ar):
            return 'для всех возрастов'

        if re.findall(r'мес|годик|рожден|малыш|реб[её]н', ar):
            return 'для малышей'

        nums = sorted(re.findall(r'\d+', ar))

        if nums:
            num = int(nums[0])
            if num >= 18:
                return 'для взрослых'
            elif 12 <= num < 18:
                return 'для подростков'
            elif 3 <= num < 12:
                return 'для детей'
            else:
                return 'для малышей'

        if re.findall(r'дет[ямскей]{2}|школ|девоч|мальчик', ar):
            return 'для детей'

        if re.findall(r'мужч|женщ|взросл', ar):
            return 'для взрослых'

        if re.findall(r'подрост', ar):
            return 'для подростков'

        return 'для всех возрастов'
