import sys
from .classification_heads import *
from .classificator import Classificator


sys.path.append('.classification_heads')

PRETRAINED_HEADS = {
    'category': 'mlp_category',
    'sub_category': 'mlp_sub_category',
    'isadult': 'mlp_isadult',
    'sex': 'mlp_sex',
    'age_restrictions': 'mlp_age_restrictions',
    'season': 'mlp_season',
    'fragility': 'mlp_fragility'
}

__all__ = ['PRETRAINED_HEADS', 'Classificator']
__version__ = '0.0.1'
