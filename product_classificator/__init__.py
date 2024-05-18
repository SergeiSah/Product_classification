import os
import sys
from .classificator import Classificator
from huggingface_hub import hf_hub_url, cached_download


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

RUCLIP_MODELS = [
    'ruclip-vit-base-patch32-224',
    'ruclip-vit-base-patch16-224',
    'ruclip-vit-large-patch14-224',
    'ruclip-vit-large-patch14-336',
    'ruclip-vit-base-patch32-384',
    'ruclip-vit-base-patch16-384'
]

RUCLIP_FILES = [
    'bpe.model', 'config.json', 'pytorch_model.bin'
]


def load(name, cache_dir='/tmp/ruclip'):
    assert name in RUCLIP_MODELS, 'Incorrect model name'

    cache_dir = os.path.join(cache_dir, name)

    for filename in RUCLIP_FILES:
        config_file_url = hf_hub_url(repo_id=f'ai-forever/{name}', filename=f'{filename}')
        cached_download(config_file_url, cache_dir=cache_dir, force_filename=filename)


__all__ = ['PRETRAINED_HEADS', 'RUCLIP_MODELS', 'Classificator']
__version__ = '0.0.2'
