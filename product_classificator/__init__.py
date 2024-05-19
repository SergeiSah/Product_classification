import os
from .classificator import Classificator
from huggingface_hub import hf_hub_url, cached_download


RUCLIP_MODELS = [
    'ruclip-vit-base-patch16-384'
]

RUCLIP_FILES = [
    'bpe.model', 'config.json', 'pytorch_model.bin'
]


def pretrained_heads():
    dir_names = os.listdir(os.path.join(os.path.dirname(__file__), 'heads'))
    return dir_names


def load(name, cache_dir='/tmp/ruclip'):
    assert name in RUCLIP_MODELS, 'Incorrect model name'

    cache_dir = os.path.join(cache_dir, name)

    for filename in RUCLIP_FILES:
        config_file_url = hf_hub_url(repo_id=f'ai-forever/{name}', filename=f'{filename}')
        cached_download(config_file_url, cache_dir=cache_dir, force_filename=filename)


__all__ = ['pretrained_heads', 'RUCLIP_MODELS', 'Classificator']
__version__ = '0.0.2'
