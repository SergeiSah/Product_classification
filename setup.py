import os
import re

from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


def get_requirements():
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
        return f.read().splitlines()


def get_version():
    with open(os.path.join(os.path.dirname(__file__), 'product_classificator', '__init__.py')) as f:
        init = f.read()
    return re.findall(r"__version__\s?=\s?['\"]([\d.]+)['\"]", init)[0]


setup(
    name='product_classificator',
    version=get_version(),
    author='Sergei Sakhonenkov',
    author_email='sergei.sakhonenkov@gmail.com',
    description='Classification of products using ruCLIP embeddings of product description and image',
    packages=['product_classificator'],
    package_dir={'': 'product_classification'},
    package_data={'product_classificator.heads': ['*.pt', '*.pkl']},
    include_package_data=True,
    long_description=readme(),
    long_description_content_type='text/markdown',
    install_requires=get_requirements(),
)
