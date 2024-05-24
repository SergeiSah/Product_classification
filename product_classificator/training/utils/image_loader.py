import zipfile
import os
from PIL import Image


def get_image(file_name: str) -> Image.Image:
    """
    Загрузка изображений по названию файла, преобразование из BGR(или GrayScale) в RGB,
    возврат изображения в формате PIL Image.
    """
    img = Image.open(file_name)
    if img.mode == 'RGB':
        # переводим из формата BGR в RGB
        B, G, R = img.split()
        return Image.merge("RGB", (R, G, B))

    elif img.mode == 'L':
        # переводим из серого в RGB
        return img.convert('RGB')

    else:
        raise ValueError('Unsupported image mode: {}'.format(img.mode))


def get_images(file_names: list[str], path_to_files: str) -> list[Image.Image]:
    """
    Загрузка изображений по списку имен файлов, преобразование из BGR(или GrayScale) в RGB,
    возврат списка всех преобразованных изображений в формате PIL Image.
    """
    images = []

    for file_name in file_names:
        if not isinstance(file_name, str):
            raise ValueError('File name must be a string')

        images.append(get_image(path_to_files + file_name))

    return images


def get_images_from_zip(file_names: list[str], path_to_zip: str) -> list[Image.Image]:
    """
    Загрузка изображений из zip-файла по списку имен файлов, преобразование из BGR(или GrayScale) в RGB,
    возврат списка всех преобразованных изображений в формате PIL Image.
    """
    images = []

    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        for file_name in file_names:
            if not isinstance(file_name, str):
                raise ValueError('File name must be a string')

            images.append(get_image(zf.open(file_name)))

    return images


def is_files_in_zip(path_to_images: str) -> bool:
    if os.path.isdir(path_to_images):
        files_in_zip = False
    elif os.path.isfile(path_to_images):
        if path_to_images.endswith('.zip'):
            files_in_zip = True
    else:
        raise ValueError('Path to images must be a directory or a zip file')

    return files_in_zip
