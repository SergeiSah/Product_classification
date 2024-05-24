import os
import torch
import pandas as pd
from more_itertools import chunked
from tqdm import tqdm

from product_classificator.utils.timer import Timer
from .utils.image_loader import get_images, get_images_from_zip, is_files_in_zip
from .utils.cleaner import clean_dataset, TextCleaner
from .utils.char_processor import CharExtractor, CharReducer
from product_classificator import Classificator


def _load_images(image_names: list, path_to_images: str):
    files_in_zip = is_files_in_zip(path_to_images)

    if files_in_zip:
        images = get_images_from_zip(image_names, path_to_images)
    else:
        images = get_images(image_names, path_to_images)

    return images


def _start_test(clf, image_names, texts, characteristics, batch_size, path_to_images, warming_up_batches_num):
    assert len(image_names) == len(texts), f'Image and text arrays must have the same length'
    assert batch_size < len(image_names), \
        f'Batch size must be less than the number of elements in the arrays'

    print('Warming up... ', end='')
    batches = list(zip(
        list(chunked(texts, batch_size)),
        list(chunked(image_names, batch_size))
    ))

    for _ in range(warming_up_batches_num):
        texts_batch, images_batch = batches[0]
        images_batch = _load_images(images_batch, path_to_images)

        clf.classify_products(texts_batch, images_batch, characteristics)

    print('Done')
    return batches


def test_preprocessing_time(path_to_dfs: str,
                            train_name: str = 'wb_school_train.parquet',
                            test_name: str = 'wb_school_test.parquet',
                            iterations_num: int = 5) -> pd.DataFrame:

    timer = Timer()
    log = pd.DataFrame(
        dtype='float64'
    )

    for it in tqdm(range(iterations_num), desc='Iterations'):
        with timer:
            train_df = pd.read_parquet(os.path.join(path_to_dfs, train_name))
            test_df = pd.read_parquet(os.path.join(path_to_dfs, test_name))
        log.loc[it, 'loading_dfs'] = timer.last_period.total_seconds()

        with timer:
            train_df = clean_dataset(train_df)
            test_df = clean_dataset(test_df, ['price', 'brand'])
        log.loc[it, 'cleaning_dfs'] = timer.last_period.total_seconds()

        with timer:
            char_extractor = CharExtractor()
            char_reducer = CharReducer()

            train_df = char_extractor.fit_transform(train_df)
            train_df = char_reducer.fit_transform(train_df)
        log.loc[it, 'processing_chars'] = timer.last_period.total_seconds()

        cleaner = TextCleaner()
        with timer:
            train_df['description'] = cleaner.fit_transform(train_df['description'])
        log.loc[it, 'cleaning_train_desc'] = timer.last_period.total_seconds()

        with timer:
            test_df['description'] = cleaner.fit_transform(test_df['description'])
        log.loc[it, 'cleaning_test_desc'] = timer.last_period.total_seconds()

    return log


def test_classifier_inference_time(clf: Classificator,
                                   image_names: list,
                                   texts: list,
                                   characteristics: list[str],
                                   batch_size: int,
                                   path_to_images: str,
                                   warming_up_batches_num: int = 20) -> pd.DataFrame:

    batches = _start_test(clf,
                          image_names,
                          texts,
                          characteristics,
                          batch_size,
                          path_to_images,
                          warming_up_batches_num)

    log = pd.DataFrame(dtype='float64')
    processor = clf.clip_predictor.clip_processor

    timer = Timer()
    timer_sum = Timer()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(batches), total=len(image_names) // batch_size):
            with timer_sum:
                texts_batch, images_batch = batch

                with timer:
                    images_batch = _load_images(images_batch, path_to_images)
                log.loc[i, 'load_images'] = timer.last_period.total_seconds()

                with timer:
                    pixel_values = processor(images=images_batch, return_tensors='pt',
                                             padding=True)['pixel_values'].to(clf.device)
                log.loc[i, 'img_process'] = timer.last_period.total_seconds()

                with timer:
                    input_ids = processor(text=texts_batch, return_tensors='pt',
                                          padding=True)['input_ids'].to(clf.device)
                log.loc[i, 'txt_process'] = timer.last_period.total_seconds()

                with timer:
                    image_latents = clf.clip_predictor.get_image_latents(pixel_values)
                log.loc[i, 'img_encode'] = timer.last_period.total_seconds()

                with timer:
                    text_latents = clf.clip_predictor.get_text_latents(input_ids)
                log.loc[i, 'txt_encode'] = timer.last_period.total_seconds()

                with timer:
                    concat_latents = torch.cat((image_latents, text_latents), dim=1)

                    if clf.reducer is not None:
                        concat_latents = torch.Tensor(
                            clf.reducer.transform(concat_latents.detach().cpu().numpy()))

                    results = {}
                    for name in characteristics:
                        indexes = clf.heads[name](concat_latents).argmax(dim=1).detach().cpu().numpy()
                        results[name] = [clf.label_to_char[name][index] for index in indexes]
                log.loc[i, 'predictions'] = timer.last_period.total_seconds()

            log.loc[i, 'cols_sum'] = log.iloc[i, 0:-1].sum()
            log.loc[i, 'sum_batch_time'] = timer_sum.last_period.total_seconds()

    return log


def test_ruclip_training_time(clf: Classificator,
                              image_names: list,
                              texts: list,
                              characteristics: list[str],
                              batch_size: int,
                              path_to_images: str,
                              optimizer: torch.optim.Optimizer,
                              loss_img: torch.nn.Module,
                              loss_txt: torch.nn.Module,
                              warming_up_batches_num: int = 20,
                              num_last_resblocks: int = 4) -> pd.DataFrame:

    batches = _start_test(clf,
                          image_names,
                          texts,
                          characteristics,
                          batch_size,
                          path_to_images,
                          warming_up_batches_num)

    log = pd.DataFrame(dtype='float64')

    timer = Timer()
    timer_sum = Timer()

    embeddings = torch.zeros((len(image_names), 1024), dtype=torch.float32)
    losses = []

    clip = clf.clip_predictor.clip_model
    processor = clf.clip_predictor.clip_processor

    for param in clip.parameters():
        param.requires_grad = False

    clip.visual.transformer.resblocks[-num_last_resblocks:].requires_grad_(True)
    clip.visual.ln_post.requires_grad_(True)
    clip.transformer.resblocks[-num_last_resblocks:].requires_grad_(True)

    clip.train()
    for i, batch in tqdm(enumerate(batches), desc='Batches', total=len(image_names) // batch_size):
        with timer_sum:
            optimizer.zero_grad()

            texts_batch, images_batch = batch

            with timer:
                images_batch = _load_images(images_batch, path_to_images)
            log.loc[i, 'load_images'] = timer.last_period.total_seconds()

            with timer:
                pixel_values = processor(images=images_batch, return_tensors='pt',
                                         padding=True)['pixel_values'].to(clf.device)
            log.loc[i, 'img_process'] = timer.last_period.total_seconds()

            with timer:
                input_ids = processor(text=texts_batch, return_tensors='pt',
                                      padding=True)['input_ids'].to(clf.device)
            log.loc[i, 'txt_process'] = timer.last_period.total_seconds()

            with timer:
                image_features = clf.clip_predictor.get_image_latents(pixel_values)
            log.loc[i, 'img_encode'] = timer.last_period.total_seconds()

            with timer:
                text_features = clf.clip_predictor.get_text_latents(input_ids)
            log.loc[i, 'txt_encode'] = timer.last_period.total_seconds()

            with timer:
                logit_scale = clip.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                labels = torch.arange(len(pixel_values), dtype=torch.long, device=clf.device)

                img_loss = loss_img(logits_per_image, labels)
                txt_loss = loss_txt(logits_per_text, labels)

                loss = (img_loss + txt_loss) / 2
            log.loc[i, 'loss_calc'] = timer.last_period.total_seconds()

            with timer:
                loss.backward()
            log.loc[i, 'loss_backprop'] = timer.last_period.total_seconds()

            with timer:
                optimizer.step()
            log.loc[i, 'optim_step'] = timer.last_period.total_seconds()

            losses.append(loss.item())

            with timer:
                idxs = list(range(i * len(batch[0]), (i + 1) * len(batch[0])))
                concat = torch.cat((image_features, text_features), dim=1).detach().cpu()

                for j, idx in enumerate(idxs):
                    embeddings[idx] = concat[j]

            log.loc[i, 'save_embeds'] = timer.last_period.total_seconds()
        log.loc[i, 'cols_sum'] = log.iloc[i, 0:-1].sum()
        log.loc[i, 'sum_batch_time'] = timer_sum.last_period.total_seconds()

    clip.eval()
    return log


__all__ = ['test_classifier_inference_time', 'test_ruclip_training_time', 'test_preprocessing_time']
