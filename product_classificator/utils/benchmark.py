import torch
import pandas as pd
from more_itertools import chunked
from tqdm import tqdm
from .timer import Timer


class SpeedTest:

    def __init__(self, clf):

        self.clf = clf
        self.tests = []

    def run_test(self,
                 texts: list,
                 images: list,
                 batch_size: int,
                 iterations_num: int = 1,
                 empty_cuda_cache: bool = True) -> pd.DataFrame:

        log = pd.DataFrame(
            columns=[f'iter_{i}' for i in range(iterations_num)],
            index=list(range(len(texts) // batch_size))
        )

        timer = Timer()

        if empty_cuda_cache:
            torch.cuda.empty_cache()

        with torch.no_grad():
            for i in range(iterations_num):
                for j, (texts_batch, images_batch) in tqdm(enumerate(zip(chunked(texts, batch_size),
                                                                         chunked(images, batch_size))),
                                                           total=len(texts) // batch_size):
                    with timer:
                        self.clf.classify_products(texts_batch, images_batch)
                    log.loc[j, f'iter_{i}'] = timer.last_period.total_seconds()

        if empty_cuda_cache:
            torch.cuda.empty_cache()

        self.tests.append({
            'info': {
                'samples_num': len(texts),
                'batch_size': batch_size,
                'iterations_num': iterations_num
            },
            'log': log,
        })

        return log
