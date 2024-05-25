import os
import gpustat
import numpy as np
import psutil
import platform
import torch
import pandas as pd
from more_itertools import chunked
from tqdm import tqdm
from .timer import Timer


class SpeedTest:

    def __init__(self, clf):

        self.clf = clf
        self.tests = []

        self.gpu_stats = gpustat.GPUStatCollection.new_query()

        self.machine_info = {
            'OS': platform.system(),
            'CPU': platform.processor(),
            'CPU cores': os.cpu_count(),
            'GPUs': {idx: f"GPU {gpu.index}: {gpu.name} ({round(gpu.memory_total / 1024)} GB)"
                     for idx, gpu in enumerate(self.gpu_stats)},
            'GPU driver': self.gpu_stats.driver_version,
            'RAM': f'{round(psutil.virtual_memory().total / 1024 ** 3)} GB',
        }

    def warming_up(self, text, image, warm_iterations):
        with torch.no_grad():
            for _ in tqdm(range(warm_iterations), desc='Warming up'):
                self.clf.classify_products(text, image)

    @staticmethod
    def batches(texts, images, batch_size):
        batches_num = len(texts) // batch_size

        indexes = np.arange(len(texts))
        np.random.shuffle(indexes)
        indexes = indexes[:batches_num * batch_size]

        texts = np.array(texts)[indexes]
        images = [images[i] for i in indexes]

        return zip(chunked(texts, batch_size), chunked(images, batch_size))

    def test_inference(self,
                       texts: list,
                       images: list,
                       batch_size: int,
                       iterations_num: int = 1,
                       warm_iterations: int = 0,
                       empty_cuda_cache: bool = True) -> pd.DataFrame:

        log = pd.DataFrame(
            columns=[f'iter_{i}' for i in range(iterations_num)],
            index=list(range(len(texts) // batch_size)),
            dtype='float64'
        )

        timer = Timer()

        if warm_iterations > 0:
            self.warming_up(texts[0], images[0], warm_iterations)

        if empty_cuda_cache:
            torch.cuda.empty_cache()

        with torch.no_grad():
            for i in range(iterations_num):
                for j, (texts_batch, images_batch) in tqdm(enumerate(self.batches(texts, images, batch_size)),
                                                           total=len(texts) // batch_size,
                                                           desc='Inference'):
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

