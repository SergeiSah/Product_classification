# Product classificator

Model for classification of a product on the basis of its image and text description. Classificator built
by the following approach: 
- using one of the [ruCLIP models](https://github.com/ai-forever/ru-clip/tree/main) embeddings of images and 
text descriptions are obtained;
- embeddings are concatenated, reduced with PCA and fed to the input of an 
[MLP](https://pytorch.org/vision/main/generated/torchvision.ops.MLP.html)
classificators, trained on dataset  with product images and descriptions from 
Wildberries online marketplace.

Development of the approach and data analysis are presented in notebooks:
1. [Explanatory Data Analysis](https://github.com/SergeiSah/Product_classification/blob/master/study/1.%20EDA.ipynb)
2. [Possible solutions](https://github.com/SergeiSah/Product_classification/blob/master/study/2.%20Possible%20solutions.ipynb)
3. [Baseline](https://github.com/SergeiSah/Product_classification/blob/master/study/3.%20Baseline.ipynb)
4. [Optimization](https://github.com/SergeiSah/Product_classification/blob/master/study/4.%20Optimization.ipynb)


Results of final tests of classificators:

| name             | F1-macro |
|:-----------------|:---------|
| category         | 0.880    |
| sub_category     | 0.847    |
| isadult          | 0.963    |
| sex              | 0.823    |
| season           | 0.622    |
| age_restrictions | 0.556    |
| fragility        | 0.704    |

# Installation

```commandline
pip install git+https://github.com/SergeiSah/Product_classification.git
```

# Usage

## Product classification
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YsjAzCdH3HaN4-TKwq9oWf_EnN-OhACX?usp=sharing)
Example of classification by *category*, *sub_category* and *isadult*

**Note**: in Colab `numpy` must be version `~1.25.0` for correct work.

```python
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from product_classificator import Classificator, load


load('ruclip-vit-base-patch16-384')

df = pd.read_csv('<path_to_df_with_dataset info>')
texts = df.descriptions.values
img_names = df.image_names.values

images = []
for img_name in img_names:
    images.append(Image.open('<path_to_images>' + img_name))

clf = Classificator()

res = clf.classify_products(texts, images)

fig, axes = plt.subplots(5, 2, figsize=(8, 20))

axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.imshow(images[i])
    ax.set_title(f'category: {res["category"][i]}\nsub_category: {res["sub_category"][i]}\nisadult: {res["isadult"][i]}',
                 fontsize=10)
    ax.axis('off')
```
![pics/example.png](./pics/example.png)

To get embeddings of products, use the following method:

```python
clf.get_products_embeddings(texts, images)
```

### Convert model to ONNX

To use model in .onnx format, first export it and then load for use:

```python
# to export model an example of input text and image is needed
clf.export_model_to_onnx(text, image)
clf.to_onnx_clip()
```

To return for using base pytorch model, use `.to_clip()` method.

## Training

By default, classificator utilizes ruCLIP model without additional training on WB dataset, only MLP head classificators 
were trained.

To train MLP head classificators, one can utilize `Trainer` from the package.
```python
from product_classificator.training import Trainer

trainer = Trainer(
    path_to_images='<folder with images or path to zip file>',
    path_to_texts='<path to .parquet files with descriptions>'
)

trainer.train_heads_only()
```
By default, `ruclip-vit-base-patch16-384` model is used. To change the model specify `ruclip_model` in `Trainer` 
constructor.

`Trainer` configured to mainly work with WB dataset, but by modifying methods `_preprocessing_texts`, 
`_loading_texts` one can train classificators on other datasets.

Before training text descriptions were cleaned from punctuation and stopwords (except *не* and *для*), 
units (*кг*, *гр*, *л*, *штук* etc.) and digits, products with empty description and/or empty (corrupted) image were
removed. Images were converted from BGR to RGB format.

One can also train last resblocks of ruCLIP vision and text transformers.
```python
trainer.train_ruclip()
```
Training procedure is based on the following [code](https://github.com/revantteotia/clip-training/blob/main/train.py).

After each epoch MLP head classificators will be trained. To speed up training the processed images and texts are cached on
disk in `Trainer().cache_dir` directory. Embeddings are cached in RAM. To delete all cached files after the training specify 
`Trainer().end_of_train['del_img_and_txt_tensors']=True`.

All training results will be saved in `Trainer().experiment_dir` directory.

### WB dataset

- Train data: image, title, description and characteristics for a product.
- Test data: image, title, description, category, subcategory, price, brand and isadult marker for a product.

Images are stored in `wb_school_horizcv_images.zip` file.

Table data are presented in `.parquet` format.
- `wb_school_train.parquet`
- `wb_school_test.parquet`

Column `nm` indicates unique identifier of a product. It coincides with image name (without extension).

### Logging

To log training with clearml set task parameter of the training methods
```python
from clearml import Task

task = Task.init('Project name', 'Task name')
trainer.train_ruclip(task)
```

**Example**: [ruCLIP training](https://github.com/SergeiSah/Product_classification/blob/master/study/Training%20example.ipynb)

# Inference

Speed tests of inference, performed on a machine with the following elements:
- CPU: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 6 cores, 12 threads
- GPU: NVIDIA GeForce RTX 2060 (6 GB)

|     | batch 1 | batch 32 | batch 64 |
|-----|---------|----------|----------|
| CPU | 495 ms  | 410 ms   | 406 ms   |
| GPU | 50 ms   | 41 ms    | 40 ms    |

In the table above time for processing of one pair text/image in batches of different sizes is shown.<br>
[Example of inference measurements](https://github.com/SergeiSah/Product_classification/blob/master/examples/Inference_time.ipynb)