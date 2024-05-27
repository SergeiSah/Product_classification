import os.path

import torch
from PIL import Image
from ..ruclip.onnx_model import Textual


def export_onnx(clf,
                text,
                image,
                params: dict):

    assert isinstance(text, str)
    assert isinstance(image, Image.Image)

    clip = clf.get_clip()
    processor = clf.get_processor()
    textual = Textual(clip)

    folder = os.path.join(clf.cache_dir, clf.model_name)

    dummy = processor(text=[text], images=[image], return_tensors='pt', padding=True)
    dummy_text = dummy['input_ids'].to(clf.clip_predictor.device)
    dummy_image = dummy['pixel_values'].to(clf.clip_predictor.device)

    torch.onnx.export(
        clip.visual,
        dummy_image,
        os.path.join(folder, "clip_visual.onnx"),
        **params
    )

    torch.onnx.export(
        textual,
        dummy_text.to(clf.clip_predictor.device),
        os.path.join(folder, "clip_transformer.onnx"),
        **params
    )
