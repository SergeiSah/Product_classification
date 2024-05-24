import os.path

import torch
from PIL import Image


def export_onnx(clf,
                text,
                image,
                opset_version: int = 14,
                dynamic_axes=None):

    assert isinstance(text, str)
    assert isinstance(image, Image.Image)

    clip = clf.get_clip()
    processor = clf.get_processor()

    folder = os.path.join(clf.cache_dir, clf.model_name)

    dynamic_axes = dynamic_axes or {
        "input": {
            0: "batch_size",
            1: "sequence_len"
        },
        "output": {
            0: "batch_size"
        }
    }

    dummy = processor(text=[text], images=[image], return_tensors='pt', padding=True)
    dummy_text = clip.token_embedding(dummy['input_ids']).type(clip.dtype).permute(1, 0, 2)
    dummy_image = dummy['pixel_values']

    torch.onnx.export(
        clip.visual,
        dummy_image,
        os.path.join(folder, "clip_visual.onnx"),
        input_names=['input'],
        output_names=['output'],
        opset_version=opset_version,
        dynamic_axes=dynamic_axes
    )

    torch.onnx.export(
        clip.transformer,
        dummy_text,
        os.path.join(folder, "clip_transformer.onnx"),
        input_names=['input'],
        output_names=['output'],
        opset_version=opset_version,
        dynamic_axes=dynamic_axes
    )
