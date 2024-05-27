import os
import torch
import numpy as np
from onnxruntime import InferenceSession, SessionOptions
from .model import CLIP


class ONNXCLIP:

    def __init__(self, clip: CLIP,
                 device: str,
                 onnx_path: str):

        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']

        self.clip = clip.to(device)
        self.device = device

        self.options = SessionOptions()
        self.session_transformer = InferenceSession(os.path.join(onnx_path, "clip_transformer.onnx"),
                                                    self.options, providers=providers)
        self.session_visual = InferenceSession(os.path.join(onnx_path, "clip_visual.onnx"),
                                               self.options, providers=providers)

        self.session_transformer.disable_fallback()
        self.session_visual.disable_fallback()

    def encode_image(self, pixel_values):
        return self.session_visual.run(["output"], {"input": pixel_values})

    def encode_text(self, input_ids):
        x = self.clip.token_embedding(input_ids).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        x = x.permute(1, 0, 2)
        x = self.session_transformer.run(["output"], {"input": x})
        x = x.permute(1, 0, 2)
        x = self.clip.ln_final(x).type(self.clip.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        x = x[torch.arange(x.shape[0]), torch.where(input_ids == self.clip.eos_id)[1]] @ self.clip.text_projection
        return x

    def forward(self, input_ids, pixel_values):
        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids)

        return text_features, image_features

