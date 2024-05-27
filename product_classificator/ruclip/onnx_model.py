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
        output, = self.session_visual.run(["output"], {"input": np.array(pixel_values)})
        return torch.from_numpy(output).to(self.device)

    def encode_text(self, input_ids):
        output, = self.session_transformer.run(["output"], {"input": np.array(input_ids)})
        return torch.from_numpy(output).to(self.device)

    def forward(self, input_ids, pixel_values):
        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids)

        return text_features, image_features


class Textual(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.token_embedding = model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.float().argmax(dim=-1)] @ self.text_projection

        return x
