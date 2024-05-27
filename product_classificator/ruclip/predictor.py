# Source: https://github.com/ai-forever/ru-clip/blob/main/ruclip/predictor.py
import torch
from PIL import Image


class Predictor:
    def __init__(self, clip_model, clip_processor, device, is_onnx):
        self.device = device
        self.clip_model = clip_model.to(self.device)
        self.clip_model.eval()
        self.clip_processor = clip_processor
        self.is_onnx = is_onnx

    def get_text_latents(self, inputs: str | list[str] | torch.Tensor):
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, torch.Tensor):
            inputs = self.clip_processor(text=inputs, return_tensors='pt', padding=True)['input_ids']
            if not self.is_onnx:
                inputs = inputs.to(self.device)

        text_latents = self.clip_model.encode_text(inputs)
        return torch.nn.functional.normalize(text_latents, p=2, dim=-1)

    def get_image_latents(self, inputs: Image.Image | list[Image.Image] | torch.Tensor):
        if isinstance(inputs, Image.Image):
            inputs = [inputs]
        if not isinstance(inputs, torch.Tensor):
            inputs = self.clip_processor(images=inputs, return_tensors='pt', padding=True)['pixel_values']
            if not self.is_onnx:
                inputs = inputs.to(self.device)

        image_latents = self.clip_model.encode_image(inputs)
        return torch.nn.functional.normalize(image_latents, p=2, dim=-1)
