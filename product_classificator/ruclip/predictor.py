# Source: https://github.com/ai-forever/ru-clip/blob/main/ruclip/predictor.py
import torch
from PIL import Image


class Predictor:
    def __init__(self, clip_model, clip_processor, device):
        self.device = device
        self.clip_model = clip_model.to(self.device)
        self.clip_model.eval()
        self.clip_processor = clip_processor

    def get_text_latents(self, inputs: str | list[str] | torch.LongTensor):
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, torch.LongTensor):
            inputs = self.clip_processor(text=inputs, return_tensors='pt', padding=True)['input_ids']

        text_latents = self.clip_model.encode_text(inputs.to(self.device))
        return text_latents / text_latents.norm(dim=-1, keepdim=True)

    def get_image_latents(self, inputs: Image.Image | list[Image.Image] | torch.Tensor):
        if isinstance(inputs, Image.Image):
            inputs = [inputs]
        if not isinstance(inputs, torch.Tensor):
            inputs = self.clip_processor(images=inputs, return_tensors='pt', padding=True)['pixel_values']

        image_latents = self.clip_model.encode_image(inputs.to(self.device))
        return image_latents / image_latents.norm(dim=-1, keepdim=True)
