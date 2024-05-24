# Source: https://github.com/ai-forever/ru-clip/blob/main/ruclip/predictor.py
import torch
from PIL import Image


class Predictor:
    def __init__(self, clip_model, clip_processor, device):
        self.device = device
        self.clip_model = clip_model.to(self.device)
        self.clip_model.eval()
        self.clip_processor = clip_processor

    def get_text_latents(self, inputs: str | list[str] | torch.Tensor):
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, torch.Tensor):
            inputs = self.clip_processor(text=inputs)['input_ids'].to(self.device)
        else:
            raise TypeError('Input must be either a string, a list of strings or torch.Tensor')

        text_latents = self.clip_model.encode_text(inputs)
        return text_latents / text_latents.norm(dim=-1, keepdim=True)

    def get_image_latents(self, inputs: Image.Image | list[Image.Image] | torch.Tensor):
        if isinstance(inputs, Image.Image):
            inputs = [inputs]
        if isinstance(inputs, list):
            inputs = self.clip_processor(images=inputs)['pixel_values'].to(self.device)
        else:
            raise TypeError('Input must be either a string, a list of strings or torch.Tensor')

        image_latents = self.clip_model.encode_image(inputs)
        image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
        return image_latents
