from typing import Callable
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class AbstractClipEncoder(nn.Module, metaclass=ABCMeta):

    def __init__(self, model_id: str):
        super().__init__()
        self.model_id = model_id
        self.encode = None

    def forward(
        self,
        tokens: torch.Tensor,
        normout: bool = True
    ) -> torch.Tensor:
        x = self.encode(tokens)
        return F.normalize(x, dim=-1) if normout else x

    def __str__(self) -> str:
        return self.model_id.replace('/', '_').replace('-', '_')

    @abstractmethod
    def __repr__(self) -> str: pass


class TextEncoderCLIP(AbstractClipEncoder):

    def __init__(self, model_id: str, device: str = 'cuda'):
        super().__init__(model_id=model_id)
        model, _ = clip.load(model_id, device=device)
        self.encode = model.encode_text

    def __repr__(self) -> str: return f'CLIP Text Encoder ({self.model_id})'


class ImageEncoderCLIP(AbstractClipEncoder):

    def __init__(self, model_id: str, device: str = 'cuda'):
        super().__init__(model_id=model_id)
        model, self.preprocess = clip.load(model_id)
        self.encode = model.encode_image

    def __repr__(self) -> str: return f'CLIP Image Encoder ({self.model_id})'