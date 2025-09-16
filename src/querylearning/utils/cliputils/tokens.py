from typing import List

import torch
from torch.utils.data import Dataset

import clip


class TokenDatasetCLIP(Dataset):

    def __init__(self, text_list: List[str]):
        self.clip_tokenizer = clip.tokenize
        self.text = text_list
        self.tokens = []
        for t in self.text:
            try:
                t = torch.tensor(self.clip_tokenizer(t)[0], dtype=torch.int64)
                self.tokens.append(t)
            except Exception as e:
                print(f"Failed to tokenize text: {t}. Error: {str(e)}")
                continue

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item: int):
        return self.tokens[item]