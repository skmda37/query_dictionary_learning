from pathlib import Path
import os
import argparse
import clip
import logging
import wandb
from collections import namedtuple
from abc import ABCMeta, abstractmethod

from typing import List, Tuple, Optional, Union
import torch.distributed as dist
import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast

from querylearning.utils.cliputils.clipdim import CLIPDIM
from querylearning.utils.cliputils.encoders import TextEncoderCLIP
from querylearning.utils.cliputils.tokens import TokenDatasetCLIP


class QueryUniverse:

    def __init__(self, config: dict):
        self.config = config
        embeddings = get_query_embeddings(
            dataroot=config['dataroot'],
            clip_model_id=config['queries']['clip_model_type'],
            queryfile=config['queries']['universe'],
            batch_size=config['train']['batch_size'],
            num_workers=config['num_workers'],
            full_precision=config['fp']
        )
        # l2 normalize embeddings
        self.embeddings = nn.functional.normalize(embeddings, dim=1)
        self.size = embeddings.shape[0]
        assert self.size == len(self.as_text)

    @property
    def as_text(self) -> List[str]:
        """Get text representation of queries in universe"""
        with open(self.config['queries']['universe'], 'r') as f:
            L = [line.strip() for line in f.readlines() if line.strip()]
        return L

    def __repr__(self) -> str:
        return 'Q(' + ', '.join(self.as_text) + ')'


class AbstractQueryDictionary(nn.Module, metaclass=ABCMeta):

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        # Embedding dim of queries
        self.d = CLIPDIM[config['queries']['clip_model_type']]
        # Initialize query dictionary embeddings
        self._init_embeddings()
        self.K = self.embeddings.shape[0]
        self.config['K'] = self.K

    @abstractmethod
    def _init_embeddings(self): pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: pass

    @property
    @abstractmethod
    def as_text(self) -> List[str]: pass

    def __repr__(self) -> str:
        return 'Q(' + ', '.join(self.as_text) + ')'

    def _get_clip_answers(
        self,
        x: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        """Computes query answers for images x and queries q
        args:
            x: torch.Tensor (bs, d) representing CLIP image embeddings
            q: torch.Tensor (K, d) representing query embeddings

        return: torch.Tensor of shape (bs, K) representing query answers
        """
        cossim = torch.matmul(x, q.T).abs()
        SA = self._min_max_norm(cossim)  # Soft answers

        if self.config['queries']['quantized_answers']:
            # Get hard (quantized) answers by thresholding soft answers
            # (using Straight-Trough Estimate for backpropagation)
            A = ((SA > 0.5).float() - SA).detach() + SA
        else:
            # Use soft answers
            A = SA
        return A

    def _min_max_norm(self, answer_scores: torch.Tensor) -> torch.Tensor:
        m = answer_scores.min(dim=-1, keepdim=True).values
        M = answer_scores.max(dim=-1, keepdim=True).values
        return (answer_scores - m) / (M-m)


class StaticQueryDictionary(AbstractQueryDictionary):

    def _init_embeddings(self) -> None:
        embeddings = get_query_embeddings(
                dataroot=self.config['dataroot'],
                clip_model_id=self.config['queries']['clip_model_type'],
                queryfile=self.config['queries']['init'],
                batch_size=self.config['train']['batch_size'],
                num_workers=self.config['num_workers'],
                full_precision=self.config['fp']
            )
        # l2 normalize query embeddings
        embeddings = nn.functional.normalize(embeddings, dim=1)
        self.register_buffer('embeddings', embeddings)  # query embeddings

    @property
    def as_text(self) -> List[str]:
        """Get text representation of queries"""
        with open(self.config['queries']['init'], 'r') as f:
            L = [line.strip() for line in f.readlines() if line.strip()]
        return L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes query answers for images x
        args:
            x: torch.Tensor (bs, d) representing CLIP image embeddings

        return: torch.Tensor of shape (bs, K) representing query answers
        """
        return self._get_clip_answers(x=x, q=self.embeddings)


class LearnableQueryDictionary(AbstractQueryDictionary):

    def __init__(self, config: dict) -> None:
        self.universe = QueryUniverse(config=config)
        super().__init__(config=config)

    def _init_embeddings(self) -> None:
        """Initializes the query embeddings from
        .txt file of queries from the query universe or
        as random queries from query universe
        """
        if self.config['queries']['init'] == 'random':
            indices = torch.randint(0, self.universe.size, (self.K,))
        elif Path(self.config['queries']['init']).exists():
            with open(self.config['queries']['init'], 'r') as f:
                text = [line.strip() for line in f.readlines() if line.strip()]
                U_as_text = self.universe.as_text
                indices = [U_as_text.index(query) for query in text]
        else:
            raise ValueError(
                f"config['queries']['init'] needs to be"
                "'random' or path to .txt file with queries"
                "from query universe"
            )
        self.embeddings = nn.Parameter(
            self.universe.embeddings[indices].detach().clone()
        )

    @property
    def as_text(self) -> List[str]:
        indices = self._get_nn_indices()
        U_as_text = self.universe.as_text
        return [U_as_text[i] for i in indices.tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes query answers for images x
        args:
            x: torch.Tensor (bs, d) representing CLIP image embeddings

        return: torch.Tensor of shape (bs, K) representing query answers
        """
        embeddings = nn.functional.normalize(self.embeddings, dim=1)

        if self.config['queries']['quantize']:
            # Use straight-through estimate to
            # learn queries from the universe
            embeddings = self._ste(embeddings)

        return self._get_clip_answers(x=x, q=embeddings)

    def _ste(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Applies Straight-Through Estimate (STE)"""

        # Get indices of nearest neighbor queries in universe
        indices = self._get_nn_indices()
        nn_embeddings = self.universe.embeddings[indices]

        # STE trick
        return (nn_embeddings - self.embeddings).detach() + self.embeddings

    def _get_nn_indices(self) -> torch.Tensor:
        """Returns indices of nearest neighbor queries in universe"""
        sim = torch.matmul(self.embeddings, self.universe.embeddings.T).abs()
        return sim.argmax(dim=1)  # Shape:(self.K,)


def get_query_embeddings(
    dataroot: str,
    clip_model_id: str,
    queryfile: str,
    batch_size: int,
    num_workers: int,
    full_precision: bool
) -> torch.Tensor:
    a = clip_model_id.replace('/', '_').replace('-', '_')
    b = os.path.basename(queryfile).replace('.txt', '.pt')
    qe_path = Path(dataroot) / 'queryembeddings' / a / b

    if qe_path.exists():
        embeddings = torch.load(qe_path, map_location=torch.device('cuda'))
    else:
        with open(queryfile, 'r') as f:
            qs = [line.strip() for line in f.readlines() if line.strip()]
        dataset = TokenDatasetCLIP(qs)
        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size
        )
        d = 'cuda'
        encoder = TextEncoderCLIP(model_id=clip_model_id, device=d)
        e_list = []
        desc = f'Precomputing query embeddings from {queryfile}'
        with tqdm(total=len(dataloader), colour='blue', desc=desc) as pbar:
            for i, tokens in enumerate(dataloader):
                tokens = tokens.to(d)
                dtype = torch.float32 if full_precision else torch.bfloat16
                with autocast(cache_enabled=False, device_type=d, dtype=dtype):
                    with torch.no_grad():
                        e_list.append(encoder(tokens, normout=True).float().cpu())
                pbar.update(1)
        embeddings = torch.cat(e_list, dim=0)
        os.makedirs(os.path.dirname(qe_path), exist_ok=True)
        torch.save(embeddings, qe_path)

    return embeddings


def build_query_dictionary(config: dict, mode: str):
    if mode in ['alt_qdl', 'joint_qdl']:
        Q = LearnableQueryDictionary(config).cuda()
    elif mode == 'vip':
        Q = StaticQueryDictionary(config).cuda()
    else:
        raise ValueError()
    return Q


if __name__ == '__main__':
    dataroot = 'data'
    clip_model_id = 'ViT-B/32'
    queryfile = 'querydict/K_LLM_cifar10.txt'
    batch_size = 32
    num_workers = 4
    full_precision = False
    e = get_query_embeddings(
        dataroot=dataroot,
        clip_model_id=clip_model_id,
        queryfile=queryfile,
        batch_size=batch_size,
        num_workers=num_workers,
        full_precision=full_precision
    )
    print(f'e.shape {e.shape}')