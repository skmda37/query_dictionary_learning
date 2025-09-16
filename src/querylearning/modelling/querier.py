from typing import Tuple

import torch
import torch.nn as nn


class ShallowMLPQuerier(nn.Module):

    def __init__(self, input_dim: int, n_queries: int, tau: float = None):
        super().__init__()
        self.input_dim = input_dim
        self.n_queries = n_queries
        # Architecture
        self.layer1 = nn.Linear(input_dim, 2000)
        self.layer2 = nn.Linear(2000, 500)
        self.norm1 = nn.LayerNorm(2000)
        self.norm2 = nn.LayerNorm(500)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.flat = nn.Flatten()
        # Set starting temperature
        self.tau = tau
        # heads
        self.head = nn.Linear(1000, self.n_queries)

    def update_tau(self, tau):
        self.tau = tau

    def hardargmax(self, query_logits):
        return nn.functional.one_hot(
            query_logits.argmax(dim=-1),
            num_classes=query_logits.shape[-1]
        )

    def get_next_query(self, history):
        mask = history[:, 1]
        x = self.relu(self.norm1(self.layer1(history)))
        x = self.relu(self.norm2(self.layer2(x)))
        x = self.flat(x)
        query_logits = self.head(x)
        # querying
        if mask is not None:
            query_logits = query_logits.masked_fill_(mask == 1, -torch.inf)
        query = self.softmax(query_logits / self.tau)
        query = (self.hardargmax(query_logits) - query).detach() + query

        return query

    def forward(self, history):
        next_query_mask = self.get_next_query(history)
        return next_query_mask

    def append_next_query(
        self,
        A: torch.Tensor,
        M: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Appends next query selected by querier to history.

        Parameters:
        -----------
        A: torch.Tensor
            Tensor of shape (bs, num_queries) representing all
            query answers.
        M: torch.Tensor
            Tensor of shape (bs, num_queries) representing history of
            previously selected queries.

        Returns:
        --------
        S_and_q: torch.Tensor
            Updated history of query answer pairs; Shape (bs, 2, num_queries).
        M: torch.Tensor
            Updated mask representing queries selected in history.
        q: torch.Tensor
            Mask representation of the query that was appended.

        """
        # Shape: (bs, 2, num_queries)
        S = torch.stack([A*M, M]).transpose(0, 1)
        # Mask representation of next query
        q = self.forward(S)  # Shape: (bs, num_queries)
        # Update Mask
        M = M + q
        # Update history
        S_and_q = torch.stack([A*M, M]).transpose(0, 1)

        return S_and_q, M, q

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True


def build_querier(config: dict) -> nn.Module:
    assert type(config['K']) is int
    querier = ShallowMLPQuerier(
        input_dim=config['K'],
        n_queries=config['K'],
        tau=config['arch']['tau']
    )
    return querier.cuda()
