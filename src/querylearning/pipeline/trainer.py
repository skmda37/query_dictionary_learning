from typing import List, Union, Dict
import argparse
from abc import ABCMeta, abstractmethod
import logging
from pathlib import Path
import os

import wandb
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp import GradScaler
from torch.amp import autocast

from querylearning.pipeline.dataset import get_split_dataset, get_dataloaders
from querylearning.pipeline.explainer import VIPExplainer


class AbstractTrainer(metaclass=ABCMeta):

    def __init__(
        self,
        config: dict,
        Q: nn.Module,
        f: nn.Module,
        g: nn.Module,
        logger: logging.Logger
    ) -> None:
        """ Base class for trainer V-IP
            (for fixed or learnable query dictionary)
        args:
            config: contains training configuration
            Q: V-IP query dictionary
            f: V-IP classifier network
            g: V-IP querier network
        """
        self.config = config
        self.Q = Q
        self.f = f
        self.g = g
        self.logger = logger

        # Number of queries in query dictionary
        self.K = self.Q.K
        # Best validation metric (AUC)
        self.best = 0.0

        # Get train and val data of (clipembeddings, labels)
        traindata, valdata = get_split_dataset(
            datasetname=config['dataset'],
            clip_model_id=config['queries']['clip_model_type'],
            dataroot=config['dataroot']
        )

        # Get dataloaders
        self.trainloader, self.valloader = get_dataloaders(
            batch_size=config['train']['batch_size'],
            num_workers=config['num_workers'],
            traindata=traindata,
            valdata=valdata
        )
        self.num_batches = len(self.trainloader)

        # Make directory to log checkpoint
        self.ckpt_dir = (Path(self.config['save_dir'])
                         / 'runs'
                         / self.config['run_id'])
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.scaler = GradScaler() if not self.config['fp'] else None

        if config['fp']:
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16

        # To visualize example explanations post-training
        self.explainer = VIPExplainer(config=config, Q=Q, f=f, g=g)

    def train(self) -> None:
        """Trains model"""
        self._setup_optim()

        for sampling_strategy in ['random', 'biased']:
            self._train_phase = f'{self._method_name} with ' \
                                f'{sampling_strategy} Sampling'
            self._sampling_strategy = sampling_strategy
            with tqdm(
                total=self.total_epochs,
                colour='red',
                desc=self._train_phase
            ) as pbar:
                self._pbar = pbar
                for epoch in range(self.total_epochs):
                    self._train_epoch(epoch=epoch)

        self._visualize()

    def _train_epoch(self, epoch: int) -> None:
        """Trains model for a single epoch
        args:
            epoch (int): current epoch
        """
        for batch_idx, (source, target) in enumerate(self.trainloader):
            source, target = source.cuda(), target.cuda()

            # Perform training step on batch
            stats = self._train_step(source=source, target=target)

            # Log training stats
            self._log_train(
                epoch=epoch,
                batch_idx=batch_idx,
                stats=stats
            )

        self._pbar.update(1)

        # Validate model
        if self._is_val_epoch(epoch=epoch):
            self._validate(epoch=epoch)

    @property
    @abstractmethod
    def _method_name(self) -> str: pass

    @abstractmethod
    def _setup_optim(self) -> None: pass

    @abstractmethod
    def _train_step(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pass

    def _vip_objective(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        optim: torch.optim,
    ) -> torch.Tensor:
        """Single optimization step on V-IP objective
        args:
            source (torch.Tensor): batch of image embeddings
            target (torch.Tensor): batch of class labels
            optim (torch.optim): optimizer
        """
        optim.zero_grad()

        with autocast(
            cache_enabled=False,
            device_type='cuda',
            dtype=self.dtype
        ):
            A = self.Q(source)  # Get query answers

            """Sample query-answer history S"""
            low = self.config['train']['sampling']['low']
            high = min(self.config['train']['sampling']['high'], self.K)
            # Shape: (bs, num_queries)
            M = self._sample_mask(A=A, low=low, high=high)
            # Append next query to history of query answers pairs
            S_and_q, _, _ = self.g.append_next_query(A=A, M=M)
            # Predict logits with classifier network f for input S_and_q
            logits = self.f(S_and_q)
            loss = F.cross_entropy(logits, target)

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(optim)
            self.scaler.update()
        else:
            loss.backward()
            optim.step()

        self._pbar.set_postfix_str(f'loss: {loss.item():.4f}')

        return loss

    @property
    def total_epochs(self) -> int:
        return self.config['train'][self._sampling_strategy]['epochs']

    def _is_val_epoch(self, epoch: int) -> bool:
        c = epoch % self.config['val_every'] == 0
        c |= epoch == self.total_epochs - 1
        return c  # condition for validation

    def _log_train(
        self,
        epoch: int,
        batch_idx: int,
        stats: dict,
    ) -> None:
        """Logs training progress in console and wandb"""
        if batch_idx % self.config['log_every'] == 0:
            self.logger.info(
                f"Training {self._train_phase} | Epoch {epoch} "
                f"| Batch {batch_idx}/{self.num_batches}"
                ' | '.join(
                    [
                        f'{key.capitalize()} {val}'
                        for key, val in stats.items()
                    ]
                )
            )
            if self.config['wandb']:
                wandb.log({'epoch': epoch, **stats})

    def _sample_mask(
        self,
        A: torch.Tensor,
        low: int,
        high: int
    ) -> torch.Tensor:
        """Samples a mask representing the V-IP query-answer history S
        args:
            A: torch.Tensor of shape (bs, num_queries) representing
                all query-answers
            low: lowest number of query-answers to be sampled
            high: highest number of query-answers to be sampled

        return: torch.Tensor of shape (bs, num_queries)
        """
        if self._sampling_strategy == 'random':
            M = self._random_mask_sampling(A=A, low=low, high=high)
        elif self._sampling_strategy == 'biased':
            M = self._biased_mask_sampling(A=A, low=low, high=high)
        else:
            raise ValueError(f'Sampling {self._sampling_strategy} not valid.')

        return M

    def _random_mask_sampling(
        self,
        A: torch.Tensor,
        low: int,
        high: int
    ) -> torch.Tensor:
        """
        Uses random sampling to generate a mask of sampled query-answer pairs
        """
        bs = A.shape[0]
        device = A.device

        # Sample number of queries per batch
        num_q = torch.randint(low=low, high=high, size=(bs,), device=device)

        # Create helper tensor of shape (bs, self.K) for indexing;
        arange_K = torch.arange(self.K, device=device).expand(bs, -1)

        # Create a random permutation per batch
        perm = torch.argsort(torch.rand(bs, self.K, device=device), dim=1)

        mask = (arange_K < num_q.unsqueeze(1)).float()
        return torch.gather(mask, 1, perm)

    def _biased_mask_sampling(
        self,
        A: torch.Tensor,
        low: int,
        high: int
    ) -> torch.Tensor:
        """
        Uses biased sampling (built up with V-IP querier) to generate a mask
        """
        bs = A.shape[0]
        device = A.device

        M = torch.zeros((bs, self.K), device=device)  # init mask
        M_final = torch.zeros((bs, self.K), device=device)

        # Sample number of queries per batch
        num_q = torch.randint(low=low, high=high, size=(bs,))
        sorted_indices = num_q.argsort()
        counter = 0

        with torch.no_grad():
            # Build up mask sequentially
            for i in range(high):
                while (counter < bs):
                    batch_idx = sorted_indices[counter]
                    if i == num_q[batch_idx]:
                        M_final[batch_idx] = M[batch_idx]
                        counter += 1
                    else:
                        break
                if counter == bs:
                    break

                # Append next query with querier to update mask
                _, M, _ = self.g.append_next_query(A=A, M=M)

        return M_final

    def _validate(self, epoch: int) -> None:
        """
        V-IP validation loop, which computes and logs
            (a) acc for variable query budget
            (b) acc for fixed query budget
        """
        device = None

        num_q = min(self.config['train']['sampling']['high'], self.K)

        # Num. correctly classified val samples after running
        # IP algorithm with variable query budget (termination
        # after posterior entropy reaches threshold)
        correct_var_budget = 0
        # Num. correctly classified val samples after running
        # IP algorithm with fixed budget
        correct_fixed_budget = [0] * num_q

        total_count = 0
        num_needed_q_list = []

        with torch.no_grad():

            for source, target in self.valloader:
                with autocast(
                    cache_enabled=False,
                    device_type='cuda',
                    dtype=self.dtype
                ):
                    source, target = source.cuda(), target.cuda()

                    bs = target.shape[0]  # Get batch size
                    A = self.Q(source)  # Get query answers

                    # Initialize mask to keep track of selected queries
                    M = torch.zeros_like(A)
                    logits_list = []
                    loss_list = []

                    for i in range(num_q):
                        # Append one query to query answer history
                        S_and_q, M, _ = self.g.append_next_query(A, M)
                        # Get prediction of classifier for history S
                        logits = self.f(S_and_q)
                        logits_list.append(logits)
                        # Update num. correct predictions after observing
                        # the first i queries selected by querier
                        correct_fixed_budget[i] += (
                            logits.argmax(dim=1) == target
                        ).float().sum().item()

                    # Shape: (bs, num_q, num_classes)
                    logits = torch.stack(logits_list).permute(1, 0, 2)

                    num_needed_q = self._get_num_needed_queries(
                        logits=logits,
                        threshold=self.config['arch']['posterior_threshold']
                    )
                    num_needed_q_list.append(num_needed_q)

                    Y_hat = logits[
                        torch.arange(bs), num_needed_q - 1
                    ].argmax(dim=1)
                    correct_var_budget += (
                        Y_hat == target
                    ).float().sum().item()
                    total_count += bs

            # Average num. of needed queries is semantic entropy
            semantic_entropy = torch.cat(
                num_needed_q_list
            ).float().mean().item()
            acc_var_budget = correct_var_budget / total_count

            acc_fixed_budget = np.array(correct_fixed_budget) / total_count
            auc_fixed_budget = acc_fixed_budget.mean()

            if self.config['wandb']:
                x_vals = np.arange(1, num_q).tolist()
                y_vals = acc_fixed_budget
                data = [[x, y] for (x, y) in zip(x_vals, y_vals)]
                table = wandb.Table(data=data, columns=['num_q', 'acc'])
                wandb.log(
                    {
                        'val_acc': wandb.plot.line(
                            table,
                            'num_q',
                            'acc',
                            title='val acc'
                        ),
                        'epoch': epoch,
                        'semantic entropy': semantic_entropy,
                        'acc_var_budget': acc_var_budget,
                        'auc_fixed_budget': auc_fixed_budget
                    }
                )
                if self.best < auc_fixed_budget:
                    wandb.log(
                        {
                            'best_auc': wandb.plot.line(
                                table,
                                'num_q',
                                'acc',
                                title='best auc'
                            )
                        }
                    )

            if self.best < auc_fixed_budget:
                self.best = auc_fixed_budget
                self._save_checkpoint()

    def _get_num_needed_queries(
        self,
        logits: torch.Tensor,
        threshold: float = 0.85
    ) -> torch.Tensor:
        """Computes number of needed queries
        args:
            logits: torch.Tensor of shape (bs, num_q, num_classes)
            threshold: probability threshold between 0 and 1
        """
        num_q = logits.shape[1]
        device = logits.device
        probs = nn.functional.softmax(logits, dim=2)
        max_probs = probs.amax(dim=2)
        # `decay` is multiplied such that argmax finds
        #  the first nonzero that is above threshold.
        threshold_indicator = (max_probs >= threshold).float().to(device)
        decay = torch.linspace(10, 1, num_q).unsqueeze(0).to(device)
        # We add one since python is 0-indexed
        num_needed_q = (threshold_indicator * decay).argmax(1) + 1
        # If threshold is never reached, then num_needed_q is num_q
        num_needed_q[threshold_indicator.sum(1) == 0] = num_q

        return num_needed_q

    def _save_checkpoint(self) -> None:
        ckpt = {
            'querier': self.g.state_dict(),
            'classifier': self.f.state_dict()
        }
        torch.save(ckpt, self.ckpt_dir / 'ckpt.pth')

    def _visualize(self) -> None:
        """Logs post-training visualizations to wandb"""
        if self.config['wandb']:
            # Plot V-IP explanation examples
            self.explainer.plot_examples()


class JointQDLTrainer(AbstractTrainer):
    """Trainer for Joint Optimization for Query Dictionary Learning"""

    def __init__(
        self,
        config: dict,
        Q: nn.Module,
        f: nn.Module,
        g: nn.Module,
        logger: logging.Logger
    ):
        super().__init__(config=config, Q=Q, f=f, g=g, logger=logger)

    @property
    def _method_name(self) -> str: return 'JointQDL'

    def _setup_optim(self) -> None:
        params = (list(self.f.parameters())
                  + list(self.g.parameters())
                  + list(self.Q.parameters()))
        self.optim = torch.optim.Adam(params, lr=self.config['train']['lr'])

    def _train_step(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """SIng training step of joint optimization of
        query dictionary and V-IP networks
        args:
            source (torch.Tensor): batch of CLIP embeddings of training images
            target (torch.Tensor): batch of training labels
        """
        loss = self._vip_objective(
            source=source,
            target=target,
            optim=self.optim
        )
        return {'loss': loss}


class AltQDLTrainer(AbstractTrainer):
    """Trainer for Alternating Optimization for Query Dictionary Learning"""

    def __init__(
        self,
        config: dict,
        Q: nn.Module,
        f: nn.Module,
        g: nn.Module,
        logger: logging.Logger
    ):
        super().__init__(config=config, Q=Q, f=f, g=g, logger=logger)

    @property
    def _method_name(self) -> str: return 'AltQDL'

    def _setup_optim(self) -> None:
        """
        For alternating optimization we define two optimizers:
            (1) an optimizer for the V-IP networks f and g
            (2) an optimizer for the query dictionary Q
        """
        lr = self.config['train']['lr']  # Learning rate
        f_and_g_params = list(self.f.parameters()) + list(self.g.parameters())

        self.optim_f_and_g = torch.optim.Adam(f_and_g_params, lr=lr)
        self.optim_Q = torch.optim.Adam(self.Q.parameters(), lr=lr)

    def _train_step(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Single training step of alternating optimization
        for query dictionary and V-IP networks
        args:
            source (torch.Tensor): batch of CLIP embeddings of training images
            target (torch.Tensor): batch of training labels
        """

        # Single query dictionary update
        loss_Q = self._vip_objective(
            source=source,
            target=target,
            optim=self.optim_Q
        )

        # t V-IP network (f and g) updates
        loss_f_and_g = 0.0
        for _ in range(self.config['train']['t']):
            loss_f_and_g += self._vip_objective(
                source=source,
                target=target,
                optim=self.optim_f_and_g
            )
        loss_f_and_g /= self.config['train']['t']

        return {'loss_Q': loss_Q, 'loss_f_and_g': loss_f_and_g}


class VIPTrainer(AbstractTrainer):
    """Trainer for Classic V-IP"""

    def __init__(
        self,
        config: dict,
        Q: nn.Module,
        f: nn.Module,
        g: nn.Module,
        logger: logging.Logger
    ):
        super().__init__(config=config, Q=Q, f=f, g=g, logger=logger)

    @property
    def _method_name(self) -> str: return 'V-IP'

    def _setup_optim(self) -> None:
        """
        In standard V-IP we optimizer classifier (f) and querier (g)
        network parameters
        """
        params = list(self.f.parameters()) + list(self.g.parameters())
        self.optim = torch.optim.Adam(params, lr=self.config['train']['lr'])

    def _train_step(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Single training step for V-IP networks
        args:
            source (torch.Tensor): batch of CLIP embeddings of training images
            target (torch.Tensor): batch of training labels
        """
        loss = self._vip_objective(
            source=source,
            target=target,
            optim=self.optim,
        )
        return {'loss': loss}
