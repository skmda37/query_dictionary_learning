from collections import namedtuple
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import seaborn as sns
import wandb

from torch.amp import autocast
import matplotlib.pyplot as plt

from querylearning.pipeline.dataset import get_img_dataset
from querylearning.utils.cliputils.encoders import ImageEncoderCLIP
from querylearning.utils.torchutils import unnormalize


QueryAnswerPair = namedtuple('QueryAnswerPair', ['query', 'answer'])


class VIPExplainer:

    def __init__(
        self,
        config: dict,
        Q: nn.Module,
        f: nn.Module,
        g: nn.Module
    ):
        self.config = config
        self.Q = Q
        self.f = f
        self.g = g

        if config['fp']:
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16

        path = Path('utils') / 'datasetutils' / 'classnames' \
                             / f"{config['dataset']}.txt"
        with open(path, 'r') as f:
            self.classnames = f.read().splitlines()

    def plot_examples(self, num_examples: int = 15) -> None:
        # Get random data for example explanations
        X, A, Y = self.get_random_examples(num_examples=num_examples)

        # Get IP trajectories
        p_trajectory, qa_trajectory, topk_classnames = self.get_IP_trajectory(
            A=A,
            Y=Y
        )

        # Plot IP explanations
        for i in range(num_examples):
            self.plot(
                img=X[i],
                ps=p_trajectory[i],
                qas=qa_trajectory[i],
                truelabel=self.classnames[Y[i].item()],
                ylabels=topk_classnames[i]
            )

    def get_random_examples(
        self,
        num_examples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_img_encoder = ImageEncoderCLIP(
            model_id=self.config['queries']['clip_model_type'],
            device='cuda'
        )
        preprocess = clip_img_encoder.preprocess

        valdata = get_img_dataset(
            datasetname=self.config['dataset'],
            dataroot=self.config['dataroot'],
            train=False,  # explanations are done on val data
            preprocess=preprocess,
        )

        rng = np.random.RandomState(42)
        indices = rng.choice(
            (len(valdata)),
            size=num_examples,
            replace=False
        ).tolist()
        sampled_data = [valdata[i] for i in indices]

        # Images
        X = torch.cat([d[0].unsqueeze(0) for d in sampled_data], dim=0).cuda()
        # Labels
        Y = torch.tensor([d[1] for d in sampled_data]).cuda()

        with torch.no_grad():
            with autocast(
                cache_enabled=False,
                device_type='cuda',
                dtype=self.dtype
            ):
                Z = clip_img_encoder(X, normout=True)  # Img embeddings
                A = self.Q(Z)  # Query answers for images X

        return X, A, Y

    def get_IP_trajectory(
        self,
        A: torch.Tensor,
        Y: torch.Tensor,
        max_num_q: int = 20,
        K: int = 10 
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        bs = Y.shape[0]
        device = torch.device('cuda')
        num_classes = self.f.n_classes
        p_trajectory, q_trajectory = [], []

        with torch.no_grad():
            M = torch.zeros((bs, A.shape[1]), device=device)
            for i in range(max_num_q):
                # Append next query q to query answers history
                S_and_q, M, q = self.g.append_next_query(A=A, M=M)
                # Make prediction with classifier for S_and_q
                logits = self.f(S_and_q)
                p = torch.softmax(logits, dim=1)
                p_trajectory.append(p.cpu())

                # Keep track of selected queries
                q_trajectory.append(q.argmax(dim=1))

        Q_as_text = self.Q.as_text
        qa_trajectory = [
            [
                QueryAnswerPair(
                    query=Q_as_text[q_trajectory[i][k]],
                    answer=int(A[k, q_trajectory[i][k]])
                ) for i in range(max_num_q)
            ] for k in range(bs)
        ]

        # Reduce probability trajectory to top-k predicted classes
        # Get indices of top-K class labels in prediction after observing
        # max_num_q queries
        topk_indices = p_trajectory[-1].topk(k=K, dim=1).indices
        p_trajectory = torch.tensor([
            [
                [
                    p[k, topk_indices[k, j]]
                    for j in range(K)
                ]
                for k in range(bs)
            ]
            for p in p_trajectory
        ])
        topk_classnames = [
            [self.classnames[topk_indices[i, j]] for j in range(K)]
            for i in range(bs)
        ]
        p_trajectory = p_trajectory.permute(1, 0, 2)

        return p_trajectory, qa_trajectory, topk_classnames

    def plot(
        self,
        img: torch.Tensor,
        ps: torch.Tensor,
        qas: List[QueryAnswerPair],
        truelabel: str,
        ylabels: List[str]
    ) -> None:
        num_q = len(qas)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        plt.subplot(1, 2, 1)
        xlabels = [qa.query for qa in qas]

        xticks = plt.xticks(ticks=range(num_q), labels=xlabels)
        colors = ['red' if qa.answer == 0 else 'green' for qa in qas]
        for xtick, color in zip(xticks[1], colors):
            xtick.set_color(color)

        M = ps.t()

        ax = sns.heatmap(
            M,
            cmap='RdPu',
            xticklabels=xlabels,
            yticklabels=ylabels,
            annot=False,
            fmt='.2%',
            cbar_kws={'label': '$P(Y|q_{1:k}(x))$'},
            vmin=0,
            vmax=1
        )

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=30,
            ha='right',
            fontsize=6
        )

        plt.subplot(1, 2, 2)
        img = unnormalize(img.cpu())
        plt.imshow(img.squeeze(0).permute(1, 2, 0))
        plt.title(truelabel)
        plt.axis('off')
        plt.tight_layout()

        wandb.log({'V-IP_Explanation': wandb.Image(plt)})
        plt.show()
        plt.close()
