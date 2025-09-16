import torch
import torch.nn as nn
from querylearning.pipeline.dataset import NUM_CLASSES


class ShallowMLPClassifier(nn.Module):

    def __init__(self, input_dim: int, n_classes: int, p: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        # Architecture
        self.layer1 = nn.Linear(input_dim, 2000)
        self.layer2 = nn.Linear(2000, 500)
        self.norm1 = nn.LayerNorm(2000)
        self.norm2 = nn.LayerNorm(500)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.flat = nn.Flatten()
        # heads
        self.head = nn.Linear(1000, self.n_classes)

    def forward(self, history):
        x = self.relu(self.norm1(self.layer1(history)))
        x = self.dropout(x)
        x = self.relu(self.norm2(self.layer2(x)))
        x = self.flat(x)
        return self.head(x)


def build_classifier(config: dict) -> nn.Module:
    classifier = ShallowMLPClassifier(
        input_dim=config['K'],
        n_classes=NUM_CLASSES[config['dataset']],
        p=config['arch']['dropout']
    )
    return classifier.cuda()
