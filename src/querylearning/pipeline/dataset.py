import os
from pathlib import Path
from collections import namedtuple
from collections.abc import Iterable
from typing import Callable, List, Union, Tuple
import time
from abc import ABCMeta, abstractmethod
import shlex
import subprocess

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as tf
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet

from querylearning.pipeline.bigfile_builder import BigFileBuilder
from querylearning.utils.cliputils.encoders import ImageEncoderCLIP
from querylearning.utils.timing import time_spent
from querylearning.utils.datasetutils.cub2011 import Cub2011
from querylearning.utils.datasetutils.subset_image_folder import SubsetImageFolder
from querylearning.utils.datasetutils.rival10 import RIVAL10


DATASETNAMES = [
    'cifar10', 'cifar100', 'rival10',
    'cub200', 'stanfordcars', 'imagenet100'
]


NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'debug': 10,
    'cub200': 200,
    'imagenette2': 10,
    'imagenet100': 100,
    'imagenet': 1000,
    'rival10': 10,
    'stanford_cars': 196
}


def get_split_dataset(
    datasetname: str,
    clip_model_id: str,
    dataroot: str,

) -> Tuple[Dataset, Dataset]:
    clip_img_encoder = ImageEncoderCLIP(
        model_id=clip_model_id,
        device='cuda'
    )
    preprocess = clip_img_encoder.preprocess

    traindata = TransformedDatasetInMemory(
        datasetname=datasetname,
        Xform=clip_img_encoder,
        train=True,
        dataroot=dataroot,
        preprocess=preprocess
    )
    valdata = TransformedDatasetInMemory(
        datasetname=datasetname,
        Xform=clip_img_encoder,
        train=False,
        dataroot=dataroot,
        preprocess=preprocess
    )
    return traindata, valdata


def get_dataloaders(
    batch_size: int,
    num_workers: int,
    traindata: Dataset,
    valdata: Dataset
) -> Tuple[DataLoader, DataLoader]:
    trainloader = DataLoader(
        traindata,
        batch_size=batch_size,
        num_workers=num_workers
    )
    valloader = DataLoader(
        valdata,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return trainloader, valloader


def get_img_dataset(
    datasetname: str,
    train: bool,
    dataroot: str,
    preprocess: tf.transforms
) -> Iterable:
    if datasetname == 'toy':
        return [
            (torch.randn(3, 224, 224), torch.randint(0, 10, size=(1,)))
            for _ in range(10)
        ]
    elif datasetname == 'cifar10':
        return CIFAR10(
            root=dataroot,
            train=train,
            download=True,
            transform=preprocess
        )
    elif datasetname == 'cifar100':
        return CIFAR100(
            root=dataroot,
            train=train,
            download=True,
            transform=preprocess
        )
    elif datasetname == 'rival10':
        if not (Path(dataroot) / 'RIVAL10').exists():
            command_line = "curl -L 'https://app.box.com/index.php?" \
                           "rm=box_download_shared_file&shared_name=" \
                           "iflviwl5rbdgtur1rru3t8f7v2vp0gww&file_id=" \
                           f"f_944375052992' -o rival10.zip"
            subprocess.run(shlex.split(command_line))
            command_line = f"unzip -d {dataroot} rival10.zip"
            subprocess.run(shlex.split(command_line))
            command_line = f"rm rival10.zip"
            subprocess.run(shlex.split(command_line))
        imagenet_root = os.path.join(dataroot, 'imagenet')
        if not Path(imagenet_root).exists():
            raise FileNotFoundError(
                f'Imagenet root {imagenet_root} does not exist'
            )
        return RIVAL10(
            root_data=dataroot,
            imagenet_root=imagenet_root,
            transform=preprocess
        )
    elif datasetname == 'cub200':
        # Potential issue downl. cub200 due to change in GDrives API
        return Cub2011(
            root=dataroot,
            train=train,
            download=True,
            transform=preprocess
        )
    elif datasetname == 'stanfordcars':
        return torchvision.datasets.StanfordCars(
            root=dataroot,
            download=True,
            split='train' if train else 'test',
            transform=preprocess
        )
    elif datasetname == 'imagenet100':
        classes_file = Path('utils') / 'datasetutils' \
                       / 'imagenet100_classes.txt'
        return SubsetImageFolder(
            root=Path(dataroot) / 'imagenet' / ('train' if train else 'val'),
            transform=preprocess,
            classes_file=classes_file
        )
    else:
        raise NotImplementedError(
            f'Dataset {datasetname} not supported! We support '
            + (', '.join(DATASETNAMES))
        )


class AbstractTransformedDataset(Dataset, metaclass=ABCMeta):

    def __init__(
        self,
        datasetname: str,
        Xform: Callable,
        train: bool,
        dataroot: str,
        preprocess: tf.transforms
    ):
        super().__init__()
        self.datasetname = datasetname
        self.Xform = Xform  # transform/embedder such as CLIP encoder
        self.train = train
        self.dataroot = dataroot
        self.preprocess = preprocess

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: pass


class TransformedDatasetInMemory(AbstractTransformedDataset):
    """
    Takes an image dataset and computes the Xform (a transform
    such as the CLIP embedding) for each entry and saves it in VRAM
    (if using GPU) or RAM else.
    """

    def __init__(
        self,
        datasetname: str,
        Xform: Callable,
        train: bool,
        dataroot: str,
        preprocess: tf.transforms
    ):
        super().__init__(
            datasetname=datasetname,
            Xform=Xform,
            train=train,
            dataroot=dataroot,
            preprocess=preprocess
        )
        dirpath = Path(dataroot) / 'imgembeddings' \
                                 / datasetname \
                                 / ('train' if train else 'val') \
                                 / str(Xform)
        if dirpath.exists():
            print(f'Loading Xformed dataset from {dirpath / "Xformed.pt"}...')
            self.Xformed = torch.load(dirpath / 'Xformed.pt')
            self.labels = torch.load(dirpath / 'labels.pt')
        else:
            self.Xformed, self.labels = self.compute_Xformed()
            os.makedirs(dirpath, exist_ok=True)
            torch.save(self.Xformed, dirpath / 'Xformed.pt')
            torch.save(self.labels, dirpath / 'labels.pt')

    def __len__(self) -> int: return len(self.labels)

    def compute_Xformed(
        self,
        batch_size: int = 32,
    ) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = get_img_dataset(
            self.datasetname,
            train=self.train,
            dataroot=self.dataroot,
            preprocess=self.preprocess
        )
        Xformed, labels = [], []
        dataloader = DataLoader(dataset, batch_size=batch_size)
        desc = f'Precomputing Xformed Dataset with {self.Xform}'
        with tqdm(total=len(dataloader), colour='blue', desc=desc) as pbar:
            for i, (x, y) in enumerate(dataloader):
                x = x.to(device)
                with torch.no_grad():
                    # Get Xform
                    Xformed.append(
                        self.Xform(x).cpu()
                    )
                # Get label
                labels.append(y)
                pbar.update(1)
        Xformed = torch.cat(Xformed, dim=0)
        labels = torch.cat(labels, dim=0)
        return Xformed, labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.Xformed[idx], self.labels[idx]


class TransformedDatasetMultiFiles(AbstractTransformedDataset):
    """
    Takes an image dataset and computes Xform (transform such as CLIP
    embedding) for each entry and saves it to a .npy file. The paths
    to each file is saved in __init__(). In __getitem__() we load the file
    corresponding to passed index. NOTE this is very inefficient because
    you have to open and close a file for each entry in the dataset.
    """

    def __init__(
        self,
        datasetname: str,
        Xform: Callable,
        dirout: str,
        train: bool,
        dataroot: str,
        preprocess: tf.transforms
    ):
        super().__init__(
            datasetname=datasetname,
            Xform=Xform,
            train=train,
            dataroot=dataroot,
            preprocess=preprocess
        )

        self.dirout = dirout

        # Dataset directory
        os.makedirs(dirout, exist_ok=True)

        # Save paths of files in list
        self.files = []

        # Save labels
        self.labels = []

        # Write clip embeddings of images to files in dirout
        self.write_Xform()

    def write_Xform(self) -> None:
        print('Getting image dataset...')
        dataset = get_img_dataset(
            self.datasetname,
            train=self.train,
            dataroot=dataroot,
            preprocess=self.preprocess
        )
        desc = f'Precomputing Xformed Dataset with {self.Xform}'
        with tqdm(total=len(dataset), colour='blue', desc=desc) as pbar:
            for i, (x, y) in enumerate(dataset):
                print(f'\rPrecomputing Xform {self.Xform} {i}', end='')
                # Get Xform
                with torch.no_grad():
                    z = self.Xform(x.unsqueeze(0)).squeeze(0).cpu().numpy()
                # Save Xform to file
                path = Path(self.dirout) / f'xformed{i}.npy'
                np.save(path, z)
                self.files.append(path)
                # Save labels in list
                self.labels.append(y)
                pbar.update(1)
        print(f'Saved Xform in {self.dirout}')

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.load(self.files[idx]))
        y = torch.tensor(self.labels[idx])
        return x, y

