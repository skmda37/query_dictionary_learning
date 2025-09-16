import argparse
import os
from pathlib import Path

import torch

from querylearning.modelling.queries import build_query_dictionary
from querylearning.modelling.classifier import build_classifier
from querylearning.modelling.querier import build_querier
from querylearning.pipeline.trainer import AbstractTrainer, VIPTrainer
from querylearning.pipeline.trainer import AltQDLTrainer, JointQDLTrainer
from querylearning.utils.logging import setup_logger, setup_wandb, get_runid
from querylearning.utils.torchutils import setup_gpu, init_seeds
from querylearning.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["alt_qdl", "joint_qdl", "vip"], required=True,
        help="Optimization mode:"
             "'alt_qdl' (alternating query dictionary learning); "
             "'joint_qdl' (joint query dictionary learning); "
             "'vip' (classic V-IP with frozen query dictionary)"
    )
    parser.add_argument("--config", help='path to config file')
    parser.add_argument(
        "--devid",
        type=int,
        default=None,
        help='Cuda device id (overwrites device id in config file!)'
    )

    args = parser.parse_args()

    # Load mode-specific config file
    config = load_config(args)

    # Get a unique run_id
    config['run_id'] = get_runid()

    # Setup GPU device
    setup_gpu(config['devid'])

    # Init seeds
    init_seeds(config['seed'])

    # Setup wandb
    setup_wandb(config, args.mode)

    # Train model
    train(config=config, mode=args.mode)


def train(config: dict, mode: str) -> None:
    Q = build_query_dictionary(config, mode)
    f = build_classifier(config)
    g = build_querier(config)

    os.makedirs('logs', exist_ok=True)
    logger = setup_logger(Path('logs') / f'run{config["run_id"]}.log')

    # Train specified optimization strategy
    if mode == 'alt_qdl':
        Trainer = AltQDLTrainer
    elif mode == 'joint_qdl':
        Trainer = JointQDLTrainer
    elif mode == 'vip':
        Trainer = VIPTrainer

    trainer = Trainer(
        config=config,
        Q=Q,
        f=f,
        g=g,
        logger=logger
    )
    trainer.train()


if __name__ == '__main__':
    main()

