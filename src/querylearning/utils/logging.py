from datetime import datetime
import logging
import uuid
from typing import Union
from pathlib import Path

import wandb


def setup_logger(logfilename: Union[str, Path]):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    formatter = logging.Formatter(
        '%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )

    file_handler = logging.FileHandler(logfilename, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def setup_wandb(config: dict, mode: str):
    if config['wandb']:
        tags = [mode, config['dataset']]
        wandb.init(project="clip_queries_learned", config=config, tags=tags)
        wandb.run.summary['run_id'] = config['run_id']


def get_runid():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    run_id = f"{timestamp}_{unique_id}"
    return run_id