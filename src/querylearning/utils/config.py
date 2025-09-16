import yaml
import argparse


def load_config(args: argparse.Namespace):
    config_file = args.config
    with open(config_file) as f:
        config = yaml.safe_load(f)
    if args.devid is not None:
        config['devid'] = args.devid
    return config