import yaml
from easydict import EasyDict as edict


def get_config(config, seed, run_name):
    config_dir = f'./config/{config}.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = seed
    config.ckpt = f'{run_name}'

    return config