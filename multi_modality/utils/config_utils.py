import logging
import os
import sys
from os.path import dirname, join

from utils.config import Config
from utils.distributed import init_distributed_mode, is_main_process
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def setup_config(args=None):
    """Conbine yaml config and command line config with OmegaConf.
    Also converts types, e.g., `'None'` (str) --> `None` (None)
    """
    config = Config.get_config(args=args)
    if config.debug:
       config.wandb.enable = False
    return config


def setup_evaluate_config(config):
    """setup evaluation default settings, e.g., disable wandb"""
    assert config.evaluate
    config.wandb.enable = False
    if config.output_dir is None:
        config.output_dir = join(dirname(config.pretrained_path), "eval")
    return config


def setup_output_dir(output_dir, excludes=["code"]):
    """ensure not overwritting an exisiting/non-empty output dir"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
    else:
        existing_dirs_files = os.listdir(output_dir)  # list
        remaining = set(existing_dirs_files) - set(excludes)
        remaining = [e for e in remaining if "slurm" not in e]
        remaining = [e for e in remaining if ".out" not in e]
        # assert len(remaining) == 0, f"remaining dirs or files: {remaining}"
        logger.warn(f"remaining dirs or files: {remaining}")


def setup_main(args=None):
    """
    Setup config, logger, output_dir, etc.
    Shared for pretrain and all downstream tasks.
    """
    if args == None:
        config = setup_config()
        if hasattr(config, "evaluate") and config.evaluate:
            config = setup_evaluate_config(config)
        init_distributed_mode(config)

        if is_main_process():
            setup_output_dir(config.output_dir, excludes=["code"])
            setup_logger(output=config.output_dir, color=True, name="umt")
            logger.info(f"config: {Config.pretty_text(config)}")
            Config.dump(config, os.path.join(config.output_dir, "config.json"))
    else:
        config = setup_config(args)

        # additional update for ours
        config.device = args.device
        config.dir_name = args.dir_name
        if config.dir_name == 'ai2':
            config.pretrained = '/net/nfs3.prior/dongjook/pretrained_models/l16_5m.pth'
            config.model.pretrained = '/net/nfs3.prior/dongjook/pretrained_models/l16_5m.pth'


        if hasattr(config, "evaluate") and config.evaluate:
            config = setup_evaluate_config(config)
        init_distributed_mode(config)

        if is_main_process():
            setup_output_dir(config.output_dir, excludes=["code"])
            setup_logger(output=config.output_dir, color=True, name="umt")
            logger.info(f"config: {Config.pretty_text(config)}")
            Config.dump(config, os.path.join(config.output_dir, "config.json"))

    return config
