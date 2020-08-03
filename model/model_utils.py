import os
import yaml
from datetime import datetime
from copy import deepcopy as dc
from prettytable import PrettyTable
from os.path import join as pjoin
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .configuration import Config


def save_model(model, prefix=None, comment=None):
    config_dict = vars(model.config)
    to_hash_dict_ = dc(config_dict)
    hashed_info = str(hash(frozenset(sorted(to_hash_dict_))))

    if prefix is None:
        prefix = 'chkpt:0'

    save_dir = pjoin(
        model.config.base_dir,
        'saved_models',
        "[{}_{:s}]".format(comment, hashed_info),
        "{}_{:s}".format(prefix, datetime.now().strftime("[%Y_%m_%d_%H:%M]")))

    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), pjoin(save_dir, 'model.bin'))

    with open(pjoin(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f)


def load_model(keyword, chkpt_id=-1, config=None, verbose=True):
    from .model import VAE

    # if load_dir is None:
    _dir = pjoin(os.environ['HOME'], 'Documents/PROJECTS/MT_LFP',  'saved_models')
    available_models = os.listdir(_dir)
    if verbose:
        print('Available models to load:\n', available_models)

    match_found = False
    model_id = -1
    for i, model_name in enumerate(available_models):
        if keyword in model_name:
            model_id = i
            match_found = True
            break

    if not match_found:
        raise RuntimeError("no match found for keyword")

    model_dir = pjoin(_dir, available_models[model_id])
    available_chkpts = os.listdir(model_dir)
    if verbose:
        print('\nAvailable chkpts to load:\n', available_chkpts)
    load_dir = pjoin(model_dir, available_chkpts[chkpt_id])

    if verbose:
        print('\nLoading from:\n{}\n'.format(load_dir))

    if config is None:
        with open(pjoin(load_dir, 'config.yaml'), 'r') as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        config = Config(**config_dict)

    loaded_model = VAE(config, verbose=verbose)
    loaded_model.load_state_dict(torch.load(pjoin(load_dir, 'model.bin')))

    chkpt = load_dir.split("/")[-1].split("_")[0]
    model_name = load_dir.split("/")[-2]
    metadata = {"chkpt": chkpt, "model_name": model_name}

    return loaded_model, metadata


def print_num_params(module: nn.Module):
    t = PrettyTable(['Module Name', 'Num Params'])

    for name, m in module.named_modules():
        total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        if '.' not in name:
            if isinstance(m, type(module)):
                t.add_row(["Total", "{}".format(total_params)])
                t.add_row(['---', '---'])
            else:
                t.add_row([name, "{}".format(total_params)])
    print(t, '\n\n')
