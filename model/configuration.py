import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from typing import List, Tuple, Union, Dict


class Config:
    def __init__(
        self,
            beta_range: Tuple[float, float] = (0.0, 1.0),
            beta_warmup_steps: int = 5000,
            nb_levels: int = 3,
            hidden_size: int = 2,
            encoder_kernel_sizes: List[int] = None,
            decoder_kernel_sizes: List[int] = None,
            nb_encoder_units: List[int] = None,
            nb_decoder_units: List[int] = None,
            encoder_strides: List[int] = None,
            decoder_strides: List[int] = None,
            initializer_range: float = 0.01,
            leaky_relu_alpha: float = 0.2,
            dropout: float = 0.0,
            layer_norm_eps: float = 1e-12,
            base_dir: str = 'Documents/MNIST',
    ):
        super(Config).__init__()

        self.beta_range = beta_range
        self.beta_warmup_steps = beta_warmup_steps
        self.nb_levels = nb_levels
        self.hidden_size = hidden_size

        if encoder_kernel_sizes is None:
            self.encoder_kernel_sizes = [2, 2, 2]
        else:
            self.encoder_kernel_sizes = encoder_kernel_sizes
        if decoder_kernel_sizes is None:
            self.decoder_kernel_sizes = [2, 3, 3, 2]
        else:
            self.decoder_kernel_sizes = decoder_kernel_sizes

        if nb_encoder_units is None:
            self.nb_encoder_units = [32, 64, 128]
        else:
            self.nb_encoder_units = nb_encoder_units
        if nb_decoder_units is None:
            self.nb_decoder_units = [256, 128, 64, 32, 1]
        else:
            self.nb_decoder_units = nb_decoder_units

        if encoder_strides is None:
            self.encoder_strides = [2, 2, 2]
        else:
            self.encoder_strides = encoder_strides
        if decoder_strides is None:
            self.decoder_strides = [2, 2, 2, 1]
        else:
            self.decoder_strides = decoder_strides

        assert self.nb_levels == \
               len(self.nb_encoder_units) == \
               len(self.encoder_kernel_sizes) == \
               len(self.encoder_strides) == \
               len(self.nb_decoder_units) - 2 == \
               len(self.decoder_kernel_sizes) - 1 == \
               len(self.decoder_strides) - 1

        self.initializer_range = initializer_range
        self.leaky_relu_alpha = leaky_relu_alpha

        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps

        # dir configs
        self.base_dir = pjoin(os.environ['HOME'], base_dir)


class TrainConfig:
    def __init__(
            self,
            optim_choice='adamax',
            lr=1e-2,
            betas=(0.9, 0.999),
            weight_decay: float = 0.0,
            warmup_steps: int = 1000,
            use_cuda: bool = True,
            log_freq: int = 10,
            chkpt_freq: int = 10,
            batch_size: int = 1024,
            runs_dir: str = 'Documents/MNIST/runs',
    ):
        super(TrainConfig).__init__()

        _allowed_optim_choices = ['lamb', 'adam', 'adam_with_warmup', 'adamax']
        assert optim_choice in _allowed_optim_choices, "Invalid optimzer choice, allowed options:\n{}".format(_allowed_optim_choices)

        self.optim_choice = optim_choice
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.use_cuda = use_cuda
        self.log_freq = log_freq
        self.chkpt_freq = chkpt_freq
        self.batch_size = batch_size
        self.runs_dir = pjoin(os.environ['HOME'], runs_dir)
