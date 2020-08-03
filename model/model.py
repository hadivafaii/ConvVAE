import os
from datetime import datetime
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch.optim import Adam

from .model_utils import print_num_params


class VAE(nn.Module):
    def __init__(self, config, verbose=True):
        super(VAE, self).__init__()

        self.config = config

        self.encoder = ConvEncoder(config, verbose=verbose)
        self.decoder = ConvDecoder(config, verbose=verbose)
        # self.encoder = FFEncoder(config, verbose=verbose)
        # self.decoder = FFDecoder(config, verbose=verbose)

        self.recon_loss_fn = nn.MSELoss(reduction="sum")
        self.init_weights()
        if verbose:
            print_num_params(self)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder(z)

        return z, mu, logvar, x_recon

    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(*mu.size()).to(mu.device)
        z = mu + std * eps
        return z

    def compute_loss(self, mu, logvar, x_recon, x_true):
        recon_term = self.recon_loss_fn(x_recon, x_true)
        kl_term = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())

        loss_dict = {
            "kl": kl_term,
            "recon": recon_term,
            "tot": recon_term + self.config.beta * kl_term,
        }
        return loss_dict

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class FFEncoder(nn.Module):
    def __init__(self, config, verbose=True):
        super(FFEncoder, self).__init__()

        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)

        self.fc1 = nn.Linear(256, config.hidden_size, bias=True)
        self.fc2 = nn.Linear(256, config.hidden_size, bias=True)

        self.activation = nn.LeakyReLU(config.leaky_relu_alpha)
        if verbose:
            print_num_params(self)

    def forward(self, x):
        x = x.flatten(start_dim=1)  # N x 784
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))

        mu = self.fc1(x)
        logvar = self.fc2(x)

        return mu, logvar


class FFDecoder(nn.Module):
    def __init__(self, config, verbose=True):
        super(FFDecoder, self).__init__()

        self.linear1 = nn.Linear(config.hidden_size, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 784)

        self.activation = nn.LeakyReLU(config.leaky_relu_alpha)
        if verbose:
            print_num_params(self)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)

        return x.view(-1, 1, 28, 28)


class ConvEncoder(nn.Module):
    def __init__(self, config, verbose=True):
        super(ConvEncoder, self).__init__()

        nb_conv_units = [1] + config.nb_encoder_units

        layers = []
        for i in range(config.nb_levels):
            layers.extend([
                nn.Conv2d(
                    in_channels=nb_conv_units[i],
                    out_channels=nb_conv_units[i + 1],
                    kernel_size=config.encoder_kernel_sizes[i],
                    stride=config.encoder_strides[i],
                    padding=int(np.ceil(config.encoder_kernel_sizes[i] / 2)) - 1,
                    bias=True),
                nn.LeakyReLU(negative_slope=config.leaky_relu_alpha),
                nn.Dropout(config.dropout),
            ])
        self.net = nn.Sequential(*layers)

        self.linear = nn.Linear(config.nb_encoder_units[-1] * 3 * 3, 32, bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=config.leaky_relu_alpha)

        self.fc1 = nn.Linear(32, config.hidden_size, bias=True)
        self.fc2 = nn.Linear(32, config.hidden_size, bias=True)

        if verbose:
            print_num_params(self)

    def forward(self, x):
        x = self.net(x)
        x = self.linear(x.flatten(start_dim=1))
        x = self.leaky_relu(x)

        mu = self.fc1(x)
        logvar = self.fc2(x)

        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, config, verbose=True):
        super(ConvDecoder, self).__init__()

        self.linear = nn.Linear(config.hidden_size, config.nb_decoder_units[0] * 3 * 3, bias=True)

        layers = []
        for i in range(1, config.nb_levels + 2):
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels=config.nb_decoder_units[i - 1],
                    out_channels=config.nb_decoder_units[i],
                    kernel_size=config.decoder_kernel_sizes[i - 1],
                    stride=config.decoder_strides[i - 1],
                    bias=False,),
                nn.LeakyReLU(negative_slope=config.leaky_relu_alpha),
                nn.Dropout(config.dropout),
            ])
        self.net = nn.Sequential(*layers)

        if verbose:
            print_num_params(self)

    def forward(self, x):

        x = self.linear(x)
        x = x.view(-1, self.linear.out_features // 9, 3, 3)
        x = self.net(x)

        return x
