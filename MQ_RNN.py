import os
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from decoder import QuantileDecoder, ContextDecoder
from encoder import Encoder


class MQ_RNN(pl.LightningModule):
    def __init__(self,
                 hist_len: int = 168,
                 horizon_size: int = 24,
                 input_size: int = 4,
                 num_layers: int = 1,
                 hidden_units: int = 8,
                 covariate_size: int = 0,
                 quantiles: List[float] = (0.5,),
                 context_size: int = 1,
                 lr=1e-3,
                 ):
        super().__init__()

        self.hist_len = hist_len
        self.horizon_size = horizon_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.covariate_size = covariate_size

        if isinstance(quantiles, (tuple, list)):
            quantiles = np.array(quantiles)

        self.quantiles = quantiles
        self.number_of_quantile = len(quantiles)
        self.context_size = context_size
        self.lr = lr

        self.encoder = Encoder(hist_len=hist_len,
                               horizon_size=horizon_size,
                               input_size=input_size,
                               num_layers=num_layers,
                               hidden_units=hidden_units)

        self.context_decoder = ContextDecoder(hidden_units=hidden_units,
                                              covariate_size=covariate_size,
                                              horizon_size=horizon_size,
                                              context_size=context_size)

        self.quantile_decoder = QuantileDecoder(covariate_size=covariate_size,
                                                number_of_quantile=self.number_of_quantile,
                                                context_size=context_size)

    def forward(self, x):
        encoder_output = self.encoder(x)
        hidden_state = encoder_output[1][0]

        context = self.context_decoder(hidden_state)
        # context = context.view(context.shape[0], self.horizon_size + 1, self.context_size)
        global_context = context[:, :self.context_size]
        local_contexts = context[:, self.context_size:]

        quantiles_output = []
        # import IPython;IPython.embed()
        for horizon in range(self.horizon_size):
            start = horizon * self.context_size
            end = start + self.context_size
            quantile_decoder_input = torch.cat([global_context, local_contexts[:, start:end]], dim=1)
            quantile_output = self.quantile_decoder(quantile_decoder_input)

            quantiles_output.append(quantile_output)

        output = torch.cat(quantiles_output, -1)

        if self.number_of_quantile == 1:
            output = output.unsqueeze(-1)

        return output

    def training_step(self, batch, batch_idx):
        combined_input = torch.cat(batch, dim=1)
        quantiles = self(combined_input[:, :self.hist_len, :])
        loss = self.loss(quantiles, combined_input[:, self.hist_len:, [0]])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        combined_input = torch.cat(batch, dim=1)

        quantiles = self(combined_input[:, :self.hist_len, :])

        loss = self.loss(quantiles, combined_input[:, self.hist_len:, [0]])
        self.log('val_logprob', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss(self, quantiles_output, y):
        quantiles = torch.tensor(self.quantiles, dtype=quantiles_output.dtype, requires_grad=False).to(quantiles_output.device)
        # Distribution with generated `mu` and `sigma`

        e = y - quantiles_output
        loss = torch.max(quantiles * e, (quantiles - 1) * e)

        return torch.mean(loss)


