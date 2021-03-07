from typing import List

import numpy as np
import torch
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
                 covariate_size: int = 3,
                 quantiles: List[float] = (0.5, 0.8),
                 context_size: int = 1,
                 lr=1e-3,
                 ):
        super().__init__()
        assert input_size == 1 + covariate_size

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

        self.encoder = Encoder(input_size=input_size,
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
        history_data = x[:, :self.hist_len, :]
        encoder_output = self.encoder(history_data)
        outputs = []

        # Forking sequence
        for i in range(self.hist_len):
            hidden_state = encoder_output[0][:, i, :]
            future_covariates = x[:, i + 1:i + 1 + self.horizon_size, 1:]
            future_covariates_flatten = future_covariates.reshape(future_covariates.shape[0],
                                                                  self.horizon_size * self.covariate_size)
            context_decoder_input = torch.cat([hidden_state,
                                               future_covariates_flatten], dim=-1)
            context = self.context_decoder(context_decoder_input)

            global_context = context[:, :self.context_size]
            local_contexts = context[:, self.context_size:]

            quantiles_output = []
            # Compute quantile for each horizon
            for horizon in range(self.horizon_size):
                start = horizon * self.context_size
                end = start + self.context_size
                covariates = future_covariates_flatten[:, horizon*self.covariate_size: (horizon+1) * self.covariate_size]
                quantile_decoder_input = torch.cat([global_context, local_contexts[:, start:end], covariates], dim=1)
                quantile_output = self.quantile_decoder(quantile_decoder_input)

                quantiles_output.append(quantile_output)

            output = torch.cat(quantiles_output, -1)
            if self.number_of_quantile == 1:
                output = output.unsqueeze(-1)

            output = output.view(x.shape[0], self.horizon_size, self.number_of_quantile)

            outputs.append(output)

        return torch.stack(outputs, dim=1)

    def training_step(self, batch, batch_idx):
        combined_input = torch.cat(batch, dim=1)
        quantiles = self(combined_input)
        loss = self.loss(quantiles, combined_input[:, :, [0]])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        combined_input = torch.cat(batch, dim=1)

        quantiles = self(combined_input)

        loss = self.loss(quantiles, combined_input[:, :, [0]])
        self.log('val_logprob', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss(self, quantiles_output, y):
        losses = []
        quantiles = torch.tensor(self.quantiles, dtype=quantiles_output[0].dtype, requires_grad=False).to(
            quantiles_output.device)

        for i in range(quantiles_output.shape[1]):
            future = y[:, 1 + i: 1 + i + self.horizon_size, :]
            quantiles_result = quantiles_output[:, i, :, :]
            e = future - quantiles_result
            loss = torch.max(quantiles * e, (quantiles - 1) * e)

            losses.append(loss)

        return torch.mean(torch.stack(losses))


