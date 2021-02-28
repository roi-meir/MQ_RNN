import torch
from torch import nn


class ContextDecoder(nn.Module):
    def __init__(self,
                 hidden_units: int,
                 covariate_size: int,
                 horizon_size: int,
                 context_size: int
                 ):
        super().__init__()
        self.horizon_size = horizon_size
        self.covariate_size = covariate_size
        self.hidden_units = hidden_units
        self.context_size = context_size

        # N horizon specific contexts + global one
        self.output_size = (horizon_size + 1) * context_size

        self.linear1 = nn.Linear(in_features=hidden_units + covariate_size*horizon_size,
                                 out_features=self.output_size * 2)

        self.linear2 = nn.Linear(in_features=self.output_size * 2,
                                 out_features=self.output_size)

        self.activation = nn.ReLU()

    def forward(self, x):
        layer1_output = self.linear1(x)
        layer1_output = self.activation(layer1_output)

        layer2_output = self.linear2(layer1_output)
        layer2_output = self.activation(layer2_output)
        return layer2_output


class QuantileDecoder(nn.Module):
    def __init__(self,
                 covariate_size: int,
                 number_of_quantile: int,
                 context_size: int,
                 ):
        super().__init__()
        self.number_of_quantile = number_of_quantile
        self.covariate_size = covariate_size
        self.context_size = context_size

        # Global context + Local context + covariates
        self.input_size = 2 * self.context_size + self.covariate_size

        self.linear1 = nn.Linear(in_features=2 * self.context_size + self.covariate_size,
                                 out_features=self.context_size)

        self.linear2 = nn.Linear(in_features=self.context_size,
                                 out_features=self.number_of_quantile)

        self.activation = nn.ReLU()

    def forward(self, x):
        layer1_output = self.linear1(x)
        layer1_output = self.activation(layer1_output)

        layer2_output = self.linear2(layer1_output)
        layer2_output = self.activation(layer2_output)
        return layer2_output

#
# class Decoder(nn.Module):
#     def __init__(self,
#                  hidden_units: int,
#                  covariate_size: int,
#                  horizon_size: int,
#                  context_size: int,
#                  number_of_quantile: int
#                  ):
#         super().__init__()
#
#         self.hidden_units = hidden_units
#         self.covariate_size = covariate_size
#         self.horizon_size = horizon_size
#         self.context_size = context_size
#         self.number_of_quantile = number_of_quantile
#
#         self.context_decoder = ContextDecoder(hidden_units=hidden_units,
#                                               covariate_size=covariate_size,
#                                               horizon_size=horizon_size,
#                                               context_size=number_of_quantile)
#
#         self.quantile_decoder = QuantileDecoder(covariate_size=covariate_size,
#                                                 context_size=context_size,
#                                                 number_of_quantile=number_of_quantile)
#
#     def forward(self, x):
#         assert x.shape[0] == self.context_size * (self.horizon_size + 1) + self.covariate_size * self.horizon_size
#
#
#
#         context = self.context_decoder(x)
#         contexts = context.split(self.number_of_quantile + 1)
#         global_context = contexts[0]
#         l_contexts = contexts[1:]
#         # TODO: handle multiple horizons
#         assert self.horizon_size == 1
#         for _ in range(self.horizon_size):
#             decoder_input = torch.cat([gdecoder_output, next_covariate_tensor], dim=2)
#             quantile_output = self.quantile_decoder(decoder_input)
#
#             seq_len = quantile_output.shape[0]
#             batch_size = quantile_output.shape[1]
#
#             quantile_output = quantile_output.view(seq_len, batch_size, self.horizon_size, self.quantile_size)
#             return quantile_output





