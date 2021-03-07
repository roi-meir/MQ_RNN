import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self,
                 input_size=4,
                 num_layers=1,
                 hidden_units=8):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_units = hidden_units

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_units,
                            num_layers=self.num_layers,
                            batch_first=True)

        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)

    def forward(self, x):
        output, (hh, cc) = self.lstm(x)
        return output, hh, cc


