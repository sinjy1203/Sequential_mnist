import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
import torchinfo

## TCN

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                      padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                      padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else hidden_size
            out_channels = hidden_size
            layers += [
                TemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size, dropout=dropout)
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout,
                 kernel_size):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(
            input_size, hidden_size, num_layers, kernel_size=kernel_size,
            dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        y1 = self.tcn(inputs)
        o = self.linear(y1[:, :, -1])
        return o



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, bi, cell='LSTM'):
        super(RNN, self).__init__()

        if cell == 'LSTM':
            self.cell = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True, dropout=dropout,
                bidirectional=bi)
        elif cell == 'GRU':
            self.cell = nn.GRU(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True, dropout=dropout,
                bidirectional=bi)
        else:
            raise Exception('wrong cell key')

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features=hidden_size * (2 if bi else 1), out_features=output_size)

    def forward(self, x):
        y, _ = self.cell(x)
        output = self.linear(y[:, -1])
        return output


if __name__ == "__main__":
    # model = RNN(input_size=28, hidden_size=50, output_size=10, num_layers=2,
    #             dropout=0.5, bi=False, cell='LSTM').to('cuda')
    model = TCN(input_size=1, hidden_size=100, output_size=10, num_layers=2, dropout=0.5, kernel_size=2).to('cuda')
    # torchinfo.summary(model, (100, 28, 28))
    # for i in model.parameters():
    #     print(i.dtype)
    for m in model.modules():
        print(m)
        print("---------------------------")
