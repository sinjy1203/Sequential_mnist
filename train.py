## import
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from sequential_mnist.dataset import Dataset
# from sequential_mnist.model import *
# from sequential_mnist.utils import EarlyStopping
from dataset import Dataset
from model import *
from utils import EarlyStopping

import argparse
import warnings
warnings.filterwarnings(action='ignore')

## parser
parser = argparse.ArgumentParser(description="training",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# dir
parser.add_argument("--root_dir", default=".", type=str, dest='root_dir')

# training hyperparameter
parser.add_argument("--lr", default=1e-3, type=float, dest='lr')
parser.add_argument("--batch_size", default=64, type=int, dest='batch_size')
parser.add_argument("--num_epochs", default=100, type=int, dest='num_epochs')
parser.add_argument("--patience", default=5, type=int, dest='patience')

parser.add_argument("--verbose", default=False, type=bool, dest='verbose')

# model hyperparameter
parser.add_argument("--model", default='LSTM', type=str, dest='model')
parser.add_argument("--hidden_size", default=50, type=int, dest='hidden_size')
parser.add_argument("--num_layers", default=5, type=int, dest='num_layers')
parser.add_argument("--dropout", default=0.5, type=float, dest='dropout')
parser.add_argument("--bi", default=False, type=bool, dest='bi')
parser.add_argument("--kernel_size", default=2, type=int, dest='kernel_size')

args = parser.parse_args()

## hyperparameter
root_dir = Path(args.root_dir)

lr = args.lr
batch_size = args.batch_size
num_epochs = args.num_epochs
patience = args.patience
verbose = args.verbose

model = args.model
hidden_size = args.hidden_size
num_layers = args.num_layers
dropout = args.dropout
bi = args.bi
kernel_size = args.kernel_size

## directory
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

data_dir = root_dir / 'data'
ckpt_dir = root_dir / 'ckpt'
log_dir = root_dir / 'log'

if not data_dir.exists():
    data_dir.mkdir()
if not ckpt_dir.exists():
    ckpt_dir.mkdir()
if not log_dir.exists():
    log_dir.mkdir()

## model
if model == 'TCN':
    rnn = False
    Model = TCN(input_size=1, hidden_size=hidden_size, output_size=10,
                num_layers=num_layers, dropout=dropout,
                kernel_size=kernel_size).double().to(device)
elif model == 'LSTM' or model == 'GRU':
    rnn = True
    Model = RNN(input_size=28, hidden_size=hidden_size, output_size=10,
                 num_layers=num_layers, dropout=dropout, bi=bi, cell=model).double().to(device)
else:
    raise Exception('wrong model key')

optim = torch.optim.Adam(Model.parameters(), lr=lr)
early_stopping = EarlyStopping(patience=patience, verbose=verbose, path=ckpt_dir, model_name=model)

## dataset
train_dataset = Dataset(data_dir=data_dir, train=True, rnn=rnn)
test_dataset = Dataset(data_dir=data_dir, train=False, rnn=rnn)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=False)

## function
loss_fn = nn.CrossEntropyLoss().to(device)
pred_fn = lambda pred: np.argmax(pred, axis=1)
acc_fn = lambda pred, target: (pred == target).mean()
tonumpy_fn = lambda x: x.detach().cpu().numpy()

## training
writer = SummaryWriter(log_dir=str(log_dir))
train_iter_num = len(train_loader)
test_iter_num = len(test_loader)

for epoch in range(num_epochs):
    epoch_train_loss = 0
    epoch_train_acc = 0
    epoch_test_loss = 0
    epoch_test_acc = 0
    Model.train()

    for i, (train_x, train_y) in enumerate(train_loader):
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        optim.zero_grad()

        output = Model(train_x)
        loss = loss_fn(output, train_y)
        loss.backward()
        optim.step()

        pred = pred_fn(tonumpy_fn(output))
        label = tonumpy_fn(train_y)
        acc = acc_fn(pred, label)

        epoch_train_loss += loss.item() / train_iter_num
        epoch_train_acc += acc / train_iter_num

    with torch.no_grad():
        Model.eval()

        for i, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_y = test_y.to(device)

            output = Model(test_x)
            loss = loss_fn(output, test_y)

            pred = pred_fn(tonumpy_fn(output))
            label = tonumpy_fn(test_y)
            acc = acc_fn(pred, label)

            epoch_test_loss += loss.item() / test_iter_num
            epoch_test_acc += acc / test_iter_num

        if verbose:
            print("EPOCH: ", epoch)
            print("TRAIN LOSS: ", epoch_train_loss)
            print("TRAIN ACC: ", epoch_train_acc)
            print("TEST LOSS: ", epoch_test_loss)
            print("TEST ACC: ", epoch_test_acc)

    early_stopping(epoch_test_loss, epoch_test_acc, Model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    writer.add_scalars("Loss", {'train': epoch_train_loss, 'test': epoch_test_loss}, epoch)
    writer.add_scalars("Acc", {'train': epoch_train_acc, 'test': epoch_test_acc}, epoch)
    # writer.add_scalars("Loss", {'GRU': epoch_test_loss}, epoch)
    # writer.add_scalars("Acc", {'GRU': epoch_test_acc}, epoch)

writer.close()
