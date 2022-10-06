## import
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

# from sequential_mnist.dataset import Dataset
# from sequential_mnist.model import *
# from sequential_mnist.utils import EarlyStopping
from dataset import Dataset
from model import *
from utils import EarlyStopping

import argparse
import warnings
warnings.filterwarnings(action='ignore')

##
## parser
parser = argparse.ArgumentParser(description="training",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# dir
parser.add_argument("--root_dir", default=".", type=str, dest='root_dir')

# hyperparameter
parser.add_argument("--k_folds", default=5, type=int, dest='k_folds')
parser.add_argument("--num_epochs", default=100, type=int, dest='num_epochs')
parser.add_argument("--batch_size", default=64, type=int, dest='batch_size')
parser.add_argument("--verbose", default=False, type=bool, dest='verbose')
parser.add_argument("--model", default='LSTM', type=str, dest='model')
parser.add_argument("--bi", default=False, type=bool, dest='bi')
parser.add_argument("--bayes_niter", default=10, type=int, dest='bayes_niter')


# optimize hyperparameter bounds
parser.add_argument("--lr", default=1e-3, type=float, dest='lr', nargs='+')
parser.add_argument("--patience", default=5, type=int, dest='patience', nargs='+')
parser.add_argument("--hidden_size", default=50, type=int, dest='hidden_size', nargs='+')
parser.add_argument("--num_layers", default=5, type=int, dest='num_layers', nargs='+')
parser.add_argument("--dropout", default=0.5, type=float, dest='dropout', nargs='+')
parser.add_argument("--kernel_size", default=2, type=int, dest='kernel_size', nargs='+')


## hyperparameter
args = parser.parse_args()

root_dir = Path(args.root_dir)

k_folds = args.k_folds
batch_size = args.batch_size
num_epochs = args.num_epochs
verbose = args.verbose
model = args.model
bi = args.bi

lr_lst = args.lr
patience_lst = args.patience
hidden_size_lst = args.hidden_size
num_layers_lst = args.num_layers
dropout_lst = args.dropout
kernel_size_lst = args.kernel_size

pbounds = {
    'lr': lr_lst,
    'patience': patience_lst,
    'hidden_size': hidden_size_lst,
    'num_layers': num_layers_lst,
    'dropout': dropout_lst,
    'kernel_size': kernel_size_lst
}

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
def generate_model(model, hidden_size, num_layers, dropout, kernel_size):
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

    return Model

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

## dataset
dataset = Dataset(data_dir=data_dir, train=True, rnn=False if model == 'TCN' else True)

## function
loss_fn = nn.CrossEntropyLoss().to(device)
pred_fn = lambda pred: np.argmax(pred, axis=1)
acc_fn = lambda pred, target: (pred == target).mean()
tonumpy_fn = lambda x: x.detach().cpu().numpy()

## train function
def model_cv(lr, patience, hidden_size, num_layers, dropout, kernel_size):
    patience, hidden_size, num_layers, kernel_size = int(patience), int(hidden_size), int(num_layers), int(kernel_size)
    # print("TRAIN | lr: {}, patience: {}, hidden_size: {}, num_layers: {}, dropout: {}, kernel_size: {}".format(lr, patience, hidden_size, num_layers, dropout, kernel_size))
    kfold = KFold(n_splits=k_folds, shuffle=True)
    cv_result = 0

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, sampler=test_subsampler)

        Model = generate_model(model, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, kernel_size=kernel_size).to(device)
        Model.apply(reset_weights)

        optim = torch.optim.Adam(Model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience, verbose=verbose, path=ckpt_dir, model_name=model, save=False)

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

            early_stopping(epoch_test_loss, epoch_test_acc, Model)
            if early_stopping.early_stop:
                break

        cv_result += early_stopping.val_acc_best / k_folds

    return cv_result

bo = BayesianOptimization(f=model_cv, pbounds=pbounds, verbose=2)
bo.maximize(init_points=2, n_iter=args.bayes_niter, acq='ei', xi=0.01)

with open("best_hyperparameter.txt", 'w') as f:
    f.write(str(bo.max) + '\n')
