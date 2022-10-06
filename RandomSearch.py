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

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import argparse
import warnings
import os

warnings.filterwarnings(action='ignore')

## directory
root_dir = 'C:/Users/sinjy/PycharmProjects/pytorch_practice/sequential_mnist'

data_dir = os.path.join(root_dir, 'data')
ckpt_dir = os.path.join(root_dir, 'ckpt')
log_dir = os.path.join(root_dir, 'log')

## hyperparameter
num_samples = 5

k_folds = 5
num_epochs = 1
verbose = 3
patience = 5

config = {
    "model": tune.choice(["LSTM", 'GRU']),
    "bi": tune.choice([True, False]),
    "batch_size": tune.choice([16, 32, 64]),
    "lr": tune.uniform(1e-2, 1e-3),
    "hidden_size": tune.randint(50, 100),
    "num_layers": tune.randint(1, 5),
    "dropout": tune.uniform(0.1, 0.9)
}

# config = {
#     "model": "TCN",
#     "batch_size": tune.choice([16, 32, 64]),
#     "lr": tune.uniform(1e-2, 1e-3),
#     "hidden_size": tune.randint(50, 100),
#     "num_layers": tune.randint(1, 5),
#     "dropout": tune.uniform(0.1, 0.9),
#     "kernel_size": tune.randint(2, 7)
# }

# device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")

## function
pred_fn = lambda pred: np.argmax(pred, axis=1)
acc_fn = lambda pred, target: (pred == target).mean()
tonumpy_fn = lambda x: x.detach().cpu().numpy()

## model
def generate_model(model, hidden_size, num_layers, dropout, kernel_size=None, bi=None):
    if model == 'TCN':
        rnn = False
        Model = TCN(input_size=1, hidden_size=hidden_size, output_size=10,
                    num_layers=num_layers, dropout=dropout,
                    kernel_size=kernel_size).double()
    elif model == 'LSTM' or model == 'GRU':
        rnn = True
        Model = RNN(input_size=28, hidden_size=hidden_size, output_size=10,
                     num_layers=num_layers, dropout=dropout, bi=bi, cell=model).double()
    else:
        raise Exception('wrong model key')

    return Model

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

## train function
def train_cv(config, checkpoint_dir=None, data_dir=None):
    # print(config)
    lr = config.pop("lr")
    batch_size = config.pop("batch_size")
    dataset = Dataset(data_dir=data_dir, train=True, rnn=False if config['model'] == 'TCN' else True)

    kfold = KFold(n_splits=k_folds, shuffle=True)
    loss_avg, acc_avg = 0, 0
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, sampler=test_subsampler)

        net = generate_model(**config)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
        net.to(device)
        net.apply(reset_weights)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss().to(device)

        early_stopping = EarlyStopping(patience=patience, verbose=False, model_name=config['model'], save=False)

        train_iter_num = len(train_loader)
        test_iter_num = len(test_loader)

        for epoch in range(num_epochs):
            epoch_train_loss = 0
            epoch_train_acc = 0
            epoch_test_loss = 0
            epoch_test_acc = 0
            net.train()

            for i, (train_x, train_y) in enumerate(train_loader):
                train_x = train_x.to(device)
                train_y = train_y.to(device)

                optim.zero_grad()

                output = net(train_x)
                loss = loss_fn(output, train_y)
                loss.backward()
                optim.step()

                pred = pred_fn(tonumpy_fn(output))
                label = tonumpy_fn(train_y)
                acc = acc_fn(pred, label)

                epoch_train_loss += loss.item() / train_iter_num
                epoch_train_acc += acc / train_iter_num

            with torch.no_grad():
                net.eval()

                for i, (test_x, test_y) in enumerate(test_loader):
                    test_x = test_x.to(device)
                    test_y = test_y.to(device)

                    output = net(test_x)
                    loss = loss_fn(output, test_y)

                    pred = pred_fn(tonumpy_fn(output))
                    label = tonumpy_fn(test_y)
                    acc = acc_fn(pred, label)

                    epoch_test_loss += loss.item() / test_iter_num
                    epoch_test_acc += acc / test_iter_num

            early_stopping(epoch_test_loss, epoch_test_acc, net)
            if early_stopping.early_stop:
                break

        loss_avg += early_stopping.val_loss_min / k_folds
        acc_avg += early_stopping.val_acc_best / k_folds

    tune.report(loss=loss_avg, accuracy=acc_avg)

## optimize
reporter = CLIReporter(
    metric_columns=["loss", "accuracy"])
result = tune.run(
    partial(train_cv, data_dir=data_dir),
    config=config,
    resources_per_trial={"cpu": 1},
    num_samples=num_samples,
    progress_reporter=reporter,
    verbose=verbose)

##
best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))

best_dict = {
    'ACC': best_trial.last_result["accuracy"],
    'LOSS': best_trial.last_result["loss"],
    'HYPERPARAMETER': best_trial.config
}

with open("best_hyperparameter.txt", 'w') as f:
    f.write(str(best_dict) + '\n')

