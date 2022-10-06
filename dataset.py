##
import numpy as np
import os
import struct
import torch
import torch.nn as nn

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train, rnn):
        self.X, self.y = self.read(dataset="training" if train else "testing", path=data_dir)
        self.rnn = rnn

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x, y = self.X[index], self.y[index]
        x = x / 255.
        if not self.rnn:
            x = x.reshape(1, -1)

        return torch.tensor(x).float(), torch.tensor(y).long()
        # return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        # return torch.tensor(x).float(), torch.tensor(y).float()

    def read(self, dataset="training", path="."):
        """
        Python function for importing the MNIST data set.  It returns an iterator
        of 2-tuples with the first element being the label and the second element
        being a numpy.uint8 2D array of pixel data for the given image.
        """

        if dataset is "training":
            fname_img = os.path.join(path, 'train-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
        elif dataset is "testing":
            fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        # Load everything in some numpy arrays
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

        get_img = lambda idx: (lbl[idx], img[idx])

        # Create an iterator which returns each image in turn
        x_data, y_data = [], []
        for i in range(len(lbl)):
            y, x = get_img(i)
            x_data += [x]
            y_data += [y]
        return np.array(x_data), np.array(y_data)

