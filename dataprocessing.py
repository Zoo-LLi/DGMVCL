import os, random, sys

import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
# from torch.nn.functional import normalize
from utils import *



class MultiviewData(Dataset):
    def __init__(self, db, device, path="datasets/"):
        self.data_views = list()


        if db == "MNIST-USPS":
            mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            # print(X1.shape)
            X1 = np.transpose(X1, (0, 3, 1, 2))
            X2 = np.transpose(X2, (0, 3, 1, 2))
            print(X1.shape)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)


        elif db == "Fashion":
            mat = sio.loadmat(os.path.join(path, 'Fashion.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            X3 = mat['X3'].astype(np.float32)
            X1 = np.transpose(X1, (0, 3, 1, 2))
            X2 = np.transpose(X2, (0, 3, 1, 2))
            X3 = np.transpose(X3, (0, 3, 1, 2))
            print(X1.shape)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "Multi-COIL-10":
            mat = sio.loadmat(os.path.join(path, 'Multi-COIL-10.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            X3 = mat['X3'].astype(np.float32)
            print(X1.shape)

            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
        elif db == "ORL":
            mat = sio.loadmat(os.path.join(path, 'ORL.mat'))
            X1 = mat['X1'].astype(np.float32)
            X1 = np.expand_dims(X1, axis=1)
            X2 = mat['X2'].astype(np.float32)
            X2 = np.expand_dims(X2, axis=1)
            X3 = mat['X3'].astype(np.float32)
            X3 = np.expand_dims(X3, axis=1)
            print(X1.shape)

            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "scene":
            mat = sio.loadmat(os.path.join(path, 'Scene15_dataset1.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            X3 = mat['X3'].astype(np.float32)
            print(X1.shape)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        else:
            raise NotImplementedError

        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sub_data_views = list()
        for view_idx in range(self.num_views):
            data_view = self.data_views[view_idx]
            sub_data_views.append(data_view[index])

        return sub_data_views, self.labels[index]


def get_multiview_data(mv_data, batch_size):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return mv_data_loader, num_views, num_samples, num_clusters


def get_all_multiview_data(mv_data):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=num_samples,
        shuffle=True,
        drop_last=True,
    )

    return mv_data_loader, num_views, num_samples, num_clusters
