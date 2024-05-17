import torch.nn as nn
from torch.nn import Parameter

from GmSa import grassmann
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def patch_len(n, epochs):

    list_len = []
    base = n // epochs
    for i in range(epochs):
        list_len.append(base)
    for i in range(n - base * epochs):
        list_len[i] += 1

    if sum(list_len) == n:
        return list_len
    else:
        return ValueError('check your epochs and axis should be split again')

def visualization(X):

    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)


    kmeans = KMeans(n_clusters=10, random_state=0)
    clusters = kmeans.fit_predict(X_tsne)


    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis')
    plt.xticks([])
    plt.yticks([])
    plt.show()

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(dims)+1):
            if i == 0:
                self.encoder.add_module('Linear%d' % i,  nn.Linear(input_dim, dims[i]))
            elif i == len(dims):
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], feature_dim))
            else:
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            self.encoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


class AutoDecoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoDecoder, self).__init__()
        self.decoder = nn.Sequential()
        dims = list(reversed(dims))
        for i in range(len(dims)+1):
            if i == 0:
                self.decoder.add_module('Linear%d' % i,  nn.Linear(feature_dim, dims[i]))
            elif i == len(dims):
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], input_dim))
            else:
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            self.decoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.decoder(x)


class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, n_z):
        super(ClusteringLayer, self).__init__()
        self.centroids = Parameter(torch.Tensor(n_clusters, n_z),requires_grad=True)
        nn.init.xavier_uniform_(self.centroids.data)

    def forward(self, x):
        epsilon = 1e-8
        q = 1.0 / (1 + torch.sum(
            torch.pow(x.unsqueeze(1) - self.centroids, 2), 2) + epsilon)
        q = (q.t() / torch.sum(q, 1)).t()

        return q

out_c = 256
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))


    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        # x = self.flatten(x)
        # x = self.classifier(x)
        x = x.view(x.shape[0], -1, x.shape[1])
        return x

class GCLMVCNetwork(nn.Module):
    def __init__(self, num_views, input_sizes, dims, dim_high_feature, dim_low_feature,num_clusters):
        super(GCLMVCNetwork, self).__init__()
        self.encoders = list()
        self.decoders = list()
        for idx in range(num_views):
            self.encoders.append(AutoEncoder(input_sizes[idx], dim_high_feature, dims))
            self.decoders.append(AutoDecoder(input_sizes[idx], dim_high_feature, dims))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.label_learning_module = nn.Sequential(
            nn.Linear(1500, dim_low_feature),
            nn.Linear(dim_low_feature, num_clusters),
            nn.Softmax(dim=1)
        )
        self.CNN = AlexNet()
        self.E2R = grassmann.Projmap()
        p = 13
        self.orth1 = grassmann.Orthmap(p)
        self.flat = nn.Flatten()
        out_size = 25
        nx = 2
        self.att2 = grassmann.GrassmanialManifold(49, out_size, p, nx)
        self.clustering = ClusteringLayer(num_clusters, out_size*out_size)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(out_size*out_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_clusters),
            nn.Softmax(dim=1)
        )


    def forward(self, data_views):
        lbps = list()
        # dvs = list()
        features = list()

        num_views = len(data_views)
        for idx in range(num_views):
            data_view = data_views[idx]
            # high_features = self.encoders[idx](data_view)
            high_features = self.CNN(data_view)
            x = high_features.unsqueeze(1)
            # print(high_features.shape)
            x = self.E2R(x)
            x = self.orth1(x)
            # # print(x.shape)
            x = self.att2(x)

            p = self.clustering(x.view(x.shape[0], -1))
            features.append(p)
            x = self.flat(x)
            # visualization(x.cpu().detach().numpy())
            # label_probs = self.label_learning_module(x)
            # data_view_recon = self.decoders[idx](high_features)
            label_probs = self.classifier(x)
            # features.append(high_features)
            # print(label_probs.shape)
            lbps.append(label_probs)
            # dvs.append(data_view_recon)

        return lbps, features
