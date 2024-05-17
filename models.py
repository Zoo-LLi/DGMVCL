import time

import torch
import torch.nn as nn
from loss import *
from metrics import *
from dataprocessing import *

def procrustes_distance(x1, x2):
    # 计算格拉斯曼流形上的投影映射
    projection = 2 ** (-0.5) * (x1 @ x1.transpose(-1, -2) - x2 @ x2.transpose(-1, -2))

    # 计算 Frobenius 范数
    frobenius_norm = torch.norm(projection, 'fro', dim=(1, 2))

    return torch.mean(frobenius_norm)



def contrastive_train(network_model, mv_data, mvc_loss, batch_size, lmd, beta, gamma, temperature_l, normalized, epoch,
                      optimizer):

    network_model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()
    total_loss = 0.
    loss_function = nn.KLDivLoss(reduction='mean')
    for batch_idx, (sub_data_views, _) in enumerate(mv_data_loader):
        # print("\tbatch_idx:{}".format(batch_idx))
        lbps, features = network_model(sub_data_views)
        loss_list = list()
        for i in range(num_views):
            # loss_list.append(-loss_function(mvc_loss.target_distribution(features[i]), mvc_loss.target_distribution(lbps[i])))
            # loss_list.append(lmd * mvc_loss.forward_label(features[i], lbps[i], temperature_l, normalized))
            for j in range(i + 1, num_views):
                loss_list.append(gamma * mvc_loss.forward_features(features[i], features[j], batch_size, temperature_l, normalized))
                loss_list.append(lmd * mvc_loss.forward_label(lbps[i], lbps[j], temperature_l, normalized))
                loss_list.append(beta * mvc_loss.forward_prob(lbps[i], lbps[j]))
            # loss_list.append(criterion(sub_data_views[i], dvs[i]))
        loss = sum(loss_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print('Contrastive_train, epoch {} loss:{:.7f}'.format(epoch, total_loss / num_samples))

    return total_loss


def inference(network_model, mv_data, batch_size):

    network_model.eval()
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    soft_vector = []
    pred_vectors = []
    labels_vector = []
    for v in range(num_views):
        pred_vectors.append([])

    for batch_idx, (sub_data_views, sub_labels) in enumerate(mv_data_loader):
        with torch.no_grad():
            lbps, _ = network_model(sub_data_views)
            lbp = sum(lbps)/num_views

        for idx in range(num_views):
            pred_label = torch.argmax(lbps[idx], dim=1)
            pred_vectors[idx].extend(pred_label.detach().cpu().numpy())

        soft_vector.extend(lbp.detach().cpu().numpy())
        labels_vector.extend(sub_labels)

    for idx in range(num_views):
        pred_vectors[idx] = np.array(pred_vectors[idx])

    # labels_vector = np.array(labels_vector).reshape(num_samples)
    actual_num_samples = len(soft_vector)
    labels_vector = np.array(labels_vector).reshape(actual_num_samples)
    total_pred = np.argmax(np.array(soft_vector), axis=1)

    return total_pred, pred_vectors, labels_vector


def valid(network_model, mv_data, batch_size):

    total_pred, pred_vectors, labels_vector = inference(network_model, mv_data, batch_size)
    num_views = len(mv_data.data_views)

    print("Clustering results on cluster assignments of each view:")
    for idx in range(num_views):
        acc, nmi, pur, ari = calculate_metrics(labels_vector,  pred_vectors[idx])
        print('ACC{} = {:.4f} NMI{} = {:.4f} PUR{} = {:.4f} ARI{}={:.4f}'.format(idx+1, acc,
                                                                                 idx+1, nmi,
                                                                                 idx+1, pur,
                                                                                 idx+1, ari))

    print("Clustering results on semantic labels: " + str(labels_vector.shape[0]))
    acc, nmi, pur, ari = calculate_metrics(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI={:.4f}'.format(acc, nmi, pur, ari))

    return acc, nmi, pur, ari
