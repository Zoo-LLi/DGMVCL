import torch
import torch.nn as nn
from GmSa.spd import Frmap
from GmSa import grassmann


def projection_metric(mx_a, mx_b):
    """计算矩阵A与B的投影度量"""
    inner_term = torch.matmul(mx_a, mx_a.permute(0, 1, 3, 2)) - torch.matmul(mx_b, mx_b.permute(0, 1, 3, 2))
    inner_multi = torch.matmul(inner_term, inner_term.permute(0, 1, 3, 2))
    return torch.norm(inner_multi, dim=[2, 3])


def w_frechet_distance_mean(B, V):
    """计算基于 PM 的加权弗雷歇均值"""
    # B: [bs, #p, #p]
    # V: [bs, #p, s, s]
    bs = V.shape[0]
    num_p = V.shape[1]
    output = torch.zeros_like(V)
    # V = V.view(bs, num_p, -1)
    for bs_e in range(bs):
        for i in range(num_p):
            for j in range(num_p):
                output[bs_e, i] = B[bs_e, i, j] * V[bs_e, j]
    return output


def patch_len(n, epochs):
    """将特征向量分为epochs个时期，返回一个列表，列表中包含每个时期中特征向量个数"""
    list_len = []
    base = n // epochs
    for i in range(epochs):
        list_len.append(base)
    for i in range(n - base * epochs):
        list_len[i] += 1
    # 验证
    if sum(list_len) == n:
        return list_len
    else:
        return ValueError('check your epochs and axis should be split again')


def qr(x):
    Q, R = torch.qr(x)
    return Q


class E2R(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.proj = grassmann.Projmap()

    def forward(self, x):
        # x with shape[bs, ch, time] batch size, channels, time
        # split feature into several epochs.
        x = x.squeeze()
        list_patch = patch_len(x.shape[-2], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-2))
        for i, item in enumerate(x_list):
            x_list[i] = self.proj(item)
        x = torch.stack(x_list)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.permute(1, 0, 2, 3)
        return x


class signal2spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cpu')

    def forward(self, x):
        # 去除维度为1的维度,便于计算
        x = x.squeeze()
        # 计算x每列均值、并保持原先维度，即 keep dim=True
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])

        # 计算协方差矩阵
        x = x - mean
        cov = x @ x.permute(0, 2, 1)  # 即 x * x.T
        cov = cov.to(self.dev)
        cov = cov / (x.shape[-1] - 1)

        # 迹归一化
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        cov /= tra

        # 确保主对角线所有元素大于0
        identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).to(self.dev).repeat(x.shape[0], 1, 1)
        cov = cov + (1e-5 * identity)
        return cov


class E2R2(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()

    def forward(self, x):
        # x with shape[bs, ch, time] batch size, channels, time
        # split feature into several epochs.
        list_patch = patch_len(x.shape[-1], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-1))
        # convert each one to a specific SPD matrix.
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)

        #  更改数据维度，将[epoch, bs, ...] 更改为[bs, epoch,...]
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        return x


class E2R_radar(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.proj = grassmann.Projmap()

    def forward(self, x):
        x_re = x[0]
        x_im = x[1]
        # x with shape[bs, ch, time] batch size, channels, time
        # split feature into several epochs.
        list_patch = patch_len(x_re.shape[-2], int(self.epochs))
        x_re_list = list(torch.split(x_re, list_patch, dim=-2))
        x_im_list = list(torch.split(x_im, list_patch, dim=-2))
        for i, item in enumerate(x_re_list):
            x_re_list[i] = self.proj(item)
        for i, item in enumerate(x_im_list):
            x_im_list[i] = self.proj(item)
        x_re = torch.stack(x_re_list)
        x_im = torch.stack(x_im_list)
        x = (x_re + x_im) / 2
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.permute(1, 0, 2, 3)
        return x


class AttentionManifold(nn.Module):
    def __init__(self, p, in_embed_size, out_embed_size):
        torch.manual_seed(3407)
        super(AttentionManifold, self).__init__()
        self.p = p
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.q_trans = grassmann.FRMap(self.d_in, self.d_out).cpu()
        self.k_trans = grassmann.FRMap(self.d_in, self.d_out).cpu()
        self.v_trans = grassmann.FRMap(self.d_in, self.d_out).cpu()
        self.qrq = grassmann.QRComposition()
        self.qrk = grassmann.QRComposition()
        self.qrv = grassmann.QRComposition()
        self.proj = grassmann.Projmap()

    def forward(self, x, shape=None):
        if len(x.shape) == 3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.p)
        x = x.to(torch.float)  # patch:[b, #patch, c, c]
        # calculate Q K V
        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs * m, self.d_in, self.p)
        # Q = w_q.T * x *w_q, K, V = ...
        Q = self.qrq(self.q_trans(x).view(bs, m, self.d_out, self.p))
        K = self.qrk(self.k_trans(x).view(bs, m, self.d_out, self.p))
        V = self.qrv(self.v_trans(x).view(bs, m, self.d_out, self.p))

        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1)
        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1)
        K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3],
                                 K_expand.shape[4])
        atten_energy = projection_metric(Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1])
        atten_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + atten_energy))).permute(0, 2, 1)

        V = self.proj(V)
        output = w_frechet_distance_mean(atten_prob, V)
        shape = list(output.shape[:2])
        shape.append(-1)
        return output, shape


class mAtt_bci(nn.Module):
    def __init__(self, epochs, p, in_size, out_size):
        super().__init__()
        # FE
        dim1 = 22
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, dim1, (22, 1)), nn.BatchNorm2d(dim1), nn.ELU())
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(dim1, epochs * in_size, (1, 12)), nn.BatchNorm2d(epochs * in_size), nn.ELU())
        self.E2R = E2R(epochs)

        # riemannian part
        self.orth1 = grassmann.Orthmap(p)
        # R2E
        self.att1 = AttentionManifold(p, in_size, out_size)
        self.flat = nn.Flatten()
        # fc
        # self.linear = nn.Linear(20 * 439, 4, bias=True)
        self.linear = nn.Linear(epochs * out_size * out_size, 4, bias=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.E2R(x)
        x = self.orth1(x)
        x, shape = self.att1(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        y = self.linear(x)
        return y


class GmAtt_mamem(nn.Module):
    def __init__(self, epochs, p, in_size, out_size):
        super().__init__()
        dim1 = 125
        # ------------------------------------------feature extraction--------------------------------------------------
        self.conv_block0 = nn.Sequential(
            nn.Conv2d(1, 4, (1, 1)), nn.BatchNorm2d(4), nn.ELU())
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(4, dim1, (8, 1)), nn.BatchNorm2d(dim1), nn.ELU())
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(dim1, epochs * in_size, (1, 64)), nn.BatchNorm2d(epochs * in_size), nn.ELU())
        # ---------------------------------------Manifold attention module----------------------------------------------
        self.E2R = E2R(epochs)
        self.orth1 = grassmann.Orthmap(p)
        self.att1 = AttentionManifold(p, in_size, out_size)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(epochs * out_size * out_size, 5, bias=True)
        # self.linear = nn.Linear(2520, 5, bias=True)

    def forward(self, x):
        x = self.conv_block0(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.E2R(x)
        x = self.orth1(x)
        x, shape = self.att1(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x


class GmAtt_cha(nn.Module):
    def __init__(self, epochs, p, in_size, out_size):
        super().__init__()
        # FE
        # bs, 1, channel, sample
        dim1 = 14
        self.conv_block0 = nn.Sequential(
            nn.Conv2d(1, 4, (1, 1)), nn.BatchNorm2d(4), nn.ELU())
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(4, dim1, (56, 1)), nn.BatchNorm2d(dim1), nn.ELU())
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(dim1, epochs * in_size, (1, 64)), nn.BatchNorm2d(epochs * in_size), nn.ELU())

        # E2R
        self.E2R = E2R(epochs)
        # riemannian part
        self.att2 = AttentionManifold(p, in_size, out_size)
        self.orth1 = grassmann.Orthmap(p)
        # R2E
        self.flat = nn.Flatten()
        # fc
        # self.linear = nn.Linear(20 * 161, 2, bias=True)
        self.linear = nn.Linear(epochs * out_size * out_size, 2, bias=True)

    def forward(self, x):
        x = self.conv_block0(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.E2R(x)
        x = self.orth1(x)
        x, shape = self.att2(x)

        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x


class GmAtt_radar(nn.Module):
    def __init__(self, epochs, p, in_size, out_size):
        super().__init__()
        self.split = grassmann.splitSignal_radar(epochs * in_size)
        self.E2R = E2R_radar(epochs)
        self.orth1 = grassmann.Orthmap(p)
        self.att2 = grassmann.AttentionManifold(p, in_size, out_size)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(epochs * out_size * out_size, 3, bias=True)
        # self.linear = nn.Linear(10 * 99, 3, bias=True)

    def forward(self, x):
        x = self.split(x)
        x = self.E2R(x)
        x = self.orth1(x)
        x, shape = self.att2(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x


class StiefelParameter:
    pass
