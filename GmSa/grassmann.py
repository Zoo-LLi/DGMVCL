import torch
from torch import nn, Tensor, ceil
from torch.optim.optimizer import Optimizer
from torch.autograd import Function
import numpy as np
import math
from GmSa.utils import *
from GmSa import StiefelParameter

dtype = torch.double
device = torch.device('cpu')


def calcuK(S):
    b, c, h = S.shape
    Sr = S.reshape(b, c, 1, h)
    Sc = S.reshape(b, c, h, 1)
    K = Sc - Sr
    K = 1.0 / K
    K[torch.isinf(K)] = 0
    return K


class FRMap(nn.Module):
    """SPD矩阵变换操作"""

    def __init__(self, input_size, output_size):
        super(FRMap, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 创建权重矩阵w,其为 input_size * output_size 的正交矩阵
        self.weight = StiefelParameter(torch.FloatTensor(input_size, output_size).to(self.device), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        # output = w.T * input * w
        output = input
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(-1, -2), output)
        return output




class QRComposition(nn.Module):
    def __init__(self):
        super(QRComposition, self).__init__()

    def forward(self, x):
        Q, R = torch.qr(x)
        return Q


class ProjMapFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.matmul(x, x.transpose(-1, -2))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # 2dL/dx * x
        return 2 * torch.matmul(grad_output, x)


class re(Function):
    def forward(ctx, x):
        x = x.reshape()
        ctx.save_for_backward(x)

    def backward(ctx, grad_output):
        return grad_output.respe()


class Projmap(nn.Module):
    def forward(self, x):
        return ProjMapFunction.apply(x)


class OrthmapFunction(Function):
    @staticmethod
    def forward(ctx, x, p):
        U, S, V = torch.svd(x)
        ctx.save_for_backward(U, S)
        if len(x.shape) == 3:
            res = U[:, :, :p]
        else:
            res = U[:, :, :, :p]
        return res

    @staticmethod
    def backward(ctx, grad_output):
        U, S = ctx.saved_tensors
        b, c, h, w = grad_output.shape
        p = h - w
        pad_zero = torch.zeros(b, c, h, p).to('cuda')
        # 调整输出格式
        grad_output = torch.cat((grad_output, pad_zero), 3)
        Ut = U.transpose(-1, -2)
        K = calcuK(S)
        # U * (K.T。(U.T * dL/dx)sym) * U.T
        # mid_1 = torch.matmul(Ut, grad_output)
        # mid_2 = K.transpose(-1, -2) * torch.add(mid_1, mid_1.transpose(-1, -2))
        # mid_3 = torch.matmul(U, mid_2)
        # return torch.matmul(mid_3, Ut), None
        mid_1 = K.transpose(-1, -2) * torch.matmul(Ut, grad_output)
        mid_2 = torch.matmul(U, mid_1)
        return torch.matmul(mid_2, Ut), None
    # def backward(ctx, grad_output):
    #     U, S = ctx.saved_tensors
    #     b, c, f = grad_output.shape  # 修改为三维数据的形状
    #     p = f - c  # 更新计算 p 的方式
    #     pad_zero = torch.zeros(b, c, p).to(grad_output.device)  # 修正维度
    #     grad_output = torch.cat((grad_output, pad_zero), -1)  # 在第三个维度上进行拼接
    #     Ut = U.transpose(-1, -2)
    #     # K = calcuK(S)
    #     b, h = S.size()  # 直接使用 S.size() 获取形状
    #     Sr = S.view(b, 1, h)
    #     Sc = S.view(b, h, 1)
    #     K = Sc - Sr
    #     K = 1.0 / K
    #     K[torch.isinf(K)] = 0
    #
    #     # 对矩阵乘法进行调整以适应三维数据
    #     mid_1 = torch.matmul(Ut.unsqueeze(-3), grad_output.unsqueeze(-1)).squeeze(-1)  # 修正维度
    #     mid_2 = torch.matmul(K.transpose(-1, -2), mid_1)
    #     # print("U shape:", U.shape)
    #     # print("mid_2 shape:", mid_2.shape)
    #     mid_3 = torch.matmul(U, mid_2)
    #     # 返回三维数据的梯度
    #     return mid_3, None


class Orthmap(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return OrthmapFunction.apply(x, self.p)


class ProjPoolLayer_A(torch.autograd.Function):
    # AProjPooling  c/n ==0
    @staticmethod
    def forward(ctx, x, n=4):
        b, c, h, w = x.shape
        ctx.save_for_backward(n)
        new_c = int(math.ceil(c / n))
        new_x = [x[:, i:i + n].mean(1) for i in range(0, c, n)]
        return torch.cat(new_x, 1).reshape(b, new_c, h, w)

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.saved_variables
        return torch.repeat_interleave(grad_output / n, n, 1)


class ProjPoolLayer(nn.Module):
    """ W-ProjPooling"""

    def __init__(self, n=4):
        super().__init__()
        self.n = n

    def forward(self, x):
        avgpool = torch.nn.AvgPool2d(int(math.sqrt(self.n)))
        return avgpool(x)


class E2R(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.proj = Projmap()

    def patch_len(self, n, epochs):
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

    def forward(self, x):
        # x with shape[bs, ch, time] batch size, channels, time
        # split feature into several epochs.
        x = x.squeeze()
        list_patch = self.patch_len(x.shape[-1], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-1))
        for i, item in enumerate(x_list):
            x_list[i] = self.proj(item)
        x = torch.stack(x_list)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.permute(1, 0, 2, 3)
        return x


class AttentionManifold(nn.Module):
    def __init__(self, p, in_embed_size, out_embed_size):
        super(AttentionManifold, self).__init__()
        self.p = p
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.q_trans = FRMap(self.d_in, self.d_out).cpu()
        self.k_trans = FRMap(self.d_in, self.d_out).cpu()
        self.v_trans = FRMap(self.d_in, self.d_out).cpu()
        # self.frmap = grassmann.FRMapping(in_embed_size, out_embed_size, 3)
        self.qrq = QRComposition()
        self.qrk = QRComposition()
        self.qrv = QRComposition()
        self.proj = Projmap()

    def projection_metric(self, mx_a, mx_b):
        """计算矩阵A与B的投影度量"""
        inner_term = torch.matmul(mx_a, mx_a.permute(0, 1, 3, 2)) - torch.matmul(mx_b, mx_b.permute(0, 1, 3, 2))
        inner_multi = torch.matmul(inner_term, inner_term.permute(0, 1, 3, 2))
        return torch.norm(inner_multi, dim=[2, 3])

    def w_frechet_distance_mean(self, B, V):
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

    def forward(self, x, shape=None):
        # x = self.frmap(x)
        # x = self.qr1(x)
        # Q = x[0]
        # K = x[1]
        # V = x[2]
        if len(x.shape) == 3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.p)
            print('12222222344')
        x = x.to(torch.float)  # patch:[b, #patch, c, c]
        # # calculate Q K V
        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs * m, self.d_in, self.p)
        # Q = w_q.T * x *w_q, K, V = ...
        Q = self.q_trans(x).view(bs, m, self.d_out, self.p)
        K = self.k_trans(x).view(bs, m, self.d_out, self.p)
        V = self.v_trans(x).view(bs, m, self.d_out, self.p)
        Q = self.qrq(Q)
        K = self.qrk(K)
        V = self.qrv(V)
        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1)
        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1)
        K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3],
                                 K_expand.shape[4])
        atten_energy = self.projection_metric(Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1])
        atten_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + atten_energy))).permute(0, 2, 1)

        V = self.proj(V)
        output = self.w_frechet_distance_mean(atten_prob, V)
        shape = list(output.shape[:2])
        shape.append(-1)
        return output, shape

class GrassmanialManifold(nn.Module):
    def __init__(self, in_embed_size, out_embed_size,p, nx=0):
        super(GrassmanialManifold, self).__init__()
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.p = p
        self.frmap = FRMap(self.d_in, self.d_out).cpu()
        self.nx = nx
        # self.frmap1 = FRMap(25, self.d_out).cpu()
        # self.frmap = grassmann.FRMapping(in_embed_size, out_embed_size, 3)
        # self.qr1 = Orthmap(15)
        self.qr1 = QRComposition()
        self.proj = Projmap()


    def forward(self, x, shape=None):
        bs = x.shape[0]
        m = x.shape[1]
        x = x.squeeze()
        # x = x.reshape(bs * m, self.d_in, self.p)

        x = self.frmap(x)#.view(bs, m, self.d_out, self.p)
        x = self.qr1(x)
        # for _ in range(self.nx):
        #     x = self.frmap1(x)
        #     x = self.qr1(x)
            # print(123456)
        # output = x
        output = self.proj(x)
        return output


# class splitSignal_radar(nn.Module):
#     def __init__(self, out_channel):
#         super(splitSignal_radar, self).__init__()
#         self.conv1 = nn.Conv1d(1, out_channel, 20, stride=10)
#         self.conv2 = nn.Conv1d(1, out_channel, 20, stride=10)
#         self.Bn1 = nn.BatchNorm1d(out_channel)
#         self.Bn2 = nn.BatchNorm1d(out_channel)
#         self.ReLu1 = nn.ReLU()
#         self.ReLu2 = nn.ReLU()
#
#     def forward(self, x):
#         x = x.split(x.shape[1] // 2, 1)
#         x_r = self.conv1(x[0])
#         x_i = self.conv1(x[1])
#         x_r = self.ReLu1(x_r)
#         x_i = self.ReLu2(x_i)
#         x_r = self.Bn1(x_r)
#         x_i = self.Bn2(x_i)
#
#         x = torch.stack((x_r, x_i), 0)
#         return x
#
# def split_features(x):
#     reshaped_data = torch.zeros(180, 96, 8, device=x.device)
#     for i in range(8):
#         # 从每个样本的1500个特征中随机选择15个
#         selected_features = torch.randint(768, (96,), device=x.device)
#         # 将选择的15个特征放入 reshaped_data 中的第 i 个位置
#         reshaped_data[:, :, i] = x[:, selected_features]
#
#     return reshaped_data