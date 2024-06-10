import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
    
    
paris = {
    'graph.nw_ucla.Graph': (
        (1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
        (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
        (20, 19)
    ),
    'graph.ntu_rgb_d.Graph': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'graph.kinetics.Graph': ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                 (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    )
}


def sparse_(tensor, sparsity, mean=0, std=0.01):
    r"""Fills the 2D input `Tensor` as a sparse matrix, where the
    non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))

    with torch.no_grad():
        tensor.normal_(mean, std)
        for col_idx in range(cols):
            row_indices = torch.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class SpatialGcnJoint(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(SpatialGcnJoint, self).__init__()
        # physical adjacency
        self.Al = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.Al, 0)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = 3
        self.conv_p = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_p.append(nn.Conv2d(in_channels, out_channels, 1))
    
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        bn_init(self.bn, 1e-6)

        for i in range(self.num_subset):
            conv_branch_init(self.conv_p[i], self.num_subset)

    def forward(self, x):
        # physical adjacency
        N, C, T, V = x.size()
        X = x.view(N, C * T, V)
        A = self.A.cuda(x.get_device())
        y = None
        for i in range(self.num_subset):
            R = A[i] + self.Al[i] # V V
            z = self.conv_p[i](torch.matmul(X, R).view(N, -1, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class SpatialGcnBone(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(SpatialGcnBone, self).__init__()
        if out_channels % 8 == 0:
            inter_channels = out_channels // 8
        else:
            raise ValueError("out_channels can not devide by 8")

        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = 3
        self.conv_p = nn.ModuleList()
        self.conv_a_0 = nn.ModuleList()
        self.conv_a_1 = nn.ModuleList()
        self.conv_a_2 = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_p.append(nn.Conv2d(in_channels, out_channels, 1))
            self.conv_a_0.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_a_1.append(nn.Conv2d(inter_channels, inter_channels, 1))
            self.conv_a_2.append(nn.Conv2d(inter_channels, inter_channels, 1))

        self.TemporalPool = nn.AdaptiveAvgPool2d((1, None))
        self.tanh = nn.Tanh()

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        bn_init(self.bn, 1e-6)

        for i in range(self.num_subset):
            conv_branch_init(self.conv_p[i], self.num_subset)
            conv_branch_init(self.conv_a_0[i], self.num_subset)
            conv_branch_init(self.conv_a_1[i], self.num_subset)
            conv_branch_init(self.conv_a_2[i], self.num_subset)

    def forward(self, x):
        # physical adjacency
        N, C, T, V = x.size()
        X = x.view(N, C * T, V)
        A = self.A.cuda(x.get_device())
        y = None
        for i in range(self.num_subset):
            temp = self.conv_a_0[i](x)
            temp = self.TemporalPool(temp)  # N C 1 V
            A1 = self.conv_a_1[i](temp).view(N, -1, V).permute(0, 2, 1).contiguous()
            A2 = self.conv_a_2[i](temp).view(N, -1, V)  # N V C || N C V
            R = self.tanh(torch.matmul(A1, A2)/A1.size(-1)) # N V V
            R = A[i].unsqueeze(0) + R  # N V V
            z = self.conv_p[i](torch.matmul(X, R).view(N, -1, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class SpatialGcnMotion(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(SpatialGcnMotion, self).__init__()
        if out_channels % 2 == 0:
            inter_channels = out_channels // 2
        else:
            raise ValueError("out_channels can not devide by 2")

        # physical adjacency
        self.Al = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        for i in range(self.Al.size(0)):
            sparse_(self.Al[i], sparsity=0.5, mean=0.01, std=0.01)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = 3
        self.conv_p = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_p.append(nn.Conv2d(in_channels, out_channels, 1))
            
        self.conv_m_1 = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_m_2 = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_m_3 = nn.Conv2d(inter_channels, out_channels, 1)
        self.conv_m_4 = nn.Conv2d(in_channels, out_channels, 1)

        self.tanh = nn.Tanh()
        self.alpha = nn.Parameter(torch.zeros(1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        bn_init(self.bn, 1e-6)

    def forward(self, x):
        # physical adjacency
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        y = None
        for i in range(self.num_subset):
            R = A[i] + self.Al[i] # V V
            X = self.conv_p[i](x)
            z = torch.einsum('uv,nctv->nctu', R, X)
            y = z + y if y is not None else z
        
        x_f = x[:, :, : T-2]
        x_b = x[:, :, 2:]
        x_f = self.conv_m_1(x_f).mean(-2) # N C V
        x_b = self.conv_m_2(x_b).mean(-2) # N C V
        R = self.tanh(self.conv_m_3(x_b.unsqueeze(-1) - x_f.unsqueeze(-2)))# N C V V
        X = self.conv_m_4(x)
        z = torch.einsum('ncuv,nctv->nctu', R, X)
        y = z * self.alpha + y

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=[1, 2, 3, 4]):

        super().__init__()
        assert out_channels % (len(dilations) + 1) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 1
        branch_channels = out_channels // self.num_branches

        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_channels, branch_channels, kernel_size=(ks, 1), padding=((ks + (ks - 1) * (dilation - 1) - 1) // 2, 0), stride=(stride, 1), dilation=(dilation, 1), bias=False),
                nn.BatchNorm2d(branch_channels),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # point and region branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(5, 5), padding=(2, 2), stride=(stride, 1), bias=False),
            nn.BatchNorm2d(branch_channels)
        ))

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        return out


class unit_tcn_lite(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn_lite, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), groups=in_channels, bias=False)
        self.pointwise  = nn.Conv2d(in_channels, out_channels, 1, bias =False)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.relu = nn.ReLU()
        conv_init(self.conv)
        conv_init(self.pointwise)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.pointwise(self.conv(x))
        return self.bn(x)
        
        
class Res(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, groups=1):
        super(Res, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, modality=0, kernel_size=5, dilations=[1, 2, 3]):
        super(TCN_GCN_unit, self).__init__()
        self.modality = modality
        if self.modality == 0:
            self.gcn1 = SpatialGcnJoint(in_channels, out_channels, A)
        elif self.modality == 1:
            self.gcn1 = SpatialGcnBone(in_channels, out_channels, A)
        else:
            self.gcn1 = SpatialGcnMotion(in_channels, out_channels, A)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = Res(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return x


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        self.dataset = graph

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False, modality=2)
        self.l2 = TCN_GCN_unit(64, 64, A, modality=2)
        self.l3 = TCN_GCN_unit(64, 64, A, modality=2)
        self.l4 = TCN_GCN_unit(64, 64, A, modality=2)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, modality=2)
        self.l6 = TCN_GCN_unit(128, 128, A, modality=2)
        self.l7 = TCN_GCN_unit(128, 128, A, modality=2)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, modality=2)
        self.l9 = TCN_GCN_unit(256, 256, A, modality=2)
        self.l10 = TCN_GCN_unit(256, 256, A, modality=2)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        # bn_init(self.data_bn_b, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)