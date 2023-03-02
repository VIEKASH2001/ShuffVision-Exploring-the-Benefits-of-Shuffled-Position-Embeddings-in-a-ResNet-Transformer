import torch
import torch.nn as nn
from math import log
from collections import OrderedDict
import torch.nn.functional as F

from .utils.model_utils import Sequential
from .utils.initializers import xavier_init_, kaiming_init_



class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention introduced in [1].
    Args:
        dims (int): The input feature dimensions.
        k_dims (int | None, optional): The dimensions of key matrix. If not
            specified, it will be the same as ``q_dims``. Default: ``None``.
        v_dims (int | None, optional): The dimensions of value matrix. If not
            specified, it will be the same as ``q_dims``. Default: ``None``.
        h_dims (int | None, optional): The hidden dimensions. If not specified,
            it will be the same as ``q_dims``. Default: ``None``.
        o_dims (int | None, optional): The output dimensions. If not specified,
            it will be the same as ``q_dims``. Default: ``None``.
        heads (int, optional): The number of attention heads. Default: ``8``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        bias (bool, optional): Whether to add the bias term. Default: ``True``.
    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 k_dims=None,
                 v_dims=None,
                 h_dims=None,
                 o_dims=None,
                 heads=8,
                 p=0.1,
                 bias=True):
        super(MultiHeadAttention, self).__init__()

        self._q_dims = dims
        self._k_dims = k_dims or dims
        self._v_dims = v_dims or dims
        self._h_dims = h_dims or dims
        self._o_dims = o_dims or dims
        self._heads = heads
        self._p = p
        self._bias = bias
        self._head_dims = self._h_dims // heads

        self.q = nn.Linear(self._q_dims, self._h_dims, bias=bias)
        self.k = nn.Linear(self._k_dims, self._h_dims, bias=bias)
        self.v = nn.Linear(self._v_dims, self._h_dims, bias=bias)
        self.m = nn.Linear(self._h_dims, self._o_dims, bias=bias)

        self.dropout = nn.Dropout(p=p)

        self.reset_parameters()

    def __repr__(self):
        return ('{}(q_dims={}, k_dims={}, v_dims={}, h_dims={}, o_dims={}, '
                'heads={}, p={}, bias={})'.format(self.__class__.__name__,
                                                  self._q_dims, self._k_dims,
                                                  self._v_dims, self._h_dims,
                                                  self._o_dims, self._heads,
                                                  self._p, self._bias))

    def reset_parameters(self):
        for m in (self.q, self.k, self.v, self.m):
            xavier_init_(m)

    def forward(self, q, k=None, v=None, mask=None):
        v = v if torch.is_tensor(v) else k if torch.is_tensor(k) else q
        k = k if torch.is_tensor(k) else q

        q = self.q(q).transpose(0, 1).contiguous()
        k = self.k(k).transpose(0, 1).contiguous()
        v = self.v(v).transpose(0, 1).contiguous()

        b = q.size(1) * self._heads

        q = q.view(-1, b, self._head_dims).transpose(0, 1)
        k = k.view(-1, b, self._head_dims).transpose(0, 1)
        v = v.view(-1, b, self._head_dims).transpose(0, 1)

        att = torch.bmm(q, k.transpose(1, 2)) / self._head_dims**0.5

        if mask is not None:
            mask = torch.where(mask > 0, .0, float('-inf'))
            mask = mask.repeat_interleave(self._heads, dim=0)
            att += mask.unsqueeze(1)

        att = att.softmax(-1)

        if self.dropout is not None:
            att = self.dropout(att)

        m = torch.bmm(att, v).transpose(0, 1).contiguous()
        m = m.view(m.size(0), -1, self._h_dims).transpose(0, 1)
        m = self.m(m)

        if self.dropout is not None:
            m = self.dropout(m)

        return m



class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network introduced in [1].
    Args:
        dims (int): The input feature dimensions.
        ratio (float, optional): The ratio of hidden layer dimensions with
            respect to the input dimensions. Default: ``4``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.
    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 ratio=4,
                 p=0.1,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(FeedForwardNetwork, self).__init__()

        self._dims = dims
        self._ratio = ratio
        self._p = p
        self._h_dims = int(dims * ratio)

        self.mapping = Sequential(
            nn.Linear(dims, self._h_dims), nn.ReLU(inplace=True),
            nn.Dropout(p=p), nn.Linear(self._h_dims, dims), nn.Dropout(p=p))

        self.reset_parameters()

    def __repr__(self):
        return '{}(dims={}, ratio={}, p={})'.format(self.__class__.__name__,
                                                    self._dims, self._ratio,
                                                    self._p)

    def reset_parameters(self):
        for m in self.mapping:
            if isinstance(m, nn.Linear):
                kaiming_init_(m)

    def forward(self, x):
        x = self.mapping(x)
        return x