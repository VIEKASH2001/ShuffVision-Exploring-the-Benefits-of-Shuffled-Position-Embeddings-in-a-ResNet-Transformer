import torch
import torch.nn as nn
from math import log
from collections import OrderedDict
import torch.nn.functional as F

from .transformer_component import FeedForwardNetwork, MultiHeadAttention
from .utils.model_utils import Parameter

class PositionalEncoding(nn.Module):
    """
    Positional Encoding introduced in [1].
    Args:
        dims (int): The input feature dimensions.
        learnable (bool, optional): Whether the positional encoding is
            learnable. Default: ``True``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        max_len (int, optional): The maximum length of the input sequence.
            Default: ``5000``.
    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self, dims, learnable=True, p=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self._dims = dims
        self._learnable = learnable
        self._p = p
        self._max_len = max_len

        if learnable:
            self.pe = Parameter(1, max_len, dims)
        else:
            pos = torch.arange(max_len).unsqueeze(1)
            div = (torch.arange(0, dims, 2) * (-log(10000.0) / dims)).exp()
            pe = torch.zeros(1, max_len, dims)
            pe[0, :, 0::2] = (pos * div).sin()
            pe[0, :, 1::2] = (pos * div).cos()
            self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(p=p)

    def __repr__(self):
        return ('{}(dims={}, learnable={}, p={}, max_len={})'.format(
            self.__class__.__name__, self._dims, self._learnable, self._p,
            self._max_len))

    def forward(self, x):
        pe = self.pe[:, :x.size(1)].repeat(x.size(0), 1, 1)
        pe = self.dropout(pe)
        return pe




class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer introduced in [1].
    Args:
        dims (int): The input feature dimensions.
        heads (int, optional): The number of attention heads. Default: ``8``.
        ratio (float, optional): The ratio of hidden layer dimensions in the
            feed forward network. Default: ``4``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        pre_norm (bool, optional): Whether to apply the normalization before
            instead of after each layer. Default: ``True``.
        norm_cfg (dict | str | None, optional): The config or name of the
            normalization layer. Default: ``dict(type='LN')``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.
    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 heads=8,
                 ratio=4,
                 p=0.1,
                 pre_norm=True,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(TransformerEncoderLayer, self).__init__()

        self._dims = dims
        self._heads = heads
        self._ratio = ratio
        self._p = p
        self._pre_norm = pre_norm

        self.att = MultiHeadAttention(dims, heads=heads, p=p)
        self.ffn = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = nn.LayerNorm(normalized_shape=dims)
        self.norm2 = nn.LayerNorm(normalized_shape=dims)

    def forward(self, x, pe=None, mask=None):
        if self._pre_norm:
            v = self.norm1(x)
            q = k = v if pe is None else v + pe
            d = self.att(q, k, v, mask=mask)
            x = x + d

            d = self.norm2(x)
            d = self.ffn(d)
            x = x + d
        else:
            q = k = x if pe is None else x + pe
            d = self.att(q, k, x, mask=mask)
            x = self.norm1(x + d)

            d = self.ffn(x)
            x = self.norm2(x + d)

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer introduced in [1].
    Args:
        dims (int): The input feature dimensions.
        heads (int, optional): The number of attention heads. Default: ``8``.
        ratio (int, optional): The ratio of hidden layer dimensions in the
            feed forward network. Default: ``4``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        pre_norm (bool, optional): Whether to apply the normalization before
            instead of after each layer. Default: ``True``.
        norm_cfg (dict | str | None, optional): The config or name of the
            normalization layer. Default: ``dict(type='LN')``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.
    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 heads=8,
                 ratio=4,
                 p=0.1,
                 pre_norm=True,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(TransformerDecoderLayer, self).__init__()

        self._dims = dims
        self._heads = heads
        self._ratio = ratio
        self._p = p
        self._pre_norm = pre_norm

        self.att1 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att2 = MultiHeadAttention(dims, heads=heads, p=p)
        self.ffn = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = nn.LayerNorm(normalized_shape=dims)
        self.norm2 = nn.LayerNorm(normalized_shape=dims)
        self.norm3 = nn.LayerNorm(normalized_shape=dims)

    def forward(self, x, mem, q_pe=None, k_pe=None, mask=None):
        if self._pre_norm:
            v = self.norm1(x)
            q = k = v if q_pe is None else v + q_pe
            d = self.att1(q, k, v, mask=mask)
            x = x + d

            q = self.norm2(x)
            q = q if q_pe is None else q + q_pe
            k = mem if k_pe is None else mem + k_pe
            d = self.att2(q, k, mem, mask=mask)
            x = x + d

            d = self.norm3(x)
            d = self.ffn(d)
            x = x + d
        else:
            q = k = x if q_pe is None else x + q_pe
            d = self.att1(q, k, x, mask=mask)
            x = self.norm1(x + d)

            q = x if q_pe is None else x + q_pe
            k = mem if k_pe is None else mem + k_pe
            d = self.att2(q, k, mem, mask=mask)
            x = self.norm2(x + d)

            d = self.ffn(x)
            x = self.norm3(x + d)

        return x



class BottleneckTransformerLayer(nn.Module):

    def __init__(self,
                 dims,
                 heads=8,
                 ratio=4,
                 p=0.1,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(BottleneckTransformerLayer, self).__init__()

        self.dims = dims
        self.heads = heads
        self.ratio = ratio
        self.p = p

        self.att1 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att2 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att3 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att4 = MultiHeadAttention(dims, heads=heads, p=p)

        self.ffn1 = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)
        self.ffn2 = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = nn.LayerNorm(normalized_shape=dims)
        self.norm2 = nn.LayerNorm(normalized_shape=dims)
        self.norm3 = nn.LayerNorm(normalized_shape=dims)
        self.norm4 = nn.LayerNorm(normalized_shape=dims)
        self.norm5 = nn.LayerNorm(normalized_shape=dims)
        self.norm6 = nn.LayerNorm(normalized_shape=dims)

    def forward(self, a, b, t, pe=None, mask=None):
        da = self.norm1(a)
        db = self.norm2(b)
        dt = self.norm3(t)

        ka = da if pe is None else da + pe
        kb = db if pe is None else db + pe

        at = self.att1(dt, ka, da, mask=mask)
        bt = self.att2(dt, kb, db, mask=mask)

        t = t + at + bt
        dt = self.norm4(t)

        qa = da if pe is None else da + pe
        qb = db if pe is None else db + pe

        a = a + self.att3(qa, dt)
        b = b + self.att4(qb, dt)

        da = self.norm5(a)
        db = self.norm6(b)

        a = a + self.ffn1(da)
        b = b + self.ffn2(db)

        return a, b, t


class BottleneckTransformer(nn.Module):

    def __init__(self, dims, num_tokens=4, num_layers=1, **kwargs):
        super(BottleneckTransformer, self).__init__()

        self.dims = dims
        self.num_tokens = num_tokens
        self.num_layers = num_layers

        self.token = Parameter(num_tokens, dims)
        self.encoder = nn.ModuleList([
            BottleneckTransformerLayer(dims, **kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, a, b, **kwargs):
        t = self.token.expand(a.size(0), -1, -1)
        for enc in self.encoder:
            a, b, t = enc(a, b, t, **kwargs)
        return a,