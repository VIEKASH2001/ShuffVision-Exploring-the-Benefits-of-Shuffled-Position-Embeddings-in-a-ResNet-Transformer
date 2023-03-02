import torch
import torch.nn as nn
from math import log
from collections import OrderedDict
import torch.nn.functional as F

from .transformer import PositionalEncoding, TransformerEncoderLayer, TransformerDecoderLayer, BottleneckTransformer
from .transformer_component import MultiHeadAttention
from .utils.model_utils import ModuleList



class TEncoder(nn.Module):

    def __init__(self,
                 dims=None,
                 p=0.5,
                 norm_cfg=None,
                 **kwargs):
        super(TEncoder, self).__init__()

        drop_cfg = dict(type='drop', p=p) if p > 0 else None
        enc_dims = dims[-1] if isinstance(dims, (list, tuple)) else dims

        self.dropout = nn.Dropout(p=p)
        self.mapping = nn.Sequential(
            nn.Linear(in_features=dims[0], out_features=dims[-1], bias=True),
        )
        self.pos_enc = PositionalEncoding(dims=enc_dims)


        self.encoder = TransformerEncoderLayer(dims=enc_dims)


        self.norm = None
        if norm_cfg:
            self.norm = nn.LayerNorm(normalized_shape=dims)

    def forward(self, x, **kwargs):
        if self.dropout is not None:
            x = self.dropout(x)
        if self.mapping is not None:
            x = self.mapping(x)
        if self.encoder is not None:
            pe = None if self.pos_enc is None else self.pos_enc(x)
            x = self.encoder(x, pe=pe, **kwargs)
        if self.norm is not None:
            x = self.norm(x)
        return x


class CrossModalEncoder(nn.Module):

    def __init__(self,
                 dims=None,
                 fusion_type='sum',
                 pos_cfg=None,
                 enc_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        super(CrossModalEncoder, self).__init__()
        assert fusion_type in ('sum', 'mean', 'concat')

        map_dims = [2 * dims, dims] if fusion_type == 'concat' else None
        self.fusion_type = fusion_type

        self.pos_enc = PositionalEncoding(dims=dims)

        self.encoder = None
        if isinstance(enc_cfg, dict):
            if enc_cfg['type'] == "MultiHeadAttention":
                self.encoder = MultiHeadAttention(dims=dims)
            elif enc_cfg['type'] == "TransformerEncoderLayer":
                self.encoder = TransformerEncoderLayer(dims=dims)
            elif enc_cfg['type'] == 'BottleneckTransformer':
                self.encoder = BottleneckTransformer(dims=dims)

        elif isinstance(enc_cfg, list):
            encs = []
            for cfg in enc_cfg:
                encT = None
                if enc_cfg['type'] == "MultiHeadAttention":
                    encT = MultiHeadAttention(dims=dims)
                elif enc_cfg['type'] == "TransformerEncoderLayer":
                    encT = TransformerEncoderLayer(dims=dims)
                elif enc_cfg['type'] == 'BottleneckTransformer':
                    encT = BottleneckTransformer(dims=dims)
                encs.append(encT)
            self.encoder = ModuleList(encs)

        self.mapping = None
        self.norm = nn.LayerNorm(normalized_shape=dims)

    def forward(self, a, b, **kwargs):
        if self.encoder is not None:
            pe = None if self.pos_enc is None else self.pos_enc(a)
            a, b = self.encoder(a, b, pe=pe, **kwargs)
        if self.fusion_type in ('sum', 'mean'):
            x = (a + b) / ((self.fusion_type == 'mean') + 1)
        else:
            x = torch.cat((a, b), dim=-1)
            x = self.mapping(x)
        if self.norm is not None:
            x = self.norm(x)
        return