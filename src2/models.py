import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import dgl
from dgl.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src2.dlutils import *

torch.manual_seed(1)


# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
    def __init__(self, feats, cfg):
        super(TranAD, self).__init__()
        self.name = cfg["model"]["name"]
        self.lr = cfg["training"]["optimizer"]["lr"]

        mcfg = cfg.get("model", {})

        self.batch = 128
        self.n_feats = feats
        self.n_window = mcfg.get("n_window", 10)
        self.n = self.n_feats * self.n_window

        d_model_factor = mcfg.get("d_model_factor", 2)
        self.d_model = d_model_factor * feats

        nhead_factor = mcfg.get("nhead_factor", 1)  # 기본: feats
        self.nhead = mcfg.get("nhead", nhead_factor * feats)

        self.dim_feedforward = mcfg.get("dim_feedforward", 16)
        self.dropout = mcfg.get("dropout", 0.1)
        self.num_encoder_layers = mcfg.get("num_encoder_layers", 1)
        self.num_decoder_layers = mcfg.get("num_decoder_layers", 1)

        # 3) Positional Encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout, self.n_window)

        # 4) Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, self.num_encoder_layers
        )

        # 5) Decoder 1
        decoder_layer1 = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        self.transformer_decoder1 = TransformerDecoder(
            decoder_layer1, self.num_decoder_layers
        )

        # 6) Decoder 2
        decoder_layer2 = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        self.transformer_decoder2 = TransformerDecoder(
            decoder_layer2, self.num_decoder_layers
        )

        self.fcn = nn.Sequential(nn.Linear(self.d_model, self.n_feats), nn.Sigmoid())
        self.input_proj = nn.Linear(2 * self.n_feats, self.d_model)

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = self.input_proj(src)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)

        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        tgt = self.input_proj(tgt)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2
