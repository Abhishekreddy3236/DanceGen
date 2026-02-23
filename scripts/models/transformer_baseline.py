# models/transformer_baseline.py
import torch, torch.nn as nn

class PoseTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, nlayers=8, in_dim=33*2+3, out_dim=33*2):
        super().__init__()
        self.input = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.out  = nn.Linear(d_model, out_dim)

    def forward(self, x):   # x: [B,T,in_dim]   (concat pose_{t-1} + beat_t)
        h = self.input(x)
        h = self.enc(h)
        y = self.out(h)     # [B,T,out_dim]  (Ŷ_pose)
        return y
