# models/diffusion_head.py
import torch, torch.nn as nn

class MotionLatent(nn.Module):
    def __init__(self, pose_dim=33*2, zdim=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(pose_dim,256), nn.SiLU(), nn.Linear(256,zdim))
        self.dec = nn.Sequential(nn.Linear(zdim,256), nn.SiLU(), nn.Linear(256,pose_dim))
    def encode(self, P): return self.enc(P)    # [B,T,pose_dim]→[B,T,zdim]
    def decode(self, Z): return self.dec(Z)

class EpsilonHead(nn.Module):
    def __init__(self, zdim=128, d_model=256, nhead=8, nlayers=4, cond_dim=3):
        super().__init__()
        self.t_embed = nn.Embedding(1024, d_model)
        self.inp = nn.Linear(zdim+cond_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.out = nn.Linear(d_model, zdim)

    def forward(self, zt, t_idx, cond):  # [B,T,z], [B] or [B,T], [B,T,cond_dim]
        h = torch.cat([zt, cond], dim=-1)
        h = self.inp(h) + self.t_embed(t_idx).unsqueeze(1)
        h = self.tr(h)
        return self.out(h)   # ε̂
