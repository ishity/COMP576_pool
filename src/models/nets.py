"""Two-headed models: shared state encoder -> win head + candidate-shot head.

All models take the batch dict from ``src.data.dataset.to_tensors`` and return
``(win_logit (B,), shot_logit (B,K))``. ``use_skill=False`` removes skill information
(the skill-agnostic ablation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from src.data.render import TableRenderer

N_SKILLS = 3
BALL_F = 7
GLOB_F = 4


def mlp(sizes, act=nn.GELU, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


class TwoHeadBase(nn.Module):
    """Shared skill embeddings and the two heads. Subclasses implement ``encode``."""

    def __init__(self, z_dim: int, tok_dim: int, cand_f: int, use_skill: bool = True, skill_dim: int = 16):
        super().__init__()
        self.use_skill = use_skill
        self.skill_dim = skill_dim
        self.skill_emb = nn.Embedding(N_SKILLS, skill_dim)
        self.pocket_emb = nn.Embedding(6, 8)
        self.win_head = mlp([z_dim + 2 * skill_dim, 256, 1])
        self.shot_head = mlp([z_dim + tok_dim + 8 + cand_f + skill_dim, 256, 256, 1])

    def skills(self, b):
        s, o = self.skill_emb(b["skill_self"]), self.skill_emb(b["skill_opp"])
        if not self.use_skill:
            s, o = torch.zeros_like(s), torch.zeros_like(o)
        return s, o

    def encode(self, b, s, o):
        """Return state embedding z (B,Z) and per-ball tokens (B,16,T)."""
        raise NotImplementedError

    def forward(self, b):
        s, o = self.skills(b)
        z, toks = self.encode(b, s, o)
        win = self.win_head(torch.cat([z, s, o], -1)).squeeze(-1)
        K = b["cand_feats"].shape[1]
        tok = torch.gather(toks, 1, b["cand_ball"][..., None].expand(-1, -1, toks.shape[-1]))
        x = torch.cat(
            [
                z[:, None].expand(-1, K, -1),
                tok,
                self.pocket_emb(b["cand_pocket"]),
                b["cand_feats"],
                s[:, None].expand(-1, K, -1),
            ],
            -1,
        )
        return win, self.shot_head(x).squeeze(-1)


class MLPNet(TwoHeadBase):
    """Model 2: MLP on the flattened, padded set representation (fixed ball order)."""

    def __init__(self, cand_f, use_skill=True, hidden=512, z_dim=256):
        super().__init__(z_dim, BALL_F, cand_f, use_skill)
        self.enc = mlp([16 * (BALL_F + 1) + GLOB_F + 2 * self.skill_dim, hidden, hidden, z_dim], last_act=True)

    def encode(self, b, s, o):
        x = torch.cat([b["ball_feats"].flatten(1), b["ball_mask"], b["glob"], s, o], -1)
        return self.enc(x), b["ball_feats"]


class DeepSets(TwoHeadBase):
    """Permutation-invariant DeepSets encoder (masked sum + max pooling)."""

    def __init__(self, cand_f, use_skill=True, d=128, z_dim=256):
        super().__init__(z_dim, d, cand_f, use_skill)
        self.phi = mlp([BALL_F, d, d, d], last_act=True)
        self.rho = mlp([2 * d + GLOB_F + 2 * self.skill_dim, z_dim, z_dim], last_act=True)

    def encode(self, b, s, o):
        h = self.phi(b["ball_feats"])
        m = b["ball_mask"][..., None]
        hsum = (h * m).sum(1)
        hmax = (h * m - 1e4 * (1 - m)).max(1).values
        return self.rho(torch.cat([hsum, hmax, b["glob"], s, o], -1)), h


class SetTransformer(TwoHeadBase):
    """Main model: Transformer over ball tokens + one global (skill/context) token.

    Pocketed balls are masked out. The global token's output is the state embedding;
    ball-token outputs feed the shot head for the candidate's object ball.
    """

    def __init__(self, cand_f, use_skill=True, d=128, layers=4, heads=4, z_dim=256, dropout=0.0):
        super().__init__(z_dim, d, cand_f, use_skill)
        self.ball_in = mlp([BALL_F, d, d])
        self.glob_in = mlp([GLOB_F + 2 * self.skill_dim, d, d])
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, z_dim), nn.GELU())

    def encode(self, b, s, o):
        g = self.glob_in(torch.cat([b["glob"], s, o], -1))[:, None]
        x = torch.cat([g, self.ball_in(b["ball_feats"])], 1)
        pad = torch.cat([torch.zeros_like(b["ball_mask"][:, :1]), 1 - b["ball_mask"]], 1).bool()
        h = self.tr(x, src_key_padding_mask=pad)
        return self.out(h[:, 0]), h[:, 1:]


class CNNNet(TwoHeadBase):
    """Model 4: ResNet-18 on the rendered (5,256,128) top-down image."""

    def __init__(self, cand_f, use_skill=True, z_dim=256, height=256, width=128):
        super().__init__(z_dim, BALL_F, cand_f, use_skill)
        self.render = TableRenderer(height, width)
        net = torchvision.models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        net.fc = nn.Identity()
        self.cnn = net
        self.proj = mlp([512 + GLOB_F + 2 * self.skill_dim, z_dim, z_dim], last_act=True)

    def encode(self, b, s, o):
        img = self.render(b["pos"], b["ball_feats"][..., 2:], b["ball_mask"])
        f = self.cnn(img)
        return self.proj(torch.cat([f, b["glob"], s, o], -1)), b["ball_feats"]


MODELS = {"mlp": MLPNet, "deepsets": DeepSets, "set_transformer": SetTransformer, "cnn": CNNNet}


def build_model(name: str, cand_f: int, **kw) -> TwoHeadBase:
    """Instantiate a model by name from the config."""
    return MODELS[name](cand_f, **kw)
