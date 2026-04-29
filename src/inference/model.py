import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.0, **kwargs):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            d_conv,
            groups=self.d_inner,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()
        self.x_proj_weight = nn.Parameter(
            torch.empty(4, (self.dt_rank + d_state * 2), self.d_inner)
        )
        self.dt_projs_weight = nn.Parameter(torch.empty(4, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(4, self.d_inner))
        self.A_logs = nn.Parameter(
            torch.log(
                repeat(
                    torch.arange(1, d_state + 1, dtype=torch.float32),
                    "n -> d n",
                    d=self.d_inner,
                )
            )
            .flatten(0, 1)
            .repeat(4, 1)
        )
        self.Ds = nn.Parameter(torch.ones(self.d_inner * 4))
        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        nn.init.xavier_uniform_(self.x_proj_weight)
        nn.init.xavier_uniform_(self.dt_projs_weight)
        nn.init.zeros_(self.dt_projs_bias)

    def forward(self, x):
        B, H, W, C = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))
        L = H * W
        xh = torch.stack(
            [
                x.reshape(B, -1, L),
                torch.transpose(x, 2, 3).contiguous().reshape(B, -1, L),
            ],
            1,
        ).reshape(B, 2, -1, L)
        xs = torch.cat([xh, torch.flip(xh, [-1])], 1)
        if self.selective_scan:
            xd = torch.einsum(
                "bkdl,kcd->bkcl", xs.reshape(B, 4, -1, L), self.x_proj_weight
            )
            dts, Bs, Cs = torch.split(xd, [self.dt_rank, 16, 16], 2)
            dts = torch.einsum(
                "bkrl,kdr->bkdl", dts.reshape(B, 4, -1, L), self.dt_projs_weight
            )
            oy = self.selective_scan(
                xs.float().reshape(B, -1, L),
                dts.float().reshape(B, -1, L),
                -torch.exp(self.A_logs.float()).reshape(-1, 16),
                Bs.float().reshape(B, 4, -1, L),
                Cs.float().reshape(B, 4, -1, L),
                self.Ds.float().reshape(-1),
                z=None,
                delta_bias=self.dt_projs_bias.float().reshape(-1),
                delta_softplus=True,
            ).reshape(B, 4, -1, L)
            y = (
                oy[:, 0]
                + torch.flip(oy[:, 2], [-1])
                + torch.transpose(
                    (oy[:, 1] + torch.flip(oy[:, 3], [-1])).reshape(B, -1, W, H), 2, 3
                ).reshape(B, -1, L)
            )
        else:
            y = xs.sum(1).reshape(B, -1, L)
        y = self.out_norm(torch.transpose(y, 1, 2).reshape(B, H, W, -1))
        y = y * F.silu(z)
        return self.out_proj(y)


class SF_Block(nn.Module):
    def __init__(self, c, H, W):
        super().__init__()
        self.sp = nn.Parameter(torch.randn(H, W // 2 + 1, c, 2) * 0.02)
        self.ma = SS2D(c)
        self.norm = nn.LayerNorm(c)
        self.fuse = nn.Conv2d(c * 2, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x.float(), norm="ortho")
        xf = xf * torch.view_as_complex(self.sp).permute(2, 0, 1).unsqueeze(0)
        xs = torch.fft.irfft2(xf, s=(H, W), norm="ortho")
        xm = self.ma(self.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return x + self.fuse(torch.cat([xs + x, xm + x], 1))


class MemoryBlock(nn.Module):
    def __init__(self, c, nr, nm, H, W):
        super().__init__()
        self.blocks = nn.ModuleList([SF_Block(c, H, W) for _ in range(nr)])
        self.gate = nn.Conv2d((nr + nm) * c, c, 1)

    def forward(self, x, ys):
        xs = []
        for b in self.blocks:
            x = b(x) + x
            xs.append(x)
        g = self.gate(torch.cat(xs + ys, 1))
        ys.append(g)
        return g


class SS_UIE_model(nn.Module):
    def __init__(self, c=32, nm=3, nr=3, H=256, W=256):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(3, c, 3, 1, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1),
        )

        self.e2 = nn.Sequential(
            nn.Conv2d(c, c * 2, 3, 1, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),
            nn.Conv2d(c * 2, c * 2, 3, 1, 1),
        )

        self.e3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, 3, 1, 1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),
            nn.Conv2d(c * 4, c * 4, 3, 1, 1),
        )

        self.p = nn.MaxPool2d(2, 2)
        self.m = nn.ModuleList(
            [MemoryBlock(c * 4, nr, i + 1, H // 4, W // 4) for i in range(nm)]
        )

        self.fuse_attn = nn.Sequential(
            nn.Conv2d(c * 4 * nm, c * 4 * nm, 1), nn.Sigmoid()
        )

        self.f = nn.Conv2d(c * 4 * nm, c * 4, 3, 1, 1)
        self.up1 = nn.Conv2d(c * 4, c * 2, 3, 1, 1)
        self.up2 = nn.Conv2d(c * 2, c, 3, 1, 1)
        self.skip1 = nn.Conv2d(c * 2, c * 2, 1)
        self.skip2 = nn.Conv2d(c, c, 1)

        self.out = nn.Conv2d(c, 3, 3, 1, 1)

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        r0 = x

        x1 = self.e1(x)
        x2 = self.p(self.e2(x1))
        x3 = self.p(self.e3(x2))

        ys = [x3]
        feats = []

        for b in self.m:
            x3 = b(x3, ys)
            feats.append(x3)

        feat_cat = torch.cat(feats, dim=1)
        attn = self.fuse_attn(feat_cat)
        p = self.f(feat_cat * attn)

        p = F.interpolate(p, scale_factor=2)
        p = self.up1(p) + self.skip1(x2)

        p = F.interpolate(p, scale_factor=2)
        p = self.up2(p) + self.skip2(x1)

        out = self.out(p)

        return r0 + self.alpha * out
