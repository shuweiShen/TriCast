"""TriCast components; MSDR follows equations (1)-(12) of TriCast.docx."""
import torch
from torch import nn
from torch.nn import functional as F


class MSDR(nn.Module):
    def __init__(self, scales=(16, 32, 64), max_lag=8, rho_floor=0.1, eps=1e-6):
        super().__init__()
        if not scales or min(scales) < 2 or max_lag < 1 or rho_floor <= 0:
            raise ValueError('Invalid MSDR scales, lag or correlation floor')
        self.scales, self.max_lag = tuple(scales), max_lag
        self.rho_floor, self.eps = rho_floor, eps

    def _stats(self, w):
        median = w.median(-1).values
        sigma = 1.4826 * (w - median.unsqueeze(-1)).abs().median(-1).values
        centered = w - w.mean(-1, keepdim=True)
        denominator = centered.square().sum(-1).clamp_min(self.eps)
        acf = torch.stack([(centered[..., :-lag] * centered[..., lag:]).sum(-1)
                           / denominator for lag in range(1, min(self.max_lag, w.size(-1) // 2) + 1)], -1)
        return median, sigma, acf

    @torch.no_grad()
    def forward(self, x):
        # x: [batch, time, variable]. Both windows are inside the observed input.
        # Boundary positions use replicated observations, never forecast targets.
        y = x.detach().float().transpose(1, 2)
        global_median = y.median(-1, keepdim=True).values
        noise = 1.4826 * (y - global_median).abs().median(-1, keepdim=True).values
        responses = []
        for h in self.scales:
            padded = F.pad(y, (h, h), mode='replicate')
            windows = padded.unfold(-1, 2 * h + 1, 1)
            ml, sl, al = self._stats(windows[..., :h])
            mr, sr, ar = self._stats(windows[..., h + 1:])
            level = (mr - ml).abs() / (sl.square() + sr.square() + noise.square() + self.eps).sqrt()
            volatility = ((sr + self.eps) / (sl + self.eps)).log().abs()
            structure = ((ar - al).abs() / torch.maximum(ar.abs(), al.abs()).clamp_min(self.rho_floor)).mean(-1)
            responses.append(((level.square() + volatility.square() + structure.square()) / 3).sqrt())
        responses = torch.stack(responses, -1)
        mean = responses.mean(-1)
        energy = mean * torch.exp(-responses.std(-1, unbiased=False) / (mean + self.eps))
        q25, q75 = torch.quantile(energy, torch.tensor([0.25, 0.75], device=x.device), dim=-1, keepdim=True)
        intensity = -torch.expm1(-(energy - q75).clamp_min(0) / (q75 - q25).clamp_min(self.eps))
        return intensity.transpose(1, 2).contiguous()


class PatchEncoder(nn.Module):
    def __init__(self, patch_len, dim):
        super().__init__()
        self.patch_len = patch_len
        self.projection = nn.Linear(patch_len, dim)

    def forward(self, x):
        y = x.transpose(1, 2)
        pad = (-y.size(-1)) % self.patch_len
        if pad:
            y = F.pad(y, (0, pad), mode='replicate')
        patches = y.unfold(-1, self.patch_len, self.patch_len)
        return self.projection(patches.flatten(0, 1))


class SpectralEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(2, dim)

    def forward(self, x):
        spectrum = torch.fft.rfft(x.float().transpose(1, 2), dim=-1, norm='ortho')
        return self.projection(torch.view_as_real(spectrum).flatten(0, 1))


class AlignBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * dim, dim))

    def forward(self, x):
        return x + self.ffn(x)


class FiLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim * 2))
        self.norm = nn.LayerNorm(dim)

    def forward(self, time, cp):
        gamma, beta = self.condition(cp).chunk(2, -1)
        return self.norm(time * (1 + gamma.tanh()) + beta)


class FourierCrossAttn(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, time, spectrum):
        filtered = spectrum * self.gate(time.mean(1, keepdim=True)).sigmoid()
        mixed, _ = self.attention(time, filtered, filtered, need_weights=False)
        return self.norm(time + mixed)


class CPRouter(nn.Module):
    def __init__(self, dim, hidden_dim, experts=4, dropout=0.1):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.router = nn.Linear(dim, experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(dim, hidden_dim)) for _ in range(experts)])
        self.last_weights = None

    def forward(self, hidden, cp, spectrum):
        prior = self.condition(torch.cat((cp.mean(1), spectrum.mean(1)), -1))
        weights = self.router(prior).softmax(-1)
        self.last_weights = weights.detach()
        return sum(expert(hidden) * weights[:, i, None, None] for i, expert in enumerate(self.experts))
