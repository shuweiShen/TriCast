"""Staged TriCast with configurable local frozen language backbones."""
import math
import torch
from torch import nn
from utils.llm_backbone import load_backbone, resolve_snapshot, PROJECT_ROOT
from layers.TriCast import MSDR, PatchEncoder, SpectralEncoder, AlignBlock, FiLM, FourierCrossAttn, CPRouter


def local_gpt2_path(path=None):
    return str(resolve_snapshot(path or PROJECT_ROOT / 'models--openai-community--gpt2'))


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.stage = getattr(configs, 'tricast_stage', 5)
        if self.stage not in range(6):
            raise ValueError('tricast_stage must be 0..5')
        self.seq_len, self.pred_len = configs.seq_len, configs.pred_len
        self.enc_in = configs.enc_in
        dim, patch = configs.d_model, configs.patch_len
        if patch < 1 or configs.seq_len < 1:
            raise ValueError('Sequence and patch lengths must be positive')
        self.llm_model, cfg, self.backbone_identity = load_backbone(configs)
        self.backbone_dtype = next(self.llm_model.parameters()).dtype
        hidden_size = cfg.hidden_size
        self.patch_nums = math.ceil(self.seq_len / patch)
        if self.patch_nums > cfg.max_position_embeddings:
            raise ValueError('Patch count exceeds LLM context length')
        self.time_encoder = PatchEncoder(patch, dim)
        if self.stage >= 1:
            self.msdr = MSDR(getattr(configs, 'msdr_scales', (16, 32, 64)),
                             getattr(configs, 'msdr_max_lag', 8), getattr(configs, 'msdr_rho_floor', 0.1))
            self.cp_encoder = PatchEncoder(patch, dim)
        if self.stage >= 2:
            self.spectral_encoder = SpectralEncoder(dim)
            self.align_time, self.align_cp, self.align_fft = [AlignBlock(dim, configs.dropout) for _ in range(3)]
        if self.stage >= 3:
            self.film = FiLM(dim)
        if self.stage >= 4:
            self.fourier = FourierCrossAttn(dim, configs.n_heads, configs.dropout)
        if self.stage >= 5:
            self.cp_router = CPRouter(dim, hidden_size, getattr(configs, 'num_experts', 4), configs.dropout)
        self.to_llm = nn.Linear(dim, hidden_size)
        self.readout = nn.Sequential(nn.Linear(hidden_size, configs.d_ff), nn.GELU(), nn.Dropout(configs.dropout))
        self.head = nn.Linear(self.patch_nums * configs.d_ff, self.pred_len)
        self.last_cp = None

    def train(self, mode=True):
        super().train(mode)
        self.llm_model.eval()
        return self

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected [batch, seq_len, enc_in] input')
        # Instance statistics are from the observed input only.
        mean = x_enc.mean(1, keepdim=True).detach()
        scale = (x_enc.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        x = (x_enc - mean) / scale
        time = self.time_encoder(x)
        if self.stage >= 1:
            cp = self.msdr(x)
            self.last_cp = cp.detach()
            cp = self.cp_encoder(cp)
        if self.stage >= 2:
            spectrum = self.align_fft(self.spectral_encoder(x))
            time, cp = self.align_time(time), self.align_cp(cp)
        if self.stage >= 3:
            time = self.film(time, cp)
        if self.stage >= 4:
            time = self.fourier(time, spectrum)
        # Do not use no_grad here: gradients must reach the input adapters.
        embeddings = self.to_llm(time).to(self.backbone_dtype)
        hidden = self.llm_model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
        hidden = hidden.to(self.to_llm.weight.dtype)
        if self.stage >= 5:
            hidden = self.cp_router(hidden, cp, spectrum)
        output = self.head(self.readout(hidden).flatten(1))
        output = output.reshape(x.size(0), self.enc_in, self.pred_len).transpose(1, 2)
        return output * scale + mean
