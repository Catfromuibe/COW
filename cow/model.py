"""Conditional speech world model, trajectory evidence, and fused detector.

The modules follow the method section:

* frozen interval encoder, phonetic condition, and quality descriptor
* genuine-only conditional Gaussian-mixture rollouts
* joint random-Fourier trajectory residuals
* capped consistency on matched neighborhoods
* dynamics score fused with a bounded acoustic score
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_ball_(param: torch.Tensor, radius: float) -> None:
    if radius <= 0:
        return
    norm = param.detach().norm()
    if float(norm) > radius:
        param.mul_(radius / (norm + 1e-8))


class IntervalEncoder(nn.Module):
    """Frozen local encoder. Independent blocks cannot see future intervals."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(64, dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, intervals: torch.Tensor) -> torch.Tensor:
        # intervals: (N, samples)
        x = self.net(intervals.unsqueeze(1)).squeeze(-1)
        return x


class StateStandardizer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0))
        self.frozen = False

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        if self.frozen:
            return
        flat = z.detach().reshape(-1, z.shape[-1])
        batch = flat.shape[0]
        if batch == 0:
            return
        mean = flat.mean(dim=0)
        var = flat.var(dim=0, unbiased=False)
        total = self.count + batch
        delta = mean - self.mean
        self.mean.add_(delta * batch / total.clamp_min(1))
        self.var.mul_(self.count / total.clamp_min(1)).add_(var * batch / total.clamp_min(1))
        self.count.copy_(total)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mean) / self.var.clamp_min(1e-4).sqrt()


class Conditioners(nn.Module):
    """Fixed phonetic codebook and history-only quality descriptors."""

    def __init__(self, n_phones: int = 40, n_mels: int = 40):
        super().__init__()
        self.n_phones = n_phones
        self.unk = n_phones
        proj = torch.randn(n_mels, n_phones)
        proj = F.normalize(proj, dim=0)
        self.register_buffer("codebook", proj)
        mel = torch.linspace(0, 1, n_mels).unsqueeze(1)
        basis = torch.arange(1, n_mels + 1).float().unsqueeze(0)
        bank = torch.exp(-0.5 * ((mel * n_mels - basis) / 4.0) ** 2)
        self.register_buffer("mel_bank", F.normalize(bank, dim=0))

    @torch.no_grad()
    def phonemes(self, intervals: torch.Tensor) -> torch.Tensor:
        # intervals: (N, samples) -> token ids, unk when the code is ambiguous
        spec = torch.stft(
            intervals,
            n_fft=256,
            hop_length=128,
            win_length=256,
            window=torch.hann_window(256, device=intervals.device),
            return_complex=True,
        )
        power = spec.abs().mean(dim=-1)[:, 1 : 1 + self.mel_bank.shape[0]]
        if power.shape[-1] != self.mel_bank.shape[0]:
            power = F.interpolate(power.unsqueeze(1), size=self.mel_bank.shape[0], mode="linear", align_corners=False).squeeze(1)
        mel = torch.log(power.clamp_min(1e-6)) @ self.mel_bank.to(power.dtype)
        scores = F.normalize(mel, dim=-1) @ self.codebook.to(mel.dtype)
        conf, idx = scores.max(dim=-1)
        idx = idx.masked_fill(conf < 0.15, self.unk)
        return idx

    @staticmethod
    def quality(history: torch.Tensor) -> torch.Tensor:
        # history waveform (B, S, samples) -> (B, S, 2): level, noise-like ratio
        level = history.pow(2).mean(dim=-1).clamp_min(1e-8).log()
        diff = history[..., 1:] - history[..., :-1]
        noise = diff.pow(2).mean(dim=-1) / history.pow(2).mean(dim=-1).clamp_min(1e-8)
        return torch.stack([level, noise.clamp(0, 50).log1p()], dim=-1)


class TransitionModel(nn.Module):
    """Causal mixture transition p(z_{t+h} | history, phonetic schedule, quality, h)."""

    def __init__(self, dim: int, n_phones: int, components: int, horizon: int, hidden: int = 128):
        super().__init__()
        self.dim = dim
        self.components = components
        self.horizon = horizon
        self.phone_emb = nn.Embedding(n_phones + 1, 32)
        self.step_emb = nn.Embedding(horizon + 1, 16)
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        in_dim = hidden + 32 + 2 + 16
        self.mix = nn.Linear(in_dim, components)
        self.mu = nn.Linear(in_dim, components * dim)
        self.log_var = nn.Linear(in_dim, components * dim)

    def _features(self, history: torch.Tensor, phones: torch.Tensor, quality: torch.Tensor, step: int) -> torch.Tensor:
        h, _ = self.gru(history)
        h = h[:, -1]
        p = self.phone_emb(phones).mean(dim=1)
        s = self.step_emb.weight[step].expand(history.shape[0], -1)
        return torch.cat([h, p, quality, s], dim=-1)

    def params(self, history, phones, quality, step):
        feat = self._features(history, phones, quality, step)
        b = history.shape[0]
        logits = self.mix(feat)
        mu = self.mu(feat).view(b, self.components, self.dim)
        log_var = self.log_var(feat).view(b, self.components, self.dim).clamp(-6.0, 1.5)
        return logits, mu, log_var

    def nll(self, history, phones, quality, target, step) -> torch.Tensor:
        logits, mu, log_var = self.params(history, phones, quality, step)
        var = log_var.exp()
        diff = target.unsqueeze(1) - mu
        log_comp = -0.5 * ((diff.pow(2) / var).sum(-1) + log_var.sum(-1) + self.dim * math.log(2 * math.pi))
        log_prob = torch.logsumexp(log_comp + F.log_softmax(logits, dim=-1), dim=-1)
        return -log_prob

    @torch.no_grad()
    def sample(self, history, phones, quality, step) -> torch.Tensor:
        logits, mu, log_var = self.params(history, phones, quality, step)
        choice = torch.distributions.Categorical(logits=logits).sample()
        idx = choice.view(-1, 1, 1).expand(-1, 1, self.dim)
        mean = mu.gather(1, idx).squeeze(1)
        std = log_var.gather(1, idx).squeeze(1).exp().sqrt()
        return mean + std * torch.randn_like(mean)


class RandomFourierPair(nn.Module):
    """Unit-norm product features psi(V) ⊗ phi_H(Y)."""

    def __init__(self, v_dim: int, y_dim: int, freqs: int, bandwidth: float):
        super().__init__()
        self.register_buffer("w_v", torch.randn(v_dim, freqs) / bandwidth)
        self.register_buffer("w_y", torch.randn(y_dim, freqs) / bandwidth)
        self.register_buffer("b_v", torch.empty(freqs).uniform_(0, 2 * math.pi))
        self.register_buffer("b_y", torch.empty(freqs).uniform_(0, 2 * math.pi))

    def _map(self, x, w, b):
        proj = x @ w + b
        feat = torch.cat([proj.sin(), proj.cos()], dim=-1)
        return F.normalize(feat, dim=-1)

    def forward(self, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        psi = self._map(context, self.w_v, self.b_v)
        phi = self._map(future, self.w_y, self.b_y)
        return (psi.unsqueeze(-1) * phi.unsqueeze(-2)).flatten(-2)


class AcousticBranch(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64, 7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(64, dim, 5, stride=4, padding=2),
            nn.GELU(),
        )
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        feat = self.net(wave.unsqueeze(1))
        pooled = torch.cat([feat.mean(-1), feat.std(-1).nan_to_num(0.0)], dim=-1)
        hidden = self.proj(pooled)
        return torch.tanh(hidden) / math.sqrt(self.dim)


class COW(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        interval: int = 1600,
        history: int = 8,
        horizon: int = 4,
        horizons: tuple[int, ...] = (1, 2, 4),
        rollouts: int = 4,
        components: int = 3,
        rff: int = 16,
        evidence_dim: int = 64,
        neighborhood: int = 3,
        alpha: float = 0.5,
        tau: float = 1.0,
        head_dyn: float = 2.0,
        head_ac: float = 2.0,
        bandwidth: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.interval = interval
        self.history = history
        self.horizon = horizon
        self.horizons = tuple(h for h in horizons if h <= horizon)
        self.rollouts = rollouts
        self.neighborhood = neighborhood
        self.alpha = alpha
        self.tau = tau
        self.head_dyn = head_dyn
        self.head_ac = head_ac
        self.evidence_dim = evidence_dim

        self.encoder = IntervalEncoder(dim)
        self.standardizer = StateStandardizer(dim)
        self.conditions = Conditioners()
        self.world = TransitionModel(dim, n_phones=40, components=components, horizon=horizon)
        v_dim = dim + 32 + 2
        self.context_phone = nn.Embedding(41, 32)
        for p in self.context_phone.parameters():
            p.requires_grad_(False)
        self.rff = nn.ModuleDict(
            {str(h): RandomFourierPair(v_dim, h * dim, rff, bandwidth) for h in self.horizons}
        )
        r_dim = len(self.horizons) * ((2 * rff) ** 2 + 1)
        self.adapter = nn.Sequential(
            nn.Linear(r_dim, evidence_dim),
            nn.GELU(),
            nn.Linear(evidence_dim, evidence_dim),
        )
        self.acoustic = AcousticBranch(dim)
        self.dyn_head = nn.Linear(evidence_dim, 1)
        self.ac_head = nn.Linear(dim, 1)

    def encode_states(self, wave: torch.Tensor, update_stats: bool = False):
        b, t = wave.shape
        n = t // self.interval
        if n < self.history + self.horizon + 1:
            raise RuntimeError(
                f"Recording of {t} samples is shorter than history+horizon. "
                f"Need at least {(self.history + self.horizon + 1) * self.interval} samples."
            )
        chunks = wave[:, : n * self.interval].reshape(b, n, self.interval)
        flat = chunks.reshape(b * n, self.interval)
        with torch.no_grad():
            z = self.encoder(flat).view(b, n, self.dim)
            if update_stats:
                self.standardizer.update(z)
            z = self.standardizer(z)
            phones = self.conditions.phonemes(flat).view(b, n)
        return z, phones, chunks

    def _starts(self, n: int, device):
        first = self.history - 1
        last = n - self.horizon - 1
        return torch.arange(first, last + 1, device=device)

    def context_and_future(self, z, phones, chunks, starts):
        hist, fut, phon, qual = [], [], [], []
        for t in starts.tolist():
            hist.append(z[:, t - self.history + 1 : t + 1])
            fut.append(z[:, t + 1 : t + 1 + self.horizon])
            phon.append(phones[:, t + 1 : t + 1 + self.horizon])
            qual.append(self.conditions.quality(chunks[:, t - self.history + 1 : t + 1]))
        return (
            torch.stack(hist, dim=1),
            torch.stack(fut, dim=1),
            torch.stack(phon, dim=1),
            torch.stack(qual, dim=1),
        )

    def world_nll(self, wave: torch.Tensor) -> torch.Tensor:
        z, phones, chunks = self.encode_states(wave, update_stats=True)
        starts = self._starts(z.shape[1], z.device)
        hist, fut, phon, qual = self.context_and_future(z, phones, chunks, starts)
        # hist (B,S,L,d), fut (B,S,H,d), phon (B,S,H), qual (B,S,L,2)
        losses = []
        b, s, _, _ = hist.shape
        running = hist.reshape(b * s, self.history, self.dim)
        phon_f = phon.reshape(b * s, self.horizon)
        qual_f = qual.mean(dim=2).reshape(b * s, 2)
        for h in range(self.horizon):
            target = fut[:, :, h].reshape(b * s, self.dim)
            losses.append(self.world.nll(running, phon_f, qual_f, target, h + 1))
            nxt = target.unsqueeze(1)
            running = torch.cat([running[:, 1:], nxt], dim=1)
        return torch.stack(losses, dim=0).mean()

    def _context_vector(self, hist, phon, qual):
        # hist (N,L,d)
        summary = hist.mean(dim=1)
        p = self.context_phone(phon).mean(dim=1)
        q = qual.mean(dim=1)
        return torch.cat([summary, p, q], dim=-1)

    def _rollouts(self, hist, phon, qual) -> torch.Tensor:
        # returns (N, K, H, d)
        n = hist.shape[0]
        k = self.rollouts
        running = hist.unsqueeze(1).expand(-1, k, -1, -1).reshape(n * k, self.history, self.dim)
        phon_k = phon.unsqueeze(1).expand(-1, k, -1).reshape(n * k, self.horizon)
        qual_k = qual.mean(dim=1).unsqueeze(1).expand(-1, k, -1).reshape(n * k, 2)
        outs = []
        for h in range(self.horizon):
            nxt = self.world.sample(running, phon_k, qual_k, h + 1)
            outs.append(nxt)
            running = torch.cat([running[:, 1:], nxt.unsqueeze(1)], dim=1)
        traj = torch.stack(outs, dim=1).view(n, k, self.horizon, self.dim)
        return traj

    def trajectory_evidence(self, wave: torch.Tensor) -> torch.Tensor:
        z, phones, chunks = self.encode_states(wave, update_stats=False)
        starts = self._starts(z.shape[1], z.device)
        hist, fut, phon, qual = self.context_and_future(z, phones, chunks, starts)
        b, s, l, d = hist.shape
        flat_hist = hist.reshape(b * s, l, d)
        flat_fut = fut.reshape(b * s, self.horizon, d)
        flat_phon = phon.reshape(b * s, self.horizon)
        flat_qual = qual.reshape(b * s, l, 2)
        with torch.no_grad():
            rollout = self._rollouts(flat_hist, flat_phon, flat_qual)
        context = self._context_vector(flat_hist, flat_phon, flat_qual)
        pieces = []
        for h in self.horizons:
            obs = flat_fut[:, :h].reshape(b * s, h * d)
            sim = rollout[:, :, :h].reshape(b * s, self.rollouts, h * d)
            feat_obs = self.rff[str(h)](context, obs)
            feat_sim = self.rff[str(h)](
                context.unsqueeze(1).expand(-1, self.rollouts, -1).reshape(b * s * self.rollouts, -1),
                sim.reshape(b * s * self.rollouts, h * d),
            )
            feat_sim = feat_sim.view(b * s, self.rollouts, -1).mean(dim=1)
            residual = feat_obs - feat_sim
            pieces.append(residual)
            pieces.append(residual.norm(dim=-1, keepdim=True))
        residual = torch.cat(pieces, dim=-1).view(b, s, -1)
        if residual.shape[-1] != self.adapter[0].in_features:
            raise RuntimeError(
                f"residual dim {residual.shape[-1]} does not match adapter {self.adapter[0].in_features}"
            )
        width = min(self.neighborhood, s)
        pooled = residual.unfold(1, width, 1).mean(dim=-1)
        evidence = torch.tanh(self.adapter(pooled)) / math.sqrt(self.evidence_dim)
        return evidence

    def scores(self, wave: torch.Tensor) -> dict[str, torch.Tensor]:
        evidence = self.trajectory_evidence(wave)
        pooled = evidence.mean(dim=1)
        dyn = self.dyn_head(pooled).squeeze(-1)
        acoustic = self.acoustic(wave)
        ac = self.ac_head(acoustic).squeeze(-1)
        fused = self.alpha * dyn + (1.0 - self.alpha) * ac
        return {"evidence": evidence, "dyn": dyn, "ac": ac, "fused": fused, "acoustic": acoustic}

    def detector_loss(self, wave, paired, labels, matched) -> dict[str, torch.Tensor]:
        left = self.scores(wave)
        right = self.scores(paired)
        bce = F.binary_cross_entropy_with_logits
        fuse = bce(left["fused"], labels) + bce(right["fused"], labels)
        dyn = bce(left["dyn"], labels) + bce(right["dyn"], labels)
        ac = bce(left["ac"], labels) + bce(right["ac"], labels)
        # Neighborhoods stay aligned because processors keep the waveform length.
        dist = (left["evidence"] - right["evidence"]).norm(dim=-1)
        cap = torch.minimum(dist.pow(2), torch.full_like(dist, self.tau ** 2))
        # matched is per recording; unmatched pairs contribute nothing.
        weight = matched.view(-1, 1)
        denom = weight.sum() * cap.shape[1]
        con = (cap * weight).sum() / denom.clamp_min(1.0)
        saturated = ((dist > self.tau).float() * weight).sum() / denom.clamp_min(1.0)
        return {
            "fuse": fuse,
            "dyn": dyn,
            "ac": ac,
            "con": con,
            "saturated": saturated.detach(),
            "score": left["fused"],
        }

    def project_heads_(self) -> None:
        project_ball_(self.dyn_head.weight, self.head_dyn)
        project_ball_(self.ac_head.weight, self.head_ac)

    def freeze_world_(self) -> None:
        self.standardizer.frozen = True
        for module in (self.encoder, self.world, self.conditions, self.context_phone, self.rff):
            for p in module.parameters():
                p.requires_grad_(False)
