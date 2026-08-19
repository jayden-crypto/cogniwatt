"""Recurrent PPO, implemented from scratch. Owner: R1.

We do not use Stable-Baselines3. Two reasons, both worth stating in the report:
(1) the project must run identically on arm64 macOS and x86_64+CUDA, and SB3's
dependency stack does not pin cleanly across both; (2) the recurrence /
device-conditioning ablations need access to the hidden state, which is
awkward through SB3's wrapper layers.

Note on partial observability: the Bayesian belief carried in the observation
is an (approximately) sufficient statistic of the interaction history, so the
POMDP is rendered nearly Markov by construction. We therefore make recurrence
*optional* and ablate it, rather than assuming it is needed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env.homegrid import HomeGridEdge, N_ACTIONS, OBS_DIM


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    # The network is tiny; kernel-launch overhead makes CPU faster than MPS here.
    # On Adit's RTX 3060 box CUDA wins once we scale envs up.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=128,
                 recurrent: bool = True):
        super().__init__()
        self.recurrent = recurrent
        self.hidden = hidden
        self.enc = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        if recurrent:
            self.gru = nn.GRU(hidden, hidden, batch_first=False)
            for n, p in self.gru.named_parameters():
                if "bias" in n:
                    nn.init.constant_(p, 0.0)
                else:
                    nn.init.orthogonal_(p, 1.0)
        self.actor = layer_init(nn.Linear(hidden, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def initial_state(self, n: int, device) -> torch.Tensor:
        return torch.zeros(1, n, self.hidden, device=device)

    def _core(self, x, h, done):
        """x: (T, N, obs). done: (T, N) - resets hidden state at episode starts."""
        z = self.enc(x)
        if not self.recurrent:
            return z, h
        T, N = z.shape[0], z.shape[1]
        outs = []
        for t in range(T):
            h = h * (1.0 - done[t].view(1, N, 1))
            o, h = self.gru(z[t:t + 1], h)
            outs.append(o)
        return torch.cat(outs, dim=0), h

    def forward(self, x, h, done, mask=None):
        z, h = self._core(x, h, done)
        logits = self.actor(z)
        if mask is not None:
            # Invalid actions are structurally impossible, not merely bad. Left
            # unmasked, exploration burns most of its budget rediscovering that
            # you cannot pick up an object you are not standing on.
            logits = logits.masked_fill(~mask, -1e8)
        return logits, self.critic(z).squeeze(-1), h

    @torch.no_grad()
    def act(self, obs_1n, h, done_1n, mask=None, deterministic=False):
        logits, value, h = self.forward(obs_1n, h, done_1n, mask)
        dist = torch.distributions.Categorical(logits=logits)
        a = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return a, dist.log_prob(a), value, h


class VecHomeGrid:
    """Minimal synchronous vector env with auto-reset."""

    def __init__(self, n_envs: int, env_kwargs: dict, seed: int = 0,
                 devices: Optional[Sequence[str]] = None):
        self.n = n_envs
        self.envs = [HomeGridEdge(seed=seed + i, **env_kwargs) for i in range(n_envs)]
        self.fixed_devices = devices

    def _dev(self, i):
        return None if self.fixed_devices is None else self.fixed_devices[i % len(self.fixed_devices)]

    def reset(self):
        obs = [e.reset(device=self._dev(i))[0] for i, e in enumerate(self.envs)]
        masks = [e.action_mask() for e in self.envs]
        return np.stack(obs).astype(np.float32), np.stack(masks)

    def step(self, actions):
        obs, masks, rews, dones, infos = [], [], [], [], []
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, info = e.step(int(a))
            d = term or trunc
            if d:
                infos.append(info)
                o, _ = e.reset(device=self._dev(i))
            obs.append(o)
            masks.append(e.action_mask())
            rews.append(r)
            dones.append(d)
        return (np.stack(obs).astype(np.float32), np.stack(masks),
                np.array(rews, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                infos)


@dataclass
class PPOConfig:
    total_steps: int = 1_500_000
    n_envs: int = 16
    horizon: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    n_minibatch: int = 4
    lr: float = 3e-4
    ent_coef: float = 0.02
    # Entropy has to do two different jobs at two different times, and a single
    # constant cannot do both (measured in docs/r1_stability.md):
    #   * early, it must be large enough to escape the uninformed policy. That
    #     basin is absorbing - at ent_coef 0.01 only 1 run in 8 escapes it.
    #   * late, it must be small enough for the policy to sharpen. Held at 0.12
    #     the converged policy stays deliberately stochastic and over-buys
    #     evidence (~4.3 queries/episode vs 2.2 for the greedy heuristic),
    #     which costs more reward than the extra accuracy is worth.
    # Annealing linearly from `ent_coef` to `ent_coef_final` asks each phase for
    # only the thing it is good at. None = hold `ent_coef` constant (old
    # behaviour, so every existing run reproduces bit-identically).
    ent_coef_final: Optional[float] = None
    vf_coef: float = 0.5
    # Env rewards live on a +-10 scale. Left unscaled the policy-gradient term
    # swamps the entropy bonus and the policy collapses within ~30 updates.
    # Training scales rewards; all *reported* metrics stay in natural units.
    reward_scale: float = 0.1
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    recurrent: bool = True
    hidden: int = 128
    seed: int = 0
    log_every: int = 10
    use_mask: bool = True


def train_ppo(cfg: PPOConfig, env_kwargs: dict,
              devices: Optional[Sequence[str]] = None,
              device_pref: str = "auto", verbose: bool = True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = pick_device(device_pref)

    venv = VecHomeGrid(cfg.n_envs, env_kwargs, seed=cfg.seed, devices=devices)
    net = ActorCritic(hidden=cfg.hidden, recurrent=cfg.recurrent).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    T, N = cfg.horizon, cfg.n_envs
    obs_b = torch.zeros(T, N, OBS_DIM, device=dev)
    act_b = torch.zeros(T, N, dtype=torch.long, device=dev)
    logp_b = torch.zeros(T, N, device=dev)
    rew_b = torch.zeros(T, N, device=dev)
    done_b = torch.zeros(T, N, device=dev)
    val_b = torch.zeros(T, N, device=dev)
    mask_b = torch.ones(T, N, N_ACTIONS, dtype=torch.bool, device=dev)

    obs_np, mask_np = venv.reset()
    obs = torch.tensor(obs_np, device=dev)
    mask = torch.tensor(mask_np, device=dev)
    done = torch.zeros(N, device=dev)
    h = net.initial_state(N, dev)

    n_updates = cfg.total_steps // (T * N)
    hist: List[dict] = []
    ep_ret, ep_succ = [], []
    t0 = time.time()
    gstep = 0

    for upd in range(1, n_updates + 1):
        if cfg.anneal_lr:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (1.0 - (upd - 1.0) / n_updates)
        ent_c = cfg.ent_coef
        if cfg.ent_coef_final is not None:
            frac = (upd - 1.0) / max(1, n_updates - 1)
            ent_c = cfg.ent_coef + (cfg.ent_coef_final - cfg.ent_coef) * frac
        h_start = h.detach().clone()

        for t in range(T):
            obs_b[t], done_b[t], mask_b[t] = obs, done, mask
            mb_mask = mask.unsqueeze(0) if cfg.use_mask else None
            a, lp, v, h = net.act(obs.unsqueeze(0), h, done.unsqueeze(0), mb_mask)
            act_b[t], logp_b[t], val_b[t] = a[0], lp[0], v[0]
            nobs, nmask, r, d, infos = venv.step(a[0].cpu().numpy())
            rew_b[t] = torch.tensor(r, device=dev) * cfg.reward_scale
            obs = torch.tensor(nobs, device=dev)
            mask = torch.tensor(nmask, device=dev)
            done = torch.tensor(d, device=dev)
            gstep += N
            for info in infos:
                ep_succ.append(float(info["outcome"] == "success"))

        with torch.no_grad():
            _, next_v, _ = net.forward(obs.unsqueeze(0), h, done.unsqueeze(0),
                                       mask.unsqueeze(0) if cfg.use_mask else None)
            next_v = next_v[0]
            adv = torch.zeros_like(rew_b)
            lastgae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nonterm = 1.0 - (done if t == T - 1 else done_b[t + 1])
                nextval = next_v if t == T - 1 else val_b[t + 1]
                delta = rew_b[t] + cfg.gamma * nextval * nonterm - val_b[t]
                lastgae = delta + cfg.gamma * cfg.gae_lambda * nonterm * lastgae
                adv[t] = lastgae
            ret = adv + val_b

        # Recurrent PPO updates whole env-sequences, so minibatches split on the
        # env axis (never the time axis) to keep the hidden state valid.
        env_idx = np.arange(N)
        mb = max(1, N // cfg.n_minibatch)
        for _ in range(cfg.epochs):
            np.random.shuffle(env_idx)
            for s in range(0, N, mb):
                idx = env_idx[s:s + mb]
                logits, v, _ = net.forward(
                    obs_b[:, idx], h_start[:, idx], done_b[:, idx],
                    mask_b[:, idx] if cfg.use_mask else None)
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(act_b[:, idx])
                ratio = (lp - logp_b[:, idx]).exp()
                a_mb = adv[:, idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = torch.max(-a_mb * ratio,
                               -a_mb * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                vloss = F.mse_loss(v, ret[:, idx])
                ent = dist.entropy().mean()
                loss = pg + cfg.vf_coef * vloss - ent_c * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
        h = h.detach()

        if verbose and (upd % cfg.log_every == 0 or upd == 1):
            sr = float(np.mean(ep_succ[-400:])) if ep_succ else float("nan")
            sps = int(gstep / (time.time() - t0))
            print(f"upd {upd:4d}/{n_updates}  steps {gstep:>9,}  success {sr:.3f}  "
                  f"ent {ent.item():.3f}  v_loss {vloss.item():.3f}  {sps:,} sps")
            hist.append({"update": upd, "step": gstep, "success": sr,
                         "entropy": ent.item(), "v_loss": vloss.item(),
                         "ent_coef": ent_c})
    return net, hist


class PPOAgent:
    """Wraps a trained policy for the scripted-baseline evaluation harness."""

    def __init__(self, net: ActorCritic, device=None, deterministic: bool = True,
                 name: str = "ppo", use_mask: bool = True):
        self.use_mask = use_mask
        self.net = net
        self.dev = device or next(net.parameters()).device
        self.deterministic = deterministic
        self.name = name
        self.h = None
        self._first = True

    def reset(self) -> None:
        self.h = self.net.initial_state(1, self.dev)
        self._first = True

    def act(self, env, obs) -> int:
        if self.h is None:
            self.reset()
        x = torch.tensor(obs, dtype=torch.float32, device=self.dev).view(1, 1, -1)
        d = torch.tensor([[1.0 if self._first else 0.0]], device=self.dev)
        m = None
        if self.use_mask:
            m = torch.tensor(env.action_mask(), device=self.dev).view(1, 1, -1)
        with torch.no_grad():
            a, _, _, self.h = self.net.act(x, self.h, d, m,
                                           deterministic=self.deterministic)
        self._first = False
        return int(a.item())
