"""Hierarchical neural random-fatigue-limit (H-NRFL) model.

Per study s with process descriptors z_s, specimen j at equivalent normalised stress u:
    ln N = beta0_s - m_s * ln(u - gamma) + s_s * eps,   eps ~ N(0,1),   if u > gamma
    N = infinity                                         otherwise (run-out for ever)
    ln gamma ~ N(mu_s, tau_s^2)                          (random fatigue limit, Pascual & Meeker 1999)
    (beta0_s, m_s, s_s, mu_s, tau_s) = g_phi(z_s) + [b1, 0, 0, b2, 0],  b ~ N(0, L L^T)
    u = sigma_a * (2/(1-R))^(1-w) / sigma_ref          (Walker mean-stress correction)

Run-outs are right-censored observations. The study-level marginal likelihood integrates b by
tensor Gauss-Hermite quadrature; the fatigue-limit integral uses Gauss-Legendre quadrature on the
probability scale, truncated at the applied stress, so every term is smooth and differentiable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import log_ndtr, logsumexp, ndtri
from jax.scipy.stats import norm

jax.config.update("jax_enable_x64", True)

LN10 = np.log(10.0)


# ----------------------------------------------------------------------------- quadrature
def gl01(K: int):
    x, w = np.polynomial.legendre.leggauss(K)
    return jnp.asarray((x + 1) / 2), jnp.asarray(w / 2)


def gh_std(n: int, q: int):
    """Tensor Gauss-Hermite nodes/weights for a standard normal in q dimensions."""
    if q == 0:
        return jnp.zeros((1, 0)), jnp.ones(1)
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    X = np.stack(np.meshgrid(*([x] * q), indexing="ij"), -1).reshape(-1, q)
    W = np.prod(np.stack(np.meshgrid(*([w] * q), indexing="ij"), -1).reshape(-1, q), 1)
    return jnp.asarray(X), jnp.asarray(W)


# ----------------------------------------------------------------------------- specimen likelihood
def rfl_terms(lu, lt, beta0, m, s, mu, tau, qn, qw, fatigue_limit=True):
    """Log density of ln N, log survival of ln N, and log P(failure ever) for broadcastable inputs."""
    lu, lt, beta0, m, s, mu, tau = jnp.broadcast_arrays(lu, lt, beta0, m, s, mu, tau)
    if not fatigue_limit:  # plain Basquin log-normal: ln N = beta0 - m ln u + s eps
        z = (lt - beta0 + m * lu) / s
        return norm.logpdf(z) - jnp.log(s), log_ndtr(-z), jnp.zeros_like(z)
    a = (lu - mu) / tau
    logpx = log_ndtr(a)  # P(gamma < u): specimen can fail at all
    log1m = log_ndtr(-a)  # P(gamma >= u): infinite life (cure fraction)
    px = jnp.exp(logpx)
    qq = jnp.clip(qn * px[..., None], 1e-300, 1 - 1e-15)
    v = mu[..., None] + tau[..., None] * ndtri(qq)  # ln gamma at conditional quantiles (< lu)
    d = jnp.minimum(v - lu[..., None], -1e-12)
    lgap = lu[..., None] + jnp.log(-jnp.expm1(d))  # ln(u - gamma)
    z = (lt[..., None] - beta0[..., None] + m[..., None] * lgap) / s[..., None]
    lw = jnp.log(qw)
    logf = logpx + logsumexp(lw + norm.logpdf(z), -1) - jnp.log(s)
    logS = jnp.logaddexp(log1m, logpx + logsumexp(lw + log_ndtr(-z), -1))
    return logf, logS, logpx


# ----------------------------------------------------------------------------- config / params
@dataclass
class RFLConfig:
    covnet: str = "mlp"  # 'mlp' | 'linear' | 'none'
    re: int = 2  # random-effect dims: 0 or 2 (beta0, mu)
    fatigue_limit: bool = True
    walker: bool = True  # learn Walker exponent; False -> w = 1 (amplitude only)
    hidden: int = 32
    n_gh_train: int = 5
    n_gh_pred: int = 13
    K: int = 12
    steps: int = 600
    lr: float = 1e-2
    wd: float = 1e-3
    batch_studies: int = 0  # 0 = full batch
    seed: int = 0
    init_theta: tuple = (5.5, 5.0, -0.6, float(np.log(0.4)), -2.6)
    init_re_sd: tuple = (0.3, 0.1)
    init_walker: float = 0.4  # logit of the initial Walker exponent (0.4 -> w = 0.6)
    verbose: bool = False
    extra: dict = field(default_factory=dict)


def softplus_inv(y):
    return float(np.log(np.expm1(y)))


def init_params(key, p, cfg: RFLConfig):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    P = {"b_out": jnp.asarray(cfg.init_theta, dtype=jnp.float64)}
    if cfg.covnet in ("linear", "mlp"):
        P["W_lin"] = jnp.zeros((p, 5))
    if cfg.covnet == "mlp":
        H = cfg.hidden
        P["W1"] = jax.random.normal(k1, (p, H)) * np.sqrt(1.0 / max(p, 1))
        P["b1"] = jnp.zeros(H)
        P["W2"] = jax.random.normal(k2, (H, H)) * np.sqrt(1.0 / H)
        P["b2"] = jnp.zeros(H)
        P["W3"] = jax.random.normal(k3, (H, 5)) * 0.01
    if cfg.re > 0:
        P["chol"] = jnp.asarray([np.log(cfg.init_re_sd[0]), np.log(cfg.init_re_sd[1]), 0.0])
    P["walker"] = jnp.asarray(float(cfg.init_walker))  # default sigmoid(0.4) ~ 0.6
    return P


def theta_raw(P, Z, cfg: RFLConfig):
    th = jnp.broadcast_to(P["b_out"], (Z.shape[0], 5))
    if cfg.covnet in ("linear", "mlp"):
        th = th + Z @ P["W_lin"]
    if cfg.covnet == "mlp":
        h = jax.nn.gelu(Z @ P["W1"] + P["b1"])
        h = jax.nn.gelu(h @ P["W2"] + P["b2"])
        th = th + h @ P["W3"]
    return th


def map_theta(th):
    beta0 = th[..., 0]
    m = jax.nn.softplus(th[..., 1]) + 0.5
    s = jax.nn.softplus(th[..., 2]) + 0.05
    mu = th[..., 3]
    tau = jax.nn.softplus(th[..., 4]) + 0.01
    return beta0, m, s, mu, tau


def chol(P, cfg):
    if cfg.re == 0:
        return jnp.zeros((0, 0))
    c = P["chol"]
    return jnp.asarray([[jnp.exp(c[0]), 0.0], [c[2], jnp.exp(c[1])]])


def walker_w(P, cfg):
    return jax.nn.sigmoid(P["walker"]) if cfg.walker else 1.0


def log_u(P, cfg, lsa, lrat, lref):
    w = walker_w(P, cfg)
    return lsa + (1.0 - w) * lrat - lref[..., None]


# ----------------------------------------------------------------------------- component grid
def components(P, Z, cfg: RFLConfig, ghX):
    """Per-study, per-node parameters. Returns arrays of shape (S, G)."""
    th = theta_raw(P, Z, cfg)
    beta0, m, s, mu, tau = map_theta(th)
    S = Z.shape[0]
    G = ghX.shape[0]
    if cfg.re > 0:
        B = ghX @ chol(P, cfg).T  # (G,2)
        beta0 = beta0[:, None] + B[None, :, 0]
        mu = mu[:, None] + B[None, :, 1]
    else:
        beta0 = jnp.broadcast_to(beta0[:, None], (S, G))
        mu = jnp.broadcast_to(mu[:, None], (S, G))
    m = jnp.broadcast_to(m[:, None], (S, G))
    s = jnp.broadcast_to(s[:, None], (S, G))
    tau = jnp.broadcast_to(tau[:, None], (S, G))
    return beta0, m, s, mu, tau


def spec_ll_grid(P, cfg, D, ghX, qn, qw):
    """Specimen log-likelihood at each quadrature node: (S, G, n)."""
    beta0, m, s, mu, tau = components(P, D["Z"], cfg, ghX)
    lu = log_u(P, cfg, D["lsa"], D["lrat"], D["lref"])  # (S,n)
    logf, logS, _ = rfl_terms(
        lu[:, None, :], D["lt"][:, None, :], beta0[..., None], m[..., None], s[..., None],
        mu[..., None], tau[..., None], qn, qw, cfg.fatigue_limit,
    )
    return jnp.where(D["d"][:, None, :] > 0, logf, logS)


def study_loglik(P, cfg, D, ghX, ghW, qn, qw, mask):
    ll = (spec_ll_grid(P, cfg, D, ghX, qn, qw) * mask[:, None, :]).sum(-1)  # (S,G)
    return logsumexp(ll + jnp.log(ghW)[None, :], -1)


# ----------------------------------------------------------------------------- data packing
def pack(studies: list[dict], feat_mean=None, feat_sd=None, nmax=None):
    """studies: list of dicts with keys z (p,), sa (n,), R (n,), N (n,), d (n,), ref (float).
    nmax pads every study to a fixed width so compiled functions are reused across calls."""
    S = len(studies)
    nmax = max(nmax or 0, max(len(st["N"]) for st in studies))
    p = len(studies[0]["z"])
    Zr = np.stack([np.asarray(st["z"], float) for st in studies])
    if feat_mean is None:
        feat_mean = Zr.mean(0)
        feat_sd = Zr.std(0) + 1e-9
    Z = (Zr - feat_mean) / feat_sd
    lsa = np.zeros((S, nmax)); lrat = np.zeros((S, nmax)); lt = np.full((S, nmax), 10.0)
    d = np.zeros((S, nmax)); mask = np.zeros((S, nmax)); lref = np.zeros(S)
    for i, st in enumerate(studies):
        n = len(st["N"])
        lref[i] = np.log(st["ref"])
        R = np.minimum(np.asarray(st["R"], float), 0.9)
        lsa[i, :n] = np.log(st["sa"]); lrat[i, :n] = np.log(2.0 / (1.0 - R))
        lt[i, :n] = np.log(st["N"]); d[i, :n] = st["d"]; mask[i, :n] = 1
        lsa[i, n:] = lref[i] - 1.0
    D = {k: jnp.asarray(v) for k, v in dict(Z=Z, lsa=lsa, lrat=lrat, lt=lt, d=d, lref=lref).items()}
    return D, jnp.asarray(mask), (feat_mean, feat_sd)


# ----------------------------------------------------------------------------- training
@lru_cache(maxsize=None)
def _optimizer(lr, steps):
    sched = optax.cosine_decay_schedule(lr, steps, alpha=0.05)
    return optax.chain(optax.clip_by_global_norm(10.0), optax.adam(sched))


def _loss(P, cfg, Db, maskb, ghX, ghW, qn, qw):
    ll = study_loglik(P, cfg, Db, ghX, ghW, qn, qw, maskb)
    reg = sum(jnp.sum(P[k] ** 2) for k in ("W_lin", "W1", "W2", "W3") if k in P)
    return -ll.sum() / maskb.sum() + cfg.wd * reg


@partial(jax.jit, static_argnums=(0, 1))
def _train_step(cfg, opt, P, state, D, mask, idx, ghX, ghW, qn, qw):
    Db = {k: v[idx] for k, v in D.items()}
    mb = mask[idx]
    val, g = jax.value_and_grad(_loss)(P, cfg, Db, mb, ghX, ghW, qn, qw)
    upd, state = opt.update(g, state, P)
    return optax.apply_updates(P, upd), state, val


def fit(studies, cfg: RFLConfig, censor: str = "proper", nmax=None):
    """censor: 'proper' (run-outs censored), 'drop' (run-outs removed), 'asfail' (run-outs as failures)."""
    studies = apply_censor_strategy(studies, censor)
    D, mask, norm_stats = pack(studies, nmax=nmax)
    ghX, ghW = gh_std(cfg.n_gh_train, cfg.re)
    qn, qw = gl01(cfg.K)
    key = jax.random.PRNGKey(cfg.seed)
    P = init_params(key, D["Z"].shape[1], cfg)
    ntot = float(mask.sum())
    opt = _optimizer(cfg.lr, cfg.steps)
    state = opt.init(P)
    S = D["Z"].shape[0]
    rng = np.random.default_rng(cfg.seed)
    bs = cfg.batch_studies if (cfg.batch_studies and cfg.batch_studies < S) else S
    t0 = time.time()
    hist = []
    for it in range(cfg.steps):
        idx = np.arange(S) if bs == S else rng.choice(S, bs, replace=False)
        P, state, val = _train_step(cfg, opt, P, state, D, mask, jnp.asarray(idx), ghX, ghW, qn, qw)
        if it % 50 == 0 or it == cfg.steps - 1:
            hist.append(float(val))
            if cfg.verbose:
                print(f"  step {it:5d} loss {float(val):.4f}  ({time.time()-t0:.1f}s)")
    return FittedRFL(P, cfg, norm_stats, hist, ntot, pad=nmax)


def apply_censor_strategy(studies, censor):
    if censor == "proper":
        return studies
    out = []
    for st in studies:
        st = dict(st)
        d = np.asarray(st["d"])
        if censor == "drop":
            keep = d > 0
            if keep.sum() == 0:
                continue
            for k in ("sa", "R", "N", "d"):
                st[k] = np.asarray(st[k])[keep]
        elif censor == "asfail":
            st["d"] = np.ones_like(d)
        out.append(st)
    return out


# ----------------------------------------------------------------------------- prediction
class FittedRFL:
    def __init__(self, P, cfg, norm_stats, hist, ntot, pad=None):
        self.P, self.cfg, self.norm_stats, self.hist, self.ntot = P, cfg, norm_stats, hist, ntot
        self.pad = pad
        self.ghX, self.ghW = gh_std(cfg.n_gh_pred, cfg.re)
        self.qn, self.qw = gl01(cfg.K)

    # --- posterior over random effects given 'shot' specimens (mask), exact on the grid
    def posterior(self, studies, shot_mask=None):
        D, mask, _ = pack(studies, *self.norm_stats, nmax=self.pad)
        sm = np.zeros(mask.shape)
        if shot_mask is not None:
            shot_mask = np.asarray(shot_mask)
            w = min(shot_mask.shape[1], sm.shape[1])
            sm[:, :w] = shot_mask[:, :w]
        ll = (self._grid_ll(D) * jnp.asarray(sm)[:, None, :]).sum(-1)
        logw = jnp.log(self.ghW)[None, :] + ll
        logw = logw - logsumexp(logw, -1, keepdims=True)
        return D, mask, logw

    def _grid_ll(self, D):
        return _grid_ll_jit(self.P, self.cfg, D, self.ghX, self.qn, self.qw)

    def logpred(self, studies, shot_mask=None):
        """Per-specimen posterior-predictive log-likelihood (S, n); padded entries are 0."""
        D, mask, logw = self.posterior(studies, shot_mask)
        ll = self._grid_ll(D)  # (S,G,n)
        lp = logsumexp(logw[:, :, None] + ll, 1)
        return np.asarray(lp * mask), np.asarray(mask)

    def comps(self, D):
        return components(self.P, D["Z"], self.cfg, self.ghX)

    def logsf(self, D, logw, lsa, lrat, lt):
        """Predictive log survival of ln N at (lsa, lrat, lt) arrays of shape (S, q)."""
        return _logsf_jit(self.P, self.cfg, D["Z"], D["lref"], logw, lsa, lrat, lt, self.ghX, self.qn, self.qw)

    def quantile(self, D, logw, lsa, lrat, p, lo=0.0, hi=60.0, iters=32):
        """p-quantile of ln N (inf if P(failure ever) < p). Shapes (S,q)."""
        q, lsh = _quantile_jit(self.P, self.cfg, iters, D["Z"], D["lref"], logw, jnp.asarray(lsa), jnp.asarray(lrat),
                               self.ghX, self.qn, self.qw, float(p), float(lo), float(hi))
        return np.where(np.asarray(lsh) > np.log1p(-p), np.inf, np.asarray(q))

    def stress_at(self, D, logw, lrat, lt_target, p, lo, hi, iters=30):
        """ln sigma_a such that P(N <= exp(lt_target)) = p (monotone in stress). Shapes (S,q)."""
        return np.asarray(_stress_jit(self.P, self.cfg, iters, D["Z"], D["lref"], logw, jnp.asarray(lrat),
                                      jnp.asarray(lt_target), self.ghX, self.qn, self.qw, float(p),
                                      jnp.asarray(lo), jnp.asarray(hi)))

    @property
    def walker(self):
        return float(walker_w(self.P, self.cfg))

    @property
    def re_cov(self):
        if self.cfg.re == 0:
            return np.zeros((0, 0))
        L = np.asarray(chol(self.P, self.cfg))
        return L @ L.T


@partial(jax.jit, static_argnums=(1,))
def _grid_ll_jit(P, cfg, D, ghX, qn, qw):
    return spec_ll_grid(P, cfg, D, ghX, qn, qw)


def _logsf_core(P, cfg, Z, lref, logw, lsa, lrat, lt, ghX, qn, qw):
    beta0, m, s, mu, tau = components(P, Z, cfg, ghX)  # (S,G)
    lu = log_u(P, cfg, lsa, lrat, lref)  # (S,q)
    _, logS, _ = rfl_terms(lu[:, None, :], lt[:, None, :], beta0[..., None], m[..., None], s[..., None],
                           mu[..., None], tau[..., None], qn, qw, cfg.fatigue_limit)
    return logsumexp(logw[:, :, None] + logS, 1)


_logsf_jit = jax.jit(_logsf_core, static_argnums=(1,))


@partial(jax.jit, static_argnums=(1, 2))
def _quantile_jit(P, cfg, iters, Z, lref, logw, lsa, lrat, ghX, qn, qw, p, lo, hi):
    target = jnp.log1p(-p)

    def body(_, ab):
        a, b = ab
        mid = 0.5 * (a + b)
        ls = _logsf_core(P, cfg, Z, lref, logw, lsa, lrat, mid, ghX, qn, qw)
        go_up = ls > target  # survival still above 1-p -> quantile is larger
        return jnp.where(go_up, mid, a), jnp.where(go_up, b, mid)

    a0 = jnp.full(lsa.shape, lo); b0 = jnp.full(lsa.shape, hi)
    a, b = jax.lax.fori_loop(0, iters, body, (a0, b0))
    lsh = _logsf_core(P, cfg, Z, lref, logw, lsa, lrat, jnp.full(lsa.shape, hi), ghX, qn, qw)
    return 0.5 * (a + b), lsh


@partial(jax.jit, static_argnums=(1, 2))
def _stress_jit(P, cfg, iters, Z, lref, logw, lrat, lt_target, ghX, qn, qw, p, lo, hi):
    target = jnp.log1p(-p)

    def body(_, ab):
        a, b = ab
        mid = 0.5 * (a + b)
        ls = _logsf_core(P, cfg, Z, lref, logw, mid, lrat, lt_target, ghX, qn, qw)
        safe = ls > target  # survival above 1-p at this stress -> can raise stress
        return jnp.where(safe, mid, a), jnp.where(safe, b, mid)

    a, b = jax.lax.fori_loop(0, iters, body, (lo, hi))
    return 0.5 * (a + b)


# make RFLConfig hashable for jit static args (every field that changes the traced computation)
def _cfg_key(self):
    return (self.covnet, self.re, self.fatigue_limit, self.walker, self.hidden, self.n_gh_train, self.n_gh_pred,
            self.K, self.wd)


RFLConfig.__hash__ = lambda self: hash(_cfg_key(self))
RFLConfig.__eq__ = lambda a, b: isinstance(b, RFLConfig) and _cfg_key(a) == _cfg_key(b)
