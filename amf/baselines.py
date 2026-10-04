"""Baselines: ML regressors with three run-out strategies, censored log-normal MLP (deep AFT) + ensembles,
and DeepSurv (Cox). Every baseline exposes the same interface on the ln N scale:
    logpred(X, lt, d) -> per-specimen censored log-likelihood (nan if unavailable)
    cdf(X, lt)        -> predicted P(ln N <= lt)
    quantile(X, p)    -> predicted p-quantile of ln N (inf allowed)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as jnorm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor

jax.config.update("jax_enable_x64", True)


def specimen_table(studies):
    X, lt, d, g = [], [], [], []
    for i, st in enumerate(studies):
        n = len(st["N"])
        R = np.minimum(np.asarray(st["R"], float), 0.9)
        Xi = np.column_stack([np.tile(st["z"], (n, 1)), np.log(st["sa"]), np.log(2 / (1 - R)),
                              np.full(n, np.log(st["ref"])), np.log(st["sa"]) - np.log(st["ref"])])
        X.append(Xi); lt.append(np.log(st["N"])); d.append(np.asarray(st["d"], float)); g.append(np.full(n, i))
    return np.vstack(X), np.concatenate(lt), np.concatenate(d), np.concatenate(g)


# ----------------------------------------------------------------------------- sklearn regressors
def make_regressor(name, seed, fast=True):
    if name == "GBM":
        return HistGradientBoostingRegressor(max_iter=200 if fast else 600, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=10, random_state=seed)
    if name == "RF":
        return RandomForestRegressor(n_estimators=150 if fast else 500, min_samples_leaf=3, n_jobs=2, random_state=seed)
    if name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=400, early_stopping=True,
                            random_state=seed)
    if name == "E739":
        return LinearRegression()
    raise ValueError(name)


class GaussRegressor:
    """Point regressor on ln N + homoscedastic Gaussian scatter estimated out-of-fold (grouped by study)."""

    def __init__(self, name, strategy="drop", seed=0, fast=True):
        self.name, self.strategy, self.seed, self.fast = name, strategy, seed, fast

    def _design(self, X, d=None):
        if self.strategy == "flag":
            flag = np.zeros(len(X)) if d is None else (1 - d)
            return np.column_stack([X, flag])
        return X

    def fit(self, X, lt, d, g):
        if self.strategy == "drop":
            keep = d > 0
            X, lt, g, dd = X[keep], lt[keep], g[keep], d[keep]
        else:
            dd = d
        Xd = self._design(X, dd)
        self.mu_, self.sd_ = Xd.mean(0), Xd.std(0) + 1e-9
        Xs = (Xd - self.mu_) / self.sd_
        # out-of-fold residual scale on failures
        res = []
        for tr, va in GroupKFold(n_splits=3).split(Xs, lt, g):
            mdl = make_regressor(self.name, self.seed, self.fast).fit(Xs[tr], lt[tr])
            fv = va[dd[va] > 0]
            if len(fv):
                res.append(lt[fv] - mdl.predict(Xs[fv]))
        self.scale = float(np.std(np.concatenate(res))) if res else 1.0
        self.model = make_regressor(self.name, self.seed, self.fast).fit(Xs, lt)
        return self

    def mean(self, X):
        Xd = self._design(X, None)
        return self.model.predict((Xd - self.mu_) / self.sd_)

    def logpred(self, X, lt, d):
        z = (lt - self.mean(X)) / self.scale
        return np.where(d > 0, stats.norm.logpdf(z) - np.log(self.scale), stats.norm.logsf(z))

    def cdf(self, X, lt):
        return stats.norm.cdf((lt - self.mean(X)) / self.scale)

    def quantile(self, X, p):
        return self.mean(X) + self.scale * stats.norm.ppf(p)


# ----------------------------------------------------------------------------- JAX MLP helpers
def _mlp_init(key, sizes):
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a), jnp.zeros(b)))
    return ps


def _mlp_apply(ps, x):
    for W, b in ps[:-1]:
        x = jax.nn.gelu(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


class CensoredLogNormalMLP:
    """Deep AFT: ln N ~ Normal(mu(x), s(x)) with exact censored likelihood. Ensemble -> mixture."""

    def __init__(self, n_members=1, hidden=64, steps=1500, lr=3e-3, wd=1e-4, seed=0):
        self.E, self.hidden, self.steps, self.lr, self.wd, self.seed = n_members, hidden, steps, lr, wd, seed

    def fit(self, X, lt, d, g=None):
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        Xs = jnp.asarray((X - self.mu_) / self.sd_)
        self.ym, self.ys = float(lt[d > 0].mean()), float(lt[d > 0].std() + 1e-9)
        y = jnp.asarray((lt - self.ym) / self.ys); dd = jnp.asarray(d)

        def loss(ps):
            out = _mlp_apply(ps, Xs)
            mu, ls = out[:, 0], jnp.clip(out[:, 1], -4, 2)
            s = jnp.exp(ls)
            z = (y - mu) / s
            ll = jnp.where(dd > 0, jnorm.logpdf(z) - ls, jax.scipy.special.log_ndtr(-z))
            reg = sum(jnp.sum(W ** 2) for W, _ in ps)
            return -ll.mean() + self.wd * reg

        self.members = []
        for e in range(self.E):
            ps = _mlp_init(jax.random.PRNGKey(self.seed * 100 + e), [Xs.shape[1], self.hidden, self.hidden, 2])
            opt = optax.adam(optax.cosine_decay_schedule(self.lr, self.steps, 0.05))
            st = opt.init(ps)

            @jax.jit
            def step(ps, st):
                v, gr = jax.value_and_grad(loss)(ps)
                up, st = opt.update(gr, st, ps)
                return optax.apply_updates(ps, up), st, v

            for _ in range(self.steps):
                ps, st, v = step(ps, st)
            self.members.append(ps)
        return self

    def _params(self, X):
        Xs = jnp.asarray((X - self.mu_) / self.sd_)
        mus, ss = [], []
        for ps in self.members:
            out = _mlp_apply(ps, Xs)
            mus.append(np.asarray(out[:, 0]) * self.ys + self.ym)
            ss.append(np.exp(np.clip(np.asarray(out[:, 1]), -4, 2)) * self.ys)
        return np.stack(mus), np.stack(ss)  # (E,n)

    def logpred(self, X, lt, d):
        mu, s = self._params(X)
        z = (lt[None] - mu) / s
        ll = np.where(d[None] > 0, stats.norm.logpdf(z) - np.log(s), stats.norm.logsf(z))
        return logsumexp(ll, 0) - np.log(len(self.members))

    def cdf(self, X, lt):
        mu, s = self._params(X)
        return stats.norm.cdf((lt[None] - mu) / s).mean(0)

    def quantile(self, X, p):
        mu, s = self._params(X)
        if len(self.members) == 1:
            return mu[0] + s[0] * stats.norm.ppf(p)
        a = (mu - 6 * s).min(0); b = (mu + 6 * s).max(0)
        for _ in range(60):
            mid = 0.5 * (a + b)
            up = stats.norm.cdf((mid[None] - mu) / s).mean(0) < p
            a = np.where(up, mid, a); b = np.where(up, b, mid)
        return 0.5 * (a + b)

    def mean(self, X):
        return self.quantile(X, 0.5)


class DeepSurv:
    """Cox proportional hazards with MLP risk (Katzman et al. 2018), Breslow baseline hazard."""

    def __init__(self, hidden=64, steps=800, lr=3e-3, wd=1e-4, seed=0):
        self.hidden, self.steps, self.lr, self.wd, self.seed = hidden, steps, lr, wd, seed

    def fit(self, X, lt, d, g=None):
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        order = np.argsort(-lt)  # descending time -> cumulative logsumexp gives risk sets
        Xs = jnp.asarray(((X - self.mu_) / self.sd_)[order]); dd = jnp.asarray(d[order])

        def loss(ps):
            r = _mlp_apply(ps, Xs)[:, 0]
            lse = jax.lax.cumlogsumexp(r)
            ll = (r - lse) * dd
            reg = sum(jnp.sum(W ** 2) for W, _ in ps)
            return -ll.sum() / dd.sum() + self.wd * reg

        ps = _mlp_init(jax.random.PRNGKey(self.seed), [Xs.shape[1], self.hidden, self.hidden, 1])
        opt = optax.adam(optax.cosine_decay_schedule(self.lr, self.steps, 0.05))
        st = opt.init(ps)

        @jax.jit
        def step(ps, st):
            v, gr = jax.value_and_grad(loss)(ps)
            up, st = opt.update(gr, st, ps)
            return optax.apply_updates(ps, up), st, v

        for _ in range(self.steps):
            ps, st, v = step(ps, st)
        self.ps = ps
        r = np.asarray(_mlp_apply(ps, jnp.asarray((X - self.mu_) / self.sd_))[:, 0])
        # Breslow: H0(t) = sum_{t_i <= t, failed} 1 / sum_{t_j >= t_i} exp(r_j)
        ev = np.sort(np.unique(lt[d > 0]))
        er = np.exp(r - r.max())
        H = []
        for t in ev:
            H.append(((lt == t) & (d > 0)).sum() / er[lt >= t].sum())
        self.times = ev
        self.H0 = np.cumsum(H) * np.exp(-r.max())
        return self

    def _risk(self, X):
        return np.asarray(_mlp_apply(self.ps, jnp.asarray((X - self.mu_) / self.sd_))[:, 0])

    def cdf(self, X, lt):
        idx = np.searchsorted(self.times, lt, side="right") - 1
        H = np.where(idx >= 0, self.H0[np.clip(idx, 0, None)], 0.0)
        return 1 - np.exp(-H * np.exp(self._risk(X)))

    def quantile(self, X, p):
        r = np.exp(self._risk(X))
        S = np.exp(-self.H0[None, :] * r[:, None])  # (n, T)
        hit = S <= 1 - p
        first = np.argmax(hit, 1)
        return np.where(hit.any(1), self.times[first], np.inf)

    def logpred(self, X, lt, d):
        return np.full(len(lt), np.nan)

    def mean(self, X):
        return self.quantile(X, 0.5)
