"""Uniform prediction interface for the hierarchical RFL and for baselines, plus allowable-stress search."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from sklearn.isotonic import IsotonicRegression

from . import baselines as bl
from . import rfl


def flat(arr2d, mask):
    return np.asarray(arr2d)[np.asarray(mask) > 0]


class RFLPredictor:
    def __init__(self, M: rfl.FittedRFL, chunk=25):
        self.M, self.chunk = M, chunk

    def predict(self, studies, shot=None, alpha=0.1):
        out = {k: [] for k in ("lp", "F", "lnQ", "lnmed")}
        for c0 in range(0, len(studies), self.chunk):
            sub = studies[c0:c0 + self.chunk]
            sh = None if shot is None else shot[c0:c0 + self.chunk]
            D, mask, logw = self.M.posterior(sub, sh)
            ll = self.M._grid_ll(D)
            lp = logsumexp(logw[:, :, None] + ll, 1)
            logS = self.M.logsf(D, logw, D["lsa"], D["lrat"], D["lt"])
            qa = self.M.quantile(D, logw, D["lsa"], D["lrat"], alpha)
            qm = self.M.quantile(D, logw, D["lsa"], D["lrat"], 0.5)
            out["lp"].append(flat(lp, mask)); out["F"].append(flat(1 - jnp.exp(logS), mask))
            out["lnQ"].append(flat(qa, mask)); out["lnmed"].append(flat(qm, mask))
        return {k: np.concatenate(v) for k, v in out.items()}

    def allowable(self, studies, shot, lt_target, p):
        """Amplitude sigma per study with predictive P(ln N <= lt_target) = p.

        lt_target may be a scalar (returns (S,)) or a 1-D array of targets (returns (S, q)), which is how
        allowables for a whole grid of conformal offsets eta are obtained in one compiled call."""
        lt_arr = np.atleast_1d(np.asarray(lt_target, float)); q = len(lt_arr)
        res = []
        for c0 in range(0, len(studies), self.chunk):
            sub = studies[c0:c0 + self.chunk]
            sh = None if shot is None else shot[c0:c0 + self.chunk]
            D, mask, logw = self.M.posterior(sub, sh)
            S = len(sub)
            lr = np.array([np.log(2 / (1 - min(st["R"][0], 0.9))) for st in sub])
            lref = np.array([np.log(st["ref"]) for st in sub])
            lrat = jnp.asarray(np.repeat(lr[:, None], q, 1))
            lo = jnp.asarray(np.repeat((lref + np.log(0.01))[:, None], q, 1))
            hi = jnp.asarray(np.repeat((lref + np.log(2.0))[:, None], q, 1))
            ltt = jnp.asarray(np.tile(lt_arr, (S, 1)))
            res.append(self.M.stress_at(D, logw, lrat, ltt, p, lo, hi))
        out = np.exp(np.concatenate(res, 0))
        return out[:, 0] if np.ndim(lt_target) == 0 else out


class BaselinePredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, studies, shot=None, alpha=0.1):
        X, lt, d, g = bl.specimen_table(studies)
        lp = self.model.logpred(X, lt, d)
        return dict(lp=lp, F=self.model.cdf(X, lt), lnQ=self.model.quantile(X, alpha), lnmed=self.model.quantile(X, 0.5))

    def allowable(self, studies, shot, lt_target, p, n_grid=120):
        """Largest grid amplitude below the first crossing of P(ln N <= lt_target) >= p (works if non-monotone)."""
        out = []
        for st in studies:
            sig = st["ref"] * np.exp(np.linspace(np.log(0.01), np.log(2.0), n_grid))
            tmp = dict(st); tmp["sa"] = sig; tmp["R"] = np.full(n_grid, st["R"][0])
            tmp["N"] = np.ones(n_grid); tmp["d"] = np.ones(n_grid)
            X, _, _, _ = bl.specimen_table([tmp])
            F = self.model.cdf(X, np.full(n_grid, lt_target))
            # monotone envelope in stress (isotonic), so non-monotone learners are not penalised by noise
            F = IsotonicRegression(increasing=True).fit_transform(np.arange(n_grid), F)
            idx = np.flatnonzero(F >= p)
            if len(idx) == 0:
                out.append(sig[-1])
            elif idx[0] == 0:
                out.append(sig[0] * 0.5)
            else:
                out.append(sig[idx[0] - 1])
        return np.array(out)


def specimen_index(studies):
    sid, pos = [], []
    for i, st in enumerate(studies):
        n = len(st["N"])
        sid.append(np.full(n, i)); pos.append(np.arange(n))
    return np.concatenate(sid), np.concatenate(pos)


def shot_masks(studies, ks, seed):
    """Nested random shot sets: shots for k are the first k of a per-study permutation."""
    rng = np.random.default_rng(seed)
    nmax = max(len(st["N"]) for st in studies)
    perms = [rng.permutation(len(st["N"])) for st in studies]
    out = {}
    for k in ks:
        m = np.zeros((len(studies), nmax))
        for i, p in enumerate(perms):
            m[i, p[:k]] = 1
        out[k] = m
    return out


def flat_shot(shot2d, studies):
    return np.concatenate([np.asarray(shot2d)[i, :len(st["N"])] for i, st in enumerate(studies)])
