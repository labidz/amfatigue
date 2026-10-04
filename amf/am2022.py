"""Loader for FatigueData-AM2022 (Zhang & Xu, Sci. Data 10:249, 2023; figshare 10.6084/m9.figshare.22337629).

The database is released as MAT, JSON and Excel. The JSON keys are documented in the paper (Tables 3-5):
articles -> scientific data -> fatigue datasets, each with
    fat_data      numeric array [N, sigma_a (or eps_a), run-out flag]  (S-N: stress AMPLITUDE in MPa)
    fdata_type    'S-N' | 'e-N' | 'dadn'
    mat_name, am_type, am_para{direction, layer_thickness, ...}, proc_para[{type, temperature, ..., method}],
    fat_type, fat_temp, fat_env, fat_r, frequency, spec_kt, load_ctrl, tensile_strength, yield_strength
Missing entries are empty lists/strings. This loader turns every usable S-N dataset into one 'study' in the
format used by amf.rfl (z, sa, R, N, d, ref) and keeps the article DOI so splits can be made by article.
Field names are matched case-insensitively and defensively, because the public file may differ in detail.
"""
from __future__ import annotations

import json
import re

import numpy as np

ALLOY_CLASSES = ["Ti-6Al-4V", "316L", "AlSi10Mg", "IN718", "17-4PH", "IN625", "Other"]
AM_CLASSES = ["L-PBF", "E-PBF", "DED", "Other"]
FEATURES = ([f"alloy={a}" for a in ALLOY_CLASSES] + [f"am={a}" for a in AM_CLASSES] +
            ["HIP", "heat_treated", "machined_or_polished", "as_built_surface", "vertical", "build_angle_known",
             "R", "log_freq", "UTS_norm", "UTS_missing", "log_layer_um", "layer_missing"])


def _get(d, *keys, default=None):
    """Case-insensitive key lookup on a dict; returns the first non-empty match."""
    if not isinstance(d, dict):
        return default
    low = {k.lower(): v for k, v in d.items()}
    for k in keys:
        v = low.get(k.lower())
        if v is None:
            continue
        if isinstance(v, (list, str)) and len(v) == 0:
            continue
        return v
    return default


def _num(v):
    """First finite number in v (scalar, list, nested), else nan. Two values = [low, high] -> their mean."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", v)
        return float(m[0]) if m else np.nan
    try:
        a = np.asarray(v, float).ravel()
        a = a[np.isfinite(a)]
        return float(a.mean()) if len(a) else np.nan
    except (TypeError, ValueError):
        return np.nan


def alloy_class(name):
    s = str(name or "").lower().replace(" ", "")
    if re.search(r"ti-?6al-?4v|tc4|ti64", s):
        return "Ti-6Al-4V"
    if "316l" in s:
        return "316L"
    if re.search(r"alsi10mg", s):
        return "AlSi10Mg"
    if re.search(r"in718|inconel718|in-718|alloy718|^718", s):
        return "IN718"
    if re.search(r"17-?4", s):
        return "17-4PH"
    if re.search(r"in625|inconel625", s):
        return "IN625"
    return "Other"


def am_class(name):
    s = str(name or "").upper().replace(" ", "")
    if "L-PBF" in s or "SLM" in s or "LPBF" in s or "DMLS" in s or "LASERPOWDER" in s:
        return "L-PBF"
    if "E-PBF" in s or "EBM" in s or "EPBF" in s:
        return "E-PBF"
    if "DED" in s or "LMD" in s or "LENS" in s or "WAAM" in s:
        return "DED"
    return "Other"


def processing_flags(proc_para):
    """HIP, heat treatment and surface state from the processing-parameter records (text-matched)."""
    txt = json.dumps(proc_para).lower() if proc_para is not None else ""
    hip = int(bool(re.search(r"\bhip\b|isostatic", txt)))
    ht = int(hip or bool(re.search(r'"ht"|heat|anneal|stress.?relie|aging|solution', txt)))
    mach = int(bool(re.search(r"machin|polish|turn|grind|milled|cnc|electropolish", txt)))
    asb = int(bool(re.search(r"as-?built|as built", txt)) or (not mach))
    return hip, ht, mach, asb


def dataset_to_study(ds, uts_ref_by_alloy, doi="", min_points=5):
    """One FatigueData-AM2022 dataset -> study dict, or None if it is not a usable uniaxial S-N set."""
    ftype = str(_get(ds, "fdata_type", default="")).upper().replace("-", "")
    if ftype not in ("SN", "S_N", "STRESSLIFE"):
        return None
    arr = np.asarray(_get(ds, "fat_data", default=[]), float)
    if arr.ndim != 2 or arr.shape[0] < min_points or arr.shape[1] < 2:
        return None
    N, sa = arr[:, 0], arr[:, 1]
    ro = arr[:, 2] if arr.shape[1] >= 3 else np.zeros(len(N))
    ok = np.isfinite(N) & np.isfinite(sa) & (N > 0) & (sa > 0)
    N, sa, ro = N[ok], sa[ok], np.nan_to_num(ro[ok])
    if len(N) < min_points:
        return None
    temp = _num(_get(ds, "fat_temp"))
    if np.isfinite(temp) and abs(temp - 25) > 30:  # room temperature only
        return None
    env = str(_get(ds, "fat_env", default="air")).lower()
    if env and "air" not in env and "lab" not in env:
        return None
    kt = _num(_get(ds, "spec_kt"))
    if np.isfinite(kt) and kt > 1.1:  # smooth specimens only
        return None
    R = _num(_get(ds, "fat_r"))
    R = -1.0 if not np.isfinite(R) else R
    if R >= 0.95:
        return None
    alloy = alloy_class(_get(ds, "mat_name"))
    am = am_class(_get(ds, "am_type"))
    hip, ht, mach, asb = processing_flags(_get(ds, "proc_para"))
    amp = _get(ds, "am_para", default={}) or {}
    ang = _num(_get(amp, "direction"))
    layer = _num(_get(amp, "layer_thickness"))
    freq = _num(_get(ds, "frequency"))
    uts = _num(_get(ds, "tensile_strength"))
    uref = uts_ref_by_alloy.get(alloy, np.nan)
    ref = uts if np.isfinite(uts) else uref
    if not np.isfinite(ref):
        ref = float(np.max(sa) * 2.5)
    # Type I limit: the smallest run-out life (all run-outs of a dataset share it in practice)
    C = float(np.min(N[ro > 0])) if (ro > 0).any() else float(np.max(N) * 1.0001)
    z = ([1.0 if alloy == a else 0.0 for a in ALLOY_CLASSES] + [1.0 if am == a else 0.0 for a in AM_CLASSES] +
         [hip, ht, mach, asb, float(np.isfinite(ang) and ang >= 60), float(np.isfinite(ang)), R,
          np.log(freq) if np.isfinite(freq) and freq > 0 else np.log(20.0),
          (uts / uref - 1.0) if (np.isfinite(uts) and np.isfinite(uref)) else 0.0, float(not np.isfinite(uts)),
          np.log(layer) if np.isfinite(layer) and layer > 0 else np.log(30.0), float(not np.isfinite(layer))])
    return dict(z=np.asarray(z, float), sa=sa, R=np.full(len(N), R), N=N, d=(ro == 0).astype(float), ref=float(ref),
                C=C, alloy=alloy, am=am, hip=hip, mach=mach, doi=doi)


def _iter_datasets(db):
    """Yield (doi, dataset) from the JSON tree, tolerant of naming differences."""
    arts = _get(db, "articles", "article", default=None) if isinstance(db, dict) else db
    if arts is None:
        arts = db if isinstance(db, list) else []
    for art in arts:
        meta = _get(art, "metadata", default={}) or {}
        doi = str(_get(meta, "doi", default="") or _get(art, "doi", default=""))
        sci = _get(art, "scientific_data", "scientific data", "scidata", default=None)
        dsets = _get(sci, "fatigue_datasets", "fatigue datasets", "datasets", default=None) if isinstance(sci, dict) else sci
        if dsets is None:
            dsets = _get(art, "fatigue_datasets", "datasets", default=[])
        if isinstance(dsets, dict):
            dsets = [dsets]
        for ds in dsets or []:
            yield doi, _flatten(ds)


def _flatten(ds):
    """Merge nested structs (fatigue/materials/AM/processing/testing/static) into one flat dict."""
    flat = {}
    for k, v in (ds.items() if isinstance(ds, dict) else []):
        if isinstance(v, dict) and k.lower() not in ("am_para",):
            for k2, v2 in v.items():
                flat.setdefault(k2, v2)
        flat.setdefault(k, v)
    return flat


def load_json(path, min_points=5):
    with open(path) as f:
        db = json.load(f)
    raw = list(_iter_datasets(db))
    # alloy-class reference UTS = median reported UTS within class (used when a dataset does not report UTS)
    by = {}
    for _, ds in raw:
        u = _num(_get(ds, "tensile_strength"))
        if np.isfinite(u):
            by.setdefault(alloy_class(_get(ds, "mat_name")), []).append(u)
    uref = {k: float(np.median(v)) for k, v in by.items()}
    studies = [s for s in (dataset_to_study(ds, uref, doi, min_points) for doi, ds in raw) if s is not None]
    for i, s in enumerate(studies):
        s["id"] = i
    return studies, dict(n_raw=len(raw), n_used=len(studies), uts_ref=uref)


def load_excel(path, min_points=5):
    """Fallback for the flattened Excel export: sheets 'S-N' (index, S, N, run-out) and 'parameter'."""
    import pandas as pd
    xl = pd.read_excel(path, sheet_name=None)
    sn = next(v for k, v in xl.items() if k.replace("-", "").replace(" ", "").lower() in ("sn", "snfatigue"))
    par = next(v for k, v in xl.items() if "param" in k.lower())

    def col(df, *pats):
        for c in df.columns:
            cl = str(c).lower()
            if any(p in cl for p in pats):
                return c
        raise KeyError(pats)

    ci, cs, cn = col(sn, "index", "idx", "dataset"), col(sn, "stress", "sigma", "s ("), col(sn, "cycle", "n (", "life")
    try:
        cr = col(sn, "run", "flag")
    except KeyError:
        cr = None
    pi = col(par, "index", "idx", "dataset")
    out = []
    for idx, g in sn.groupby(ci):
        prow = par[par[pi] == idx]
        ds = {} if prow.empty else {str(k): v for k, v in prow.iloc[0].items() if not (isinstance(v, float) and np.isnan(v))}
        ro = g[cr].values if cr is not None else np.zeros(len(g))
        ds["fat_data"] = np.column_stack([g[cn].values, g[cs].values, ro]).tolist()
        ds.setdefault("fdata_type", "S-N")
        ds["proc_para"] = [str(v) for k, v in ds.items() if any(t in str(k).lower() for t in ("proc", "surf", "heat", "hip", "treat"))]
        ds["am_para"] = {"direction": ds.get("direction"), "layer_thickness": ds.get("layer_thickness")}
        out.append((str(ds.get("doi", "")), ds))
    by = {}
    for _, ds in out:
        u = _num(_get(ds, "tensile_strength"))
        if np.isfinite(u):
            by.setdefault(alloy_class(_get(ds, "mat_name")), []).append(u)
    uref = {k: float(np.median(v)) for k, v in by.items()}
    studies = [s for s in (dataset_to_study(ds, uref, doi, min_points) for doi, ds in out) if s is not None]
    for i, s in enumerate(studies):
        s["id"] = i
    return studies, dict(n_raw=len(out), n_used=len(studies), uts_ref=uref)


def load(path, min_points=5):
    return load_json(path, min_points) if str(path).lower().endswith(".json") else load_excel(path, min_points)
