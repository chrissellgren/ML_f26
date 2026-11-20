#### CLAUDE GENERATED

"""Loading the slimmed samples: file discovery, train/val/test split by hard-scatter event,
truth-column guard, cross-section weights.

Deliberately contains NO physics choices (tau selection, vertex matching, algorithms):
those live in Chris's modules.

    from vtx.samples import load, split_of, reco_only
    ev = load(mu=50, slices=["A_m60_120"], columns=["event_id", "trk_pt", "trk_dz", "truth_hs_z"])
    train = ev[split_of(ev.event_id) == "train"]
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SLIM = ROOT / "samples" / "slim" / "prod"
RUNS_CSV = ROOT / "samples" / "hepmc" / "runs.csv"
MUS = (0, 30, 50, 100, 140, 200)
SLICES = ("A_m60_120", "B_m120_300", "C_m300")
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
TRUTH_MARKER = "truth"  # every generator-truth column contains this substring


def files(mu: int, slices=SLICES) -> list[Path]:
    out = []
    for s in slices:
        out += sorted((SLIM / s).glob(f"run*_mu{mu}.parquet"))
    return out


def load(mu: int, slices=SLICES, columns=None) -> ak.Array:
    """All events at one mu (concatenated over runs/slices). Raises if a file is missing."""
    runs = pd.read_csv(RUNS_CSV)
    runs = runs[runs.slice.isin(slices)]
    fs = files(mu, slices)
    expected = {SLIM / r.slice / f"run{r.run_id}_mu{mu}.parquet" for r in runs.itertuples()}
    missing = expected - set(fs)
    if missing:
        raise FileNotFoundError(f"missing slim files: {sorted(missing)}")
    arr = ak.concatenate([ak.from_parquet(f, columns=columns) for f in fs])
    if columns is None or "event_id" in columns:
        n_expected = int(runs.nevents.sum())
        if len(arr) != n_expected:  # guard against silently dropped events
            raise RuntimeError(f"mu={mu}: loaded {len(arr)} events, runs.csv says {n_expected}")
    return arr


def split_of(event_id) -> np.ndarray:
    """Deterministic train/val/test assignment from event_id (stable across mu, machines, runs).
    Uses a hash rather than event_id % 100 so that the split is not correlated with run or
    position in the file."""
    ids = np.asarray(ak.to_numpy(event_id) if isinstance(event_id, ak.Array) else event_id, dtype=np.int64)
    h = np.array([int.from_bytes(hashlib.blake2b(int(i).to_bytes(8, "little"), digest_size=4).digest(), "little")
                  for i in ids], dtype=np.uint64) % 10000
    u = h / 10000.0
    f_tr, f_va = SPLIT_FRACTIONS["train"], SPLIT_FRACTIONS["val"]
    return np.where(u < f_tr, "train", np.where(u < f_tr + f_va, "val", "test"))


def truth_columns(fields) -> list[str]:
    return [f for f in fields if TRUTH_MARKER in f]


def reco_only(arr: ak.Array) -> ak.Array:
    """Drop every truth column: use this on whatever is turned into model/algorithm inputs."""
    return arr[[f for f in arr.fields if TRUTH_MARKER not in f]]


def assert_no_truth(feature_names) -> None:
    """Leak audit: call on the list of input feature names before training."""
    bad = truth_columns(feature_names)
    if bad:
        raise AssertionError(f"truth columns used as inputs: {bad}")


def event_weights(arr: ak.Array) -> np.ndarray:
    """Per-event weight sigma_slice / N_slice [pb/event], to recombine mass slices into the
    physical gamma*->tautau spectrum. N_slice = all events of that slice at this mu."""
    sl = np.asarray(ak.to_list(arr.slice))
    xs = ak.to_numpy(arr.xsec_pb)
    w = np.empty(len(arr))
    runs = pd.read_csv(RUNS_CSV)
    for s in np.unique(sl):
        m = sl == s
        r = runs[runs.slice == s]
        sigma = np.average(r.xsec_pb, weights=r.nevents)  # runs of a slice share one sigma
        w[m] = sigma / m.sum()
    assert np.allclose(xs[np.isfinite(xs)] > 0, True)
    return w
