"""Intel Berkeley Research Lab trace (54 Mica2Dot motes, 2004).

Expected files in cfg.intel_dir:
  data.txt or data.txt.gz : "date time epoch moteid temperature humidity light voltage"
  mote_locs.txt           : "moteid x y"   (metres)

One simulation round = one Intel epoch (~31 s). Temperature is the sensed quantity.

Cleaning (all thresholds are Config fields, reported in the paper):
  * readings outside [intel_t_min, intel_t_max] C or with mote voltage < intel_v_min V
    are treated as missing (failing-battery readings are a known artefact of this trace);
  * optional automatic choice of the epoch window with the most valid readings;
  * motes with valid-reading coverage below intel_min_coverage in the window are dropped;
  * gaps up to intel_max_gap epochs are linearly interpolated; longer gaps hold the last
    value (sample-and-hold) and are marked as having no ground truth, so they are excluded
    from the reconstruction-error metric.
"""
import os
import numpy as np
import pandas as pd


class TraceEnvironment:
    """Replays a (rounds x nodes) matrix of readings. No ground-truth events."""

    def __init__(self, V, valid):
        self.V = V
        self.valid = valid
        n = V.shape[1]
        self.ev_start = np.zeros(0, int)
        self.ev_end = np.zeros(0, int)
        self.detectable = np.zeros((0, n), bool)

    def readings(self, r):
        return self.V[min(r, len(self.V) - 1)].copy(), np.zeros(0, int)

    def truth_mask(self, r):
        return self.valid[min(r, len(self.valid) - 1)]


def _read_raw(intel_dir):
    data_path = next((os.path.join(intel_dir, f) for f in ("data.txt", "data.txt.gz")
                      if os.path.exists(os.path.join(intel_dir, f))), None)
    if data_path is None:
        raise FileNotFoundError(f"data.txt(.gz) not found in {intel_dir}")
    locs = pd.read_csv(os.path.join(intel_dir, "mote_locs.txt"), sep=r"\s+", header=None,
                       names=["mote", "x", "y"])
    df = pd.read_csv(data_path, sep=r"\s+", header=None, on_bad_lines="skip",
                     names=["date", "time", "epoch", "mote", "temp", "hum", "light", "volt"],
                     usecols=["epoch", "mote", "temp", "volt"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["epoch", "mote", "temp"])
    df["epoch"] = df["epoch"].astype(int)
    df["mote"] = df["mote"].astype(int)
    return df[df["mote"].isin(locs["mote"])], locs


_CACHE = {}


def load_intel(intel_dir, epoch_start, n_rounds, t_min=10.0, t_max=40.0, v_min=2.3,
               min_coverage=0.5, max_gap=10, auto_window=True):
    key = (os.path.abspath(intel_dir), epoch_start, n_rounds, t_min, t_max, v_min,
           min_coverage, max_gap, auto_window)
    if key in _CACHE:
        return _CACHE[key]
    df, locs = _read_raw(intel_dir)
    bad = (df["temp"] < t_min) | (df["temp"] > t_max)
    if v_min:
        bad |= df["volt"].notna() & (df["volt"] < v_min)
    df = df[~bad]

    if auto_window:
        per_epoch = df.groupby("epoch")["mote"].nunique()
        full = per_epoch.reindex(range(int(per_epoch.index.min()), int(per_epoch.index.max()) + 1),
                                 fill_value=0)
        roll = full.rolling(n_rounds).sum().shift(-(n_rounds - 1))
        if roll.notna().any():
            epoch_start = int(roll.idxmax())

    w = df[(df["epoch"] >= epoch_start) & (df["epoch"] < epoch_start + n_rounds)]
    M = w.pivot_table(index="epoch", columns="mote", values="temp", aggfunc="mean")
    M = M.reindex(range(epoch_start, epoch_start + n_rounds))
    coverage = M.notna().mean()
    keep = coverage.index[coverage >= min_coverage]
    if len(keep) == 0:
        raise ValueError(f"Intel trace: no mote has >= {min_coverage:.0%} valid readings in the "
                         f"{n_rounds}-epoch window starting at {epoch_start}; the trace may be shorter "
                         f"than max_rounds or the cleaning thresholds too strict")
    M = M[keep]
    raw_valid = M.notna()
    interp = M.interpolate(limit=max_gap, limit_area="inside")
    valid = interp.notna()                      # real readings + short interpolated gaps
    M = interp.ffill().bfill()
    locs = locs.set_index("mote").loc[keep]
    out = dict(x=locs["x"].to_numpy(float), y=locs["y"].to_numpy(float),
               V=M.to_numpy(float), valid=valid.to_numpy(bool), motes=np.asarray(keep),
               coverage=coverage.loc[keep].to_numpy(), epoch_start=epoch_start,
               raw_valid_frac=float(raw_valid.to_numpy().mean()),
               n_dropped=int((coverage < min_coverage).sum()))
    _CACHE[key] = out
    return out
