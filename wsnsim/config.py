"""Single source of truth for every simulation parameter.

Every number reported in the paper's parameter table must come from here.
"""
from dataclasses import dataclass, asdict, replace
import math


@dataclass(frozen=True)
class Config:
    # Deployment
    n_nodes: int = 100
    area_x: float = 100.0          # m
    area_y: float = 100.0          # m
    bs_x: float = 50.0             # m
    bs_y: float = 175.0            # m (outside the field, classic LEACH setting)

    # First-order radio model
    e_init: float = 0.5            # J per node
    e_elec: float = 50e-9          # J/bit
    eps_fs: float = 10e-12         # J/bit/m^2
    eps_mp: float = 0.0013e-12     # J/bit/m^4
    e_da: float = 5e-9             # J/bit/signal (aggregation)
    data_bits: int = 4000          # data packet
    ctrl_bits: int = 200           # ADV / JOIN / TDMA-schedule packet
    data_rate_bps: float = 250e3   # IEEE 802.15.4 rate, used by TDMA delay model

    # Clustering
    p_ch: float = 0.05             # desired CH fraction (K = round(p * alive))
    recluster_every: int = 1       # rounds between set-up phases

    # Traffic
    traffic: str = "periodic"      # "periodic" | "event"
    theta0: float = 0.5            # base event threshold
    theta_lambda: float = 1.0      # energy sensitivity of threshold
    heartbeat: int = 25            # max rounds without a report (event mode)

    # Synthetic environment
    noise_std: float = 0.1         # per-reading sensor noise
    drift_rho: float = 0.99        # AR(1) background drift coefficient
    drift_std: float = 0.02        # AR(1) innovation std
    event_rate: float = 0.05       # events per round (Poisson)
    event_amp: float = 2.0         # peak event amplitude
    event_sigma: float = 10.0      # spatial spread of an event (m)
    event_mean_dur: float = 20.0   # mean event duration (rounds, exponential)

    # Adversary (selective forwarding / grey-hole CHs)
    malicious_ratio: float = 0.2
    drop_low: float = 0.3
    drop_high: float = 0.8
    attack_period: int = 0         # on-off attack: cycle length in rounds (0 = always on)
    attack_on_frac: float = 0.5    # fraction of each cycle the attacker misbehaves

    # Channel
    per_base: float = 0.02         # per-attempt data frame loss probability (noise/collisions)
    max_retx: int = 3              # link-layer retransmissions (IEEE 802.15.4 macMaxFrameRetries default)

    # Environment source
    env: str = "synthetic"         # "synthetic" | "intel"
    intel_dir: str = ""            # folder with data.txt(.gz) and mote_locs.txt
    intel_epoch_start: int = 0     # first Intel epoch used as round 0 (ignored if auto window)
    intel_auto_window: bool = True # pick the n-round window with the most valid readings
    intel_t_min: float = 10.0      # plausible indoor temperature range (C)
    intel_t_max: float = 40.0
    intel_v_min: float = 2.3       # readings below this mote voltage are unreliable
    intel_min_coverage: float = 0.5
    intel_max_gap: int = 10        # epochs; longer gaps are held and excluded from error

    # Trust (Beta reputation at the BS)
    trust_forget: float = 0.9
    trust_threshold: float = 0.4

    # Coverage
    sense_radius: float = 15.0     # m; a grid point is covered if an alive node is within this radius
    coverage_grid: int = 20        # grid points per side used to estimate covered area

    # Run control
    max_rounds: int = 20000

    @property
    def d0(self) -> float:
        return math.sqrt(self.eps_fs / self.eps_mp)

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["d0"] = self.d0
        return d
