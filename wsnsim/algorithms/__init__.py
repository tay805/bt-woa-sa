from .base import ClusteringAlgorithm, Direct
from .leach import LEACH
from .ewoasa import EWOASA
from .baselines import LEACHC, DEEC, PSOC, WOAC, IWOASA

REGISTRY = {
    "Direct": Direct,
    "LEACH": LEACH,
    "BT-WOA-SA": EWOASA,
    "LEACH-C": LEACHC,
    "DEEC": DEEC,
    "PSO-C": PSOC,
    "WOA-C": WOAC,
    "IWOA-SA": IWOASA,
}


def make(name, **kw):
    cls = REGISTRY[name]
    if "label" in kw and cls in (Direct, LEACH):
        kw = {k: v for k, v in kw.items() if k != "label"}
    return cls(**kw)
