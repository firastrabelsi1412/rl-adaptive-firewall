"""
NSL-KDD label remapping — single source of truth.

Phase 2 stored 40-class fine-grained ``LabelEncoder`` integers in
``y_train.npy`` / ``y_test.npy`` (alphabetical order: 0=apache2 … 39=xterm).
Phase 3 collapses those into 5 coarse classes (Normal/DoS/Probe/R2L/U2R).

Lifted out of ``training/train_lstm_cnn.py`` so Phase 4 (PPO) can apply the
same mapping inside ``FirewallEnv`` without importing from a script directory.
The mapping itself was reconstructed by re-fitting the Phase 2 LabelEncoder
on the combined NSL-KDD train + test label columns and cross-referencing with
the ``ATTACK_MAP`` from ``notebooks/01_eda.ipynb`` — see §12.1 of the project
documentation for the full discovery story.
"""

from __future__ import annotations

import numpy as np


CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


LABEL_REMAP: dict[int, int] = {
    0: 1,   # apache2         → DoS
    1: 1,   # back            → DoS
    2: 4,   # buffer_overflow → U2R
    3: 3,   # ftp_write       → R2L
    4: 3,   # guess_passwd    → R2L
    5: 3,   # httptunnel      → R2L
    6: 3,   # imap            → R2L
    7: 2,   # ipsweep         → Probe
    8: 1,   # land            → DoS
    9: 4,   # loadmodule      → U2R
    10: 1,  # mailbomb        → DoS
    11: 2,  # mscan           → Probe
    12: 3,  # multihop        → R2L
    13: 3,  # named           → R2L
    14: 1,  # neptune         → DoS
    15: 2,  # nmap            → Probe
    16: 0,  # normal          → Normal
    17: 4,  # perl            → U2R
    18: 3,  # phf             → R2L
    19: 1,  # pod             → DoS
    20: 2,  # portsweep       → Probe
    21: 1,  # processtable    → DoS
    22: 4,  # ps              → U2R
    23: 4,  # rootkit         → U2R
    24: 2,  # saint           → Probe
    25: 2,  # satan           → Probe
    26: 3,  # sendmail        → R2L
    27: 1,  # smurf           → DoS
    28: 3,  # snmpgetattack   → R2L
    29: 3,  # snmpguess       → R2L
    30: 3,  # spy             → R2L
    31: 4,  # sqlattack       → U2R
    32: 1,  # teardrop        → DoS
    33: 1,  # udpstorm        → DoS
    34: 3,  # warezclient     → R2L
    35: 3,  # warezmaster     → R2L
    36: 3,  # worm            → R2L
    37: 3,  # xlock           → R2L
    38: 3,  # xsnoop          → R2L
    39: 4,  # xterm           → U2R
}


def remap_labels(y_raw: np.ndarray) -> np.ndarray:
    """Convert 40-class fine-grained integers to 5-class coarse integers."""
    return np.vectorize(LABEL_REMAP.get)(y_raw).astype(np.int64)
