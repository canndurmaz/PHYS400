"""Shared logging factory for the NNIP pipeline.

All modules use the parent logger "nnip" so handlers are attached once.
StreamHandler mirrors print() output; FileHandler captures DEBUG detail.
"""

import logging
import os
from datetime import datetime

# Session timestamp — shared across all loggers for a single pipeline run
SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

_CONFIGURED = False


def setup_logger(name, log_dir=None):
    """Return a child logger under 'nnip' (e.g. 'nnip.pipeline').

    Args:
        name: logger suffix (e.g. 'pipeline', 'dft', 'nn_optimizer')
        log_dir: directory for structured log file (default: src/NNIP/logs/)

    Returns:
        logging.Logger
    """
    global _CONFIGURED
    parent = logging.getLogger("nnip")

    if not _CONFIGURED:
        parent.setLevel(logging.DEBUG)

        # Console — INFO, mimics print() output
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(message)s"))
        parent.addHandler(sh)

        # File — DEBUG, structured
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"pipeline_{SESSION_TS}.structured.log")
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s %(message)s"
        ))
        parent.addHandler(fh)

        _CONFIGURED = True

    return parent.getChild(name)
