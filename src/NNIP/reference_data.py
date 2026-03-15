"""Load reference data and optimization config."""

import json
import os


def load_config(path=None):
    """Load optimization configuration.

    Args:
        path: path to config_optimize.json.  If None, uses the default
              adjacent to this file.

    Returns:
        dict with all config keys.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config_optimize.json")
    with open(path) as f:
        return json.load(f)


def get_reference(config):
    """Extract reference targets dict from config.

    Returns:
        dict[str, dict[str, float]] — e.g. {"Al": {"a_lat": 4.05, ...}}
    """
    return config["reference"]


def get_weights(config):
    """Extract property weights dict from config.

    Returns:
        dict[str, float] — e.g. {"a_lat": 10.0, "E_coh": 10.0, ...}
    """
    return config["weights"]


def get_bounds(config, names):
    """Build scipy-compatible bounds list for the parameter vector.

    Args:
        config: loaded config dict
        names: list of parameter names from params_to_vector()

    Returns:
        list of (lo, hi) tuples in the same order as names
    """
    bounds_map = config["bounds"]
    bounds = []
    for name in names:
        if name in bounds_map:
            bounds.append(tuple(bounds_map[name]))
        else:
            raise KeyError(f"No bounds defined for parameter: {name}")
    return bounds
