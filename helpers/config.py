"""
Minimal config handling for train.py / evaluate.py.

The repo only ever needed "load one YAML, allow key=value overrides on the CLI,
and save the exact config a run used to a single reusable file". That is done
here with OmegaConf directly -- no Hydra. `cfg` stays a DictConfig, so the dot
and bracket access the rest of the code relies on is unchanged.

Author: Aritra Bal (ETP)
2026-09-05
"""
import argparse

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = "configs/VQC/base.yaml"


def load_config(default_config: str = DEFAULT_CONFIG) -> DictConfig:
    """
    Parse the CLI and return the merged config.

    Usage: ``python train.py [--config PATH] [--print-config] [key=value ...]``
    Overrides use OmegaConf dotlist syntax, e.g. ``epochs=5 aux_weights.bias=0.2``.
    A saved run config (``<save_dir>/<seed>/config.yaml``) can be passed straight
    back in via ``--config`` to reproduce that run.

    Args:
        default_config: base YAML used when ``--config`` is not given.

    Returns:
        The base config merged with any CLI overrides.
    """
    parser = argparse.ArgumentParser(description="Load a YAML config with CLI overrides.")
    parser.add_argument(
        "--config", default=default_config,
        help="path to the base YAML config (default: " + default_config + ")",
    )
    parser.add_argument(
        "--print-config", action="store_true",
        help="print the resolved config and exit",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf overrides, e.g. seed=run1 epochs=5 aux_weights.bias=0.2",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        raise SystemExit(0)

    return cfg


def save_config(cfg: DictConfig, path: str) -> None:
    """Write the exact config used for a run to a single reusable YAML file."""
    OmegaConf.save(cfg, path)
