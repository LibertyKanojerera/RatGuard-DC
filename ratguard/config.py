"""Paths and config loading."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"                      # real DC downloads
RAW_SYNTHETIC = ROOT / "data" / "raw_synthetic"  # fake data in the same format, for testing only
PROCESSED = ROOT / "data" / "processed"
APP_DATA = ROOT / "app_data"
REPORTS = ROOT / "reports"

for _p in (RAW, RAW_SYNTHETIC, PROCESSED, APP_DATA, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)


def raw_dir(synthetic: bool) -> Path:
    return RAW_SYNTHETIC if synthetic else RAW


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def norm_type(name: str) -> str:
    """Normalise a 311 service-type label for matching (case, spaces)."""
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def type_to_group(cfg: dict) -> dict[str, str]:
    """Map normalised service-type label -> group name (rodent/food/shelter/containers)."""
    out = {}
    for group, names in cfg["service_groups"].items():
        for n in names:
            out[norm_type(n)] = group
    return out
