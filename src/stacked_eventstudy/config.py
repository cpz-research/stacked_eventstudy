"""Configuration helpers."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProjectPaths:
    """Project paths used by the package."""

    root: Path = ROOT
    src: Path = SRC
