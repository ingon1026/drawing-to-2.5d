from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class AnchorState:
    homography: Optional[np.ndarray]  # 3x3, None if fully lost
    confidence: float                 # 0.0 .. 1.0
    lost: bool


class WorldAnchor(Protocol):
    def initialize(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> None: ...
    def update(self, frame: np.ndarray) -> AnchorState: ...
    def is_lost(self) -> bool: ...
