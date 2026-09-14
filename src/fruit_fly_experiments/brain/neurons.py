from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NeuronIndex:
    body_ids: np.ndarray
    annotations: pd.DataFrame

    def __post_init__(self) -> None:
        if self.body_ids.ndim != 1:
            raise ValueError("body_ids must be one-dimensional")
        if len(self.body_ids) != len(self.annotations):
            raise ValueError("body_ids and annotations must have equal length")

    @property
    def size(self) -> int:
        return int(self.body_ids.size)

    def indices_for_types(self, types: Iterable[str], side: str | None = None) -> np.ndarray:
        wanted = {str(x) for x in types}
        df = self.annotations
        mask = df["type"].fillna("").astype(str).isin(wanted)
        if side is not None:
            side = side.upper()
            side_cols = [c for c in ("somaSide", "rootSide") if c in df.columns]
            if side_cols:
                side_mask = np.zeros(len(df), dtype=bool)
                aliases = {"L": {"L", "LHS", "LEFT"}, "R": {"R", "RHS", "RIGHT"}}
                accepted = aliases.get(side, {side})
                for col in side_cols:
                    side_mask |= df[col].fillna("").astype(str).str.upper().isin(accepted).to_numpy()
                mask &= side_mask
        return np.flatnonzero(mask.to_numpy())

    def indices_for_superclass(self, superclass: str) -> np.ndarray:
        mask = self.annotations["superclass"].fillna("").astype(str).eq(superclass)
        return np.flatnonzero(mask.to_numpy())

    def positions_xyz(self) -> np.ndarray:
        """Return MaleCNS soma/soma-tract coordinates in the released EM frame.

        `somaLocation` is preferred. `tosomaLocation` is used as a fallback for
        neurons without a soma coordinate. Missing positions remain NaN. These are
        measured dataset coordinates, not a generated visualization layout.
        """
        positions = np.full((self.size, 3), np.nan, dtype=np.float32)
        df = self.annotations
        soma = df["somaLocation"] if "somaLocation" in df.columns else [None] * self.size
        to_soma = df["tosomaLocation"] if "tosomaLocation" in df.columns else [None] * self.size

        def vector3(value) -> np.ndarray | None:
            if isinstance(value, np.ndarray):
                arr = value
            elif isinstance(value, (list, tuple)):
                arr = np.asarray(value)
            else:
                return None
            if arr.size != 3:
                return None
            try:
                arr = arr.astype(np.float32, copy=False).reshape(3)
            except (TypeError, ValueError):
                return None
            return arr if np.all(np.isfinite(arr)) else None

        for i, (primary, fallback) in enumerate(zip(soma, to_soma)):
            loc = vector3(primary)
            if loc is None:
                loc = vector3(fallback)
            if loc is not None:
                positions[i] = loc
        return positions
