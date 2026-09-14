from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fruit_fly_experiments.brain.neurons import NeuronIndex


@dataclass(frozen=True)
class EncodedVision:
    neuron_indices: np.ndarray
    drive: np.ndarray
    motion_centroid: float
    motion_energy: float
    retinal_count: int
    lc10a_count: int


class FlyVisualEncoder:
    """Pixel-only visual interface into real MaleCNS visual neuron identities.

    V0.1 rotates the Pong sensor image by 90 degrees so the game's vertical
    control axis maps onto the fly's left/right steering axis. Retinal columns
    receive local luminance/motion. LC10a receives a coarse moving-object signal
    inferred only from frame differencing; no Pong state is available here.
    """

    def __init__(self, neurons: NeuronIndex, optic_columns: pd.DataFrame | None = None) -> None:
        self.neurons = neurons
        self._previous: np.ndarray | None = None
        self._body_to_idx = {int(b): i for i, b in enumerate(neurons.body_ids)}
        self.retinal = self._build_retinal_map(optic_columns) if optic_columns is not None else []
        self.lc10a_l = neurons.indices_for_types(["LC10a"], side="L")
        self.lc10a_r = neurons.indices_for_types(["LC10a"], side="R")
        if len(self.lc10a_l) == 0 or len(self.lc10a_r) == 0:
            # Still permits pure-retina runs, but gives an explicit failure if neither route exists.
            if not self.retinal:
                raise ValueError("no LC10a or optic-column visual neurons found in retained MaleCNS annotations")

    @classmethod
    def from_data(cls, neurons: NeuronIndex, optic_path: Path) -> "FlyVisualEncoder":
        return cls(neurons, pd.read_excel(optic_path))

    def _build_retinal_map(self, optics: pd.DataFrame) -> list[tuple[int, float, float]]:
        rows: list[tuple[int, float, float]] = []
        parsed = []
        for _, row in optics.iterrows():
            name = str(row.get("column", ""))
            match = re.search(r"ME_([LR])_col_(\d+)_(\d+)", name)
            if not match:
                continue
            side, a, b = match.group(1), int(match.group(2)), int(match.group(3))
            parsed.append((row, side, a, b))
        if not parsed:
            return rows
        aa = np.array([x[2] for x in parsed], dtype=float)
        bb = np.array([x[3] for x in parsed], dtype=float)
        amin, amax = aa.min(), aa.max()
        bmin, bmax = bb.min(), bb.max()
        for row, side, a, b in parsed:
            u = (a - amin) / max(1.0, amax - amin)
            v = (b - bmin) / max(1.0, bmax - bmin)
            if side == "L":
                u = 1.0 - u
            for col in ("L1", "R7", "R8"):
                body = row.get(col)
                if pd.isna(body):
                    continue
                try:
                    body = int(body)
                except (TypeError, ValueError):
                    continue
                if body < 0 or body not in self._body_to_idx:
                    continue
                rows.append((self._body_to_idx[body], float(u), float(v)))
        return rows

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        arr = arr.astype(np.float32)
        if arr.max(initial=0) > 1.0:
            arr /= 255.0
        # Rotate so Pong vertical displacement becomes horizontal/azimuthal displacement.
        return np.rot90(arr)

    def encode(self, frame: np.ndarray) -> EncodedVision:
        image = self._gray(frame)
        if self._previous is None or self._previous.shape != image.shape:
            motion = np.zeros_like(image)
        else:
            motion = np.abs(image - self._previous)
        self._previous = image.copy()
        h, w = image.shape
        indices: list[int] = []
        drives: list[float] = []

        retinal_count = 0
        for idx, u, v in self.retinal:
            x = min(w - 1, max(0, int(round(u * (w - 1)))))
            y = min(h - 1, max(0, int(round(v * (h - 1)))))
            drive = 0.32 * image[y, x] + 0.85 * motion[y, x]
            if drive > 0.03:
                indices.append(idx)
                drives.append(float(min(0.9, drive)))
                retinal_count += 1

        # Coarse moving-object detector for LC10a. This is image-derived motion
        # energy, not ball coordinates or velocity from Pong's internal state.
        weights = motion * (0.25 + image)
        energy = float(weights.sum())
        centroid = 0.0
        lc_count = 0
        if energy > 1e-5 and (len(self.lc10a_l) or len(self.lc10a_r)):
            xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
            profile = weights.sum(axis=0)
            centroid = float((profile * xs).sum() / (profile.sum() + 1e-8))
            strength = float(np.clip(energy / max(1.0, 0.015 * h * w), 0.0, 1.0))
            left_drive = strength * max(0.0, -centroid)
            right_drive = strength * max(0.0, centroid)
            for group, amount in ((self.lc10a_l, left_drive), (self.lc10a_r, right_drive)):
                if amount > 0.02 and len(group):
                    indices.extend(group.tolist())
                    drives.extend([0.25 + 0.75 * amount] * len(group))
                    lc_count += len(group)

        if not indices:
            return EncodedVision(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), centroid, energy, 0, 0)
        idx = np.asarray(indices, dtype=np.int64)
        drv = np.asarray(drives, dtype=np.float32)
        unique, inv = np.unique(idx, return_inverse=True)
        combined = np.zeros(len(unique), dtype=np.float32)
        np.maximum.at(combined, inv, drv)
        return EncodedVision(unique, combined, centroid, energy, retinal_count, lc_count)
