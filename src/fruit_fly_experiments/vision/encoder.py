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


class NoVisionEncoder:
    """Negative-control encoder that removes all task visual drive."""

    def encode(self, frame: np.ndarray) -> EncodedVision:
        return EncodedVision(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            0.0,
            0.0,
            0,
            0,
        )


class ShuffledVisionEncoder:
    """Control preserving visual timing/strength while scrambling entry neurons.

    A fixed permutation maps every biologically selected visual-neuron index to a
    different retained MaleCNS neuron. Pixel processing, number of driven neurons,
    drive amplitudes and temporal structure are unchanged; only the biological
    identity of the connectome entry points is destroyed.
    """

    def __init__(self, base, neuron_count: int, shuffle_seed: int = 424242) -> None:
        self.base = base
        self.permutation = np.random.default_rng(shuffle_seed).permutation(int(neuron_count)).astype(np.int64)

    def encode(self, frame: np.ndarray) -> EncodedVision:
        out = self.base.encode(frame)
        mapped = self.permutation[out.neuron_indices] if len(out.neuron_indices) else out.neuron_indices.copy()
        return EncodedVision(
            mapped,
            out.drive.copy(),
            out.motion_centroid,
            out.motion_energy,
            out.retinal_count,
            out.lc10a_count,
        )


class VisionSubsetEncoder:
    """Ablation wrapper retaining only retinal or only LC10a visual drive."""

    def __init__(self, base, *, keep_retina: bool, keep_lc10a: bool) -> None:
        if not keep_retina and not keep_lc10a:
            raise ValueError("VisionSubsetEncoder must retain at least one visual pathway")
        self.base = base
        self.keep_retina = bool(keep_retina)
        self.keep_lc10a = bool(keep_lc10a)
        allowed: list[int] = []
        if self.keep_retina:
            allowed.extend(int(idx) for idx, _, _ in base.retinal)
        if self.keep_lc10a:
            allowed.extend(np.asarray(base.lc10a_l, dtype=np.int64).tolist())
            allowed.extend(np.asarray(base.lc10a_r, dtype=np.int64).tolist())
        self.allowed = np.unique(np.asarray(allowed, dtype=np.int64))

    def encode(self, frame: np.ndarray) -> EncodedVision:
        out = self.base.encode(frame)
        if len(out.neuron_indices):
            mask = np.isin(out.neuron_indices, self.allowed, assume_unique=False)
            indices = out.neuron_indices[mask].copy()
            drive = out.drive[mask].copy()
        else:
            indices = out.neuron_indices.copy()
            drive = out.drive.copy()
        return EncodedVision(
            indices,
            drive,
            out.motion_centroid,
            out.motion_energy,
            out.retinal_count if self.keep_retina else 0,
            out.lc10a_count if self.keep_lc10a else 0,
        )


class FlyVisualEncoder:
    """Pixel-only visual interface into real MaleCNS visual neuron identities.

    Pong is rendered egocentrically and rotated by 90 degrees so the game's
    vertical control axis maps onto the fly's left/right steering axis. Retinal
    columns receive local luminance/motion. LC10a receives a coarse small-target
    motion signal inferred only from image pixels; no Pong state is available.
    """

    def __init__(self, neurons: NeuronIndex, optic_columns: pd.DataFrame | None = None) -> None:
        self.neurons = neurons
        self._previous: np.ndarray | None = None
        self._body_to_idx = {int(b): i for i, b in enumerate(neurons.body_ids)}
        self.retinal = self._build_retinal_map(optic_columns) if optic_columns is not None else []
        self.lc10a_l = neurons.indices_for_types(["LC10a"], side="L")
        self.lc10a_r = neurons.indices_for_types(["LC10a"], side="R")
        if len(self.lc10a_l) == 0 or len(self.lc10a_r) == 0:
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

        # LC10a is a moving-target detector in the biological fly. For Pong we use
        # only rendered pixels to emphasize the bright moving target over the dimmer
        # paddles/centre line. This is an engineered sensory interface, not hidden
        # ball coordinates or velocity from the game state.
        bright_target = np.clip((image - 0.60) / 0.40, 0.0, 1.0)
        weights = motion * (0.12 + 1.88 * bright_target)
        energy = float(weights.sum())
        centroid = 0.0
        lc_count = 0
        if energy > 1e-5 and (len(self.lc10a_l) or len(self.lc10a_r)):
            xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
            profile = weights.sum(axis=0)
            centroid = float((profile * xs).sum() / (profile.sum() + 1e-8))
            strength = float(np.clip(energy / max(1.0, 0.008 * h * w), 0.0, 1.0))
            left_drive = strength * max(0.0, -centroid)
            right_drive = strength * max(0.0, centroid)
            for group, amount in ((self.lc10a_l, left_drive), (self.lc10a_r, right_drive)):
                if amount > 0.015 and len(group):
                    indices.extend(group.tolist())
                    drives.extend([0.30 + 0.70 * amount] * len(group))
                    lc_count += len(group)

        if not indices:
            return EncodedVision(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), centroid, energy, 0, 0)
        idx = np.asarray(indices, dtype=np.int64)
        drv = np.asarray(drives, dtype=np.float32)
        unique, inv = np.unique(idx, return_inverse=True)
        combined = np.zeros(len(unique), dtype=np.float32)
        np.maximum.at(combined, inv, drv)
        return EncodedVision(unique, combined, centroid, energy, retinal_count, lc_count)
