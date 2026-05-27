"""
temporal_smoothing.py — ParcelVision Stage 7
=============================================
Per-object dimension history and EMA smoothing.

With ZoeDepth giving metric depth, smoothing is the primary noise-suppression
mechanism — no scale hacks anywhere in this file.

Classes
-------
DimensionSmoother   — single-object EMA with jump gate
SceneSmoother       — manages one DimensionSmoother per tracked object_id
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class DimensionSmoother:
    """
    Exponential moving average smoother for a single object's dimensions.

    Jump gate:  if any dimension changes by more than `jump_threshold`
    (as a fraction of the previous value), the new measurement is accepted
    raw rather than blended.  This avoids EMA lag when the object genuinely
    moves or rotates into a new configuration.

    Parameters
    ----------
    alpha          : EMA weight on the NEW measurement  (0 < alpha < 1).
                     Higher = faster tracking, more noise.
    jump_threshold : accept raw if any dim changes > this fraction (default 0.40)
    history_len    : number of raw measurements kept for std / debug queries
    """

    def __init__(
        self,
        alpha:          float = 0.35,
        jump_threshold: float = 0.40,
        history_len:    int   = 10,
    ):
        self.alpha          = alpha
        self.jump_threshold = jump_threshold
        self._prev: Optional[dict] = None
        self._history: deque = deque(maxlen=history_len)

    # ─────────────────────────────────────────────────────────────────────────

    def update(self, new: dict) -> dict:
        """
        Smooth a new measurement.

        `new` must contain keys: length, width, height, volume_m3.
        Any extra keys (e.g. point_count, mean_depth_m) are passed through
        unchanged from `new`.

        Returns the smoothed dict.
        """
        self._history.append(new.copy())

        if self._prev is None:
            self._prev = {k: new[k] for k in ("length", "width", "height", "volume_m3")}
            return new.copy()

        # ── Jump gate ─────────────────────────────────────────────────────────
        for key in ("length", "width", "height"):
            ratio = abs(new[key] - self._prev[key]) / (self._prev[key] + 1e-8)
            if ratio > self.jump_threshold:
                self._prev = {k: new[k] for k in ("length", "width", "height", "volume_m3")}
                return new.copy()

        # ── EMA blend ─────────────────────────────────────────────────────────
        a = self.alpha
        smoothed_dims = {
            "length": a * new["length"] + (1 - a) * self._prev["length"],
            "width":  a * new["width"]  + (1 - a) * self._prev["width"],
            "height": a * new["height"] + (1 - a) * self._prev["height"],
        }
        smoothed_dims["volume_m3"] = (
            smoothed_dims["length"]
            * smoothed_dims["width"]
            * smoothed_dims["height"]
        )
        smoothed_dims = {
            k: round(v, 3) if k != "volume_m3" else round(v, 4)
            for k, v in smoothed_dims.items()
        }

        self._prev = smoothed_dims.copy()

        # Merge smoothed dims back with pass-through fields from `new`
        result = new.copy()
        result.update(smoothed_dims)
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear state — call when the tracked object disappears."""
        self._prev = None
        self._history.clear()

    @property
    def history(self) -> list[dict]:
        """Raw measurement history (up to history_len entries)."""
        return list(self._history)

    @property
    def std(self) -> dict:
        """Per-dimension standard deviation over the kept history."""
        if len(self._history) < 2:
            return {"length": 0.0, "width": 0.0, "height": 0.0}
        return {
            k: round(float(np.std([h[k] for h in self._history])), 4)
            for k in ("length", "width", "height")
        }


# ─────────────────────────────────────────────────────────────────────────────

class SceneSmoother:
    """
    Manages one DimensionSmoother per tracked object_id.

    Typical usage in the server:

        _smoother = SceneSmoother(alpha=0.35, jump_threshold=0.40)
        ...
        dims = _smoother.update(object_id, raw_dims)

    Stale smoothers (objects that have left the scene) are pruned
    automatically when reset() is called at the start of each scan.
    """

    def __init__(self, **smoother_kwargs):
        self._kwargs:    dict                       = smoother_kwargs
        self._smoothers: dict[int, DimensionSmoother] = {}

    def update(self, object_id: int, dims: dict) -> dict:
        """Smooth `dims` for `object_id`.  Creates a smoother on first call."""
        if object_id not in self._smoothers:
            self._smoothers[object_id] = DimensionSmoother(**self._kwargs)
        return self._smoothers[object_id].update(dims)

    def reset(self, object_id: int | None = None) -> None:
        """
        Reset smoothing state.

        reset()          → clear ALL object smoothers (e.g. new scan session)
        reset(object_id) → clear only one object (e.g. track ID reassignment)
        """
        if object_id is None:
            self._smoothers.clear()
        elif object_id in self._smoothers:
            self._smoothers[object_id].reset()

    def stdev(self, object_id: int) -> dict:
        """Return per-dimension std for a specific object, or {} if unknown."""
        smoother = self._smoothers.get(object_id)
        return smoother.std if smoother is not None else {}

    def tracked_ids(self) -> list[int]:
        """List of currently tracked object IDs."""
        return list(self._smoothers.keys())