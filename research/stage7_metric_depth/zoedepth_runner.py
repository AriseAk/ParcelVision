"""
zoedepth_runner.py — ParcelVision Stage 7
==========================================
Wraps ZoeDepth to produce absolute metric depth maps (metres).

Unlike DepthAnything / MiDaS, ZoeDepth outputs *metric* depth directly.
No scale factor, no camera-height hack, no floor-disparity calibration.

Model variants (preference order for indoor parcel scanning):
  ZoeD_NK  — trained on NYU (indoor) + KITTI (outdoor)  ← default, best general use
  ZoeD_N   — NYU only (indoor), lighter, still fully metric
  ZoeD_K   — KITTI only (outdoor) — skip for indoor use

Public API
----------
  ZoeDepthRunner          — class, instantiate once, call .infer(bgr) per frame
  get_runner(variant)     — module-level singleton accessor
  infer_metric_depth(bgr) — one-liner convenience wrapper

CLI quick-test
--------------
  python zoedepth_runner.py path/to/image.jpg
  python zoedepth_runner.py          # live webcam
"""

from __future__ import annotations

import time
import traceback

import cv2
import numpy as np
import torch
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_ZOEDEPTH_REPO = "isl-org/ZoeDepth"

# Sane depth clamps for indoor parcel scanning
DEPTH_MIN_M = 0.10
DEPTH_MAX_M = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Runner class
# ─────────────────────────────────────────────────────────────────────────────

class ZoeDepthRunner:
    """
    Lazy-loading ZoeDepth wrapper.

    Instantiate once at server startup; call .infer(bgr) per frame.
    Thread-safe for read (inference) after initial load.
    Do NOT call .load() from multiple threads simultaneously.
    """

    def __init__(self, variant: str = "ZoeD_NK", device: str | None = None):
        """
        Parameters
        ----------
        variant : "ZoeD_NK" (default) | "ZoeD_N" | "ZoeD_K"
        device  : "cuda" | "cpu" | None (auto-detect)
        """
        self.variant = variant
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model  = None
        self._loaded = False

    # ─────────────────────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Download / load model weights.

        Called automatically on first .infer() call.
        Pre-call at server startup to avoid first-request latency.

        Falls back through the variant chain if the requested model fails.
        """
        if self._loaded:
            return

        fallback_order = [self.variant, "ZoeD_NK", "ZoeD_N"]
        # Deduplicate while preserving order
        seen     = set()
        attempts = [v for v in fallback_order if not (v in seen or seen.add(v))]

        last_err = None
        for variant in attempts:
            try:
                print(f"[ZoeDepth] Loading {variant} on {self.device} …")
                t0 = time.perf_counter()
                model = torch.hub.load(
                    _ZOEDEPTH_REPO,
                    variant,
                    pretrained=True,
                    verbose=False,
                )
                model.eval().to(self.device)
                self._model  = model
                self.variant = variant
                self._loaded = True
                print(
                    f"[ZoeDepth] {variant} ready in "
                    f"{time.perf_counter() - t0:.1f}s  device={self.device}"
                )
                return
            except Exception as exc:
                last_err = exc
                print(f"[ZoeDepth] {variant} failed: {exc}")

        raise RuntimeError(
            f"All ZoeDepth variants failed.  Last error: {last_err}\n"
            "Ensure the following are installed:\n"
            "  pip install torch torchvision timm einops"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """
        Run ZoeDepth on one BGR frame.

        Parameters
        ----------
        bgr : np.ndarray  shape (H, W, 3), dtype uint8, OpenCV BGR order

        Returns
        -------
        depth : np.ndarray  shape (H, W), dtype float32
                Absolute depth in **metres** along the optical axis (Z-depth,
                not Euclidean distance).
                Clamped to [DEPTH_MIN_M, DEPTH_MAX_M].
                Use directly — no normalisation, no scale factor needed.
        """
        self.load()   # no-op after first call

        rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            depth: np.ndarray = self._model.infer_pil(rgb_pil)

        depth = np.asarray(depth, dtype=np.float32)
        depth = np.clip(depth, DEPTH_MIN_M, DEPTH_MAX_M)

        # Guard: resize if hub returns a different spatial size
        h, w = bgr.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def model_type(self) -> str:
        return self.variant if self._loaded else "not_loaded"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def depth_stats(self, depth: np.ndarray) -> dict:
        """
        Quick sanity-check statistics.  Log these on every frame while
        validating the build (step 2 of the README build order).

        Expected ranges for indoor parcel scanning at 0.5–3 m:
            mean_m  ~ 0.5–3.0
            std_m   < 1.0 (for a single object scene)
        """
        return {
            "min_m":  round(float(depth.min()),  3),
            "max_m":  round(float(depth.max()),  3),
            "mean_m": round(float(depth.mean()), 3),
            "std_m":  round(float(depth.std()),  3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_singleton: ZoeDepthRunner | None = None


def get_runner(variant: str = "ZoeD_NK") -> ZoeDepthRunner:
    """
    Return the shared ZoeDepthRunner, creating it on first call.

    The singleton is created with the first variant requested.
    Subsequent calls with a different variant return the existing instance
    unchanged — call ZoeDepthRunner() directly if you need a separate instance.
    """
    global _singleton
    if _singleton is None:
        _singleton = ZoeDepthRunner(variant=variant)
    return _singleton


def infer_metric_depth(bgr: np.ndarray) -> np.ndarray:
    """
    One-liner depth inference using the shared singleton.

    Equivalent to get_runner().infer(bgr).
    Import this in the server for a single-line depth call.
    """
    return get_runner().infer(bgr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI quick-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("ZoeDepth quick-test  (ParcelVision Stage 7)")
    print("=" * 55)

    runner = ZoeDepthRunner()

    if len(sys.argv) > 1:
        # ── Image file mode ───────────────────────────────────────────────────
        path = sys.argv[1]
        bgr  = cv2.imread(path)
        if bgr is None:
            print(f"ERROR: cannot read '{path}'")
            sys.exit(1)

        t0    = time.perf_counter()
        depth = runner.infer(bgr)
        ms    = (time.perf_counter() - t0) * 1000

        stats = runner.depth_stats(depth)
        print(f"Image   : {path}  ({bgr.shape[1]}×{bgr.shape[0]})")
        print(f"Inference: {ms:.0f} ms")
        print(f"Stats   : {stats}")
        print()
        print("Sanity check — expected for indoor scenes at 0.5–3 m:")
        ok = 0.3 <= stats["mean_m"] <= 5.0
        print(f"  mean_m in [0.3, 5.0]: {'✅' if ok else '❌  CHECK LIGHTING / MODEL VARIANT'}")

        # Save colourmap for visual inspection
        vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        vis = (vis * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        out = path.rsplit(".", 1)[0] + "_zoedepth_depth.png"
        cv2.imwrite(out, vis)
        print(f"Colourmap saved → {out}")

    else:
        # ── Webcam mode ───────────────────────────────────────────────────────
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: cannot open webcam")
            sys.exit(1)

        print("Webcam mode — press Q to quit")
        frame_n = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0    = time.perf_counter()
            depth = runner.infer(frame)
            ms    = (time.perf_counter() - t0) * 1000

            frame_n += 1
            stats = runner.depth_stats(depth)
            print(f"Frame {frame_n:4d}  {ms:5.0f}ms  {stats}")

            vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            vis = (vis * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
            cv2.imshow("ZoeDepth — metric depth (INFERNO)", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()