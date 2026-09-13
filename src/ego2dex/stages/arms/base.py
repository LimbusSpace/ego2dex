"""Base class for arm / upper-limb pose stages.

Concrete stages override :meth:`infer`. The base owns frame iteration and the
deterministic **dry-run** path (synthetic but topologically valid ARM4
shoulder-elbow-wrist-hip chains so the whole graph runs with no weights).
"""

from __future__ import annotations

import numpy as np

from ...schema.core import ArmPose, HandSide
from ...topology import (
    ARM4_SHOULDER,
    NUM_ARM_KEYPOINTS,
    NUM_POSE33_KEYPOINTS,
    POSE33_TO_ARM4_LEFT,
    POSE33_TO_ARM4_RIGHT,
    ArmConvention,
)
from ..base import ClipAnnotation, Stage

# Canonical first-person ARM4 layout in a unit square (x right, y down).
# Shoulder near the image edge, wrist toward the bottom-center (egocentric
# reach). Used only to synthesize deterministic arms for dry-run / tests / viz.
_CANONICAL_ARM_2D: np.ndarray = np.array(
    [
        [0.18, 0.42],  # 0 shoulder
        [0.28, 0.58],  # 1 elbow
        [0.40, 0.72],  # 2 wrist
        [0.22, 0.78],  # 3 hip
    ],
    dtype=np.float64,
)


def canonical_arm_3d(scale: float = 0.35) -> np.ndarray:
    """A shoulder-origin metric-ish 4x3 arm (meters), for synthetic outputs."""
    xy = (_CANONICAL_ARM_2D - _CANONICAL_ARM_2D[ARM4_SHOULDER]) * np.array([1.0, -1.0])
    z = np.array([0.0, 0.04, 0.08, -0.05])
    pts = np.concatenate([xy * scale, z[:, None]], axis=1)
    return pts.astype(np.float64)


def _synthetic_pose33(arm_left: np.ndarray, arm_right: np.ndarray) -> np.ndarray:
    """Lift two ARM4 chains into a 33-point MediaPipe Pose array (zeros elsewhere)."""
    pose = np.zeros((NUM_POSE33_KEYPOINTS, 3), dtype=np.float64)
    for src, dst in zip(range(NUM_ARM_KEYPOINTS), POSE33_TO_ARM4_LEFT, strict=True):
        pose[dst] = arm_left[src]
    for src, dst in zip(range(NUM_ARM_KEYPOINTS), POSE33_TO_ARM4_RIGHT, strict=True):
        pose[dst] = arm_right[src]
    return pose


class ArmStageBase(Stage):
    family = "arms"
    keypoint_convention: ArmConvention = ArmConvention.ARM4

    def infer(self, image: np.ndarray, frame_id: int) -> list[ArmPose]:  # pragma: no cover
        """Run the real model on one BGR image -> list of ArmPose. Override me."""
        raise NotImplementedError(
            f"{self.cls_name()}.infer() is not wired to weights yet. "
            "Run with run.dry_run=true for the synthetic path, or install the "
            "model extra + weights (see docs/install.md) and implement the call."
        )

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        num_arms = int(self.param("num_arms", 2))
        for fa, img in self.iter_frames(clip):
            if self.dry_run or img is None:
                arms = self._synthetic(num_arms, img, fa.frame_id, clip)
            else:
                arms = self.infer(img, fa.frame_id)
            fa.arms.extend(arms)
        return clip

    def _synthetic(
        self, num_arms: int, image: np.ndarray | None, frame_id: int, clip: ClipAnnotation
    ) -> list[ArmPose]:
        h = image.shape[0] if image is not None else clip.video_meta.height or 256
        w = image.shape[1] if image is not None else clip.video_meta.width or 256
        out: list[ArmPose] = []
        sides = [HandSide.RIGHT, HandSide.LEFT][:num_arms]
        drift = ((frame_id % 10) - 5) * 0.008
        built: dict[str, np.ndarray] = {}
        for i, side in enumerate(sides):
            tmpl = _CANONICAL_ARM_2D.copy()
            if side == HandSide.LEFT:
                tmpl[:, 0] = 1.0 - tmpl[:, 0]
            cx = (0.32 + 0.36 * i) + drift
            kp2d = np.empty((NUM_ARM_KEYPOINTS, 3))
            kp2d[:, 0] = cx * w + (tmpl[:, 0] - 0.5) * (0.55 * w)
            kp2d[:, 1] = 0.52 * h + (tmpl[:, 1] - 0.5) * (0.55 * h)
            kp2d[:, 2] = 0.95
            kp3d = canonical_arm_3d() + np.array([0.12 * (i - 0.5), 0.0, 0.55])
            built[side.value if isinstance(side, HandSide) else str(side)] = kp2d
            out.append(
                ArmPose(
                    side=side,
                    keypoints_2d=kp2d,
                    keypoints_3d=kp3d,
                    keypoint_convention=self.keypoint_convention,
                    score=0.99,
                )
            )
        if len(out) == 2:
            pose33 = _synthetic_pose33(
                built.get("left", out[-1].kp2d_array()),
                built.get("right", out[0].kp2d_array()),
            )
            for arm in out:
                arm.pose33 = pose33.tolist()
        return out
