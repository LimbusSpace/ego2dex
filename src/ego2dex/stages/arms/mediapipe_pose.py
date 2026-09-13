"""MediaPipe Pose Landmarker stage (lightweight, CPU, no SMPL).

License: Apache-2.0. BlazePose 33 landmarks; we slice each detected person
into per-side ARM4 (shoulder, elbow, wrist, hip) and keep the full 33-point
body on ``ArmPose.pose33``. Wrist (ARM4 index 2) is the same anatomical point
as hand keypoint 0.

Live path needs ``pip install 'ego2dex[mediapipe]'`` and the
``pose_landmarker_lite.task`` (or full/heavy) bundle:
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
Set its path via ``params.model_path`` (or ``EGO2DEX_MEDIAPIPE_POSE_TASK``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ...schema.core import ArmPose, HandSide
from ...topology import (
    NUM_POSE33_KEYPOINTS,
    ArmConvention,
    arm4_from_pose,
)
from ..base import ARMS
from .base import ArmStageBase


def _lm_conf(lm: object) -> float:
    vis = getattr(lm, "visibility", None)
    pres = getattr(lm, "presence", None)
    if vis is not None and pres is not None:
        return float(min(vis, pres))
    if vis is not None:
        return float(vis)
    if pres is not None:
        return float(pres)
    return 1.0


@ARMS.register("mediapipe_pose", aliases=("mp_pose", "blazepose"))
class MediaPipePoseArms(ArmStageBase):
    name = "mediapipe_pose"
    requires = ("mediapipe",)
    extra = "mediapipe"
    license = "Apache-2.0"
    license_url = "https://github.com/google-ai-edge/mediapipe"
    keypoint_convention = ArmConvention.ARM4

    def load(self) -> None:
        mp = self.import_or_raise("mediapipe")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = self.param("model_path") or os.environ.get("EGO2DEX_MEDIAPIPE_POSE_TASK")
        if not model_path or not Path(model_path).exists():
            raise ImportError(
                "MediaPipe PoseLandmarker needs the 'pose_landmarker_*.task' bundle. "
                "Download it (see this module's docstring) and pass "
                "params.model_path=/path/to/pose_landmarker_lite.task, or set "
                "EGO2DEX_MEDIAPIPE_POSE_TASK. (Or run with run.dry_run=true.)"
            )
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_poses=int(self.param("num_poses", 1)),
            min_pose_detection_confidence=float(self.param("min_detection_confidence", 0.5)),
            min_pose_presence_confidence=float(self.param("min_presence_confidence", 0.5)),
            min_tracking_confidence=float(self.param("min_tracking_confidence", 0.5)),
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._mp = mp

    def infer(self, image: np.ndarray, frame_id: int) -> list[ArmPose]:
        import cv2

        mp = self._mp
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        arms: list[ArmPose] = []
        for i, lms in enumerate(result.pose_landmarks):
            if len(lms) != NUM_POSE33_KEYPOINTS:
                continue
            pose33 = np.array(
                [[lm.x * w, lm.y * h, _lm_conf(lm)] for lm in lms],
                dtype=np.float64,
            )
            world = None
            if result.pose_world_landmarks and i < len(result.pose_world_landmarks):
                wl = result.pose_world_landmarks[i]
                world = np.array([[lm.x, lm.y, lm.z] for lm in wl], dtype=np.float64)

            for side in (HandSide.LEFT, HandSide.RIGHT):
                kp2d = arm4_from_pose(pose33, ArmConvention.MEDIAPIPE_POSE, side.value)
                if float(np.mean(kp2d[:, 2])) < float(self.param("min_arm_confidence", 0.2)):
                    continue
                kp3d = None
                if world is not None:
                    kp3d = arm4_from_pose(world, ArmConvention.MEDIAPIPE_POSE, side.value)
                score = float(np.mean(kp2d[:, 2]))
                arms.append(
                    ArmPose(
                        side=side,
                        keypoints_2d=kp2d,
                        keypoints_3d=kp3d,
                        keypoint_convention=self.keypoint_convention,
                        pose33=pose33,
                        score=score,
                    )
                )
        return arms
