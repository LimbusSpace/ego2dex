"""EgoForce unified hand + forearm stage (PRIMARY / default GPU path).

"EgoForce: Forearm-Guided Camera-Space 3D Hand Pose from a Monocular
Egocentric Camera", SIGGRAPH 2026 (DFKI). One transformer emits 21 hand
keypoints + MANO + a 3-joint forearm (elbow, mid, wrist). Shoulder / hip
are out of view in first-person footage; they stay unobserved in ARM4
until a later IK / headset-calib stage fills them.

This stage writes **both** ``FrameAnnotation.hands`` and
``FrameAnnotation.arms`` so the default pipeline does not need a second
full-body pose model (MediaPipe Pose remains an optional ``arms/`` backend).

License: CC-BY-NC 4.0 (code + weights) + MANO (research-only, gated).
Not for commercial use. Live path needs GPU + the official EgoForce repo
and ``_DATA/`` weights; dry-run emits synthetic hands + forearm-derived
ARM4 with no weights.

Install:
    bash scripts/install_models.sh egoforce
    # or clone https://github.com/dfki-av/EgoForce and run its
    # scripts/install.sh + scripts/download_model_weights.sh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...schema.core import ArmPose, HandPose, HandSide, MANOParams
from ...topology import (
    ARM4_ELBOW,
    ARM4_HIP,
    ARM4_SHOULDER,
    ARM4_WRIST,
    NUM_ARM_KEYPOINTS,
    NUM_FOREARM3_KEYPOINTS,
    NUM_HAND_KEYPOINTS,
    ArmConvention,
    HandConvention,
    arm4_from_forearm3,
)
from ..arms.base import canonical_arm_3d
from ..base import HANDS, ClipAnnotation
from .base import HandStageBase


def _xy_to_xyconf(xy: np.ndarray, conf: np.ndarray | float = 1.0) -> np.ndarray:
    """``(N, 2)`` pixels + scalar/vector conf -> ``(N, 3)``."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    out = np.empty((pts.shape[0], 3), dtype=np.float64)
    out[:, :2] = pts
    if np.isscalar(conf):
        out[:, 2] = float(conf)
    else:
        c = np.asarray(conf, dtype=np.float64).reshape(-1)
        out[:, 2] = c[: pts.shape[0]]
    return out


def arm4_from_egoforce(
    forearm3_2d: np.ndarray,
    forearm3_3d: np.ndarray | None = None,
    *,
    conf: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Map EgoForce forearm3 -> ARM4 ``(4, 3)`` 2D (with conf) and optional 3D.

    Shoulder and hip rows have conf 0 (unobserved). Elbow / wrist take the
    EgoForce joints 0 and 2.
    """
    f2d = np.asarray(forearm3_2d, dtype=np.float64)
    if f2d.ndim != 2 or f2d.shape[0] != NUM_FOREARM3_KEYPOINTS:
        raise ValueError(f"forearm3_2d must be ({NUM_FOREARM3_KEYPOINTS}, D), got {f2d.shape}")
    xy = f2d[:, :2]
    if f2d.shape[1] >= 3:
        src_conf = f2d[:, 2]
    elif np.isscalar(conf):
        src_conf = np.full(NUM_FOREARM3_KEYPOINTS, float(conf), dtype=np.float64)
    else:
        src_conf = np.asarray(conf, dtype=np.float64).reshape(-1)[:NUM_FOREARM3_KEYPOINTS]

    kp2d = np.zeros((NUM_ARM_KEYPOINTS, 3), dtype=np.float64)
    kp2d[ARM4_ELBOW, :2] = xy[0]
    kp2d[ARM4_ELBOW, 2] = float(src_conf[0])
    kp2d[ARM4_WRIST, :2] = xy[2]
    kp2d[ARM4_WRIST, 2] = float(src_conf[2])
    # ARM4_SHOULDER / ARM4_HIP stay 0,0,0 (unobserved)

    kp3d = None
    if forearm3_3d is not None:
        kp3d = arm4_from_forearm3(np.asarray(forearm3_3d, dtype=np.float64)[:, :3])
    return kp2d, kp3d


@HANDS.register("egoforce", aliases=("ego_force", "halo"))
class EgoForce(HandStageBase):
    """Unified first-person hand + forearm estimator.

    Registered under ``hands/`` because the pipeline selects the primary
    reconstruction backend from that family. It also writes ARM4 arms so
    a separate ``arms/mediapipe_pose`` stage is not required.
    """

    name = "egoforce"
    requires = ("torch",)
    extra = "egoforce"
    license = "CC-BY-NC 4.0 (code+weights) + MANO non-commercial"
    license_url = "https://github.com/dfki-av/EgoForce"
    keypoint_convention = HandConvention.STANDARD21

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        num_hands = int(self.param("num_hands", 2))
        for fa, img in self.iter_frames(clip):
            if self.dry_run or img is None:
                hands, arms = self._synthetic_limb(num_hands, img, fa.frame_id, clip)
            else:
                hands, arms = self.infer_limb(img, fa.frame_id)
            fa.hands.extend(hands)
            fa.arms.extend(arms)
        return clip

    def infer(self, image: np.ndarray, frame_id: int) -> list[HandPose]:  # pragma: no cover
        hands, _arms = self.infer_limb(image, frame_id)
        return hands

    def load(self) -> None:  # pragma: no cover - needs GPU + EgoForce repo
        torch = self.import_or_raise("torch")
        self._torch = torch
        self.device = self.param("device", self.ctx.device)

        repo = self.param("repo_dir") or self._discover_repo()
        if not repo:
            raise ImportError(
                "EgoForce live path needs the official repo cloned and on "
                "PYTHONPATH (https://github.com/dfki-av/EgoForce). Pass "
                "params.repo_dir=/path/to/EgoForce or set EGO2DEX_EGOFORCE_DIR. "
                "Then run its scripts/install.sh + scripts/download_model_weights.sh. "
                "(Or run with run.dry_run=true.)"
            )
        repo_path = Path(repo)
        import sys

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            from demo.inference import Inference  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Could not import EgoForce `demo.inference.Inference`. Clone "
                "https://github.com/dfki-av/EgoForce, install via "
                "scripts/install.sh, and pass params.repo_dir."
            ) from e

        ckpt = self.param("checkpoint")
        if ckpt:
            # Official loader reads cfg.POSE_3D.CHECKPOINT_PATH; we only
            # surface the path so the user can point at a local _DATA copy.
            import os

            os.environ.setdefault("EGO2DEX_EGOFORCE_CKPT", str(ckpt))

        self._engine = Inference()

    def _discover_repo(self) -> str | None:
        import os

        env = os.environ.get("EGO2DEX_EGOFORCE_DIR")
        if env and Path(env).is_dir():
            return env
        sibling = Path.cwd() / "EgoForce"
        if (sibling / "demo" / "inference.py").is_file():
            return str(sibling)
        return None

    def infer_limb(
        self, image: np.ndarray, frame_id: int
    ) -> tuple[list[HandPose], list[ArmPose]]:  # pragma: no cover
        """BGR image -> (hands, arms) from EgoForce ``Inference.run_outputs``."""
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        engine = getattr(self, "_engine", None)
        if engine is None:
            raise RuntimeError("EgoForce.load() did not create Inference()")

        if hasattr(engine, "run_outputs"):
            outs = engine.run_outputs(rgb, getattr(self, "device", self.ctx.device))
        else:
            raise NotImplementedError(
                "EgoForce demo.inference.Inference is missing run_outputs(); "
                "pin a current dfki-av/EgoForce checkout."
            )
        return self.poses_from_outputs(outs)

    @staticmethod
    def poses_from_outputs(
        outs: dict,
    ) -> tuple[list[HandPose], list[ArmPose]]:
        """Map EgoForce ``run_outputs`` dict -> (HandPose list, ArmPose list).

        Expected keys (from demo/inference.py): ``pred_j2d``, ``pred_j3d``,
        ``pred_arm_j2d``, ``pred_arm_j3d``, ``pred_vertices``, ``visible_hand``.
        Batch dim 0 = left, 1 = right when both sides are present.
        """
        j2d = np.asarray(outs["pred_j2d"])
        j3d = np.asarray(outs["pred_j3d"]) if outs.get("pred_j3d") is not None else None
        arm_j2d = np.asarray(outs["pred_arm_j2d"])
        arm_j3d = np.asarray(outs["pred_arm_j3d"]) if outs.get("pred_arm_j3d") is not None else None
        verts = outs.get("pred_vertices")
        visible = outs.get("visible_hand")
        if visible is None:
            visible = np.ones((j2d.shape[0],), dtype=bool)
        else:
            visible = np.asarray(visible).reshape(-1).astype(bool)

        n = int(j2d.shape[0])
        sides = [HandSide.LEFT, HandSide.RIGHT][:n]
        if n == 1:
            explicit = outs.get("hand_side")
            if explicit in ("left", "right"):
                sides = [HandSide(explicit)]

        hands: list[HandPose] = []
        arms: list[ArmPose] = []
        for i in range(n):
            if i < len(visible) and not bool(visible[i]):
                continue
            kp2d = _as_hand_xyconf(j2d[i])
            kp3d = None if j3d is None else np.asarray(j3d[i], dtype=np.float64).reshape(-1, 3)
            vertices = None
            if verts is not None:
                vertices = np.asarray(verts[i], dtype=np.float64)
            mano = None
            betas = outs.get("pred_betas")
            go = outs.get("pred_global_orient")
            hp = outs.get("pred_hand_pose")
            if betas is not None and go is not None and hp is not None:
                mano = MANOParams(
                    global_orient=np.asarray(go[i]).reshape(-1)[:3].tolist(),
                    pose=np.asarray(hp[i]).reshape(-1)[:45].tolist(),
                    betas=np.asarray(betas[i]).reshape(-1)[:10].tolist(),
                    trans=(kp3d[0].tolist() if kp3d is not None else [0.0, 0.0, 0.0]),
                )
            score = float(np.mean(kp2d[:, 2]))
            hands.append(
                HandPose(
                    side=sides[i] if i < len(sides) else HandSide.UNKNOWN,
                    keypoints_2d=kp2d,
                    keypoints_3d=kp3d,
                    mano=mano,
                    vertices=vertices,
                    keypoint_convention=HandConvention.STANDARD21,
                    score=score,
                )
            )
            a2d_src = arm_j2d[i]
            a3d_src = None if arm_j3d is None else arm_j3d[i]
            arm_kp2d, arm_kp3d = arm4_from_egoforce(a2d_src, a3d_src, conf=score)
            arms.append(
                ArmPose(
                    side=sides[i] if i < len(sides) else HandSide.UNKNOWN,
                    keypoints_2d=arm_kp2d,
                    keypoints_3d=arm_kp3d,
                    keypoint_convention=ArmConvention.ARM4,
                    score=float(np.mean(arm_kp2d[[ARM4_ELBOW, ARM4_WRIST], 2])),
                )
            )
        return hands, arms

    def _synthetic_limb(
        self,
        num_hands: int,
        image: np.ndarray | None,
        frame_id: int,
        clip: ClipAnnotation,
    ) -> tuple[list[HandPose], list[ArmPose]]:
        """Dry-run: canonical 21-kpt hands + forearm-derived ARM4 (no shoulder/hip)."""
        h = image.shape[0] if image is not None else clip.video_meta.height or 256
        w = image.shape[1] if image is not None else clip.video_meta.width or 256
        hands = self._synthetic(num_hands, image, frame_id, clip)
        arms: list[ArmPose] = []
        for hand in hands:
            kp2d_h = hand.kp2d_array()
            wrist_xy = kp2d_h[0, :2]
            side = HandSide(hand.side)
            # Elbow sits toward the image edge (egocentric forearm coming in
            # from the matching side); mid is unused (dropped by ARM4 map).
            edge_x = 0.08 * w if side == HandSide.RIGHT else 0.92 * w
            edge_y = 0.55 * h
            elbow_xy = np.array(
                [0.65 * edge_x + 0.35 * wrist_xy[0], 0.65 * edge_y + 0.35 * wrist_xy[1]]
            )
            mid_xy = 0.5 * (elbow_xy + wrist_xy)
            forearm = np.stack([elbow_xy, mid_xy, wrist_xy], axis=0)
            kp2d, _ = arm4_from_egoforce(forearm, conf=float(hand.score))
            kp3d_h = hand.kp3d_array()
            if kp3d_h is not None:
                # Place the 3D elbow along -Y from the wrist so ARM4 wrist
                # coincides with hand kpt 0; shoulder/hip remain origin.
                kp3d = np.zeros((NUM_ARM_KEYPOINTS, 3), dtype=np.float64)
                kp3d[ARM4_WRIST] = kp3d_h[0]
                offset = np.array([0.12 if side == HandSide.RIGHT else -0.12, 0.18, 0.04])
                kp3d[ARM4_ELBOW] = kp3d_h[0] + offset
            else:
                kp3d = canonical_arm_3d()
                kp3d[ARM4_SHOULDER] = 0.0
                kp3d[ARM4_HIP] = 0.0
            arms.append(
                ArmPose(
                    side=side,
                    keypoints_2d=kp2d,
                    keypoints_3d=kp3d,
                    keypoint_convention=ArmConvention.ARM4,
                    score=float(hand.score),
                )
            )
        return hands, arms


def _as_hand_xyconf(src: np.ndarray) -> np.ndarray:
    arr = np.asarray(src, dtype=np.float64)
    if arr.ndim != 2:
        arr = arr.reshape(NUM_HAND_KEYPOINTS, -1)
    if arr.shape[0] != NUM_HAND_KEYPOINTS:
        raise ValueError(f"hand keypoints must be (21, D), got {arr.shape}")
    if arr.shape[1] >= 3:
        out = np.empty((NUM_HAND_KEYPOINTS, 3), dtype=np.float64)
        out[:, :2] = arr[:, :2]
        out[:, 2] = arr[:, 2]
        return out
    return _xy_to_xyconf(arr[:, :2], 1.0)
