"""EgoForce unified hand+forearm mapping (no weights)."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from ego2dex.config import _resolve_stages
from ego2dex.pipeline import Pipeline
from ego2dex.stages.base import RunContext, build_stage
from ego2dex.stages.hands.egoforce import EgoForce, arm4_from_egoforce
from ego2dex.topology import ARM4_ELBOW, ARM4_HIP, ARM4_SHOULDER, ARM4_WRIST, NUM_ARM_KEYPOINTS


def test_arm4_from_egoforce_conf_and_unobserved():
    forearm = np.array(
        [
            [10.0, 20.0],
            [15.0, 30.0],
            [40.0, 50.0],
        ],
        dtype=np.float64,
    )
    kp2d, kp3d = arm4_from_egoforce(forearm, conf=0.8)
    assert kp3d is None
    assert kp2d.shape == (NUM_ARM_KEYPOINTS, 3)
    assert kp2d[ARM4_SHOULDER, 2] == 0.0
    assert kp2d[ARM4_HIP, 2] == 0.0
    assert kp2d[ARM4_ELBOW, 0] == 10.0 and kp2d[ARM4_ELBOW, 2] == 0.8
    assert kp2d[ARM4_WRIST, 0] == 40.0 and kp2d[ARM4_WRIST, 2] == 0.8


def test_poses_from_outputs_left_right():
    outs = {
        "pred_j2d": np.zeros((2, 21, 2)),
        "pred_j3d": np.zeros((2, 21, 3)),
        "pred_arm_j2d": np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ]
        ),
        "pred_arm_j3d": np.array(
            [
                [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.4, 0.5, 0.6]],
                [[1.1, 1.2, 1.3], [0.0, 0.0, 0.0], [1.4, 1.5, 1.6]],
            ]
        ),
        "visible_hand": np.array([True, True]),
    }
    hands, arms = EgoForce.poses_from_outputs(outs)
    assert [str(h.side) for h in hands] == ["left", "right"]
    assert [str(a.side) for a in arms] == ["left", "right"]
    assert hands[0].keypoints_2d[0][2] == 1.0
    assert arms[0].keypoints_2d[ARM4_ELBOW][0] == 1.0
    assert arms[0].keypoints_2d[ARM4_WRIST][0] == 5.0
    assert arms[0].keypoints_3d[ARM4_ELBOW] == [0.1, 0.2, 0.3]
    assert arms[0].keypoints_3d[ARM4_WRIST] == [0.4, 0.5, 0.6]
    assert arms[0].keypoints_2d[ARM4_SHOULDER][2] == 0.0


def test_egoforce_dry_run_writes_hands_and_arms(synthetic_dir, tmp_path):
    cfg = OmegaConf.create(
        {
            "name": "egoforce_dry",
            "io": {"source": "generic", "max_frames": 3},
            "run": {
                "device": "cpu",
                "dry_run": True,
                "strict": True,
                "output_dir": str(tmp_path),
            },
            "mano": {"model_dir": None},
            "stages": [{"family": "hands", "name": "egoforce", "params": {"num_hands": 2}}],
        }
    )
    cfg.stages = _resolve_stages(OmegaConf.to_container(cfg.stages), tmp_path / "x.yaml")
    clip = Pipeline.from_config(cfg).run(synthetic_dir, output_dir=tmp_path)
    assert len(clip.frames) == 3
    for fa in clip.frames:
        assert len(fa.hands) == 2
        assert len(fa.arms) == 2
        for hand, arm in zip(fa.hands, fa.arms, strict=True):
            assert hand.side == arm.side
            hw = hand.kp2d_array()[0, :2]
            aw = arm.kp2d_array()[ARM4_WRIST, :2]
            assert np.allclose(hw, aw)
            assert arm.kp2d_array()[ARM4_SHOULDER, 2] == 0.0
            assert arm.kp2d_array()[ARM4_HIP, 2] == 0.0


def test_egoforce_stage_metadata():
    stage = build_stage("hands", "egoforce", {"num_hands": 2})
    stage.bind(RunContext(dry_run=True, device="cpu"))
    assert stage.name == "egoforce"
    assert stage.family == "hands"
    assert stage.extra == "egoforce"
    assert "CC-BY-NC" in (stage.license or "")
