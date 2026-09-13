"""Topology + MANO constant correctness (the #1 interop surface)."""

from __future__ import annotations

import numpy as np
import pytest

from ego2dex import topology as T
from ego2dex.topology import (
    ARM4_NAMES,
    ARM_EDGES,
    HAND_EDGES,
    MANO_FULL_POSE_DIM,
    MANO_POSE_DIM,
    MANOPTH_TO_STANDARD21,
    NUM_ARM_KEYPOINTS,
    NUM_HAND_KEYPOINTS,
    NUM_POSE33_KEYPOINTS,
    POSE33_TO_ARM4_LEFT,
    POSE33_TO_ARM4_RIGHT,
    STANDARD21_NAMES,
    STANDARD21_TO_MANOPTH,
    ArmConvention,
    HandConvention,
    apply_remap,
    arm4_from_pose,
    remap_keypoints,
    split_full_pose,
    wrist_relative,
)


def test_standard21_names_count_and_order():
    assert len(STANDARD21_NAMES) == NUM_HAND_KEYPOINTS == 21
    assert STANDARD21_NAMES[0] == "WRIST"
    # thumb..pinky, base->tip, contiguous blocks
    assert STANDARD21_NAMES[4] == "THUMB_TIP"
    assert STANDARD21_NAMES[8] == "INDEX_TIP"
    assert STANDARD21_NAMES[20] == "PINKY_TIP"


def test_manopth_map_is_a_permutation():
    assert sorted(MANOPTH_TO_STANDARD21) == list(range(21))
    assert sorted(STANDARD21_TO_MANOPTH) == list(range(21))


def test_manopth_to_standard21_is_inverse_roundtrip():
    x = np.arange(21 * 3).reshape(21, 3)
    y = apply_remap(x, HandConvention.MANO_NATIVE, HandConvention.STANDARD21)
    z = apply_remap(y, HandConvention.STANDARD21, HandConvention.MANO_NATIVE)
    assert np.array_equal(x, z)


def test_mano_native_wrist_and_thumb_mapping():
    # standard21[0]=wrist comes from mano native 0; thumb base from mano 13
    assert MANOPTH_TO_STANDARD21[0] == 0
    assert MANOPTH_TO_STANDARD21[1] == 13  # THUMB_CMC <- mano thumb base
    assert MANOPTH_TO_STANDARD21[4] == 16  # THUMB_TIP <- first appended tip


def test_standard_family_is_identity():
    for a in (HandConvention.STANDARD21, HandConvention.MEDIAPIPE, HandConvention.OPENPOSE):
        for b in (HandConvention.STANDARD21, HandConvention.MEDIAPIPE, HandConvention.OPENPOSE):
            assert remap_keypoints(a, b) == tuple(range(21))


def test_unknown_remap_raises():
    with pytest.raises(ValueError):
        remap_keypoints("standard21", "bogus")


def test_pose_dims_45_vs_48():
    assert MANO_POSE_DIM == 45
    assert MANO_FULL_POSE_DIM == 48
    g, p = split_full_pose(np.zeros(48))
    assert g.shape == (3,) and p.shape == (45,)
    with pytest.raises(ValueError):
        split_full_pose(np.zeros(45))  # missing the 3 global -> must fail loudly


def test_hand_edges_valid():
    assert len(HAND_EDGES) == 20  # 5 fingers x 4 segments
    for a, b in HAND_EDGES:
        assert 0 <= a < 21 and 0 <= b < 21


def test_tip_vertex_id_sets():
    assert T.MANO_TIP_VERTEX_IDS["manopth"]["right"] == [745, 317, 444, 556, 673]
    assert T.MANO_TIP_VERTEX_IDS["manopth"]["left"][2] == 445  # left differs at middle
    assert T.MANO_TIP_VERTEX_IDS["otaheri"]["right"] == [744, 320, 443, 554, 671]


def test_wrist_relative():
    kp = np.random.RandomState(3).rand(21, 3)
    wr = wrist_relative(kp)
    assert np.allclose(wr[0], 0.0)


def test_arm4_names_and_edges():
    assert len(ARM4_NAMES) == NUM_ARM_KEYPOINTS == 4
    assert ARM4_NAMES == ("SHOULDER", "ELBOW", "WRIST", "HIP")
    assert ARM_EDGES == ((3, 0), (0, 1), (1, 2))
    for a, b in ARM_EDGES:
        assert 0 <= a < 4 and 0 <= b < 4


def test_arm4_from_mediapipe_pose():
    pose = np.arange(NUM_POSE33_KEYPOINTS * 3).reshape(NUM_POSE33_KEYPOINTS, 3).astype(np.float64)
    left = arm4_from_pose(pose, ArmConvention.MEDIAPIPE_POSE, "left")
    right = arm4_from_pose(pose, ArmConvention.MEDIAPIPE_POSE, "right")
    assert left.shape == (4, 3)
    assert np.array_equal(left, pose[list(POSE33_TO_ARM4_LEFT)])
    assert np.array_equal(right, pose[list(POSE33_TO_ARM4_RIGHT)])
    # person-centric: left shoulder is BlazePose 11, right is 12
    assert POSE33_TO_ARM4_LEFT == (11, 13, 15, 23)
    assert POSE33_TO_ARM4_RIGHT == (12, 14, 16, 24)


def test_arm4_from_coco17():
    pose = np.arange(17 * 2).reshape(17, 2).astype(np.float64)
    left = arm4_from_pose(pose, ArmConvention.COCO17, "left")
    assert np.array_equal(left, pose[[5, 7, 9, 11]])


def test_arm4_passthrough_and_bad_side():
    arm = np.ones((4, 3))
    assert np.array_equal(arm4_from_pose(arm, ArmConvention.ARM4, "left"), arm)
    with pytest.raises(ValueError):
        arm4_from_pose(arm, ArmConvention.ARM4, "unknown")
