"""WBF unit tests for E03 late fusion."""
import math

import torch

from gfyolo.wbf import wbf_merge


def _box(x, y, w, h):
    return [x, y, x + w, y + h]


def test_identical_boxes_fuse_to_one():
    b = _box(10, 10, 20, 20)
    boxes, scores, labels = wbf_merge(
        [b], [0.8], [0],
        [b], [0.6], [0],
        iou_thr=0.55)
    assert boxes.shape[0] == 1
    assert math.isclose(scores[0], 0.7, abs_tol=1e-6)
    assert labels[0] == 0
    for k in range(4):
        assert math.isclose(float(boxes[0, k]), b[k], abs_tol=1e-4)


def test_disjoint_boxes_stay_separate():
    boxes, scores, labels = wbf_merge(
        [_box(0, 0, 10, 10)], [0.9], [0],
        [_box(50, 50, 10, 10)], [0.8], [0])
    assert boxes.shape[0] == 2
    assert scores[0] >= scores[1]  # sorted desc


def test_one_box_per_source_per_cluster():
    # source A has two mutually overlapping boxes; only one can join a cluster.
    boxes, scores, labels = wbf_merge(
        [_box(0, 0, 10, 10), _box(1, 0, 10, 10)], [0.9, 0.5], [0, 0],
        [_box(0, 0, 10, 10)], [0.8], [0],
        iou_thr=0.55)
    # cluster 1: A[0] + B; cluster 2: A[1] alone (A already in cluster 1)
    assert boxes.shape[0] == 2
    # fused top box is the weighted mean of the two matched members
    assert math.isclose(scores[0], 0.85, abs_tol=1e-6)


def test_label_mismatch_prevents_fusion():
    b = _box(10, 10, 20, 20)
    boxes, scores, labels = wbf_merge(
        [b], [0.9], [0],
        [b], [0.8], [1],
        iou_thr=0.55, class_agnostic=False)
    assert boxes.shape[0] == 2
    assert sorted(labels) == [0, 1]


def test_scale_members_downweights_single_source():
    b = _box(10, 10, 20, 20)
    _, s_off, _ = wbf_merge([b], [0.8], [0], [], [], [], scale_members=False)
    _, s_on, _ = wbf_merge([b], [0.8], [0], [], [], [], scale_members=True)
    assert math.isclose(s_off[0], 0.8, abs_tol=1e-6)
    assert math.isclose(s_on[0], 0.4, abs_tol=1e-6)  # 0.8 * 1/2


def test_empty_inputs():
    boxes, scores, labels = wbf_merge([], [], [], [], [], [])
    assert boxes.shape == (0, 4) and scores == [] and labels == []
