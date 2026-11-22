"""Evaluator and geometry tests for experiment 09."""
import pytest
import torch

from eval.coco_eval import evaluate
from eval.geometry import box_iou_xywh, unletterbox


def _gt(boxes, names=("person",), h=512, w=640):
    anns = [{"id": i + 1, "image_id": 0, "category_id": 1, "bbox": b,
             "area": b[2] * b[3], "iscrowd": 0} for i, b in enumerate(boxes)]
    return {"images": [{"id": 0, "file_name": "x.jpg", "height": h, "width": w}],
            "annotations": anns,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}


def test_gt_as_pred_ap_one():
    m = evaluate(_gt([[10, 20, 30, 40]]),
                 [{"image_id": 0, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.9}])
    assert m["AP"] == pytest.approx(1.0, abs=1e-6)
    assert m["AP50"] == pytest.approx(1.0, abs=1e-6)
    assert m["AP75"] == pytest.approx(1.0, abs=1e-6)


def test_empty_predictions_zero():
    m = evaluate(_gt([[10, 20, 30, 40]]), [])
    assert m["AP"] == 0.0 and m["AP50"] == 0.0


def test_categories_and_names():
    gt = {
        "images": [{"id": 0, "file_name": "x.jpg", "height": 512, "width": 640}],
        "annotations": [
            {"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 0, "category_id": 2, "bbox": [100, 100, 20, 20], "area": 400, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
    }
    preds = [{"image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
             {"image_id": 0, "category_id": 2, "bbox": [100, 100, 20, 20], "score": 0.8}]
    m = evaluate(gt, preds)
    assert m["AP"] == pytest.approx(1.0, abs=1e-6)
    assert set(m["per_class_AP"]) == {"person", "car"}


def test_letterbox_roundtrip():
    boxes = torch.tensor([[100.0, 164.0, 200.0, 264.0]])
    out = unletterbox(boxes, 512, 640, 640)
    assert torch.allclose(out, torch.tensor([[100.0, 100.0, 200.0, 200.0]]), atol=1e-6)


def test_known_shift_iou():
    a = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    b = torch.tensor([[10.0, 0.0, 100.0, 100.0]])
    iou = box_iou_xywh(a, b)[0, 0]
    assert 0.80 < float(iou) < 0.83


def test_hflip_gt_as_pred():
    box = [10.0, 20.0, 30.0, 40.0]
    flipped = [640.0 - (10.0 + 30.0), 20.0, 30.0, 40.0]
    m = evaluate(_gt([flipped]),
                 [{"image_id": 0, "category_id": 1, "bbox": flipped, "score": 0.9}])
    assert m["AP"] == pytest.approx(1.0, abs=1e-6)
