"""Regression tests for the custom COCO evaluator and letterbox mapping."""
import pytest
import torch

from graphkd.training.evaluate import coco_eval, resolve_device, unletterbox


def _gt_one(box, name="person", cid=1):
    return {
        "images": [{"id": 0, "file_name": "x.jpg", "height": 512, "width": 640}],
        "annotations": [{"id": 1, "image_id": 0, "category_id": cid,
                         "bbox": list(box), "area": box[2] * box[3], "iscrowd": 0}],
        "categories": [{"id": cid, "name": name}],
    }


def _pred_one(box, score=0.9, cid=1):
    return [{"image_id": 0, "category_id": cid, "bbox": list(box), "score": score}]


def test_gt_as_pred_single_image_ap_is_one():
    gt = _gt_one([10.0, 20.0, 30.0, 40.0])
    m = coco_eval(gt, _pred_one([10.0, 20.0, 30.0, 40.0]))
    assert m["AP"] == pytest.approx(1.0, abs=1e-6)
    assert m["AP50"] == pytest.approx(1.0, abs=1e-6)
    assert m["AP75"] == pytest.approx(1.0, abs=1e-6)
    assert m["per_class_AP"]["person"] == pytest.approx(1.0, abs=1e-6)


def test_gt_as_pred_two_classes_names_match():
    gt = {
        "images": [{"id": 0, "file_name": "x.jpg", "height": 512, "width": 640}],
        "annotations": [
            {"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20],
             "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 0, "category_id": 2, "bbox": [100, 100, 20, 20],
             "area": 400, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "dog"}],
    }
    preds = [
        {"image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 0, "category_id": 2, "bbox": [100, 100, 20, 20], "score": 0.8},
    ]
    m = coco_eval(gt, preds)
    assert m["AP"] == pytest.approx(1.0, abs=1e-6)
    assert set(m["per_class_AP"]) == {"car", "dog"}


def test_empty_predictions_return_zero():
    m = coco_eval(_gt_one([10.0, 20.0, 30.0, 40.0]), [])
    assert m["AP"] == 0.0 and m["AP50"] == 0.0


def test_unletterbox_640x512():
    # 640x512 -> square 640 canvas pads 64 px top/bottom, gain 1.
    boxes = torch.tensor([[100.0, 164.0, 200.0, 264.0]])
    out = unletterbox(boxes, h0=512, w0=640, imgsz=640)
    assert torch.allclose(out, torch.tensor([[100.0, 100.0, 200.0, 200.0]]), atol=1e-6)


def test_letterbox_roundtrip_smaller_image():
    # 320x640 -> gain 1 (limited by width), pad 160 top/bottom.
    boxes = torch.tensor([[50.0, 200.0, 150.0, 260.0]])
    out = unletterbox(boxes, h0=320, w0=640, imgsz=640)
    assert torch.allclose(out, torch.tensor([[50.0, 40.0, 150.0, 100.0]]), atol=1e-6)


def test_resolve_device_cpu_on_gpu_node():
    assert resolve_device("cpu", cuda_available=True).type == "cpu"
    assert resolve_device("0", cuda_available=True) == torch.device("cuda:0")
    assert resolve_device("0", cuda_available=False).type == "cpu"


def test_hflip_gt_as_pred_still_ap_one():
    # A horizontal flip applied identically to GT and prediction must not
    # change AP; this guards the top-left xywh convention used by the writer.
    gt = _gt_one([10.0, 20.0, 30.0, 40.0])
    flipped = [640.0 - (10.0 + 30.0), 20.0, 30.0, 40.0]
    gt["annotations"][0]["bbox"] = flipped
    m = coco_eval(gt, _pred_one(flipped))
    assert m["AP"] == pytest.approx(1.0, abs=1e-6)
