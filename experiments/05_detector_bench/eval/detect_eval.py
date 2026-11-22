"""The single scorer every GATE-IR experiment and baseline reports through.

Two things it does that Ultralytics' own `.val()` cannot:

1. **Per-condition breakdown.** Ultralytics scores a split as one pool; the whole
   paper is about clear vs fog vs rain, so the split is scored again per
   condition from the recovered labels in conditions.csv.
2. **AP against object size, per condition.** Non-negotiable here. In UNIRI-TID
   condition is confounded with camera distance (companion paper Table 2: fog
   0-30 m, clear 50-160 m, rain 30-215 m), so a per-condition AP gap may simply
   be a large-vs-small-object gap. Reporting APs/APm/APl and the GT size
   distribution side by side is what lets a reader tell the two apart -- and
   what stops us claiming a weather effect that is really a scale effect.

Scoring is pycocotools, so the numbers are COCO-comparable rather than
Ultralytics-flavoured; the overall figure is cross-checked against `.val()` and
the delta is recorded in the output. Inference runs once over the split and the
detections are re-scored per subset, so the breakdown is nearly free.

    python bench/eval/detect_eval.py --run v8s --split test
"""

import argparse
import contextlib
import csv
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
ROOT = BENCH.parents[1]
sys.path.insert(0, str(BENCH))
import debugmode

DATA_ROOT = ROOT / "data" / "uniri_tid"
RUNS = BENCH / "runs"
RESULTS = BENCH / "results"
CONDITIONS = ("clear", "fog", "rain")

os.environ.setdefault("YOLO_CONFIG_DIR", str(BENCH / ".ultralytics"))

# COCO's own area thresholds, kept as-is so APs/APm/APl mean what they mean
# everywhere else in the literature.
SMALL, LARGE = 32 ** 2, 96 ** 2


def dataset_dir():
    return DATA_ROOT / ("yolo_debug" if debugmode.enabled() else "yolo")


def data_yaml():
    d = dataset_dir()
    return d / ("uniri_debug.yaml" if debugmode.enabled() else "uniri.yaml")


def conditions_by_frame():
    path = DATA_ROOT / "conditions.csv"
    return {int(r["frame_id"]): r["condition"] for r in csv.DictReader(open(path))}


def load_gt(images, sizes):
    """YOLO label files -> COCO ground truth. Boxes are in original pixel coords."""
    imgs, anns, aid = [], [], 1
    for i, p in enumerate(images):
        h, w = sizes[i]
        imgs.append(dict(id=i, file_name=p.name, width=int(w), height=int(h)))
        lbl = p.parent.parent.parent / "labels" / p.parent.name / f"{p.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in f[1:5])
            x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
            bw, bh = bw * w, bh * h
            anns.append(dict(id=aid, image_id=i, category_id=1, iscrowd=0,
                             bbox=[x, y, bw, bh], area=bw * bh))
            aid += 1
    return dict(images=imgs, annotations=anns,
                categories=[dict(id=1, name="Human")])


def score(gt, dets, image_ids):
    """COCOeval over a subset of image ids. Returns None if the subset is empty."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ids = set(image_ids)
    sub = dict(categories=gt["categories"],
               images=[im for im in gt["images"] if im["id"] in ids],
               annotations=[a for a in gt["annotations"] if a["image_id"] in ids])
    sub_dets = [d for d in dets if d["image_id"] in ids]
    if not sub["images"] or not sub["annotations"]:
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = sub
        coco.createIndex()
        if not sub_dets:
            # a model that predicts nothing is a legitimate outcome, not a crash
            return dict(images=len(sub["images"]), gt_boxes=len(sub["annotations"]),
                        detections=0, mAP50_95=0.0, mAP50=0.0, mAP75=0.0,
                        AP_small=0.0, AP_medium=0.0, AP_large=0.0, AR_100=0.0)
        dt = coco.loadRes([dict(d) for d in sub_dets])
        e = COCOeval(coco, dt, "bbox")
        e.params.imgIds = sorted(ids)
        e.evaluate()
        e.accumulate()
        e.summarize()
    s = [None if np.isnan(v) else round(float(v), 5) for v in e.stats]
    return dict(images=len(sub["images"]), gt_boxes=len(sub["annotations"]),
                detections=len(sub_dets),
                mAP50_95=s[0], mAP50=s[1], mAP75=s[2],
                AP_small=s[3], AP_medium=s[4], AP_large=s[5], AR_100=s[8])


def size_profile(gt, image_ids):
    """GT box-size distribution -- the evidence for or against the distance confound."""
    ids = set(image_ids)
    areas = np.array([a["area"] for a in gt["annotations"] if a["image_id"] in ids])
    if areas.size == 0:
        return None
    side = np.sqrt(areas)
    return dict(
        boxes=int(areas.size),
        median_box_side_px=round(float(np.median(side)), 1),
        p10_box_side_px=round(float(np.percentile(side, 10)), 1),
        p90_box_side_px=round(float(np.percentile(side, 90)), 1),
        frac_small=round(float((areas < SMALL).mean()), 4),
        frac_medium=round(float(((areas >= SMALL) & (areas < LARGE)).mean()), 4),
        frac_large=round(float((areas >= LARGE).mean()), 4),
    )


def evaluate(weights, split="test", name=None, imgsz=640, conf=0.001,
             iou=0.7, max_det=300, batch=16, write=True,
             data_root=None, yaml_path=None):
    from ultralytics import YOLO

    debugmode.banner("detect_eval")
    if debugmode.enabled():
        imgsz = debugmode.IMGSZ
        batch = debugmode.BATCH

    img_dir = (Path(data_root) if data_root else dataset_dir()) / "images" / split
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {img_dir}")

    model = YOLO(str(weights))

    # one inference pass; every subset is scored from these same detections
    dets, sizes = [], []
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        preds = model.predict(source=[str(p) for p in chunk], imgsz=imgsz, conf=conf,
                              iou=iou, max_det=max_det, verbose=False)
        for j, r in enumerate(preds):
            i = start + j
            sizes.append(r.orig_shape)  # (h, w)
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            cnf = b.conf.cpu().numpy()
            for (x1, y1, x2, y2), s in zip(xyxy, cnf):
                dets.append(dict(image_id=i, category_id=1, score=float(s),
                                 bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]))

    gt = load_gt(images, sizes)
    cond_of = conditions_by_frame()
    by_cond = defaultdict(list)
    for i, p in enumerate(images):
        by_cond[cond_of.get(int(p.stem), "unknown")].append(i)

    all_ids = list(range(len(images)))
    out = dict(
        name=name or Path(weights).parent.parent.name,
        weights=str(weights), split=split, imgsz=imgsz, conf=conf, iou=iou,
        debug=debugmode.enabled(),
        overall=score(gt, dets, all_ids),
        per_condition={c: score(gt, dets, by_cond[c]) for c in CONDITIONS if by_cond[c]},
        size_profile={c: size_profile(gt, by_cond[c]) for c in CONDITIONS if by_cond[c]},
    )
    out["size_profile"]["overall"] = size_profile(gt, all_ids)

    # Cross-check the overall figure against Ultralytics' own scorer. They use
    # different interpolation, so they never agree exactly; a large gap means one
    # of the two is being fed the wrong thing and every number here is suspect.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m = model.val(data=str(yaml_path) if yaml_path else str(data_yaml()),
                          split=split, imgsz=imgsz, verbose=False,
                          project=str(RUNS), name=f"_xcheck_{split}", exist_ok=True)
        ul = round(float(m.box.map), 5)
        out["crosscheck"] = dict(ultralytics_mAP50_95=ul,
                                 pycocotools_mAP50_95=out["overall"]["mAP50_95"],
                                 delta=round(abs(ul - out["overall"]["mAP50_95"]), 5))
    except Exception as e:  # a failed cross-check must not lose the real numbers
        out["crosscheck"] = dict(error=f"{type(e).__name__}: {e}")

    if write:
        dest = RESULTS / ("debug" if debugmode.enabled() else "")
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"eval_{out['name']}_{split}.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}")
    return out


def report(out):
    o = out["overall"]
    print(f"\n{out['name']}  split={out['split']}  "
          f"{o['images']} images, {o['gt_boxes']} GT boxes")
    hdr = f"{'subset':>9} {'mAP50-95':>9} {'mAP50':>7} {'APs':>7} {'APm':>7} {'APl':>7} {'med px':>7} {'%small':>7}"
    print(hdr)
    print("-" * len(hdr))
    for key, row in [("overall", o)] + list(out["per_condition"].items()):
        sp = out["size_profile"].get(key) or {}
        def f(v):
            return "   -   " if v is None else f"{v:7.4f}"
        print(f"{key:>9} {f(row['mAP50_95'])[1:]:>9} {f(row['mAP50'])} {f(row['AP_small'])} "
              f"{f(row['AP_medium'])} {f(row['AP_large'])} "
              f"{sp.get('median_box_side_px', 0):7.1f} {sp.get('frac_small', 0):7.1%}")
    x = out.get("crosscheck", {})
    if "delta" in x:
        print(f"\ncross-check vs ultralytics .val(): {x['ultralytics_mAP50_95']:.4f} "
              f"vs {x['pycocotools_mAP50_95']:.4f}  (delta {x['delta']:.4f})")
    elif x:
        print(f"\ncross-check failed: {x.get('error')}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run name under bench/runs/, uses its weights/best.pt")
    g.add_argument("--weights", help="explicit path to a .pt")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    weights = args.weights or (RUNS / args.run / "weights" / "best.pt")
    if not Path(weights).exists():
        raise SystemExit(f"no weights at {weights}")
    report(evaluate(weights, split=args.split, name=args.run, imgsz=args.imgsz,
                    conf=args.conf, iou=args.iou, batch=args.batch))


if __name__ == "__main__":
    main()
