import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = {r["frame"]: r for r in csv.DictReader((root / "audit" / "flir3_shifts_robust.csv").open())}
out = []
for split in ("train", "val", "test"):
    frames = [p.stem for p in (root / "data" / "flir3" / "warp" / "ir" / "images" / split).glob("*")]
    missing = sum(1 for f in frames if f not in rows)
    flagged = sum(1 for f in frames if rows.get(f, {}).get("flagged") == "1")
    cropped = sum(1 for f in frames if rows.get(f, {}).get("cropped") == "1")
    out.append(f"{split}: n={len(frames)} missing_shift={missing} flagged={flagged} ({flagged/len(frames):.3f}) cropped={cropped} ({cropped/len(frames):.3f})")
print("\n".join(out))
