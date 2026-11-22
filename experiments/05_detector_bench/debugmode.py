"""Shared DEBUG switch: `DEBUG=1` shrinks every stage to a handful of images.

The point is to prove the plumbing works -- splits load, training steps, the
evaluator produces a per-condition breakdown, results serialise -- in a couple of
minutes, instead of discovering a typo four hours into a GPU run that only
reports back the next morning.

Debug numbers are meaningless. Everything written in debug mode is kept out of
bench/results/ so it can never be mistaken for a real result.
"""

import os

_TRUTHY = {"1", "true", "yes", "on"}


def enabled() -> bool:
    return os.environ.get("DEBUG", "").strip().lower() in _TRUTHY


# Two frames per condition per split. Tiny, but every condition stays non-empty,
# so the per-condition breakdown is actually exercised rather than silently
# skipped -- which is the part most likely to be broken.
FRAMES_PER_CONDITION = 2
EPOCHS = 2
BATCH = 2
IMGSZ = 320


def banner(where: str) -> None:
    if enabled():
        print(f"*** DEBUG MODE [{where}]: tiny subset, {EPOCHS} epochs, "
              f"imgsz {IMGSZ} -- numbers are meaningless ***", flush=True)
