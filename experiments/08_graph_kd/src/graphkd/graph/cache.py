"""Teacher output cache: sharded, lazily loaded, fp16."""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path

import torch


class CacheWriter:
    def __init__(self, out_dir: Path, meta: dict, shard_size: int = 512):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.shard_size = shard_size
        self.records: dict[str, dict] = {}
        self.shard = 0

    def add(self, stem: str, rec: dict) -> None:
        self.records[stem] = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
                              for k, v in rec.items()}
        if len(self.records) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        path = self.out / f"shard_{self.shard:04d}.pt"
        torch.save({"records": self.records}, path)
        self.shard += 1
        self.records = {}

    def close(self) -> None:
        self.flush()
        self.meta = {**self.meta, "shards": self.shard}
        (self.out / "meta.json").write_text(json.dumps(self.meta, indent=1))


class TeacherCache:
    """Lazy reader with a small LRU over shards."""

    def __init__(self, root: Path, lru: int = 3):
        self.root = Path(root)
        meta_path = self.root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing teacher cache metadata: {meta_path}")
        self.meta = json.loads(meta_path.read_text())
        self.index = {}
        self.stem_to_shard = {}
        for p in sorted(self.root.glob("shard_*.pt")):
            recs = torch.load(p, map_location="cpu", weights_only=False)["records"]
            self.index[p] = list(recs.keys())
            for name in recs:
                self.stem_to_shard[name] = p
        self.lru = lru
        self._open: OrderedDict[Path, dict] = OrderedDict()

    def _shard_for(self, stem: str) -> Path | None:
        return self.stem_to_shard.get(stem)

    def get(self, stem: str) -> dict | None:
        p = self._shard_for(stem)
        if p is None:
            return None
        if p not in self._open:
            self._open[p] = torch.load(p, map_location="cpu", weights_only=False)["records"]
            if len(self._open) > self.lru:
                self._open.popitem(last=False)
        else:
            self._open.move_to_end(p)
        return self._open[p].get(stem)


_CACHES: dict[str, TeacherCache] = {}


def get_cache(path: str) -> TeacherCache:
    key = str(path)
    if key not in _CACHES:
        _CACHES[key] = TeacherCache(Path(path))
    return _CACHES[key]
