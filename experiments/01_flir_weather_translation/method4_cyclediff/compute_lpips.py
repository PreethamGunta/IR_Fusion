import os, sys, glob
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from taming.modules.losses.lpips import LPIPS

trans_dir = str(HERE / "sample_results" / "clear2fog_model_12_timestep50")
src_dir = str(ROOT / "data" / "flir_adas_v2" / "FLIR_ADAS_v2" / "test" / "clear")

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = LPIPS().to(dev).eval()

tfm = T.Compose([T.Resize((256,256)), T.ToTensor()])

# build paired list: translated filename -> source filename (same stem, .jpg/.png)
src_files = {}
for f in os.listdir(src_dir):
    stem = os.path.splitext(f)[0]
    src_files[stem] = os.path.join(src_dir, f)

trans_files = sorted(os.listdir(trans_dir))
pairs = []
for f in trans_files:
    if not f.lower().endswith((".png",".jpg",".jpeg")):
        continue
    stem = os.path.splitext(f)[0]
    # source stem may match directly or with suffix variations
    if stem in src_files:
        pairs.append((os.path.join(trans_dir, f), src_files[stem]))
    else:
        # try matching by removing trailing hash-ish suffix: source names like "xxx.jpg"
        # fallback: find source with same prefix up to first '-'
        pass

print(f"paired {len(pairs)} images", flush=True)

total = 0.0
n = 0
batch = 8
buf_a, buf_b = [], []
def run_batch():
    global total, n
    if not buf_a:
        return
    a = torch.stack(buf_a).to(dev)
    b = torch.stack(buf_b).to(dev)
    with torch.no_grad():
        v = model(a, b)
    total += float(v.mean().item()) * len(buf_a)
    n += len(buf_a)
    buf_a.clear(); buf_b.clear()

for tp, sp in pairs:
    img_t = Image.open(tp).convert("RGB")
    img_s = Image.open(sp).convert("RGB")
    buf_a.append(tfm(img_t))
    buf_b.append(tfm(img_s))
    if len(buf_a) >= batch:
        run_batch()
        if n % 200 == 0:
            print(f"  {n}/{len(pairs)}  running_mean={total/n:.6f}", flush=True)
run_batch()
print(f"LPIPS_mean={total/n:.6f} over {n} pairs", flush=True)
