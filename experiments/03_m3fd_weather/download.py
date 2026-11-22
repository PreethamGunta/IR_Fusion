from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve()
while not (ROOT / "data" / "flir_adas_v2").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

snapshot_download(repo_id='nonameplease/M3FD_Detecion', repo_type='dataset',
                  local_dir=str(ROOT / 'data' / 'm3fd' / 'M3FD'))
