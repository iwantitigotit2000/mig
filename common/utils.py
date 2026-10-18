"""Small helpers shared by all three environments (only stdlib + yaml + numpy)."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_paths():
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "paths.yaml", encoding="utf-8"))
    root = os.environ.get("MIG_ASSETS", cfg["assets_root"])

    def fmt(v):
        if isinstance(v, dict):
            return {k: fmt(x) for k, x in v.items()}
        if isinstance(v, str):
            return v.replace("{assets_root}", root)
        return v

    cfg = fmt(cfg)
    cfg["assets_root"] = root
    return cfg


def load_yaml(path):
    return yaml.safe_load(open(path, encoding="utf-8"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sha1_file(path, chunk=1 << 22, limit=None):
    """sha1 of a file; `limit` bytes only if given (fast fingerprint of huge checkpoints)."""
    h = hashlib.sha1()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit and n >= limit:
                break
    return h.hexdigest()


def ckpt_fingerprint(path):
    st = os.stat(path)
    return {"path": str(path), "size": st.st_size, "sha1_first_64MB": sha1_file(path, limit=64 << 20)}


def config_hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]


def git_commit(repo):
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def env_info(packages=("torch", "torchvision", "transformers", "diffusers", "huggingface_hub",
                       "accelerate", "numpy", "opencv-python-headless", "supervision", "pyiqa")):
    from importlib import metadata
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": {}}
    for p in packages:
        try:
            info["packages"][p] = metadata.version(p)
        except Exception:
            pass
    try:
        import torch
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    except Exception:
        pass
    return info


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0


def image_name(sample_id):
    return f"{sample_id}.png"


def initial_noise(seed, shape=(1, 4, 64, 64)):
    """Shared starting latent: identical tensor for every method given the same seed."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=g, dtype=torch.float32)
