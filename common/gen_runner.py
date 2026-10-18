"""Generic manifest -> images loop shared by the GLIGEN and MIGC entry points.

For every sample it writes <out>/images/<sample_id>.png and <out>/meta/<sample_id>.json
(inputs, full config, checkpoint, seconds, peak GPU memory). Samples whose png+json
already exist are skipped, so interrupted runs resume where they stopped.
"""
import time
import traceback
from pathlib import Path

import torch

from common.utils import append_jsonl, dump_json, env_info, initial_noise, read_jsonl


def run_generation(manifest, out_dir, method, run_config, generate_fn, limit=None, warmup=True):
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(manifest)
    if limit:
        rows = rows[:limit]
    run_info = {"method": method, "manifest": str(manifest), "n_samples": len(rows),
                "config": run_config, "env": env_info(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    dump_json(out / f"run_config_{time.strftime('%Y%m%d_%H%M%S')}.json", run_info)

    todo = [r for r in rows if not ((out / "images" / f"{r['sample_id']}.png").exists()
                                   and (out / "meta" / f"{r['sample_id']}.json").exists())]
    print(f"[{method}] {len(rows) - len(todo)} done, {len(todo)} to generate -> {out}")
    if todo and warmup:
        # one untimed call so CUDA/cuDNN initialisation does not pollute the first timing
        r = todo[0]
        generate_fn(r, initial_noise(r["seed"]))
    for i, r in enumerate(todo):
        sid = r["sample_id"]
        for attempt in range(3):  # transient host-RAM / CUDA OOM: free caches and retry, then record
            try:
                _generate_one(out, method, run_config, generate_fn, r, i, len(todo))
                break
            except (MemoryError, torch.cuda.OutOfMemoryError) as e:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                append_jsonl(out / "errors.jsonl", {"sample_id": sid, "error": repr(e), "attempt": attempt,
                                                    "trace": traceback.format_exc()})
                print(f"  [OOM retry {attempt + 1}] {sid}: {e!r}", flush=True)
                time.sleep(10)
            except Exception as e:  # recorded, never silently dropped
                append_jsonl(out / "errors.jsonl", {"sample_id": sid, "error": repr(e), "trace": traceback.format_exc()})
                print(f"  [ERROR] {sid}: {e!r}", flush=True)
                break


def _generate_one(out, method, run_config, generate_fn, r, i, n_todo):
    sid = r["sample_id"]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    img = generate_fn(r, initial_noise(r["seed"]))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    img.save(out / "images" / f"{sid}.png")
    meta = {
        "sample_id": sid, "method": method, "seed": r["seed"], "layout_id": r["layout_id"],
        "caption": r["caption"],
        "objects": [{"desc": o["desc"], "box": o["box"]} for o in r["objects"]],
        "seconds": round(dt, 3),
        "peak_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "config": run_config,
    }
    dump_json(out / "meta" / f"{sid}.json", meta)
    print(f"  [{i + 1}/{n_todo}] {sid} {dt:.1f}s peak {meta['peak_mem_allocated_gb']}GB", flush=True)
