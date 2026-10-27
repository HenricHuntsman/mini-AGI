#!/usr/bin/env python3
"""Benchmark causal paged evaluation against the old chunk admission path.

Read-only synthetic experts: no model downloads, checkpoint writes or training.
The old path is for timing comparison ONLY; it is not causally correct.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from minagi.paged import PagedPool
from minagi.pool import PooledMLP


def benchmark(device, tokens, experts, resident, width, repeats):
    if experts < 1 or resident < 1 or resident > experts:
        raise ValueError("require 1 <= resident <= experts")
    if tokens < 1 or width < 1 or repeats < 1:
        raise ValueError("tokens, width and repeats must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.manual_seed(17)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(17)
        for i in range(experts):
            np.savez(root / f"e{i:05d}.npz",
                     w1=rng.standard_normal((width, width)).astype("float32") * .02,
                     w3=rng.standard_normal((width, width)).astype("float32") * .02,
                     w2=rng.standard_normal((width, width)).astype("float32") * .02)
        pool = PagedPool(tmp, width, width, experts, resident=resident,
                         ram_capacity=experts, device=device, read_only=True)
        pool.to(device)
        model = PooledMLP(pool, width, top_k=1, grad_checkpoint=False,
                          capacity_factor=0).to(device).eval()
        x = torch.randn(1, tokens, width, device=device)
        def sync():
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
        results = {}
        with torch.no_grad():
            for label, route in (("causal", model.forward),
                                 ("legacy_noncausal", model._route)):
                times = []
                for _ in range(repeats):
                    pool.begin_text(False)
                    sync()
                    start = time.perf_counter()
                    out = route(x)
                    sync()
                    times.append(time.perf_counter() - start)
                results[label] = {
                    "median_seconds": round(float(np.median(times)), 6),
                    "tokens_per_second": round(tokens / float(np.median(times)), 2),
                    "shape": list(out.shape),
                }
        return {"device": device, "tokens": tokens, "experts": experts,
                "resident": resident, "width": width, "repeats": repeats,
                "timings": results,
                "note": "legacy_noncausal is a timing baseline, not a correctness reference"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--resident", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.device, args.tokens, args.experts,
                               args.resident, args.width, args.repeats),
                     indent=2))


if __name__ == "__main__":
    main()
