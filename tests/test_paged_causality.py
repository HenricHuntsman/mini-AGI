"""CPU-only diagnostic for upstream issue #33: paged admission looks ahead.

A chunk's first prediction should not change just because future characters
change. Paged evaluation now admits in position order, so future suffixes must not
change the first position. Training still uses batched admission and needs
separate design and regression coverage.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from minagi.paged import PagedPool
from minagi.pool import PooledMLP


def first_outputs(resident):
    with tempfile.TemporaryDirectory() as td:
        for expert in range(2):
            np.savez(
                Path(td) / f"e{expert:05d}.npz",
                w1=np.eye(2, dtype=np.float32),
                w3=np.eye(2, dtype=np.float32),
                w2=(expert + 1) * np.eye(2, dtype=np.float32),
            )
        pool = PagedPool(
            td, 2, 2, 2, resident=resident, ram_capacity=2,
            device="cpu", read_only=True,
        )
        model = PooledMLP(
            pool, 2, top_k=1, grad_checkpoint=False, capacity_factor=0,
        ).eval()
        with torch.no_grad():
            model.router.weight.copy_(torch.eye(2))
            model.depth_emb.zero_()
            outputs = []
            for suffix in (
                [[0.0, 8.0], [0.0, 8.0]],
                [[8.0, 0.0], [8.0, 0.0]],
            ):
                x = torch.tensor([[[1.0, 0.0]] + suffix])
                pool.begin_text(False)
                outputs.append(model(x)[0, 0].clone())
        return outputs


class PagedCausalityTests(unittest.TestCase):
    def test_full_residency_is_suffix_invariant(self):
        first_a, first_b = first_outputs(resident=2)
        torch.testing.assert_close(first_a, first_b, rtol=0, atol=1e-6)

    def test_inactive_future_positions_do_not_change_first_output(self):
        with tempfile.TemporaryDirectory() as td:
            for expert in range(2):
                np.savez(
                    Path(td) / f"e{expert:05d}.npz",
                    w1=np.eye(2, dtype=np.float32),
                    w3=np.eye(2, dtype=np.float32),
                    w2=(expert + 1) * np.eye(2, dtype=np.float32),
                )
            pool = PagedPool(td, 2, 2, 2, resident=1, ram_capacity=2,
                             device="cpu", read_only=True)
            model = PooledMLP(pool, 2, top_k=1, grad_checkpoint=False,
                              capacity_factor=0).eval()
            with torch.no_grad():
                model.router.weight.copy_(torch.eye(2))
                model.depth_emb.zero_()
                x = torch.tensor([[[1., 0.], [0., 8.], [0., 8.]]])
                pool.begin_text(False)
                first = model(x[:, :1])[0, 0].clone()
                pool.begin_text(False)
                mask = torch.tensor([[True, False, False]])
                output = model(x, active=mask)
                torch.testing.assert_close(first, output[0, 0], rtol=0, atol=1e-6)
                torch.testing.assert_close(
                    output[0, 1:], torch.zeros_like(output[0, 1:]),
                    rtol=0, atol=0)

    def test_paged_middle_position_is_suffix_invariant(self):
        # Same prefix, different future tokens: not just position zero.
        with tempfile.TemporaryDirectory() as td:
            for expert in range(2):
                np.savez(
                    Path(td) / f"e{expert:05d}.npz",
                    w1=np.eye(2, dtype=np.float32),
                    w3=np.eye(2, dtype=np.float32),
                    w2=(expert + 1) * np.eye(2, dtype=np.float32),
                )
            pool = PagedPool(td, 2, 2, 2, resident=1, ram_capacity=2,
                             device="cpu", read_only=True)
            model = PooledMLP(pool, 2, top_k=1, grad_checkpoint=False,
                              capacity_factor=0).eval()
            with torch.no_grad():
                model.router.weight.copy_(torch.eye(2))
                model.depth_emb.zero_()
                prefix = [[1.0, 0.0], [0.0, 1.0]]
                outputs = []
                for suffix in ([[0.0, 8.0]], [[8.0, 0.0]]):
                    pool.begin_text(False)
                    x = torch.tensor([prefix + suffix])
                    outputs.append(model(x)[0, 1].clone())
                torch.testing.assert_close(outputs[0], outputs[1],
                                           rtol=0, atol=1e-6)

    def test_paged_first_position_is_suffix_invariant(self):
        # Regression originally identified in volotat/mini-AGI#33.
        # Different future suffixes must not change the same first position.
        first_a, first_b = first_outputs(resident=1)
        torch.testing.assert_close(first_a, first_b, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
