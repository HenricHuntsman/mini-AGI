"""Document the remaining paged *training* lookahead separately from eval.

The expected failure is intentional. Do not remove the marker until training
admission is redesigned to be causal without breaking gradient accumulation
or the fixed resident-expert budget.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from minagi.paged import PagedPool
from minagi.pool import PooledMLP


class PagedTrainingCausalityTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_training_mode_first_position_ignores_future_suffix(self):
        # Training-mode admission is still chunk-wide, even if this test
        # disables autograd to isolate the routing selection itself.
        with tempfile.TemporaryDirectory() as td:
            for expert in range(2):
                np.savez(Path(td) / f"e{expert:05d}.npz",
                         w1=np.eye(2, dtype=np.float32),
                         w3=np.eye(2, dtype=np.float32),
                         w2=(expert + 1) * np.eye(2, dtype=np.float32))
            pool = PagedPool(td, 2, 2, 2, resident=1,
                             ram_capacity=2, device="cpu", read_only=True)
            model = PooledMLP(pool, 2, top_k=1, grad_checkpoint=False,
                              capacity_factor=0).train()
            with torch.no_grad():
                model.router.weight.copy_(torch.eye(2))
                model.depth_emb.zero_()
                first_outputs = []
                for suffix in ([[0., 8.], [0., 8.]],
                               [[8., 0.], [8., 0.]]):
                    pool.begin_text(True)
                    x = torch.tensor([[[1., 0.]] + suffix])
                    first_outputs.append(model(x)[0, 0].clone())
                torch.testing.assert_close(first_outputs[0], first_outputs[1],
                                           rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
