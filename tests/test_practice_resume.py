import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from minagi.random_state import restore_rng
from minagi.live import LiveLearner


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(block=32, vocab_size=265)
        self.emb = torch.nn.Embedding(265, 8)
        self.head = torch.nn.Linear(8, 265)

    def forward(self, x, y=None, **kw):
        logits = self.head(self.emb(x))
        loss = None if y is None else torch.nn.functional.cross_entropy(
            logits.reshape(-1, 265), y.reshape(-1))
        return logits, loss


class PracticeResumeTests(unittest.TestCase):
    def test_json_round_trip_preserves_both_private_streams(self):
        for seed in ([7, 7041, 30], [7, 7042, 30]):
            rng = restore_rng(None, seed)
            rng.integers(10000, size=19)
            state = json.loads(json.dumps(rng.bit_generator.state))
            resumed = restore_rng(state, [999])
            np.testing.assert_array_equal(rng.integers(10000, size=100),
                                          resumed.integers(10000, size=100))

    def test_malformed_state_does_not_silently_reseed(self):
        with self.assertRaises(ValueError):
            restore_rng({"bit_generator": "wrong"}, 1)

    def test_live_checkpoint_saves_rng_and_resumes_absolute_cadence(self):
        with tempfile.TemporaryDirectory() as root:
            learner = LiveLearner(Tiny(), weights_dir=root,
                                  manifest={"step": 5}, practice_every=2)
            learner.prng.integers(1000, size=17)
            self.assertTrue(learner.save())
            manifest = json.loads((Path(root) / 'manifest.json').read_text())
            resumed = LiveLearner(Tiny(), manifest=manifest, practice_every=2)
            np.testing.assert_array_equal(learner.prng.integers(1000, size=20),
                                          resumed.prng.integers(1000, size=20))
            # The first new reading is absolute step 6, so practice is due.
            with patch.object(resumed, '_practice', return_value=None) as practice:
                resumed._step(ids=list(range(12)), new=12)
            self.assertEqual(practice.call_count, 1)


if __name__ == '__main__':
    unittest.main()
