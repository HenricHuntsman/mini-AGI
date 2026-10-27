"""The checkpoint loader must reject malformed expert metadata before disk reads."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from minagi import store


class CheckpointManifestSafetyTests(unittest.TestCase):
    def assert_rejected(self, manifest, message):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            # No core/router bundles exist: a safe rejection must occur
            # before the loader tries to read any weight files.
            with self.assertRaisesRegex(ValueError, message):
                store.load(torch.nn.Linear(1, 1), tmp)

    def test_missing_experts_list(self):
        self.assert_rejected({"n_experts": 0}, "experts list")

    def test_experts_must_be_list(self):
        self.assert_rejected({"experts": {}, "n_experts": 0}, "experts list")

    def test_entry_must_be_object(self):
        self.assert_rejected({"experts": [None], "n_experts": 1}, "entry")

    def test_reject_parent_traversal(self):
        self.assert_rejected({"experts": [{"id": 0, "file": "../outside.npz"}],
                              "n_experts": 1}, "filename")

    def test_reject_windows_traversal(self):
        self.assert_rejected({"experts": [{"id": 0, "file": "..\\\\outside.npz"}],
                              "n_experts": 1}, "filename")

    def test_reject_absolute_path(self):
        self.assert_rejected({"experts": [{"id": 0, "file": "/tmp/outside.npz"}],
                              "n_experts": 1}, "filename")

    def test_reject_missing_id(self):
        self.assert_rejected({"experts": [{"file": "e00000.npz"}],
                              "n_experts": 1}, "ID")

    def test_reject_boolean_id(self):
        self.assert_rejected({"experts": [{"id": True, "file": "e00000.npz"}],
                              "n_experts": 1}, "ID")

    def test_reject_duplicate_ids(self):
        self.assert_rejected({"experts": [
            {"id": 0, "file": "e00000.npz"},
            {"id": 0, "file": "e00001.npz"}],
            "n_experts": 2}, "duplicate")

    def test_reject_duplicate_filenames(self):
        self.assert_rejected({"experts": [
            {"id": 0, "file": "e00000.npz"},
            {"id": 1, "file": "e00000.npz"}],
            "n_experts": 2}, "duplicate")

    def test_reject_count_mismatch(self):
        self.assert_rejected({"experts": [], "n_experts": 1}, "lists 0")

    def test_reject_invalid_count(self):
        self.assert_rejected({"experts": [], "n_experts": "broken"}, "n_experts")

    def test_reject_negative_count(self):
        self.assert_rejected({"experts": [], "n_experts": -1}, "lists 0")


if __name__ == "__main__":
    unittest.main()
