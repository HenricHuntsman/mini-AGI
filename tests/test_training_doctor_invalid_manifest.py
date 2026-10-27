"""Malformed checkpoint metadata must be diagnosed, not crash the doctor."""
import tempfile
import unittest
from pathlib import Path

from minagi.training_doctor import resume_preflight


class InvalidManifestPreflightTests(unittest.TestCase):
    def assert_invalid(self, contents):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "manifest.json").write_text(contents, encoding="utf-8")
            report = resume_preflight(root, build=lambda *_: self.fail(
                "Must not try to construct a model from an invalid manifest"))
        self.assertEqual(report["health"], "critical")
        self.assertIn("INVALID_CHECKPOINT_MANIFEST",
                      {finding["code"] for finding in report["findings"]})

    def test_truncated_json(self):
        self.assert_invalid('{"experts": [')

    def test_manifest_must_be_object(self):
        self.assert_invalid('[]')

    def test_experts_must_be_list(self):
        self.assert_invalid('{"experts": {}}')

    def test_expert_entry_must_have_filename(self):
        self.assert_invalid('{"experts": [{}]}')

    def test_expert_filename_must_be_string(self):
        self.assert_invalid('{"experts": [{"file": 123}]}')

    def test_empty_expert_filename(self):
        self.assert_invalid('{"experts": [{"file": ""}]}')

    def test_parent_directory_escape_is_rejected(self):
        self.assert_invalid('{"experts": [{"file": "../outside.npz"}]}')

    def test_windows_directory_escape_is_rejected(self):
        self.assert_invalid('{"experts": [{"file": "..\\\\outside.npz"}]}')


if __name__ == "__main__":
    unittest.main()
