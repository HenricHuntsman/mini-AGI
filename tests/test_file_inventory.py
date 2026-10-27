"""Tests for the dated, read-only repository inventory validator."""
import tempfile
import unittest
from pathlib import Path

from tools.check_file_inventory import inspect_inventory


class InventoryValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "README.md").write_text("hi", encoding="utf-8")
        (self.root / "docs").mkdir()
        (self.root / "docs" / "HISTORY.md").write_text("history", encoding="utf-8")

    def inventory(self, root_count=1, docs_count=1, total=2, paths=None):
        if paths is None:
            paths = ["README.md", "docs/HISTORY.md"]
        return (
            "# Inventory\n\n"
            f"**{total} tracked files** across 2 top-level groups.\n"
            "## Tracked files by folder\n\n"
            f"### `(root)` — {root_count} files\n"
            + "".join(f"- `{p}`\n" for p in paths if "/" not in p)
            + f"\n### `docs` — {docs_count} files\n"
            + "".join(f"- `{p}`\n" for p in paths if "/" in p)
            + "\n## Next section\n"
        )

    def test_valid_snapshot(self):
        self.assertEqual(inspect_inventory(self.inventory(), self.root), [])

    def test_new_files_do_not_invalidate_snapshot(self):
        (self.root / "NEW.md").write_text("new", encoding="utf-8")
        self.assertEqual(inspect_inventory(self.inventory(), self.root), [])

    def test_missing_file(self):
        (self.root / "README.md").unlink()
        self.assertTrue(any("missing tracked file" in error for error in
                            inspect_inventory(self.inventory(), self.root)))

    def test_wrong_declared_count(self):
        self.assertTrue(any("folder docs" in error for error in
                            inspect_inventory(self.inventory(docs_count=3), self.root)))

    def test_wrong_total(self):
        self.assertTrue(any("total:" in error for error in
                            inspect_inventory(self.inventory(total=10), self.root)))

    def test_duplicate_entry(self):
        self.assertTrue(any("duplicate file entry" in error for error in
                            inspect_inventory(self.inventory(root_count=2, total=3,
                                                             paths=["README.md", "README.md",
                                                                    "docs/HISTORY.md"]), self.root)))

    def test_reject_path_traversal(self):
        errors = inspect_inventory(
            self.inventory(docs_count=2, total=3,
                           paths=["README.md", "docs/HISTORY.md", "docs/../secret.txt"]),
            self.root)
        self.assertTrue(any("unsafe inventory path" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
