"""Regression coverage for scientific-notation pool.balance in self-knowledge."""
import os
import unittest

import yaml
from unittest.mock import patch

from corpora import chat


class SelfFactsBalanceTests(unittest.TestCase):
    def facts_from(self, value):
        with patch.dict(os.environ, {"MINAGI_FACTS_WEIGHTS": "config"}), \
             patch("minagi.config.load", return_value={"pool": {"balance": value}}):
            return chat._self_facts()

    def test_scientific_notation_string_is_numeric(self):
        raw = yaml.safe_load("pool: {balance: 1e-5}")["pool"]["balance"]
        self.assertIsInstance(raw, str)
        balance = self.facts_from(raw)["balance"]
        self.assertIsInstance(balance, float)
        self.assertAlmostEqual(balance, 1e-5)

    def test_numeric_balance_remains_numeric(self):
        balance = self.facts_from(0.000065)["balance"]
        self.assertIsInstance(balance, float)
        self.assertAlmostEqual(balance, 6.5e-5)

    def test_empty_balance_falls_back_to_zero(self):
        self.assertEqual(self.facts_from(None)["balance"], 0.0)


if __name__ == "__main__":
    unittest.main()
