"""CPU smoke coverage for the opt-in paged routing benchmark."""
import unittest

from tools.benchmark_paged_causality import benchmark


class PagedCausalityBenchmarkTests(unittest.TestCase):
    def test_small_read_only_benchmark(self):
        result = benchmark("cpu", tokens=2, experts=2, resident=1,
                           width=2, repeats=1)
        self.assertEqual(result["device"], "cpu")
        for label in ("causal", "legacy_noncausal"):
            self.assertEqual(result["timings"][label]["shape"], [1, 2, 2])
            self.assertGreater(result["timings"][label]["median_seconds"], 0)
            self.assertGreater(result["timings"][label]["tokens_per_second"], 0)

    def test_invalid_residency(self):
        with self.assertRaisesRegex(ValueError, "resident"):
            benchmark("cpu", tokens=2, experts=2, resident=3,
                      width=2, repeats=1)


if __name__ == "__main__":
    unittest.main()
