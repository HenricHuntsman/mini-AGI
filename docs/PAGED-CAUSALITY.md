# Paged expert causality — evaluation repair

The paged router historically selected its resident experts using a vote over
**every token in the current chunk**. This let later prompt positions change
the expert chosen for an earlier position. See the regression in
`tests/test_paged_causality.py` and upstream issue #33.

## Implemented in this draft branch

When `PooledMLP` is in evaluation mode, gradients are disabled, and the
pool supports paged admission, it routes a multi-position chunk **one position
at a time**. The same admission state is retained across positions, and
inactive positions do not vote or produce outputs. This prevents a later
position from participating in an earlier position's expert admission.

The original batched dispatch remains in place for training, fully resident
pools, and evaluation with gradients enabled. This avoids silently changing
the training graph, optimizer state, or expert residency constraints.

## Tests

```bash
python -m unittest tests.test_paged_causality -v
python -m unittest discover -s tests -v
```

The CPU regressions cover suffix invariance at the first and a middle prompt
position, full residency, and inactive future positions. CI runs on Python
3.11 and 3.14.

## Remaining work before integration

- Training-window admission still votes over the whole chunk. It needs a
  separate causality-preserving algorithm compatible with autograd and the
  fixed resident expert budget.
- Benchmark no-grad prompt evaluation against the old batched path. Sequential
  admission trades throughput for causal correctness; do not assume an RTX
  5060 Ti result from CPU CI. An opt-in, read-only synthetic benchmark is
  available:
  ```bash
  python tools/benchmark_paged_causality.py --device cpu --tokens 32
  python tools/benchmark_paged_causality.py --device cuda --tokens 32
  ```
  The `legacy_noncausal` measurement is only a speed baseline; its outputs
  are not a correctness reference. CUDA must be available for the second
  command. This script never writes checkpoints or trained weights.
- Evaluate multiple independent sequences per batch: a shared resident budget
  still allows one sequence's requests to influence another's admitted set.
- Test stochastic expert selection (`select_temperature > 0`) with
  controlled RNG and compare reproducibility.

No checkpoint format, model weight file, or training config changes are part
of this draft.
