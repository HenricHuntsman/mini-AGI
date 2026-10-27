# Training-mode paged causality: regression and design checkpoint

The no-gradient evaluation repair in PR #51 routes positions sequentially.
**Training-mode admission is still chunk-wide.** It aggregates all positions'
expert requests before dispatching the earliest token, allowing future tokens
to influence earlier outputs.

`tests/test_paged_training_causality.py` captures that bug with an explicit
`@unittest.expectedFailure`. The test runs the module in training mode with
autograd disabled **only to isolate the admission decision**. It is not a
gradient correctness test or proof of training stability.

## Design constraints for the eventual fix

1. Training must preserve autograd through the selected experts and router.
2. An expert needed by an earlier position must not be evicted before its
   backward pass; the resident limit is a fixed hardware budget.
3. Sequentially selecting a different resident set for every position cannot
   simply reuse the existing in-place slot swapping code during a single
   backward graph.
4. A causal training solution may require smaller forward/backward segments,
   gradient accumulation, and explicit optimizer-step boundaries. Benchmark
   memory and throughput before selecting an approach.
5. Multi-sequence batches must not let another sequence's *future* tokens
   influence a given sequence's earlier expert choices.
6. Preserve checkpoint and optimizer-moment ownership across paging.

## Review gate

Replace `expectedFailure` with a normal passing assertion only after a
training-compatible algorithm is implemented. Add tests for actual gradients,
optimizer updates, checkpoint save/resume, and multiple batch sequences.
The existing evaluation fix is not evidence that this training bug is fixed.
