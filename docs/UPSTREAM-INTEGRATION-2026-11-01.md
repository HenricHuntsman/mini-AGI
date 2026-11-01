# Greenlight upstream integration — 1 November 2026

Prepared from the previously reviewed candidate `2063613a6c93fbaf1fb613e5a1d4c7da38d12db6` and pinned upstream `9b2bc5a9f1b381bee738b624dee331e543f80917`. The old PR #43 and its branch remain intact. This is a new draft candidate; main and production checkpoints are unchanged.

## Integration

Resolved conflicts in `.gitignore`, configuration, paged storage, checkpoint helpers, stream evaluation and training. Retained the fork's configuration aliases, read-only/temporary weight overlays, history recorder, audition identity and growth-clock safeguards. Kept upstream's write-behind storage, bf16 moment storage, routing state and new training/evaluation helpers.

Deletion waits for queued expert writes before removing an overlay file, preventing a background eviction from recreating it. The eviction and resume regressions now check persistence after the async writer drains and compare persisted bf16 moments with widened AdamW state. Audition pruning also checks the current upstream routing buffers instead of removed buffers. Training introspection follows the implementation beneath the dry-run lifetime wrapper.

The corpus dispatcher now retains the fork's separate `self-knowledge` lane, writing to the directory that expansion reads while honouring an explicit output override.

## Test execution

Python 3.12.14 and PyTorch 2.15.0+cpu were used in an isolated test environment. Python compilation passed. The following **23 executable regression files passed** through the offline runner, with two workers. File count is not an individual assertion count:

`test_notify`, `test_paged_auditions`, `test_write_behind`, `test_own_moments`, `test_store_shadow`, `test_optimizer_resume`, `test_checkpoint_reload_resume`, `test_read_only_paged`, `test_prune`, `test_prune_identity`, `test_config_profile`, `test_history`, `test_stream_evaluation`, `test_attention_logit`, `test_drop_quietest`, `test_dropped_report`, `test_growth_gap`, `test_live_learning`, `test_arc_corpus`, `test_agent_corpus`, `test_context_growth`, `test_corpora_contract`, `test_governor`.

Upstream includes executable regression scripts that call `sys.exit` during import. The earlier unittest-discovery command and pytest collection cannot run that mixed suite correctly. `tools/run_test_suite.py` executes each file in its own process, preserves nonzero failures, bounds runtime and allows loopback fixtures while blocking external Python socket connections. It strips inherited proxy and notification settings. The notification transport failure uses an offline mock; runtime notification behaviour remains opt-in. CI installs the corpus fixture dependencies explicitly.

The broader initial attempt was interrupted after 55 completed file records. Automatic approval review blocked resuming it because the upstream notification regression attempted a Telegram request with dummy settings. That original broad run is **not a full-suite pass**. The mocked notification test and offline checks above replace the relevant unsafe attempt; no live notification delivery was authorised or verified.

Reproduce the selected checks with the runner, for example:

```bash
python tools/run_test_suite.py --workers 2 --timeout 90 \
  tests/test_notify.py tests/test_store_shadow.py \
  tests/test_checkpoint_reload_resume.py tests/test_write_behind.py
```

The complete offline file suite is configured in CI, but its full result, Python 3.11/3.14 qualification, CUDA/consumer-GPU training, existing checkpoint compatibility and the separate paged-causality research PRs remain unresolved gates. No full-suite or release-readiness claim is made from these focused results.
