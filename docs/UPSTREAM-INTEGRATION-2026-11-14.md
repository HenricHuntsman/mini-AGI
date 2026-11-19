# Greenlight integration candidate — 14 November 2026

Draft [PR #55](https://github.com/HenricHuntsman/mini-AGI/pull/55) builds on
#54 (`f4fd38f40fbe4e6d14e013855704d4ad7ebaa7cd`) and integrates upstream
`2848bfaeacbf992b9b95d12812b7c6155e7b576b`. It is not a production release.

## Changes

- Import one-moment Muon, router floor/balance changes, configurable sampling,
  optional loop practice/self-play, character-based lane scheduling and joined
  passages. Practice and self-play remain disabled by default.
- Accept upstream `MINAGI_CONFIG` after `GREENLIGHT_CONFIG` and `MINI_AGI_CONFIG`;
  an explicit config path takes precedence over all environment variables.
- Determine checkpoint router width and expert count before config validation,
  so a stale growth ceiling does not reject a valid grown-pool checkpoint.
- Preserve private practice/self-play RNG state in checkpoints and use absolute
  reading steps for live practice cadence. Non-paged saves now retain extra
  manifest metadata, matching the paged save path.
- Keep read-only and temporary checkpoint overlays and strict missing-counter
  rejection. Moving Muon to AdamW seeds a new second moment from momentum squared;
  this does not reconstruct the original AdamW second moment.
- Remove unused legacy working-set CLI options and their consumers. Update plot,
  signal-meter, optimizer and serving-window fixtures to the current interfaces.
- Install zstandard in CI, retain per-file logs and JSON summaries as artifacts,
  and kill trainer subprocess groups when a regression times out on POSIX.

## Evidence and limits

The implementation tree was published at `bcb8cd6e6894b77c85f342edd7fb9c2cac259190`.
Local checks run in the ChatGPT execution environment on Python 3.12.14 and
PyTorch 2.15.0+cpu, using disposable synthetic fixtures. No user-PC access or
production training was performed. Compilation passed. The final full run passed
115 of 115 test files with 0 failed files. One opt-in command-lifecycle
test was skipped; this is not a count of individual tests.

Command for the final local regression run:

```sh
TMPDIR="$PWD/.test-tmp" .venv/bin/python tools/run_test_suite.py \
  --workers 2 --timeout 180 --report-dir test-results/20261114-final
```

File exits are not individual test counts. The opt-in full command lifecycle
is skipped unless `GREENLIGHT_LIFECYCLE=1`. The lane-character regression does
execute a small CPU train/save/restart sequence. Checkpoint and optimizer
regressions use synthetic checkpoints; they do not qualify a production GPU run.

Python 3.11/3.14 execution belongs to the separate
[PR CI run](https://github.com/HenricHuntsman/mini-AGI/actions/runs/43531755004).
Read its job outcomes and retained artifacts; a configured matrix is not a pass.

## Remaining gates

- No measured speed or capability superiority over upstream.
- No validation of the user's production checkpoint on the RTX 5060 Ti.
- Private RNG continuity is not complete deterministic training replay: input
  reader position, all Torch randomness and CUDA behavior are separate concerns.
- Paged future-token influence and cross-sequence interference are not resolved
  by this integration. Causality drafts #51/#52 remain separate research work.
- Historical upstream samples/figures in this repository are upstream evidence,
  not Greenlight benchmark results.
