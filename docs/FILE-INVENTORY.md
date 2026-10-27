# Greenlight repository file inventory

Snapshot of the **main** branch taken 2026-10-28. This is a maintenance
inventory, not a claim that every file has been modified or tested.

**137 tracked files** across 15 top-level groups.
Generated model weights, private training datasets, and untracked local files
are **not** included in the Git tree and must be backed up separately.

## Renewal policy

- Keep `main` and all historical checkpoints unchanged until focused PRs pass review.
- Update documentation and indexes in isolation; never mass-reformat model source,
  model tensors, archived upstream material, or experiment results.
- Treat `README.md.clean`, `mini-AGI-rebuild.zip`, `runs/samples.txt`,
  `replication/results/` and media assets as archival unless provenance is
  established and a replacement is specifically requested.
- No automatic dependency major-version upgrades, GPU training runs, or
  destructive cleanup. CI checks Python 3.11/3.14 but not the target GPU.
- The folder listings below describe what is tracked, **not** a health pass
  for every module. Check each changed path's dedicated tests before merging.

## Automatic consistency check

Run `python tools/check_file_inventory.py` from the repository root.
The checker validates this dated snapshot's declared counts, duplicate paths,
unsafe paths and missing files. It runs in Python 3.11/3.14 CI.
It does **not** require newly added files to appear in this historical snapshot;
update the snapshot date and counts deliberately when publishing a new inventory.
This is a read-only file check, not a dependency, security, or GPU audit.

## Tracked files by folder

### `(root)` — 18 files

Entrypoints, configs, rebuild instructions, change history and archival source ZIP.

- `.gitignore`
- `CHANGELOG.md`
- `LICENSE`
- `README.md`
- `README.md.clean`
- `REBUILD.md`
- `UPSTREAM-HARDENING.md`
- `config-16gb.yaml`
- `config.yaml`
- `greenlight.py`
- `greenlight_lab.py`
- `mini-AGI-rebuild.zip`
- `requirements-optional.txt`
- `requirements.txt`
- `serve.py`
- `train-16gb.sh`
- `train.py`
- `training_doctor.py`

### `.github` — 3 files

CI workflows and repository settings.

- `.github/FUNDING.yml`
- `.github/workflows/android.yml`
- `.github/workflows/tests.yml`

### `android` — 8 files

Android companion application source and build files.

- `android/README.md`
- `android/app/build.gradle.kts`
- `android/app/src/main/AndroidManifest.xml`
- `android/app/src/main/java/org/dragonforge/companion/MainActivity.kt`
- `android/app/src/main/res/values/styles.xml`
- `android/build.gradle.kts`
- `android/gradle.properties`
- `android/settings.gradle.kts`

### `assets` — 9 files

Images and other project presentation assets.

- `assets/dashboard.png`
- `assets/generate.gif`
- `assets/layers.svg`
- `assets/mitigations.png`
- `assets/probe_interleaved.png`
- `assets/probe_massed.png`
- `assets/scaling.png`
- `assets/shape.gif`
- `assets/training_progress.png`

### `companion` — 1 files

Local read-only companion server.

- `companion/server.py`

### `corpora` — 12 files

Corpus acquisition, formatting, sampling and self-knowledge.

- `corpora/__init__.py`
- `corpora/__main__.py`
- `corpora/arithmetic.py`
- `corpora/build.py`
- `corpora/chat.py`
- `corpora/chess_games.py`
- `corpora/code.py`
- `corpora/expand.py`
- `corpora/fetch.py`
- `corpora/pg19.py`
- `corpora/reasoning.py`
- `corpora/self_knowledge.yaml`

### `docs` — 5 files

Research plans, history, training doctor and integration contracts.

- `docs/AGI-ROADMAP.md`
- `docs/CLI-TRAINING-2026-09-27.md`
- `docs/EXPERT-BALANCE-AB-CONTRACT.md`
- `docs/HISTORY.md`
- `docs/TRAINING-DOCTOR-V2.md`

### `iphone` — 3 files

iOS SwiftUI companion application source.

- `iphone/DragonForge/CompanionModel.swift`
- `iphone/DragonForge/ContentView.swift`
- `iphone/DragonForge/DragonForgeApp.swift`

### `minagi` — 27 files

Model, routing, paging, optimizer, storage, evaluation and Training Doctor engine.

- `minagi/__init__.py`
- `minagi/benchmark.py`
- `minagi/config.py`
- `minagi/corpus.py`
- `minagi/create.py`
- `minagi/decode.py`
- `minagi/evaluate.py`
- `minagi/history.py`
- `minagi/ingest.py`
- `minagi/live.py`
- `minagi/model.py`
- `minagi/optim.py`
- `minagi/paged.py`
- `minagi/plasticity.py`
- `minagi/pool.py`
- `minagi/precision.py`
- `minagi/recur.py`
- `minagi/report.py`
- `minagi/store.py`
- `minagi/stream.py`
- `minagi/tokenizer.py`
- `minagi/training/__init__.py`
- `minagi/training/cli.py`
- `minagi/training/ponder_probe.py`
- `minagi/training_doctor.py`
- `minagi/training_policy.py`
- `minagi/verify_model.py`

### `mini_agent` — 5 files

Separate local Ollama assistant.

- `mini_agent/__init__.py`
- `mini_agent/__main__.py`
- `mini_agent/agent.py`
- `mini_agent/ollama.py`
- `mini_agent/tools.py`

### `mobile` — 1 files

Cross-platform companion overview.

- `mobile/README.md`

### `replication` — 7 files

Reproducibility experiments and frozen research results.

- `replication/cl_summary.py`
- `replication/forgetting_probe.py`
- `replication/plot_figures.py`
- `replication/results/control.json`
- `replication/results/none.json`
- `replication/results/static.json`
- `replication/results/trunk01.json`

### `runs` — 1 files

Tracked historical sample record; do not rewrite as part of cleanup.

- `runs/samples.txt`

### `tests` — 32 files

CPU regressions and recovery safety gates.

- `tests/test_adamw_orphan_steps.py`
- `tests/test_balance_report_gate.py`
- `tests/test_benchmark.py`
- `tests/test_checkpoint_reload_resume.py`
- `tests/test_command_contract.py`
- `tests/test_companion_api.py`
- `tests/test_config_profile.py`
- `tests/test_context_ramp.py`
- `tests/test_corpora_build.py`
- `tests/test_fetch_cache.py`
- `tests/test_grad_snr.py`
- `tests/test_gradsnr_regression.py`
- `tests/test_greenlight_cli.py`
- `tests/test_greenlight_lab.py`
- `tests/test_history.py`
- `tests/test_local_agent.py`
- `tests/test_model_evaluate.py`
- `tests/test_model_stress.py`
- `tests/test_optimizer_resume.py`
- `tests/test_paged_auditions.py`
- `tests/test_paged_layout_detection.py`
- `tests/test_ponder_probe.py`
- `tests/test_read_only_paged.py`
- `tests/test_serve.py`
- `tests/test_serve_context.py`
- `tests/test_store_shadow.py`
- `tests/test_stream_evaluation.py`
- `tests/test_training_cli.py`
- `tests/test_training_doctor.py`
- `tests/test_training_policy.py`
- `tests/test_upstream_balance_ab.py`
- `tests/test_verify_model.py`

### `tools` — 5 files

Local analysis, diagnostics, reporting and plots.

- `tools/compare_balance_ab.py`
- `tools/device_check.py`
- `tools/plot_dashboard.py`
- `tools/plot_experts.py`
- `tools/plot_progress.py`

## Open development review queue

- PR #43: upstream integration candidate; hardware and behavior checks remain.
- PR #44: self-knowledge YAML balance normalization (branch CI passed).
- PR #45: CPU reproduction of paged-admission causality bug (expected failure).
- PR #46: Training Doctor invalid-manifest handling (branch CI passed).
- PR #47: checkpoint loader expert manifest validation (branch CI passed).
- PR #48: checkpoint recovery documentation (branch CI passed).

The PRs above are draft candidates, not merged functionality on `main`.
A full repository update must proceed through reviewed, scoped changes.
