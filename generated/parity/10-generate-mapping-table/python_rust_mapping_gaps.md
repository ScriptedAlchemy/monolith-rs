# Mapping Gaps / Cleanup Report

This file is generated from the computed mapping JSON and is intentionally deterministic (no timestamps).

## Invariants (Pre-Normalization Checklist)

- [ ] No missing python files (records=334, unique_python_paths=334, expected_python_files=334).
- [ ] Every record has at least one crate mapping (non-empty `rustTargets`) OR an explicit N/A justification (`status`=`N/A` and `notes` start with `N/A:`).
- [ ] Every Rust target has a stable representation (prefer `{crate, path}`); avoid ambiguous entries that only specify `src`/`tests` without crate context.
- [ ] Every Rust target crate exists under `monolith-rs/crates/`.
- [ ] No Rust targets are empty or non-pathlike unless `status` is `N/A`.

Computed checks:

- Missing python files check: PASS.
- Records with empty rustTargets but missing/invalid N/A justification: 0.

## Rust Targets Empty Or Non-Pathlike (Up To 100)

- Empty rustTargets: 53
- Non-pathlike rustTargets: 0

| Python File | Status | Notes |
| --- | --- | --- |
| monolith/native_training/data/training_instance/python/test_data_utils.py | N/A | N/A: Unspecified justification. |
| monolith/native_training/layers/sparse_nas.py | N/A | N/A: stub |
| monolith/native_training/layers/sparse_nas_test.py | N/A | N/A: empty test |
| monolith/native_training/learning_rate_functions.py | N/A | N/A: no Rust schedule yet |
| monolith/native_training/learning_rate_functions_test.py | N/A | N/A: no Rust schedule yet |
| monolith/native_training/logging_ops.py | N/A | N/A: TF custom ops |
| monolith/native_training/logging_ops_test.py | N/A | N/A: TF custom ops |
| monolith/native_training/losses/batch_softmax_loss.py | N/A | N/A: no Rust loss yet |
| monolith/native_training/losses/batch_softmax_loss_test.py | N/A | N/A: no Rust loss yet |
| monolith/native_training/losses/inbatch_auc_loss.py | N/A | N/A: TF custom op |
| monolith/native_training/losses/inbatch_auc_loss_test.py | N/A | N/A: TF custom op |
| monolith/native_training/losses/ltr_losses.py | N/A | N/A: no Rust ranking losses yet |
| monolith/native_training/metric/cli.py | N/A | N/A: stub |
| monolith/native_training/metric/deep_insight_ops.py | N/A | N/A: TF custom ops |
| monolith/native_training/metric/deep_insight_ops_test.py | N/A | N/A: empty test |
| monolith/native_training/metric/exit_hook.py | N/A | N/A: no Rust hook |
| monolith/native_training/metric/kafka_utils.py | N/A | N/A: no Rust Kafka wrapper |
| monolith/native_training/metric/metric_hook.py | N/A | N/A: TF hooks + Kafka |
| monolith/native_training/metric/metric_hook_test.py | N/A | N/A: TF hooks |
| monolith/native_training/metric/utils.py | N/A | N/A: TF custom ops |
| monolith/native_training/metric/utils_test.py | N/A | N/A: TF custom ops |
| monolith/native_training/mlp_utils.py | N/A | N/A: TF distributed runtime |
| monolith/native_training/model.py | N/A | N/A: test model |
| monolith/native_training/model_comp_test.py | N/A | N/A: TF/Horovod test |
| monolith/native_training/model_dump/dump_utils.py | N/A | N/A: TF model dump |
| monolith/native_training/model_dump/graph_utils.py | N/A | N/A: TF graph utils |
| monolith/native_training/model_dump/graph_utils_test.py | N/A | N/A: TF graph utils |
| monolith/native_training/model_export/__init__.py | N/A | N/A: module alias |
| monolith/native_training/model_export/data_gen_utils.py | N/A | N/A: data generator |
| monolith/native_training/model_export/data_gen_utils_test.py | N/A | N/A: no tests |
| monolith/native_training/model_export/demo_export.py | N/A | N/A: demo exporter |
| monolith/native_training/model_export/demo_export_test.py | N/A | N/A: TF export test |
| monolith/native_training/model_export/demo_predictor.py | N/A | N/A: demo predictor |
| monolith/native_training/model_export/demo_predictor_client.py | N/A | N/A: demo gRPC client |
| monolith/native_training/model_export/export_context.py | N/A | N/A: export context |
| monolith/native_training/model_export/export_hooks.py | N/A | N/A: TF export hook |
| monolith/native_training/model_export/export_hooks_test.py | N/A | N/A: export hook test |
| monolith/native_training/model_export/export_state_utils.py | N/A | N/A: export state |
| monolith/native_training/model_export/export_state_utils_test.py | N/A | N/A: export state test |
| monolith/native_training/model_export/export_utils.py | N/A | N/A: remote predict helper |
| monolith/native_training/model_export/export_utils_test.py | N/A | N/A: remote predict test |
| monolith/native_training/model_export/saved_model_exporters.py | N/A | N/A: SavedModel exporters |
| monolith/native_training/model_export/saved_model_exporters_test.py | N/A | N/A: exporter tests |
| monolith/native_training/model_export/saved_model_visulizer.py | N/A | N/A: tensorboard visualizer |
| monolith/native_training/model_export/warmup_data_decoder.py | N/A | N/A: warmup decoder |
| monolith/native_training/model_export/warmup_data_gen.py | N/A | N/A: warmup generator |
| monolith/native_training/model_export/warmup_example_batch.py | N/A | N/A: warmup example batch |
| monolith/native_training/monolith_export.py | N/A | N/A: decorator |
| monolith/native_training/multi_hash_table_ops.py | N/A | N/A: TF custom ops |
| monolith/native_training/multi_hash_table_ops_test.py | N/A | N/A: TF custom ops |
| monolith/native_training/multi_type_hash_table.py | N/A | N/A: hash table wrapper |
| monolith/native_training/multi_type_hash_table_test.py | N/A | N/A: hash table tests |
| monolith/native_training/remote_predict_ops.py | N/A | N/A: empty stub |

## Rust Targets Referencing Missing Crates (Up To 100)

Crates present under `monolith-rs/crates/`: 12

| Python File | Missing Crate | Raw Target |
| --- | --- | --- |
| monolith/core/auto_checkpoint_feed_hook.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/core/tpu_variable.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/native_training/distribution_ops_test.py | monolith-tf | monolith-rs/crates/monolith-tf/tests |
| monolith/native_training/gen_seq_mask.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/native_training/gen_seq_mask_test.py | monolith-tf | monolith-rs/crates/monolith-tf/tests |
| monolith/native_training/graph_meta.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/native_training/graph_utils.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/native_training/hash_filter_ops.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/native_training/hash_filter_ops_test.py | monolith-tf | monolith-rs/crates/monolith-tf/tests |
| monolith/native_training/hash_table_ops.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/native_training/hash_table_ops_test.py | monolith-tf | monolith-rs/crates/monolith-tf/tests |
| monolith/native_training/runtime/ops/gen_monolith_ops.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/native_training/static_reshape_op.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/native_training/static_reshape_op_test.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/native_training/summary/summary_ops.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/native_training/summary/summary_ops_test.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/native_training/summary/utils.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/native_training/summary/utils_test.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/utils.py | monolith-tf | monolith-rs/crates/monolith-tf/src |
| monolith/utils_test.py | monolith-tf | monolith-rs/crates/monolith-tf/tests |

## Next Actions (For normalize-mapping)

- Normalize target format: prefer `{crate, path}` with `path` relative to the crate root (e.g. `src/...`), and keep `raw` only as a fallback.
- Crate normalization: resolve references to missing crates (e.g. `monolith-tf`) by either adding the crate under `monolith-rs/crates/` or remapping those targets to the correct existing crate(s).
- N/A normalization: enforce `status: N/A`, require `notes` to start with `N/A:` + a short reason, and require `rustTargets: []`.
- Naming cleanup: remove annotations like ` (new)` from `raw` targets (keep that metadata elsewhere if needed).
- Rendering cleanup: when producing markdown tables, render Rust targets as `crate:path` (or `crate/raw`) to avoid ambiguity between common paths like `src` and `tests`.
