# Python-Rust Parity Mapping Table

## Summary

- Python files: 334
- Rust crates: 12
- Rust targets: seeded=281, heuristic=0, seeded_from_checklists=0
- N/A: 53
- Unknown mappings: 0
- Warnings:
  - No parity checklists found under monolith-rs/parity/**/*.md.

## Mapping Table

Records are sorted lexicographically by Python path.

| Python File | Lines | Status | Rust Targets | Source | Notes |
| --- | ---: | --- | --- | --- | --- |
| monolith/__init__.py | 56 | IN PROGRESS | monolith-rs/crates/monolith-core | seed_master_plan |  |
| monolith/agent_service/__init__.py | 0 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/agent.py | 101 | IN PROGRESS | monolith-rs/crates/monolith-cli, `src` | seed_master_plan |  |
| monolith/agent_service/agent_base.py | 89 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/agent_client.py | 217 | IN PROGRESS | `src/bin/agent_client.rs` | seed_master_plan |  |
| monolith/agent_service/agent_controller.py | 146 | IN PROGRESS | `src/bin/agent_controller.rs` | seed_master_plan |  |
| monolith/agent_service/agent_controller_test.py | 96 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/agent_service.py | 156 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/agent_service_test.py | 108 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/agent_v1.py | 391 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/agent_v3.py | 211 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/agent_v3_test.py | 114 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/backends.py | 519 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/backends_test.py | 135 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/client.py | 127 | IN PROGRESS | `src/bin/serving_client.rs` | seed_master_plan |  |
| monolith/agent_service/constants.py | 16 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/data_def.py | 172 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/data_def_test.py | 53 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/mocked_tfserving.py | 400 | IN PROGRESS | `tests/support` | seed_master_plan |  |
| monolith/agent_service/mocked_tfserving_test.py | 93 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/mocked_zkclient.py | 378 | IN PROGRESS | `tests/support` | seed_master_plan |  |
| monolith/agent_service/mocked_zkclient_test.py | 131 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/model_manager.py | 372 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/model_manager_test.py | 114 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/replica_manager.py | 836 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/replica_manager_test.py | 127 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/resource_utils.py | 270 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/resource_utils_test.py | 37 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/run.py | 40 | IN PROGRESS | `src/bin` | seed_master_plan |  |
| monolith/agent_service/svr_client.py | 71 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/tfs_client.py | 504 | IN PROGRESS | `src/bin/tfs_client.rs` | seed_master_plan |  |
| monolith/agent_service/tfs_client_test.py | 51 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/tfs_monitor.py | 304 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/tfs_monitor_test.py | 183 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/tfs_wrapper.py | 203 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/utils.py | 1168 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/utils_test.py | 171 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/agent_service/zk_mirror.py | 673 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/agent_service/zk_mirror_test.py | 230 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/base_runner.py | 47 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/common/python/mem_profiling.py | 52 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/__init__.py | 0 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/auto_checkpoint_feed_hook.py | 377 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/base_embedding_host_call.py | 644 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/base_embedding_host_call_test.py | 78 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/core/base_embedding_task.py | 612 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/base_host_call.py | 146 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/base_layer.py | 162 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/base_layer_test.py | 42 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/core/base_model_params.py | 26 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/base_task.py | 96 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/base_tpu_test.py | 74 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/core/core_test_suite.py | 36 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/core/dense.py | 180 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/dense_test.py | 109 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/core/feature.py | 612 | IN PROGRESS | `src/feature.rs` | seed_master_plan |  |
| monolith/core/feature_test.py | 179 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/core/host_call.py | 249 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/hyperparams.py | 440 | IN PROGRESS | `src/hyperparams.rs` | seed_master_plan |  |
| monolith/core/hyperparams_test.py | 278 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/core/mixed_emb_op_comb_nws.py | 422 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/model.py | 321 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/model_imports.py | 105 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/model_registry.py | 175 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/optimizers.py | 26 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/py_utils.py | 314 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/testing_utils.py | 204 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/core/tpu_variable.py | 215 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/util.py | 270 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/core/util_test.py | 150 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/core/variance_scaling.py | 189 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/gpu_runner.py | 227 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/alert/alert_manager.py | 32 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/alert/alert_manager_test.py | 33 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/barrier_ops.py | 159 | IN PROGRESS | `src/barrier.rs` | seed_master_plan |  |
| monolith/native_training/barrier_ops_test.py | 105 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/basic_restore_hook.py | 73 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/basic_restore_hook_test.py | 138 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/clip_ops.py | 81 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/clip_ops_test.py | 93 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/cluster_manager.py | 185 | IN PROGRESS | `src/distributed.rs` | seed_master_plan |  |
| monolith/native_training/cluster_manager_test.py | 36 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/consul.py | 150 | IN PROGRESS | `src/discovery.rs` | seed_master_plan |  |
| monolith/native_training/consul_test.py | 60 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/cpu_sync_training_test.py | 361 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/cpu_training.py | 2450 | IN PROGRESS | `src/cpu_training.rs`, `src/distributed.rs`, `src/local.rs` | seed_master_plan |  |
| monolith/native_training/cpu_training_distributed_test_binary.py | 227 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/cpu_training_test.py | 598 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/__init__.py | 21 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/data/data_ops_test.py | 503 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/data_service_parquet_test.py | 146 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/data_service_test.py | 99 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/datasets.py | 1643 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/data/eager_mode_test.py | 187 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/extract_fid_test.py | 31 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/feature_list.py | 410 | IN PROGRESS | `src/feature_list.rs` | seed_master_plan |  |
| monolith/native_training/data/feature_list_test.py | 0 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/feature_utils.py | 1071 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/data/feature_utils_test.py | 1415 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/item_pool_hook.py | 110 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/data/item_pool_test.py | 59 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/kafka_dataset_test.py | 240 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/multi_flow_test.py | 126 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/negative_gen_test.py | 254 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/parse_sparse_feature_test.py | 1834 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/parsers.py | 783 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/data/tf_example_to_example_test.py | 184 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/training_instance/python/instance_dataset_op.py | 167 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/data/training_instance/python/instance_dataset_op_test_stdin.py | 59 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/training_instance/python/instance_negative_gen_dataset_op_test.py | 284 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/training_instance/python/parse_instance_ops.py | 246 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/data/training_instance/python/parse_instance_ops_test.py | 186 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/training_instance/python/parser_utils.py | 86 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/data/training_instance/python/pb_datasource_ops.py | 49 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/data/training_instance/python/test_data_utils.py | 16 | N/A |  | seed_master_plan | N/A: Unspecified justification. |
| monolith/native_training/data/transform/transforms.py | 251 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/data/transform/transforms_test.py | 71 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/transform_dataset_test.py | 169 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/data/utils.py | 56 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/debugging/debugging_client.py | 99 | IN PROGRESS | `src/debugging` | seed_master_plan |  |
| monolith/native_training/debugging/debugging_server.py | 218 | IN PROGRESS | `src/debugging` | seed_master_plan |  |
| monolith/native_training/demo.py | 58 | IN PROGRESS | `examples` | seed_master_plan |  |
| monolith/native_training/dense_reload_utils.py | 458 | IN PROGRESS | `src/checkpoint` | seed_master_plan |  |
| monolith/native_training/dense_reload_utils_test.py | 193 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/device_utils.py | 232 | IN PROGRESS | `src/device` | seed_master_plan |  |
| monolith/native_training/device_utils_test.py | 105 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/distribute/distributed_dataset.py | 82 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/distribute/distributed_dataset_test.py | 125 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/distribute/str_queue.py | 115 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/distribute/str_queue_test.py | 68 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/distributed_ps.py | 2109 | IN PROGRESS | `src/ps` | seed_master_plan |  |
| monolith/native_training/distributed_ps_benchmark.py | 169 | IN PROGRESS | `benches` | seed_master_plan |  |
| monolith/native_training/distributed_ps_factory.py | 263 | IN PROGRESS | `src/ps` | seed_master_plan |  |
| monolith/native_training/distributed_ps_factory_test.py | 88 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/distributed_ps_sync.py | 532 | IN PROGRESS | `src/ps` | seed_master_plan |  |
| monolith/native_training/distributed_ps_sync_test.py | 110 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/distributed_ps_test.py | 980 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/distributed_serving_ops.py | 161 | IN PROGRESS | `src/serving` | seed_master_plan |  |
| monolith/native_training/distributed_serving_ops_test.py | 143 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/distribution_ops.py | 890 | IN PROGRESS | `src/ops` | seed_master_plan |  |
| monolith/native_training/distribution_ops_benchmark.py | 119 | IN PROGRESS | `benches` | seed_master_plan |  |
| monolith/native_training/distribution_ops_fused_benchmark.py | 62 | IN PROGRESS | `benches` | seed_master_plan |  |
| monolith/native_training/distribution_ops_fused_test.py | 149 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/distribution_ops_test.py | 537 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/distribution_utils.py | 444 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/embedding_combiners.py | 103 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/embedding_combiners_test.py | 48 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/entry.py | 631 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/entry_test.py | 85 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/env_utils.py | 33 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/env_utils_test.py | 24 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/estimator.py | 668 | IN PROGRESS | `src/estimator.rs` | seed_master_plan |  |
| monolith/native_training/estimator_dist_test.py | 167 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/estimator_mode_test.py | 418 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/estimator_test.py | 113 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/feature.py | 664 | IN PROGRESS | `src/feature.rs` | seed_master_plan |  |
| monolith/native_training/feature_test.py | 267 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/feature_utils.py | 420 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/feature_utils_test.py | 145 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/file_ops.py | 52 | IN PROGRESS | `src/file_ops.rs` | seed_master_plan |  |
| monolith/native_training/file_ops_test.py | 57 | IN PROGRESS | `tests/file_ops.rs` | seed_master_plan |  |
| monolith/native_training/fused_embedding_to_layout_test.py | 1334 | IN PROGRESS | `tests/fused_embedding_to_layout.rs` | seed_master_plan |  |
| monolith/native_training/gen_seq_mask.py | 27 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/gen_seq_mask_test.py | 43 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/gflags_utils.py | 283 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/gflags_utils_test.py | 218 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/graph_meta.py | 31 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/graph_utils.py | 27 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/hash_filter_ops.py | 327 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/hash_filter_ops_test.py | 229 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/hash_table_ops.py | 739 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/hash_table_ops_benchmark.py | 149 | IN PROGRESS | `src/bin/hash_table_ops_benchmark.rs` | seed_master_plan |  |
| monolith/native_training/hash_table_ops_test.py | 1201 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/hash_table_utils.py | 51 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/hash_table_utils_test.py | 46 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/hooks/ckpt_hooks.py | 194 | IN PROGRESS | `src/hooks` | seed_master_plan |  |
| monolith/native_training/hooks/ckpt_hooks_test.py | 182 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/hooks/ckpt_info.py | 99 | IN PROGRESS | `src/hooks` | seed_master_plan |  |
| monolith/native_training/hooks/ckpt_info_test.py | 46 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/hooks/controller_hooks.py | 171 | IN PROGRESS | `src/hooks` | seed_master_plan |  |
| monolith/native_training/hooks/controller_hooks_test.py | 83 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/hooks/feature_engineering_hooks.py | 100 | IN PROGRESS | `src/hooks` | seed_master_plan |  |
| monolith/native_training/hooks/hook_utils.py | 42 | IN PROGRESS | `src/hooks` | seed_master_plan |  |
| monolith/native_training/hooks/hook_utils_test.py | 36 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/hooks/ps_check_hooks.py | 98 | IN PROGRESS | `src/hooks` | seed_master_plan |  |
| monolith/native_training/hooks/ps_check_hooks_test.py | 113 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/hooks/server/client_lib.py | 31 | IN PROGRESS | `src/hooks/server` | seed_master_plan |  |
| monolith/native_training/hooks/server/constants.py | 15 | IN PROGRESS | `src/hooks/server` | seed_master_plan |  |
| monolith/native_training/hooks/server/server_lib.py | 96 | IN PROGRESS | `src/hooks/server` | seed_master_plan |  |
| monolith/native_training/hooks/server/server_lib_test.py | 55 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/hooks/session_hooks.py | 45 | IN PROGRESS | `src/hooks` | seed_master_plan |  |
| monolith/native_training/hooks/session_hooks_test.py | 34 | IN PROGRESS | `tests` | seed_master_plan |  |
| monolith/native_training/hvd_lib.py | 66 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/input.py | 46 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/layers/__init__.py | 47 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/layers/add_bias.py | 111 | IN PROGRESS | `src/add_bias.rs` | seed_master_plan |  |
| monolith/native_training/layers/add_bias_test.py | 66 | IN PROGRESS | `tests/add_bias_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/advanced_activations.py | 218 | IN PROGRESS | `src/activation.rs` | seed_master_plan |  |
| monolith/native_training/layers/advanced_activations_test.py | 85 | IN PROGRESS | `tests/advanced_activations_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/agru.py | 296 | IN PROGRESS | `src/agru.rs` | seed_master_plan |  |
| monolith/native_training/layers/agru_test.py | 113 | IN PROGRESS | `tests/agru_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/dense.py | 308 | IN PROGRESS | `src/dense.rs` | seed_master_plan |  |
| monolith/native_training/layers/dense_test.py | 148 | IN PROGRESS | `tests/dense_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/feature_cross.py | 806 | IN PROGRESS | `src/feature_cross.rs` | seed_master_plan |  |
| monolith/native_training/layers/feature_cross_test.py | 287 | IN PROGRESS | `tests/feature_cross_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/feature_seq.py | 362 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/layers/feature_seq_test.py | 127 | IN PROGRESS | `tests/feature_seq_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/feature_trans.py | 341 | IN PROGRESS | `src/feature_trans.rs` | seed_master_plan |  |
| monolith/native_training/layers/feature_trans_test.py | 141 | IN PROGRESS | `tests/feature_trans_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/layer_ops.py | 132 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/layers/layer_ops_test.py | 233 | IN PROGRESS | `tests/layer_ops_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/lhuc.py | 297 | IN PROGRESS | `src/lhuc.rs` | seed_master_plan |  |
| monolith/native_training/layers/lhuc_test.py | 74 | IN PROGRESS | `tests/lhuc_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/logit_correction.py | 89 | IN PROGRESS | `src/logit_correction.rs` | seed_master_plan |  |
| monolith/native_training/layers/logit_correction_test.py | 66 | IN PROGRESS | `tests/logit_correction_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/mlp.py | 212 | IN PROGRESS | `src/mlp.rs` | seed_master_plan |  |
| monolith/native_training/layers/mlp_test.py | 79 | IN PROGRESS | `tests/mlp_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/multi_task.py | 449 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/layers/multi_task_test.py | 129 | IN PROGRESS | `tests/multi_task_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/norms.py | 344 | IN PROGRESS | `src/normalization.rs` | seed_master_plan |  |
| monolith/native_training/layers/norms_test.py | 85 | IN PROGRESS | `tests/norms_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/pooling.py | 102 | IN PROGRESS | `src/pooling.rs` | seed_master_plan |  |
| monolith/native_training/layers/pooling_test.py | 142 | IN PROGRESS | `tests/pooling_test.rs` | seed_master_plan |  |
| monolith/native_training/layers/sparse_nas.py | 32 | N/A |  | seed_master_plan | N/A: stub |
| monolith/native_training/layers/sparse_nas_test.py | 24 | N/A |  | seed_master_plan | N/A: empty test |
| monolith/native_training/layers/utils.py | 160 | IN PROGRESS | `src/merge.rs` | seed_master_plan |  |
| monolith/native_training/learning_rate_functions.py | 113 | N/A |  | seed_master_plan | N/A: no Rust schedule yet |
| monolith/native_training/learning_rate_functions_test.py | 77 | N/A |  | seed_master_plan | N/A: no Rust schedule yet |
| monolith/native_training/logging_ops.py | 57 | N/A |  | seed_master_plan | N/A: TF custom ops |
| monolith/native_training/logging_ops_test.py | 58 | N/A |  | seed_master_plan | N/A: TF custom ops |
| monolith/native_training/losses/batch_softmax_loss.py | 58 | N/A |  | seed_master_plan | N/A: no Rust loss yet |
| monolith/native_training/losses/batch_softmax_loss_test.py | 36 | N/A |  | seed_master_plan | N/A: no Rust loss yet |
| monolith/native_training/losses/inbatch_auc_loss.py | 42 | N/A |  | seed_master_plan | N/A: TF custom op |
| monolith/native_training/losses/inbatch_auc_loss_test.py | 72 | N/A |  | seed_master_plan | N/A: TF custom op |
| monolith/native_training/losses/ltr_losses.py | 1234 | N/A |  | seed_master_plan | N/A: no Rust ranking losses yet |
| monolith/native_training/metric/cli.py | 29 | N/A |  | seed_master_plan | N/A: stub |
| monolith/native_training/metric/deep_insight_ops.py | 135 | N/A |  | seed_master_plan | N/A: TF custom ops |
| monolith/native_training/metric/deep_insight_ops_test.py | 33 | N/A |  | seed_master_plan | N/A: empty test |
| monolith/native_training/metric/exit_hook.py | 49 | N/A |  | seed_master_plan | N/A: no Rust hook |
| monolith/native_training/metric/kafka_utils.py | 120 | N/A |  | seed_master_plan | N/A: no Rust Kafka wrapper |
| monolith/native_training/metric/metric_hook.py | 564 | N/A |  | seed_master_plan | N/A: TF hooks + Kafka |
| monolith/native_training/metric/metric_hook_test.py | 190 | N/A |  | seed_master_plan | N/A: TF hooks |
| monolith/native_training/metric/utils.py | 105 | N/A |  | seed_master_plan | N/A: TF custom ops |
| monolith/native_training/metric/utils_test.py | 51 | N/A |  | seed_master_plan | N/A: TF custom ops |
| monolith/native_training/mlp_utils.py | 445 | N/A |  | seed_master_plan | N/A: TF distributed runtime |
| monolith/native_training/model.py | 183 | N/A |  | seed_master_plan | N/A: test model |
| monolith/native_training/model_comp_test.py | 184 | N/A |  | seed_master_plan | N/A: TF/Horovod test |
| monolith/native_training/model_dump/dump_utils.py | 758 | N/A |  | seed_master_plan | N/A: TF model dump |
| monolith/native_training/model_dump/graph_utils.py | 846 | N/A |  | seed_master_plan | N/A: TF graph utils |
| monolith/native_training/model_dump/graph_utils_test.py | 87 | N/A |  | seed_master_plan | N/A: TF graph utils |
| monolith/native_training/model_export/__init__.py | 23 | N/A |  | seed_master_plan | N/A: module alias |
| monolith/native_training/model_export/data_gen_utils.py | 733 | N/A |  | seed_master_plan | N/A: data generator |
| monolith/native_training/model_export/data_gen_utils_test.py | 0 | N/A |  | seed_master_plan | N/A: no tests |
| monolith/native_training/model_export/demo_export.py | 101 | N/A |  | seed_master_plan | N/A: demo exporter |
| monolith/native_training/model_export/demo_export_test.py | 49 | N/A |  | seed_master_plan | N/A: TF export test |
| monolith/native_training/model_export/demo_predictor.py | 111 | N/A |  | seed_master_plan | N/A: demo predictor |
| monolith/native_training/model_export/demo_predictor_client.py | 94 | N/A |  | seed_master_plan | N/A: demo gRPC client |
| monolith/native_training/model_export/export_context.py | 142 | N/A |  | seed_master_plan | N/A: export context |
| monolith/native_training/model_export/export_hooks.py | 138 | N/A |  | seed_master_plan | N/A: TF export hook |
| monolith/native_training/model_export/export_hooks_test.py | 142 | N/A |  | seed_master_plan | N/A: export hook test |
| monolith/native_training/model_export/export_state_utils.py | 47 | N/A |  | seed_master_plan | N/A: export state |
| monolith/native_training/model_export/export_state_utils_test.py | 37 | N/A |  | seed_master_plan | N/A: export state test |
| monolith/native_training/model_export/export_utils.py | 99 | N/A |  | seed_master_plan | N/A: remote predict helper |
| monolith/native_training/model_export/export_utils_test.py | 44 | N/A |  | seed_master_plan | N/A: remote predict test |
| monolith/native_training/model_export/saved_model_exporters.py | 740 | N/A |  | seed_master_plan | N/A: SavedModel exporters |
| monolith/native_training/model_export/saved_model_exporters_test.py | 154 | N/A |  | seed_master_plan | N/A: exporter tests |
| monolith/native_training/model_export/saved_model_visulizer.py | 90 | N/A |  | seed_master_plan | N/A: tensorboard visualizer |
| monolith/native_training/model_export/warmup_data_decoder.py | 56 | N/A |  | seed_master_plan | N/A: warmup decoder |
| monolith/native_training/model_export/warmup_data_gen.py | 254 | N/A |  | seed_master_plan | N/A: warmup generator |
| monolith/native_training/model_export/warmup_example_batch.py | 57 | N/A |  | seed_master_plan | N/A: warmup example batch |
| monolith/native_training/monolith_export.py | 19 | N/A |  | seed_master_plan | N/A: decorator |
| monolith/native_training/multi_hash_table_ops.py | 696 | N/A |  | seed_master_plan | N/A: TF custom ops |
| monolith/native_training/multi_hash_table_ops_test.py | 250 | N/A |  | seed_master_plan | N/A: TF custom ops |
| monolith/native_training/multi_type_hash_table.py | 436 | N/A |  | seed_master_plan | N/A: hash table wrapper |
| monolith/native_training/multi_type_hash_table_test.py | 327 | N/A |  | seed_master_plan | N/A: hash table tests |
| monolith/native_training/native_model.py | 1110 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/native_task.py | 214 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/native_task_context.py | 59 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/nested_tensors.py | 110 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/nested_tensors_test.py | 58 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/net_utils.py | 134 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/net_utils_test.py | 95 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/optimizers/adamom.py | 69 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/optimizers/adamom_test.py | 57 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/optimizers/rmsprop.py | 103 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/optimizers/rmsprop_test.py | 78 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/optimizers/rmspropv2_test.py | 113 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/optimizers/shampoo.py | 208 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/prefetch_queue.py | 380 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/prefetch_queue_test.py | 306 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/ps_benchmark.py | 274 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/ps_benchmark_test.py | 58 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/ragged_utils.py | 30 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/ragged_utils_test.py | 33 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/remote_predict_ops.py | 0 | N/A |  | seed_master_plan | N/A: empty stub |
| monolith/native_training/restore_test.py | 241 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/runner_utils.py | 397 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/runner_utils_test.py | 109 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/runtime/ops/gen_monolith_ops.py | 24 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/save_utils.py | 1310 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/save_utils_test.py | 1741 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/service_discovery.py | 482 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/service_discovery_test.py | 408 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/serving_ps_test.py | 232 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/session_run_hooks.py | 172 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/session_run_hooks_test.py | 145 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/signal_utils.py | 38 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/signal_utils_test.py | 31 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/static_reshape_op.py | 59 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/static_reshape_op_test.py | 80 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/summary/summary_ops.py | 79 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/summary/summary_ops_test.py | 123 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/summary/utils.py | 115 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/summary/utils_test.py | 44 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/sync_hooks.py | 177 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/sync_hooks_test.py | 120 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/sync_training_hooks.py | 356 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/sync_training_hooks_test.py | 93 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/tensor_utils.py | 163 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/tensor_utils_test.py | 176 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/test_utils.py | 66 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/touched_key_set_ops.py | 62 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/touched_key_set_ops_test.py | 52 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/native_training/utils.py | 321 | IN PROGRESS | `src/utils.rs` | seed_master_plan |  |
| monolith/native_training/utils_test.py | 71 | IN PROGRESS | `tests/utils.rs` | seed_master_plan |  |
| monolith/native_training/variables.py | 148 | IN PROGRESS | `src/variables.rs` | seed_master_plan |  |
| monolith/native_training/variables_test.py | 90 | IN PROGRESS | `tests/variables.rs` | seed_master_plan |  |
| monolith/native_training/yarn_runtime.py | 128 | IN PROGRESS | `src/yarn_runtime.rs` | seed_master_plan |  |
| monolith/native_training/yarn_runtime_test.py | 134 | IN PROGRESS | `tests/yarn_runtime.rs` | seed_master_plan |  |
| monolith/native_training/zk_utils.py | 97 | IN PROGRESS | `src/zk_utils.rs` | seed_master_plan |  |
| monolith/path_utils.py | 48 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/tpu_runner.py | 430 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/utils.py | 82 | IN PROGRESS | `src` | seed_master_plan |  |
| monolith/utils_test.py | 66 | IN PROGRESS | `tests` | seed_master_plan |  |
