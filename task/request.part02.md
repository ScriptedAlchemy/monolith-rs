<!--
Source: task/request.md
Lines: 343-683 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
## Line-Level Inventory (All Python Files)

This table enumerates **every** Python file under `monolith/` with line counts and a direct link to its checklist section.

| Python File | Lines | Status | Rust Mapping | Notes |
|---|---:|---|---|---|
| [`monolith/__init__.py`](#monolith-init-py) | 55 | IN PROGRESS | monolith-rs/crates/monolith-core | |
| [`monolith/agent_service/__init__.py`](#monolith-agent-service-init-py) | 0 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/agent.py`](#monolith-agent-service-agent-py) | 100 | IN PROGRESS | monolith-rs/crates/monolith-cli, monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/agent_base.py`](#monolith-agent-service-agent-base-py) | 88 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/agent_client.py`](#monolith-agent-service-agent-client-py) | 216 | IN PROGRESS | monolith-rs/crates/monolith-cli/src/bin/agent_client.rs | |
| [`monolith/agent_service/agent_controller.py`](#monolith-agent-service-agent-controller-py) | 145 | IN PROGRESS | monolith-rs/crates/monolith-cli/src/bin/agent_controller.rs | |
| [`monolith/agent_service/agent_controller_test.py`](#monolith-agent-service-agent-controller-test-py) | 95 | IN PROGRESS | monolith-rs/crates/monolith-cli/tests | |
| [`monolith/agent_service/agent_service.py`](#monolith-agent-service-agent-service-py) | 155 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/agent_service_test.py`](#monolith-agent-service-agent-service-test-py) | 107 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/agent_v1.py`](#monolith-agent-service-agent-v1-py) | 390 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/agent_v3.py`](#monolith-agent-service-agent-v3-py) | 210 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/agent_v3_test.py`](#monolith-agent-service-agent-v3-test-py) | 114 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/backends.py`](#monolith-agent-service-backends-py) | 518 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/backends_test.py`](#monolith-agent-service-backends-test-py) | 134 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/client.py`](#monolith-agent-service-client-py) | 126 | IN PROGRESS | monolith-rs/crates/monolith-cli/src/bin/serving_client.rs | |
| [`monolith/agent_service/constants.py`](#monolith-agent-service-constants-py) | 15 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/data_def.py`](#monolith-agent-service-data-def-py) | 171 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/data_def_test.py`](#monolith-agent-service-data-def-test-py) | 52 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/mocked_tfserving.py`](#monolith-agent-service-mocked-tfserving-py) | 399 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests/support | |
| [`monolith/agent_service/mocked_tfserving_test.py`](#monolith-agent-service-mocked-tfserving-test-py) | 92 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/mocked_zkclient.py`](#monolith-agent-service-mocked-zkclient-py) | 377 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests/support | |
| [`monolith/agent_service/mocked_zkclient_test.py`](#monolith-agent-service-mocked-zkclient-test-py) | 130 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/model_manager.py`](#monolith-agent-service-model-manager-py) | 371 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/model_manager_test.py`](#monolith-agent-service-model-manager-test-py) | 113 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/replica_manager.py`](#monolith-agent-service-replica-manager-py) | 835 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/replica_manager_test.py`](#monolith-agent-service-replica-manager-test-py) | 126 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/resource_utils.py`](#monolith-agent-service-resource-utils-py) | 269 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/resource_utils_test.py`](#monolith-agent-service-resource-utils-test-py) | 36 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/run.py`](#monolith-agent-service-run-py) | 39 | IN PROGRESS | monolith-rs/crates/monolith-cli/src/bin | |
| [`monolith/agent_service/svr_client.py`](#monolith-agent-service-svr-client-py) | 70 | IN PROGRESS | monolith-rs/crates/monolith-cli/src | |
| [`monolith/agent_service/tfs_client.py`](#monolith-agent-service-tfs-client-py) | 503 | IN PROGRESS | monolith-rs/crates/monolith-cli/src/bin/tfs_client.rs | |
| [`monolith/agent_service/tfs_client_test.py`](#monolith-agent-service-tfs-client-test-py) | 50 | IN PROGRESS | monolith-rs/crates/monolith-cli/tests | |
| [`monolith/agent_service/tfs_monitor.py`](#monolith-agent-service-tfs-monitor-py) | 303 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/tfs_monitor_test.py`](#monolith-agent-service-tfs-monitor-test-py) | 182 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/tfs_wrapper.py`](#monolith-agent-service-tfs-wrapper-py) | 202 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/utils.py`](#monolith-agent-service-utils-py) | 1167 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/utils_test.py`](#monolith-agent-service-utils-test-py) | 170 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/zk_mirror.py`](#monolith-agent-service-zk-mirror-py) | 672 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/zk_mirror_test.py`](#monolith-agent-service-zk-mirror-test-py) | 229 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/base_runner.py`](#monolith-base-runner-py) | 46 | IN PROGRESS | monolith-rs/crates/monolith-training/src | |
| [`monolith/common/python/mem_profiling.py`](#monolith-common-python-mem-profiling-py) | 51 | IN PROGRESS | monolith-rs/crates/monolith-training/src | |
| [`monolith/core/__init__.py`](#monolith-core-init-py) | 0 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/auto_checkpoint_feed_hook.py`](#monolith-core-auto-checkpoint-feed-hook-py) | 376 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/core/base_embedding_host_call.py`](#monolith-core-base-embedding-host-call-py) | 643 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/base_embedding_host_call_test.py`](#monolith-core-base-embedding-host-call-test-py) | 77 | IN PROGRESS | monolith-rs/crates/monolith-core/tests | |
| [`monolith/core/base_embedding_task.py`](#monolith-core-base-embedding-task-py) | 611 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/base_host_call.py`](#monolith-core-base-host-call-py) | 145 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/base_layer.py`](#monolith-core-base-layer-py) | 161 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/base_layer_test.py`](#monolith-core-base-layer-test-py) | 41 | IN PROGRESS | monolith-rs/crates/monolith-core/tests | |
| [`monolith/core/base_model_params.py`](#monolith-core-base-model-params-py) | 25 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/base_task.py`](#monolith-core-base-task-py) | 95 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/base_tpu_test.py`](#monolith-core-base-tpu-test-py) | 73 | IN PROGRESS | monolith-rs/crates/monolith-training/tests | |
| [`monolith/core/core_test_suite.py`](#monolith-core-core-test-suite-py) | 35 | IN PROGRESS | monolith-rs/crates/monolith-core/tests | |
| [`monolith/core/dense.py`](#monolith-core-dense-py) | 179 | IN PROGRESS | monolith-rs/crates/monolith-layers/src | |
| [`monolith/core/dense_test.py`](#monolith-core-dense-test-py) | 108 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests | |
| [`monolith/core/feature.py`](#monolith-core-feature-py) | 611 | IN PROGRESS | monolith-rs/crates/monolith-core/src/feature.rs |  |
| [`monolith/core/feature_test.py`](#monolith-core-feature-test-py) | 178 | IN PROGRESS | monolith-rs/crates/monolith-core/tests |  |
| [`monolith/core/host_call.py`](#monolith-core-host-call-py) | 248 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/hyperparams.py`](#monolith-core-hyperparams-py) | 439 | IN PROGRESS | monolith-rs/crates/monolith-core/src/hyperparams.rs |  |
| [`monolith/core/hyperparams_test.py`](#monolith-core-hyperparams-test-py) | 277 | IN PROGRESS | monolith-rs/crates/monolith-core/tests |  |
| [`monolith/core/mixed_emb_op_comb_nws.py`](#monolith-core-mixed-emb-op-comb-nws-py) | 421 | IN PROGRESS | monolith-rs/crates/monolith-layers/src |  |
| [`monolith/core/model.py`](#monolith-core-model-py) | 320 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/core/model_imports.py`](#monolith-core-model-imports-py) | 104 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/model_registry.py`](#monolith-core-model-registry-py) | 174 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/optimizers.py`](#monolith-core-optimizers-py) | 25 | IN PROGRESS | monolith-rs/crates/monolith-optimizer/src | |
| [`monolith/core/py_utils.py`](#monolith-core-py-utils-py) | 313 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/testing_utils.py`](#monolith-core-testing-utils-py) | 203 | IN PROGRESS | monolith-rs/crates/monolith-core/tests | |
| [`monolith/core/tpu_variable.py`](#monolith-core-tpu-variable-py) | 214 | IN PROGRESS | monolith-rs/crates/monolith-tf/src | |
| [`monolith/core/util.py`](#monolith-core-util-py) | 269 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/util_test.py`](#monolith-core-util-test-py) | 149 | IN PROGRESS | monolith-rs/crates/monolith-core/tests | |
| [`monolith/core/variance_scaling.py`](#monolith-core-variance-scaling-py) | 188 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/gpu_runner.py`](#monolith-gpu-runner-py) | 226 | IN PROGRESS | monolith-rs/crates/monolith-training/src | |
| [`monolith/native_training/alert/alert_manager.py`](#monolith-native-training-alert-alert-manager-py) | 31 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/alert/alert_manager_test.py`](#monolith-native-training-alert-alert-manager-test-py) | 32 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/barrier_ops.py`](#monolith-native-training-barrier-ops-py) | 158 | IN PROGRESS | monolith-rs/crates/monolith-training/src/barrier.rs |  |
| [`monolith/native_training/barrier_ops_test.py`](#monolith-native-training-barrier-ops-test-py) | 104 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/basic_restore_hook.py`](#monolith-native-training-basic-restore-hook-py) | 72 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/basic_restore_hook_test.py`](#monolith-native-training-basic-restore-hook-test-py) | 137 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/clip_ops.py`](#monolith-native-training-clip-ops-py) | 80 | IN PROGRESS | monolith-rs/crates/monolith-optimizer/src |  |
| [`monolith/native_training/clip_ops_test.py`](#monolith-native-training-clip-ops-test-py) | 92 | IN PROGRESS | monolith-rs/crates/monolith-optimizer/tests |  |
| [`monolith/native_training/cluster_manager.py`](#monolith-native-training-cluster-manager-py) | 184 | IN PROGRESS | monolith-rs/crates/monolith-training/src/distributed.rs |  |
| [`monolith/native_training/cluster_manager_test.py`](#monolith-native-training-cluster-manager-test-py) | 35 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/consul.py`](#monolith-native-training-consul-py) | 149 | IN PROGRESS | monolith-rs/crates/monolith-training/src/discovery.rs |  |
| [`monolith/native_training/consul_test.py`](#monolith-native-training-consul-test-py) | 59 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/cpu_sync_training_test.py`](#monolith-native-training-cpu-sync-training-test-py) | 360 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/cpu_training.py`](#monolith-native-training-cpu-training-py) | 2449 | IN PROGRESS | monolith-rs/crates/monolith-training/src/cpu_training.rs (new), monolith-rs/crates/monolith-training/src/distributed.rs (new), monolith-rs/crates/monolith-training/src/local.rs (new) |  |
| [`monolith/native_training/cpu_training_distributed_test_binary.py`](#monolith-native-training-cpu-training-distributed-test-binary-py) | 226 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/cpu_training_test.py`](#monolith-native-training-cpu-training-test-py) | 597 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/data/__init__.py`](#monolith-native-training-data-init-py) | 20 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/data_ops_test.py`](#monolith-native-training-data-data-ops-test-py) | 502 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/data_service_parquet_test.py`](#monolith-native-training-data-data-service-parquet-test-py) | 145 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/data_service_test.py`](#monolith-native-training-data-data-service-test-py) | 98 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/datasets.py`](#monolith-native-training-data-datasets-py) | 1642 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/eager_mode_test.py`](#monolith-native-training-data-eager-mode-test-py) | 186 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/extract_fid_test.py`](#monolith-native-training-data-extract-fid-test-py) | 30 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/feature_list.py`](#monolith-native-training-data-feature-list-py) | 409 | IN PROGRESS | monolith-rs/crates/monolith-data/src/feature_list.rs |  |
| [`monolith/native_training/data/feature_list_test.py`](#monolith-native-training-data-feature-list-test-py) | 0 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/feature_utils.py`](#monolith-native-training-data-feature-utils-py) | 1070 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/feature_utils_test.py`](#monolith-native-training-data-feature-utils-test-py) | 1414 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/item_pool_hook.py`](#monolith-native-training-data-item-pool-hook-py) | 109 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/item_pool_test.py`](#monolith-native-training-data-item-pool-test-py) | 58 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/kafka_dataset_test.py`](#monolith-native-training-data-kafka-dataset-test-py) | 239 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/multi_flow_test.py`](#monolith-native-training-data-multi-flow-test-py) | 125 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/negative_gen_test.py`](#monolith-native-training-data-negative-gen-test-py) | 253 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/parse_sparse_feature_test.py`](#monolith-native-training-data-parse-sparse-feature-test-py) | 1833 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/parsers.py`](#monolith-native-training-data-parsers-py) | 782 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/tf_example_to_example_test.py`](#monolith-native-training-data-tf-example-to-example-test-py) | 183 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/training_instance/python/instance_dataset_op.py`](#monolith-native-training-data-training-instance-python-instance-dataset-op-py) | 166 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/training_instance/python/instance_dataset_op_test_stdin.py`](#monolith-native-training-data-training-instance-python-instance-dataset-op-test-stdin-py) | 58 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/training_instance/python/instance_negative_gen_dataset_op_test.py`](#monolith-native-training-data-training-instance-python-instance-negative-gen-dataset-op-test-py) | 283 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/training_instance/python/parse_instance_ops.py`](#monolith-native-training-data-training-instance-python-parse-instance-ops-py) | 245 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/training_instance/python/parse_instance_ops_test.py`](#monolith-native-training-data-training-instance-python-parse-instance-ops-test-py) | 185 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/training_instance/python/parser_utils.py`](#monolith-native-training-data-training-instance-python-parser-utils-py) | 85 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/training_instance/python/pb_datasource_ops.py`](#monolith-native-training-data-training-instance-python-pb-datasource-ops-py) | 48 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/training_instance/python/test_data_utils.py`](#monolith-native-training-data-training-instance-python-test-data-utils-py) | 15 | IN PROGRESS | none |  |
| [`monolith/native_training/data/transform/transforms.py`](#monolith-native-training-data-transform-transforms-py) | 250 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/transform/transforms_test.py`](#monolith-native-training-data-transform-transforms-test-py) | 70 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/transform_dataset_test.py`](#monolith-native-training-data-transform-dataset-test-py) | 168 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/utils.py`](#monolith-native-training-data-utils-py) | 55 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/debugging/debugging_client.py`](#monolith-native-training-debugging-debugging-client-py) | 98 | IN PROGRESS | monolith-rs/crates/monolith-training/src/debugging |  |
| [`monolith/native_training/debugging/debugging_server.py`](#monolith-native-training-debugging-debugging-server-py) | 217 | IN PROGRESS | monolith-rs/crates/monolith-training/src/debugging |  |
| [`monolith/native_training/demo.py`](#monolith-native-training-demo-py) | 57 | IN PROGRESS | monolith-rs/crates/monolith-training/examples |  |
| [`monolith/native_training/dense_reload_utils.py`](#monolith-native-training-dense-reload-utils-py) | 457 | IN PROGRESS | monolith-rs/crates/monolith-training/src/checkpoint |  |
| [`monolith/native_training/dense_reload_utils_test.py`](#monolith-native-training-dense-reload-utils-test-py) | 192 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/device_utils.py`](#monolith-native-training-device-utils-py) | 231 | IN PROGRESS | monolith-rs/crates/monolith-training/src/device |  |
| [`monolith/native_training/device_utils_test.py`](#monolith-native-training-device-utils-test-py) | 104 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/distribute/distributed_dataset.py`](#monolith-native-training-distribute-distributed-dataset-py) | 81 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/distribute/distributed_dataset_test.py`](#monolith-native-training-distribute-distributed-dataset-test-py) | 124 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/distribute/str_queue.py`](#monolith-native-training-distribute-str-queue-py) | 114 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/distribute/str_queue_test.py`](#monolith-native-training-distribute-str-queue-test-py) | 67 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/distributed_ps.py`](#monolith-native-training-distributed-ps-py) | 2108 | IN PROGRESS | monolith-rs/crates/monolith-training/src/ps |  |
| [`monolith/native_training/distributed_ps_benchmark.py`](#monolith-native-training-distributed-ps-benchmark-py) | 168 | IN PROGRESS | monolith-rs/crates/monolith-training/benches |  |
| [`monolith/native_training/distributed_ps_factory.py`](#monolith-native-training-distributed-ps-factory-py) | 262 | IN PROGRESS | monolith-rs/crates/monolith-training/src/ps |  |
| [`monolith/native_training/distributed_ps_factory_test.py`](#monolith-native-training-distributed-ps-factory-test-py) | 87 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/distributed_ps_sync.py`](#monolith-native-training-distributed-ps-sync-py) | 531 | IN PROGRESS | monolith-rs/crates/monolith-training/src/ps |  |
| [`monolith/native_training/distributed_ps_sync_test.py`](#monolith-native-training-distributed-ps-sync-test-py) | 109 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/distributed_ps_test.py`](#monolith-native-training-distributed-ps-test-py) | 979 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/distributed_serving_ops.py`](#monolith-native-training-distributed-serving-ops-py) | 160 | IN PROGRESS | monolith-rs/crates/monolith-training/src/serving |  |
| [`monolith/native_training/distributed_serving_ops_test.py`](#monolith-native-training-distributed-serving-ops-test-py) | 142 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/distribution_ops.py`](#monolith-native-training-distribution-ops-py) | 889 | IN PROGRESS | monolith-rs/crates/monolith-training/src/ops |  |
| [`monolith/native_training/distribution_ops_benchmark.py`](#monolith-native-training-distribution-ops-benchmark-py) | 118 | IN PROGRESS | monolith-rs/crates/monolith-training/benches |  |
| [`monolith/native_training/distribution_ops_fused_benchmark.py`](#monolith-native-training-distribution-ops-fused-benchmark-py) | 61 | IN PROGRESS | monolith-rs/crates/monolith-training/benches |  |
| [`monolith/native_training/distribution_ops_fused_test.py`](#monolith-native-training-distribution-ops-fused-test-py) | 148 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/distribution_ops_test.py`](#monolith-native-training-distribution-ops-test-py) | 536 | IN PROGRESS | monolith-rs/crates/monolith-tf/tests |  |
| [`monolith/native_training/distribution_utils.py`](#monolith-native-training-distribution-utils-py) | 443 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/embedding_combiners.py`](#monolith-native-training-embedding-combiners-py) | 102 | IN PROGRESS | monolith-rs/crates/monolith-layers/src |  |
| [`monolith/native_training/embedding_combiners_test.py`](#monolith-native-training-embedding-combiners-test-py) | 47 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests |  |
| [`monolith/native_training/entry.py`](#monolith-native-training-entry-py) | 630 | IN PROGRESS | monolith-rs/crates/monolith-hash-table/src |  |
| [`monolith/native_training/entry_test.py`](#monolith-native-training-entry-test-py) | 84 | IN PROGRESS | monolith-rs/crates/monolith-hash-table/tests |  |
| [`monolith/native_training/env_utils.py`](#monolith-native-training-env-utils-py) | 32 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/env_utils_test.py`](#monolith-native-training-env-utils-test-py) | 23 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/estimator.py`](#monolith-native-training-estimator-py) | 667 | IN PROGRESS | monolith-rs/crates/monolith-training/src/estimator.rs |  |
| [`monolith/native_training/estimator_dist_test.py`](#monolith-native-training-estimator-dist-test-py) | 166 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/estimator_mode_test.py`](#monolith-native-training-estimator-mode-test-py) | 417 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/estimator_test.py`](#monolith-native-training-estimator-test-py) | 112 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/feature.py`](#monolith-native-training-feature-py) | 663 | IN PROGRESS | monolith-rs/crates/monolith-core/src/feature.rs |  |
| [`monolith/native_training/feature_test.py`](#monolith-native-training-feature-test-py) | 266 | IN PROGRESS | monolith-rs/crates/monolith-core/tests |  |
| [`monolith/native_training/feature_utils.py`](#monolith-native-training-feature-utils-py) | 419 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/feature_utils_test.py`](#monolith-native-training-feature-utils-test-py) | 144 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/file_ops.py`](#monolith-native-training-file-ops-py) | 51 | IN PROGRESS | monolith-rs/crates/monolith-training/src/file_ops.rs (new) |  |
| [`monolith/native_training/file_ops_test.py`](#monolith-native-training-file-ops-test-py) | 56 | IN PROGRESS | monolith-rs/crates/monolith-training/tests/file_ops.rs (new) |  |
| [`monolith/native_training/fused_embedding_to_layout_test.py`](#monolith-native-training-fused-embedding-to-layout-test-py) | 1333 | IN PROGRESS | monolith-rs/crates/monolith-training/tests/fused_embedding_to_layout.rs (new) |  |
| [`monolith/native_training/gen_seq_mask.py`](#monolith-native-training-gen-seq-mask-py) | 26 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/native_training/gen_seq_mask_test.py`](#monolith-native-training-gen-seq-mask-test-py) | 42 | IN PROGRESS | monolith-rs/crates/monolith-tf/tests |  |
| [`monolith/native_training/gflags_utils.py`](#monolith-native-training-gflags-utils-py) | 282 | IN PROGRESS | monolith-rs/crates/monolith-cli/src |  |
| [`monolith/native_training/gflags_utils_test.py`](#monolith-native-training-gflags-utils-test-py) | 217 | IN PROGRESS | monolith-rs/crates/monolith-cli/tests |  |
| [`monolith/native_training/graph_meta.py`](#monolith-native-training-graph-meta-py) | 30 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/native_training/graph_utils.py`](#monolith-native-training-graph-utils-py) | 26 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/native_training/hash_filter_ops.py`](#monolith-native-training-hash-filter-ops-py) | 326 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/native_training/hash_filter_ops_test.py`](#monolith-native-training-hash-filter-ops-test-py) | 228 | IN PROGRESS | monolith-rs/crates/monolith-tf/tests |  |
| [`monolith/native_training/hash_table_ops.py`](#monolith-native-training-hash-table-ops-py) | 738 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/native_training/hash_table_ops_benchmark.py`](#monolith-native-training-hash-table-ops-benchmark-py) | 148 | IN PROGRESS | monolith-rs/crates/monolith-examples/src/bin/hash_table_ops_benchmark.rs (new) |  |
| [`monolith/native_training/hash_table_ops_test.py`](#monolith-native-training-hash-table-ops-test-py) | 1200 | IN PROGRESS | monolith-rs/crates/monolith-tf/tests |  |
| [`monolith/native_training/hash_table_utils.py`](#monolith-native-training-hash-table-utils-py) | 50 | IN PROGRESS | monolith-rs/crates/monolith-hash-table/src |  |
| [`monolith/native_training/hash_table_utils_test.py`](#monolith-native-training-hash-table-utils-test-py) | 45 | IN PROGRESS | monolith-rs/crates/monolith-hash-table/tests |  |
| [`monolith/native_training/hooks/ckpt_hooks.py`](#monolith-native-training-hooks-ckpt-hooks-py) | 193 | IN PROGRESS | monolith-rs/crates/monolith-training/src/hooks |  |
| [`monolith/native_training/hooks/ckpt_hooks_test.py`](#monolith-native-training-hooks-ckpt-hooks-test-py) | 181 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/hooks/ckpt_info.py`](#monolith-native-training-hooks-ckpt-info-py) | 98 | IN PROGRESS | monolith-rs/crates/monolith-training/src/hooks |  |
| [`monolith/native_training/hooks/ckpt_info_test.py`](#monolith-native-training-hooks-ckpt-info-test-py) | 45 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/hooks/controller_hooks.py`](#monolith-native-training-hooks-controller-hooks-py) | 170 | IN PROGRESS | monolith-rs/crates/monolith-training/src/hooks |  |
| [`monolith/native_training/hooks/controller_hooks_test.py`](#monolith-native-training-hooks-controller-hooks-test-py) | 82 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/hooks/feature_engineering_hooks.py`](#monolith-native-training-hooks-feature-engineering-hooks-py) | 99 | IN PROGRESS | monolith-rs/crates/monolith-training/src/hooks |  |
| [`monolith/native_training/hooks/hook_utils.py`](#monolith-native-training-hooks-hook-utils-py) | 41 | IN PROGRESS | monolith-rs/crates/monolith-training/src/hooks |  |
| [`monolith/native_training/hooks/hook_utils_test.py`](#monolith-native-training-hooks-hook-utils-test-py) | 35 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/hooks/ps_check_hooks.py`](#monolith-native-training-hooks-ps-check-hooks-py) | 97 | IN PROGRESS | monolith-rs/crates/monolith-training/src/hooks |  |
| [`monolith/native_training/hooks/ps_check_hooks_test.py`](#monolith-native-training-hooks-ps-check-hooks-test-py) | 112 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/hooks/server/client_lib.py`](#monolith-native-training-hooks-server-client-lib-py) | 30 | IN PROGRESS | monolith-rs/crates/monolith-training/src/hooks/server |  |
| [`monolith/native_training/hooks/server/constants.py`](#monolith-native-training-hooks-server-constants-py) | 15 | IN PROGRESS | monolith-rs/crates/monolith-training/src/hooks/server |  |
| [`monolith/native_training/hooks/server/server_lib.py`](#monolith-native-training-hooks-server-server-lib-py) | 95 | IN PROGRESS | monolith-rs/crates/monolith-training/src/hooks/server |  |
| [`monolith/native_training/hooks/server/server_lib_test.py`](#monolith-native-training-hooks-server-server-lib-test-py) | 54 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/hooks/session_hooks.py`](#monolith-native-training-hooks-session-hooks-py) | 44 | IN PROGRESS | monolith-rs/crates/monolith-training/src/hooks |  |
| [`monolith/native_training/hooks/session_hooks_test.py`](#monolith-native-training-hooks-session-hooks-test-py) | 33 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/hvd_lib.py`](#monolith-native-training-hvd-lib-py) | 65 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/input.py`](#monolith-native-training-input-py) | 45 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/layers/__init__.py`](#monolith-native-training-layers-init-py) | 46 | IN PROGRESS | monolith-rs/crates/monolith-layers/src |  |
| [`monolith/native_training/layers/add_bias.py`](#monolith-native-training-layers-add-bias-py) | 110 | IN PROGRESS | monolith-rs/crates/monolith-layers/src/add_bias.rs |  |
| [`monolith/native_training/layers/add_bias_test.py`](#monolith-native-training-layers-add-bias-test-py) | 65 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/add_bias_test.rs |  |
| [`monolith/native_training/layers/advanced_activations.py`](#monolith-native-training-layers-advanced-activations-py) | 217 | IN PROGRESS | monolith-rs/crates/monolith-layers/src/activation.rs |  |
| [`monolith/native_training/layers/advanced_activations_test.py`](#monolith-native-training-layers-advanced-activations-test-py) | 84 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/advanced_activations_test.rs |  |
| [`monolith/native_training/layers/agru.py`](#monolith-native-training-layers-agru-py) | 295 | IN PROGRESS | monolith-rs/crates/monolith-layers/src/agru.rs |  |
| [`monolith/native_training/layers/agru_test.py`](#monolith-native-training-layers-agru-test-py) | 112 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/agru_test.rs |  |
| [`monolith/native_training/layers/dense.py`](#monolith-native-training-layers-dense-py) | 307 | IN PROGRESS | monolith-rs/crates/monolith-layers/src/dense.rs |  |
| [`monolith/native_training/layers/dense_test.py`](#monolith-native-training-layers-dense-test-py) | 147 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/dense_test.rs |  |
| [`monolith/native_training/layers/feature_cross.py`](#monolith-native-training-layers-feature-cross-py) | 805 | IN PROGRESS | monolith-rs/crates/monolith-layers/src/feature_cross.rs |  |
| [`monolith/native_training/layers/feature_cross_test.py`](#monolith-native-training-layers-feature-cross-test-py) | 286 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/feature_cross_test.rs |  |
| [`monolith/native_training/layers/feature_seq.py`](#monolith-native-training-layers-feature-seq-py) | 361 | IN PROGRESS | monolith-rs/crates/monolith-layers/src |  |
| [`monolith/native_training/layers/feature_seq_test.py`](#monolith-native-training-layers-feature-seq-test-py) | 126 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/feature_seq_test.rs |  |
| [`monolith/native_training/layers/feature_trans.py`](#monolith-native-training-layers-feature-trans-py) | 340 | IN PROGRESS | monolith-rs/crates/monolith-layers/src/feature_trans.rs |  |
| [`monolith/native_training/layers/feature_trans_test.py`](#monolith-native-training-layers-feature-trans-test-py) | 140 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/feature_trans_test.rs |  |
| [`monolith/native_training/layers/layer_ops.py`](#monolith-native-training-layers-layer-ops-py) | 131 | IN PROGRESS | monolith-rs/crates/monolith-layers/src |  |
| [`monolith/native_training/layers/layer_ops_test.py`](#monolith-native-training-layers-layer-ops-test-py) | 232 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/layer_ops_test.rs |  |
| [`monolith/native_training/layers/lhuc.py`](#monolith-native-training-layers-lhuc-py) | 296 | IN PROGRESS | monolith-rs/crates/monolith-layers/src/lhuc.rs |  |
| [`monolith/native_training/layers/lhuc_test.py`](#monolith-native-training-layers-lhuc-test-py) | 73 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/lhuc_test.rs |  |
| [`monolith/native_training/layers/logit_correction.py`](#monolith-native-training-layers-logit-correction-py) | 88 | IN PROGRESS | monolith-rs/crates/monolith-layers/src/logit_correction.rs |  |
| [`monolith/native_training/layers/logit_correction_test.py`](#monolith-native-training-layers-logit-correction-test-py) | 65 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/logit_correction_test.rs |  |
| [`monolith/native_training/layers/mlp.py`](#monolith-native-training-layers-mlp-py) | 211 | IN PROGRESS | monolith-rs/crates/monolith-layers/src/mlp.rs |  |
| [`monolith/native_training/layers/mlp_test.py`](#monolith-native-training-layers-mlp-test-py) | 78 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/mlp_test.rs |  |
| [`monolith/native_training/layers/multi_task.py`](#monolith-native-training-layers-multi-task-py) | 448 | IN PROGRESS | monolith-rs/crates/monolith-layers/src |  |
| [`monolith/native_training/layers/multi_task_test.py`](#monolith-native-training-layers-multi-task-test-py) | 128 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/multi_task_test.rs |  |
| [`monolith/native_training/layers/norms.py`](#monolith-native-training-layers-norms-py) | 343 | IN PROGRESS | monolith-rs/crates/monolith-layers/src/normalization.rs |  |
| [`monolith/native_training/layers/norms_test.py`](#monolith-native-training-layers-norms-test-py) | 84 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/norms_test.rs |  |
| [`monolith/native_training/layers/pooling.py`](#monolith-native-training-layers-pooling-py) | 101 | IN PROGRESS | monolith-rs/crates/monolith-layers/src/pooling.rs |  |
| [`monolith/native_training/layers/pooling_test.py`](#monolith-native-training-layers-pooling-test-py) | 141 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests/pooling_test.rs |  |
| [`monolith/native_training/layers/sparse_nas.py`](#monolith-native-training-layers-sparse-nas-py) | 31 | IN PROGRESS | N/A (stub) |  |
| [`monolith/native_training/layers/sparse_nas_test.py`](#monolith-native-training-layers-sparse-nas-test-py) | 23 | IN PROGRESS | N/A (empty test) |  |
| [`monolith/native_training/layers/utils.py`](#monolith-native-training-layers-utils-py) | 159 | IN PROGRESS | monolith-rs/crates/monolith-layers/src/merge.rs |  |
| [`monolith/native_training/learning_rate_functions.py`](#monolith-native-training-learning-rate-functions-py) | 112 | IN PROGRESS | N/A (no Rust schedule yet) |  |
| [`monolith/native_training/learning_rate_functions_test.py`](#monolith-native-training-learning-rate-functions-test-py) | 76 | IN PROGRESS | N/A (no Rust schedule yet) |  |
| [`monolith/native_training/logging_ops.py`](#monolith-native-training-logging-ops-py) | 56 | IN PROGRESS | N/A (TF custom ops) |  |
| [`monolith/native_training/logging_ops_test.py`](#monolith-native-training-logging-ops-test-py) | 57 | IN PROGRESS | N/A (TF custom ops) |  |
| [`monolith/native_training/losses/batch_softmax_loss.py`](#monolith-native-training-losses-batch-softmax-loss-py) | 57 | IN PROGRESS | N/A (no Rust loss yet) |  |
| [`monolith/native_training/losses/batch_softmax_loss_test.py`](#monolith-native-training-losses-batch-softmax-loss-test-py) | 35 | IN PROGRESS | N/A (no Rust loss yet) |  |
| [`monolith/native_training/losses/inbatch_auc_loss.py`](#monolith-native-training-losses-inbatch-auc-loss-py) | 41 | IN PROGRESS | N/A (TF custom op) |  |
| [`monolith/native_training/losses/inbatch_auc_loss_test.py`](#monolith-native-training-losses-inbatch-auc-loss-test-py) | 71 | IN PROGRESS | N/A (TF custom op) |  |
| [`monolith/native_training/losses/ltr_losses.py`](#monolith-native-training-losses-ltr-losses-py) | 1233 | IN PROGRESS | N/A (no Rust ranking losses yet) |  |
| [`monolith/native_training/metric/cli.py`](#monolith-native-training-metric-cli-py) | 28 | IN PROGRESS | N/A (stub) |  |
| [`monolith/native_training/metric/deep_insight_ops.py`](#monolith-native-training-metric-deep-insight-ops-py) | 134 | IN PROGRESS | N/A (TF custom ops) |  |
| [`monolith/native_training/metric/deep_insight_ops_test.py`](#monolith-native-training-metric-deep-insight-ops-test-py) | 33 | IN PROGRESS | N/A (empty test) |  |
| [`monolith/native_training/metric/exit_hook.py`](#monolith-native-training-metric-exit-hook-py) | 48 | IN PROGRESS | N/A (no Rust hook) |  |
| [`monolith/native_training/metric/kafka_utils.py`](#monolith-native-training-metric-kafka-utils-py) | 119 | IN PROGRESS | N/A (no Rust Kafka wrapper) |  |
| [`monolith/native_training/metric/metric_hook.py`](#monolith-native-training-metric-metric-hook-py) | 563 | IN PROGRESS | N/A (TF hooks + Kafka) |  |
| [`monolith/native_training/metric/metric_hook_test.py`](#monolith-native-training-metric-metric-hook-test-py) | 189 | IN PROGRESS | N/A (TF hooks) |  |
| [`monolith/native_training/metric/utils.py`](#monolith-native-training-metric-utils-py) | 104 | IN PROGRESS | N/A (TF custom ops) |  |
| [`monolith/native_training/metric/utils_test.py`](#monolith-native-training-metric-utils-test-py) | 50 | IN PROGRESS | N/A (TF custom ops) |  |
| [`monolith/native_training/mlp_utils.py`](#monolith-native-training-mlp-utils-py) | 444 | IN PROGRESS | N/A (TF distributed runtime) |  |
| [`monolith/native_training/model.py`](#monolith-native-training-model-py) | 182 | IN PROGRESS | N/A (test model) |  |
| [`monolith/native_training/model_comp_test.py`](#monolith-native-training-model-comp-test-py) | 183 | IN PROGRESS | N/A (TF/Horovod test) |  |
| [`monolith/native_training/model_dump/dump_utils.py`](#monolith-native-training-model-dump-dump-utils-py) | 757 | IN PROGRESS | N/A (TF model dump) |  |
| [`monolith/native_training/model_dump/graph_utils.py`](#monolith-native-training-model-dump-graph-utils-py) | 845 | IN PROGRESS | N/A (TF graph utils) |  |
| [`monolith/native_training/model_dump/graph_utils_test.py`](#monolith-native-training-model-dump-graph-utils-test-py) | 86 | IN PROGRESS | N/A (TF graph utils) |  |
| [`monolith/native_training/model_export/__init__.py`](#monolith-native-training-model-export-init-py) | 22 | IN PROGRESS | N/A (module alias) |  |
| [`monolith/native_training/model_export/data_gen_utils.py`](#monolith-native-training-model-export-data-gen-utils-py) | 732 | IN PROGRESS | N/A (data generator) |  |
| [`monolith/native_training/model_export/data_gen_utils_test.py`](#monolith-native-training-model-export-data-gen-utils-test-py) | 0 | IN PROGRESS | N/A (no tests) |  |
| [`monolith/native_training/model_export/demo_export.py`](#monolith-native-training-model-export-demo-export-py) | 100 | IN PROGRESS | N/A (demo exporter) |  |
| [`monolith/native_training/model_export/demo_export_test.py`](#monolith-native-training-model-export-demo-export-test-py) | 48 | IN PROGRESS | N/A (TF export test) |  |
| [`monolith/native_training/model_export/demo_predictor.py`](#monolith-native-training-model-export-demo-predictor-py) | 110 | IN PROGRESS | N/A (demo predictor) |  |
| [`monolith/native_training/model_export/demo_predictor_client.py`](#monolith-native-training-model-export-demo-predictor-client-py) | 93 | IN PROGRESS | N/A (demo gRPC client) |  |
| [`monolith/native_training/model_export/export_context.py`](#monolith-native-training-model-export-export-context-py) | 141 | IN PROGRESS | N/A (export context) |  |
| [`monolith/native_training/model_export/export_hooks.py`](#monolith-native-training-model-export-export-hooks-py) | 137 | IN PROGRESS | N/A (TF export hook) |  |
| [`monolith/native_training/model_export/export_hooks_test.py`](#monolith-native-training-model-export-export-hooks-test-py) | 141 | IN PROGRESS | N/A (export hook test) |  |
| [`monolith/native_training/model_export/export_state_utils.py`](#monolith-native-training-model-export-export-state-utils-py) | 46 | IN PROGRESS | N/A (export state) |  |
| [`monolith/native_training/model_export/export_state_utils_test.py`](#monolith-native-training-model-export-export-state-utils-test-py) | 36 | IN PROGRESS | N/A (export state test) |  |
| [`monolith/native_training/model_export/export_utils.py`](#monolith-native-training-model-export-export-utils-py) | 98 | IN PROGRESS | N/A (remote predict helper) |  |
| [`monolith/native_training/model_export/export_utils_test.py`](#monolith-native-training-model-export-export-utils-test-py) | 43 | IN PROGRESS | N/A (remote predict test) |  |
| [`monolith/native_training/model_export/saved_model_exporters.py`](#monolith-native-training-model-export-saved-model-exporters-py) | 739 | IN PROGRESS | N/A (SavedModel exporters) |  |
| [`monolith/native_training/model_export/saved_model_exporters_test.py`](#monolith-native-training-model-export-saved-model-exporters-test-py) | 153 | IN PROGRESS | N/A (exporter tests) |  |
| [`monolith/native_training/model_export/saved_model_visulizer.py`](#monolith-native-training-model-export-saved-model-visulizer-py) | 89 | IN PROGRESS | N/A (tensorboard visualizer) |  |
| [`monolith/native_training/model_export/warmup_data_decoder.py`](#monolith-native-training-model-export-warmup-data-decoder-py) | 55 | IN PROGRESS | N/A (warmup decoder) |  |
| [`monolith/native_training/model_export/warmup_data_gen.py`](#monolith-native-training-model-export-warmup-data-gen-py) | 253 | IN PROGRESS | N/A (warmup generator) |  |
| [`monolith/native_training/model_export/warmup_example_batch.py`](#monolith-native-training-model-export-warmup-example-batch-py) | 57 | IN PROGRESS | N/A (warmup example batch) |  |
| [`monolith/native_training/monolith_export.py`](#monolith-native-training-monolith-export-py) | 18 | IN PROGRESS | N/A (decorator) |  |
| [`monolith/native_training/multi_hash_table_ops.py`](#monolith-native-training-multi-hash-table-ops-py) | 695 | IN PROGRESS | N/A (TF custom ops) |  |
| [`monolith/native_training/multi_hash_table_ops_test.py`](#monolith-native-training-multi-hash-table-ops-test-py) | 249 | IN PROGRESS | N/A (TF custom ops) |  |
| [`monolith/native_training/multi_type_hash_table.py`](#monolith-native-training-multi-type-hash-table-py) | 435 | IN PROGRESS | N/A (hash table wrapper) |  |
| [`monolith/native_training/multi_type_hash_table_test.py`](#monolith-native-training-multi-type-hash-table-test-py) | 326 | IN PROGRESS | N/A (hash table tests) |  |
| [`monolith/native_training/native_model.py`](#monolith-native-training-native-model-py) | 1109 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/native_task.py`](#monolith-native-training-native-task-py) | 213 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/native_task_context.py`](#monolith-native-training-native-task-context-py) | 58 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/nested_tensors.py`](#monolith-native-training-nested-tensors-py) | 110 | IN PROGRESS | monolith-rs/crates/monolith-tensor/src |  |
| [`monolith/native_training/nested_tensors_test.py`](#monolith-native-training-nested-tensors-test-py) | 57 | IN PROGRESS | monolith-rs/crates/monolith-tensor/src |  |
| [`monolith/native_training/net_utils.py`](#monolith-native-training-net-utils-py) | 133 | IN PROGRESS | monolith-rs/crates/monolith-core/src |  |
| [`monolith/native_training/net_utils_test.py`](#monolith-native-training-net-utils-test-py) | 94 | IN PROGRESS | monolith-rs/crates/monolith-core/src |  |
| [`monolith/native_training/optimizers/adamom.py`](#monolith-native-training-optimizers-adamom-py) | 68 | IN PROGRESS | monolith-rs/crates/monolith-optimizer/src |  |
| [`monolith/native_training/optimizers/adamom_test.py`](#monolith-native-training-optimizers-adamom-test-py) | 57 | IN PROGRESS | monolith-rs/crates/monolith-optimizer/src |  |
| [`monolith/native_training/optimizers/rmsprop.py`](#monolith-native-training-optimizers-rmsprop-py) | 102 | IN PROGRESS | monolith-rs/crates/monolith-optimizer/src |  |
| [`monolith/native_training/optimizers/rmsprop_test.py`](#monolith-native-training-optimizers-rmsprop-test-py) | 77 | IN PROGRESS | monolith-rs/crates/monolith-optimizer/src |  |
| [`monolith/native_training/optimizers/rmspropv2_test.py`](#monolith-native-training-optimizers-rmspropv2-test-py) | 112 | IN PROGRESS | monolith-rs/crates/monolith-optimizer/src |  |
| [`monolith/native_training/optimizers/shampoo.py`](#monolith-native-training-optimizers-shampoo-py) | 207 | IN PROGRESS | monolith-rs/crates/monolith-optimizer/src |  |
| [`monolith/native_training/prefetch_queue.py`](#monolith-native-training-prefetch-queue-py) | 379 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/prefetch_queue_test.py`](#monolith-native-training-prefetch-queue-test-py) | 305 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/ps_benchmark.py`](#monolith-native-training-ps-benchmark-py) | 273 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/ps_benchmark_test.py`](#monolith-native-training-ps-benchmark-test-py) | 57 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/ragged_utils.py`](#monolith-native-training-ragged-utils-py) | 29 | IN PROGRESS | monolith-rs/crates/monolith-tensor/src |  |
| [`monolith/native_training/ragged_utils_test.py`](#monolith-native-training-ragged-utils-test-py) | 32 | IN PROGRESS | monolith-rs/crates/monolith-tensor/src |  |
| [`monolith/native_training/remote_predict_ops.py`](#monolith-native-training-remote-predict-ops-py) | 0 | IN PROGRESS | N/A (empty stub) |  |
| [`monolith/native_training/restore_test.py`](#monolith-native-training-restore-test-py) | 240 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/runner_utils.py`](#monolith-native-training-runner-utils-py) | 396 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/runner_utils_test.py`](#monolith-native-training-runner-utils-test-py) | 108 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/runtime/ops/gen_monolith_ops.py`](#monolith-native-training-runtime-ops-gen-monolith-ops-py) | 23 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/native_training/save_utils.py`](#monolith-native-training-save-utils-py) | 1309 | IN PROGRESS | monolith-rs/crates/monolith-checkpoint/src |  |
| [`monolith/native_training/save_utils_test.py`](#monolith-native-training-save-utils-test-py) | 1740 | IN PROGRESS | monolith-rs/crates/monolith-checkpoint/src |  |
| [`monolith/native_training/service_discovery.py`](#monolith-native-training-service-discovery-py) | 481 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/service_discovery_test.py`](#monolith-native-training-service-discovery-test-py) | 407 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/serving_ps_test.py`](#monolith-native-training-serving-ps-test-py) | 231 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/session_run_hooks.py`](#monolith-native-training-session-run-hooks-py) | 171 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/session_run_hooks_test.py`](#monolith-native-training-session-run-hooks-test-py) | 144 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/signal_utils.py`](#monolith-native-training-signal-utils-py) | 37 | IN PROGRESS | monolith-rs/crates/monolith-core/src |  |
| [`monolith/native_training/signal_utils_test.py`](#monolith-native-training-signal-utils-test-py) | 30 | IN PROGRESS | monolith-rs/crates/monolith-core/src |  |
| [`monolith/native_training/static_reshape_op.py`](#monolith-native-training-static-reshape-op-py) | 58 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/native_training/static_reshape_op_test.py`](#monolith-native-training-static-reshape-op-test-py) | 79 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/native_training/summary/summary_ops.py`](#monolith-native-training-summary-summary-ops-py) | 78 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/native_training/summary/summary_ops_test.py`](#monolith-native-training-summary-summary-ops-test-py) | 122 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/native_training/summary/utils.py`](#monolith-native-training-summary-utils-py) | 114 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/native_training/summary/utils_test.py`](#monolith-native-training-summary-utils-test-py) | 43 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/native_training/sync_hooks.py`](#monolith-native-training-sync-hooks-py) | 176 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/sync_hooks_test.py`](#monolith-native-training-sync-hooks-test-py) | 119 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/sync_training_hooks.py`](#monolith-native-training-sync-training-hooks-py) | 355 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/sync_training_hooks_test.py`](#monolith-native-training-sync-training-hooks-test-py) | 92 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/tensor_utils.py`](#monolith-native-training-tensor-utils-py) | 162 | IN PROGRESS | monolith-rs/crates/monolith-tensor/src |  |
| [`monolith/native_training/tensor_utils_test.py`](#monolith-native-training-tensor-utils-test-py) | 175 | IN PROGRESS | monolith-rs/crates/monolith-tensor/src |  |
| [`monolith/native_training/test_utils.py`](#monolith-native-training-test-utils-py) | 65 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/touched_key_set_ops.py`](#monolith-native-training-touched-key-set-ops-py) | 61 | IN PROGRESS | monolith-rs/crates/monolith-hash-table/src |  |
| [`monolith/native_training/touched_key_set_ops_test.py`](#monolith-native-training-touched-key-set-ops-test-py) | 51 | IN PROGRESS | monolith-rs/crates/monolith-hash-table/src |  |
| [`monolith/native_training/utils.py`](#monolith-native-training-utils-py) | 320 | IN PROGRESS | monolith-rs/crates/monolith-training/src/utils.rs (new) |  |
| [`monolith/native_training/utils_test.py`](#monolith-native-training-utils-test-py) | 70 | IN PROGRESS | monolith-rs/crates/monolith-training/tests/utils.rs (new) |  |
| [`monolith/native_training/variables.py`](#monolith-native-training-variables-py) | 147 | IN PROGRESS | monolith-rs/crates/monolith-training/src/variables.rs (new) |  |
| [`monolith/native_training/variables_test.py`](#monolith-native-training-variables-test-py) | 89 | IN PROGRESS | monolith-rs/crates/monolith-training/tests/variables.rs (new) |  |
| [`monolith/native_training/yarn_runtime.py`](#monolith-native-training-yarn-runtime-py) | 127 | IN PROGRESS | monolith-rs/crates/monolith-training/src/yarn_runtime.rs (new) |  |
| [`monolith/native_training/yarn_runtime_test.py`](#monolith-native-training-yarn-runtime-test-py) | 133 | IN PROGRESS | monolith-rs/crates/monolith-training/tests/yarn_runtime.rs (new) |  |
| [`monolith/native_training/zk_utils.py`](#monolith-native-training-zk-utils-py) | 96 | IN PROGRESS | monolith-rs/crates/monolith-training/src/zk_utils.rs (new) |  |
| [`monolith/path_utils.py`](#monolith-path-utils-py) | 47 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/tpu_runner.py`](#monolith-tpu-runner-py) | 429 | IN PROGRESS | monolith-rs/crates/monolith-training/src | |
| [`monolith/utils.py`](#monolith-utils-py) | 81 | IN PROGRESS | monolith-rs/crates/monolith-tf/src | |
| [`monolith/utils_test.py`](#monolith-utils-test-py) | 65 | IN PROGRESS | monolith-rs/crates/monolith-tf/tests | |
