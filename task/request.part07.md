<!--
Source: task/request.md
Lines: 15823-19762 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/layers/utils.py`
<a id="monolith-native-training-layers-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 159
- Purpose/role: Shared utilities for layer code: merge semantics, shape helpers, and Gumbel-based subset sampling.
- Key symbols/classes/functions: `MergeType`, `DCNType`, `check_dim`, `dim_size`, `merge_tensor_list`, `gumbel_keys`, `continuous_topk`, `sample_subset`.
- External dependencies: TensorFlow, NumPy.
- Side effects: None.

**Required Behavior (Detailed)**
- `MergeType`: string constants `concat`, `stack`, `None`.
- `DCNType`: string constants `vector`, `matrix`, `mixed`.
- `check_dim(dim)`:
  - `None` → `-1`, `int` → itself, `tf.compat.v1.Dimension` → `.value`, else raise.
- `dim_size(inputs, axis)`:
  - Uses static shape; if unknown (`-1`), returns dynamic `array_ops.shape(inputs)[axis]`.
- `merge_tensor_list(tensor_list, merge_type='concat', num_feature=None, axis=1, keep_list=False)`:
  - Accepts tensor or list; if single tensor, uses shape to decide:
    - 3D: `stack` returns `[tensor]` or tensor; `concat` reshapes to `[B, num_feat*emb]`; `None` unstack on axis.
    - 2D with `num_feature>1`: `stack` reshapes to `[B, num_feature, emb]`; `concat` returns as-is; `None` unstack.
    - 2D without `num_feature`: returns as-is.
    - Else: raise shape error.
  - For list length >1: `stack`, `concat`, or return list.
- `gumbel_keys(w)`: samples Gumbel noise and adds to `w`.
- `continuous_topk(w, k, t, separate=False)`:
  - Iteratively computes soft top-k masks; returns sum or list.
- `sample_subset(w, k, t=0.1)`:
  - `w = gumbel_keys(w)` then `continuous_topk`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/merge.rs` for merge utilities; DCNType maps to `monolith-layers/src/dcn.rs` (`DCNMode`).
- Rust public API surface: `MergeType`, `merge_tensor_list`, `merge_tensor_list_tensor`.
- Data model mapping:
  - `MergeType::None` corresponds to `MergeOutput::List`.
  - `check_dim`/`dim_size` are implicit in Rust shape handling; consider helper utilities.
  - Gumbel subset sampling functions not currently present in Rust.
- Feature gating: None.
- Integration points: feature_cross, feature_trans, senet, etc.

**Implementation Steps (Detailed)**
1. Verify `merge_tensor_list` semantics in Rust match Python (including single-tensor reshape/unstack cases).
2. Add Rust equivalents for `check_dim`/`dim_size` if needed for dynamic shapes.
3. Implement Gumbel subset sampling helpers if required by future layers.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: add unit tests in `monolith-rs/crates/monolith-layers/tests/merge_test.rs` if not present.
- Cross-language parity test:
  - Compare merge outputs for 2D/3D inputs with `num_feature` and `keep_list` settings.

**Gaps / Notes**
- Gumbel subset sampling utilities are missing in Rust; add if used elsewhere.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/learning_rate_functions.py`
<a id="monolith-native-training-learning-rate-functions-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 112
- Purpose/role: Defines learning rate schedule function objects (base class + polynomial decay) for optimizers and embedding slice configs.
- Key symbols/classes/functions: `LearningRateFunction`, `PolynomialDecay`.
- External dependencies: TensorFlow v1 (`tf.compat.v1.train.polynomial_decay`, `get_or_create_global_step`), `abc`.
- Side effects: None; uses global step when called.

**Required Behavior (Detailed)**
- `LearningRateFunction`:
  - Abstract `__call__` that must be overridden.
  - `__str__` prints class name and sorted `__dict__` params.
- `PolynomialDecay`:
  - Stores init params: `initial_learning_rate`, `decay_steps`, `end_learning_rate`, `power`, `cycle`, `name`.
  - `__call__` fetches `global_step = tf.compat.v1.train.get_or_create_global_step()` and returns `tf.compat.v1.train.polynomial_decay(...)`.
  - Uses TF’s polynomial decay semantics (including `cycle`).

**Rust Mapping (Detailed)**
- Target crate/module: N/A (no Rust equivalent yet).
- Rust public API surface: None.
- Data model mapping: If implemented, use a trait + struct for polynomial decay tied to training step.
- Feature gating: None.
- Integration points: Optimizer configs and embedding slice configs.

**Implementation Steps (Detailed)**
1. Decide where to place LR schedules in Rust (optimizer module or training crate).
2. Implement `PolynomialDecay` with explicit step parameter (Rust lacks TF global_step).
3. Provide string formatting for config parity if required.

**Tests (Detailed)**
- Python tests: None in repo.
- Rust tests: Add unit tests for decay values at known steps.
- Cross-language parity test: Compare decay outputs for fixed steps.

**Gaps / Notes**
- Python relies on TF global step; Rust will need explicit step input or global trainer context.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/learning_rate_functions_test.py`
<a id="monolith-native-training-learning-rate-functions-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 76
- Purpose/role: Tests PolynomialDecay schedule and its integration with an optimizer.
- Key symbols/classes/functions: `PolynomialDecayTest.test_basic`, `test_dense_optimizer`.
- External dependencies: TensorFlow v1 session/optimizers, NumPy.
- Side effects: Uses global_step and updates variables via Adagrad.

**Required Behavior (Detailed)**
- `test_basic`:
  - Creates global_step, increments twice, and checks decay outputs.
  - With `initial_learning_rate=0.01`, `decay_steps=10`, `end_learning_rate=0.11`:
    - At global_step=1: expects 0.02.
    - At global_step=2: expects 0.03.
  - Ensures `__str__` equality between two identical PolynomialDecay instances.
- `test_dense_optimizer`:
  - Uses PolynomialDecay as `learning_rate` for `AdagradOptimizer`.
  - Applies grads to two variables for 3 steps.
  - Verifies updated values match expected arrays.

**Rust Mapping (Detailed)**
- Target crate/module: N/A until learning rate schedules are implemented.
- Rust public API surface: PolynomialDecay schedule + optimizer integration.
- Data model mapping: global_step must be explicit in Rust.
- Feature gating: None.
- Integration points: Optimizer implementations (e.g., Adagrad).

**Implementation Steps (Detailed)**
1. Implement PolynomialDecay in Rust with explicit step input.
2. Add tests validating decay values for known steps (0,1,2).
3. If an optimizer exists, add integration test similar to Adagrad update.

**Tests (Detailed)**
- Python tests: `monolith/native_training/learning_rate_functions_test.py`.
- Rust tests: `monolith-rs/crates/monolith-optim/tests/learning_rate_functions_test.rs` (new) or similar.
- Cross-language parity test:
  - Compare decay values at fixed steps.

**Gaps / Notes**
- Python uses TF global_step; Rust needs explicit step or trainer context.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/logging_ops.py`
<a id="monolith-native-training-logging-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 56
- Purpose/role: Thin wrappers around custom logging/metrics TF ops (timestamps, timers, machine health).
- Key symbols/classes/functions: `tensors_timestamp`, `emit_timer`, `machine_info`, `check_machine_health`.
- External dependencies: TensorFlow, absl flags, custom ops `gen_monolith_ops`.
- Side effects: Registers a global flag `monolith_default_machine_info_mem_limit`.

**Required Behavior (Detailed)**
- `tensors_timestamp(tensors)`: returns `(tensors, timestamp)` via `monolith_tensors_timestamp`.
- `emit_timer(key, value, tags=None)`:
  - Formats tags as `"k=v|k2=v2"`, passes to `monolith_metric_v2`.
  - Returns TF op.
- `machine_info(mem_limit=None, shared_name=None)`:
  - Uses default flag if `mem_limit` is None.
  - Calls `monolith_machine_info` with `mem_limit`, `name`, `shared_name`.
- `check_machine_health(machine_info_tensor)`:
  - Returns scalar string tensor from `monolith_check_machine_health`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (custom TF ops not wired in Rust).
- Rust public API surface: None.
- Data model mapping: Would require TF runtime bindings.
- Feature gating: TF-runtime only if added.
- Integration points: metrics/logging pipeline.

**Implementation Steps (Detailed)**
1. Decide whether to expose these custom ops in Rust TF-runtime backend.
2. If yes, add FFI bindings and wrappers with identical signatures.
3. Provide a config/flag equivalent for default mem_limit.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add once bindings exist.
- Cross-language parity test: validate emitted tags and machine health output.

**Gaps / Notes**
- Requires custom TF ops; currently no Rust bindings.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/logging_ops_test.py`
<a id="monolith-native-training-logging-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 57
- Purpose/role: Tests logging ops custom TF wrappers (timestamp, timer, machine health).
- Key symbols/classes/functions: `LoggingOpsTest.test_tensors_timestamp`, `test_emit_timer`, `test_machine_health`, `test_machine_health_oom`.
- External dependencies: TensorFlow v1, absl flags, `logging_ops_pb2`.
- Side effects: Mutates global flag `monolith_default_machine_info_mem_limit`.

**Required Behavior (Detailed)**
- `test_tensors_timestamp`:
  - Calls `tensors_timestamp` twice and asserts newer timestamp >= old.
- `test_emit_timer`:
  - Calls `emit_timer("test", 0.0)` and evaluates op.
- `test_machine_health`:
  - Sets mem_limit high; `check_machine_health` returns empty bytes.
- `test_machine_health_oom`:
  - Sets mem_limit=0; `check_machine_health` returns serialized proto with status `OUT_OF_MEMORY`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (custom TF ops not implemented).
- Rust public API surface: None.
- Data model mapping: Would require TF runtime bindings and protobuf parsing.
- Feature gating: TF-runtime only.
- Integration points: logging/metrics pipeline.

**Implementation Steps (Detailed)**
1. Add Rust bindings for logging ops if TF runtime backend is enabled.
2. Add tests mirroring timestamp monotonicity and machine health outcomes.
3. Parse protobuf in Rust to validate OOM status.

**Tests (Detailed)**
- Python tests: `monolith/native_training/logging_ops_test.py`.
- Rust tests: N/A until bindings exist.
- Cross-language parity test: Validate proto outputs for machine health.

**Gaps / Notes**
- Depends on custom ops and protobufs not yet exposed in Rust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/losses/batch_softmax_loss.py`
<a id="monolith-native-training-losses-batch-softmax-loss-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 57
- Purpose/role: Computes batch softmax loss for retrieval-style training.
- Key symbols/classes/functions: `batch_softmax_loss`.
- External dependencies: TensorFlow.
- Side effects: None.

**Required Behavior (Detailed)**
- Inputs:
  - `query` shape `(batch_size, k)`, `item` shape `(batch_size, k)`.
  - `item_step_interval` shape `(batch_size,)`.
  - `r` weights (interest) same length as batch.
  - `normalize` (default True), `temperature` (default 1.0).
- Validation:
  - `temperature` must be > 0 else raise `ValueError("temperature should be positive, while got ...")`.
- Computation:
  - Optional L2-normalize query/item along axis 1.
  - `similarity = query @ item^T / temperature`.
  - Clamp `item_step_interval` to at least 1.0, compute `item_frequency = 1 / item_step_interval`.
  - Adjust similarity: `exp(similarity - log(item_frequency))`.
  - Loss: `-sum(r * log(diag(similarity) / reduce_sum(similarity, axis=1)))`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (no Rust loss implementation yet).
- Rust public API surface: loss function in training/optimizer crate.
- Data model mapping: Tensor ops for matmul, diag, log, exp.
- Feature gating: None.
- Integration points: training loss computation.

**Implementation Steps (Detailed)**
1. Implement batch_softmax_loss in Rust with the same math and shape checks.
2. Ensure numerical stability around log/exp and item_frequency.
3. Add input normalization option.

**Tests (Detailed)**
- Python tests: `monolith/native_training/losses/batch_softmax_loss_test.py`.
- Rust tests: new test in `monolith-rs/crates/monolith-training/tests`.
- Cross-language parity test: compare loss for fixed inputs.

**Gaps / Notes**
- Requires loss module placement decision in Rust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/losses/batch_softmax_loss_test.py`
<a id="monolith-native-training-losses-batch-softmax-loss-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 35
- Purpose/role: Single test for batch_softmax_loss numeric output.
- Key symbols/classes/functions: `BatchSoftmaxLossTest.test_batch_softmax_loss`.
- External dependencies: TensorFlow, NumPy.
- Side effects: None.

**Required Behavior (Detailed)**
- Creates random `query` and `item` tensors `(batch=4, dim=3)`.
- `item_step_interval` is random integers in `[1,10)`, `r` is ones.
- Calls `batch_softmax_loss` and asserts loss equals `6.5931373`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A until loss is implemented.
- Rust public API surface: batch_softmax_loss.
- Data model mapping: Tensor operations and RNG.
- Feature gating: None.
- Integration points: training loss module.

**Implementation Steps (Detailed)**
1. Implement loss and a deterministic test by seeding RNG or using fixed inputs.
2. Match Python numeric output if using the same fixed inputs.

**Tests (Detailed)**
- Python tests: `monolith/native_training/losses/batch_softmax_loss_test.py`.
- Rust tests: add deterministic equivalent.
- Cross-language parity test: compare loss for fixed inputs.

**Gaps / Notes**
- Python test uses random inputs without setting a seed but asserts a fixed value; likely flaky. Prefer fixing inputs in Rust and note the discrepancy.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/losses/inbatch_auc_loss.py`
<a id="monolith-native-training-losses-inbatch-auc-loss-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 41
- Purpose/role: Wrapper for custom TF op `InbatchAucLoss` and its gradient registration.
- Key symbols/classes/functions: `inbatch_auc_loss`, `_inbatch_auc_loss_grad`.
- External dependencies: `gen_monolith_ops` custom op.
- Side effects: Registers gradient for `InbatchAucLoss`.

**Required Behavior (Detailed)**
- `inbatch_auc_loss(label, logit, neg_weight=1.0)`:
  - Calls `inbatch_auc_loss_ops.inbatch_auc_loss`.
- Gradient:
  - `InbatchAucLoss` gradient returns `None` for label and computed gradient for logit via `inbatch_auc_loss_grad`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (custom TF op not bound in Rust).
- Rust public API surface: None.
- Data model mapping: Would need TF runtime binding.
- Feature gating: TF-runtime only if implemented.
- Integration points: loss computation in training.

**Implementation Steps (Detailed)**
1. Add Rust binding for `InbatchAucLoss` op if TF runtime backend is used.
2. Expose gradient or compute manually if training is supported.

**Tests (Detailed)**
- Python tests: `monolith/native_training/losses/inbatch_auc_loss_test.py`.
- Rust tests: N/A until binding exists.
- Cross-language parity test: compare loss/grad values for fixed inputs.

**Gaps / Notes**
- Depends on custom TF ops; currently missing in Rust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/losses/inbatch_auc_loss_test.py`
<a id="monolith-native-training-losses-inbatch-auc-loss-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 71
- Purpose/role: Unit tests for inbatch_auc_loss and its gradient op.
- Key symbols/classes/functions: `InbatchAucLossTest.test_inbatch_auc_loss`, `test_inbatch_auc_loss_grad`.
- External dependencies: TensorFlow, Python math.
- Side effects: None.

**Required Behavior (Detailed)**
- `test_inbatch_auc_loss`:
  - Uses labels `[1,0,0,1]` and logits `[0.5,-0.2,-0.4,0.8]`.
  - Computes expected loss by summing `log(sigmoid(diff))` over all pos-neg pairs.
  - Asserts almost equal to TF op output.
- `test_inbatch_auc_loss_grad`:
  - Calls custom op grad with `grad=2`.
  - Computes expected gradient by pairwise contributions.
  - Asserts close.

**Rust Mapping (Detailed)**
- Target crate/module: N/A until custom op binding exists.
- Rust public API surface: inbatch_auc_loss and grad.
- Data model mapping: pairwise log-sigmoid over pos-neg pairs.
- Feature gating: TF runtime only.
- Integration points: training loss.

**Implementation Steps (Detailed)**
1. Implement loss (or bind op) and deterministic tests with the same inputs.
2. Validate gradient against manual computation.

**Tests (Detailed)**
- Python tests: `monolith/native_training/losses/inbatch_auc_loss_test.py`.
- Rust tests: add once implementation exists.
- Cross-language parity test: compare loss/grad for fixed inputs.

**Gaps / Notes**
- Depends on custom TF op; no Rust binding yet.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/losses/ltr_losses.py`
<a id="monolith-native-training-losses-ltr-losses-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1233
- Purpose/role: Implements learning-to-rank loss functions (pairwise, listwise, approx-NDCG) and LambdaWeight models (DCG, Precision, ListMLE) based on TF-Ranking style utilities.
- Key symbols/classes/functions: `_EPSILON`, `label_valid_fn`, `sort_by_scores`, `organize_valid_indices`, `shuffle_valid_indices`, `reshape_first_ndims`, `approx_ranks`, `inverse_max_dcg`, `get_batch_idx_size`, `RankingLossKey`, `make_loss_fn`, `create_ndcg_lambda_weight`, `create_reciprocal_rank_lambda_weight`, `create_p_list_mle_lambda_weight`, `_LambdaWeight`, `IdentityLambdaWeight`, `DCGLambdaWeight`, `PrecisionLambdaWeight`, `ListMLELambdaWeight`, `_sort_and_normalize`, `_pairwise_comparison`, `_pairwise_loss`, `_pairwise_hinge_loss`, `_pairwise_logistic_loss`, `_pairwise_soft_zero_one_loss`, `_softmax_loss`, `_sigmoid_cross_entropy_loss`, `_mean_squared_loss`, `_list_mle_loss`, `_approx_ndcg_loss`.
- External dependencies: TensorFlow (core ops, math_ops, nn_ops, array_ops, random_ops, sparse_ops), `tf.losses` reduction and loss helpers, `abc`.
- Side effects: None (no I/O). Uses op-level RNG for shuffling (`random_uniform`) and TF name scopes.

**Required Behavior (Detailed)**
- Constants:
  - `_EPSILON = 1e-10` used to set invalid logits to a log probability for softmax/ListMLE.
- `label_valid_fn(labels)`:
  - Returns boolean tensor for label validity: `labels >= 0`.
  - Accepts any convertible tensor; shape preserved.
- `sort_by_scores(scores, features_list, topn=None)`:
  - `scores` must be rank-2 `[batch_size, list_size]`; asserts rank 2.
  - `features_list` must be list of tensors with same shape as `scores`.
  - `topn` defaults to `list_size`; clamped to `<= list_size`.
  - Uses `tf.nn.top_k(scores, topn, sorted=True)` and gathers each feature after flattening; output shapes `[batch_size, topn]`.
  - Preserves per-batch ordering by using list offsets (`batch_id * list_size`).
- `organize_valid_indices(is_valid, shuffle=True, seed=None)`:
  - `is_valid` must be rank-2 boolean `[batch, list]`; asserts rank 2.
  - If `shuffle=True`, uses `random_uniform` with op seed; if False, uses reversed range to preserve original order among valids.
  - Invalid entries get sentinel score `-1e-6` so they appear last.
  - Returns gather/scatter indices tensor `[batch, list, 2]` containing `(batch_id, index)` pairs.
- `shuffle_valid_indices(is_valid, seed=None)`:
  - Wrapper around `organize_valid_indices(..., shuffle=True)`.
- `reshape_first_ndims(tensor, first_ndims, new_shape)`:
  - Asserts tensor has at least `first_ndims` (if static rank known).
  - `new_shape` replaces first `first_ndims` dims; remaining dims preserved.
  - Uses `sparse_reshape` for `SparseTensor`, else `reshape`.
- `approx_ranks(logits, alpha=10.)`:
  - `logits` shape `[batch, list]`.
  - Computes approximate ranks via generalized sigmoid of pairwise differences:
    - `pairs = sigmoid(alpha * (s_j - s_i))`
    - `rank_i = sum_j pairs + 0.5`.
  - Produces `[batch, list]` tensor; O(list_size^2) memory.
- `inverse_max_dcg(labels, gain_fn=2^label-1, rank_discount_fn=1/log1p(rank), topn=None)`:
  - Sorts labels by label values (descending) using `sort_by_scores`.
  - Computes `discounted_gain = sum(gain * discount)` over ranks.
  - Returns `1/discounted_gain` when `discounted_gain > 0`, else 0; shape `[batch,1]`.
- `get_batch_idx_size(logits, labels, rank_id, name_prefix)`:
  - Expects `logits`/`labels` with leading dimension `batch_size`; `rank_id` length `batch_size`.
  - Groups examples by `rank_id` using `tf.unique_with_counts`.
  - Computes `(row, col)` indices for each example into a padded `[num_unique, max_count]` grid.
  - `logits_idx = scatter_nd(logits, indices=batch_idx, shape=[num_unique, max_count])`.
  - `label_idx = scatter_nd(labels, indices=batch_idx, shape=[num_unique, max_count]) - 1e-6`.
  - `mask_idx = scatter_nd(ones_like(logits), indices=batch_idx, shape=[batch_size, batch_size])`.
  - Returns `(logits_idx, label_idx, mask_idx, unique_idx)` where `unique_idx = argmax(list_id_mask, axis=1)`.
  - Keep exact scatter shapes and `-1e-6` offset; even if unused elsewhere.
- `RankingLossKey`: string constants for all supported loss names.
- `make_loss_fn(...)`:
  - Validates `reduction` is in `tf.losses.Reduction.all()` and not `NONE`, else raises `ValueError("Invalid reduction: ...")`.
  - Raises `ValueError` if `loss_keys` empty or `loss_weights` length mismatch.
  - Accepts `loss_keys` as string or list; normalizes to list.
  - `_loss_fn(labels, logits, features)`:
    - Optional `weights` from `features[weights_feature_name]`.
    - Builds kwargs and dispatches to loss functions; `extra_args` merged in for all losses.
    - Uses `lambda_weight` for pairwise/softmax/listmle losses; `seed` passed to ListMLE.
    - Unknown `loss_key` raises `ValueError("Invalid loss_key: ...")`.
    - Returns weighted sum (`math_ops.add_n`), optionally applying `loss_weights`.
- `create_ndcg_lambda_weight(topn=None, smooth_fraction=0.)`:
  - Returns `DCGLambdaWeight` with `gain_fn = 2^label - 1`, `rank_discount_fn = 1/log1p(rank)`, `normalized=True`.
- `create_reciprocal_rank_lambda_weight(topn=None, smooth_fraction=0.)`:
  - Returns `DCGLambdaWeight` with `gain_fn = labels`, `rank_discount_fn = 1/rank`, `normalized=True`.
- `create_p_list_mle_lambda_weight(list_size)`:
  - Returns `ListMLELambdaWeight` with `rank_discount_fn = 2^(list_size - rank) - 1`.
- `_LambdaWeight` (abstract):
  - `_get_valid_pairs_and_clean_labels(sorted_labels)`:
    - Ensures rank-2; returns `valid_pairs` mask and labels with invalids zeroed.
  - `pair_weights(sorted_labels)`: abstract; must be implemented.
  - `individual_weights(sorted_labels)`: default returns `sorted_labels` (no transform).
- `IdentityLambdaWeight`:
  - `pair_weights` returns scalar `1.0`.
- `DCGLambdaWeight`:
  - `__init__` stores `topn`, `gain_fn`, `rank_discount_fn`, `normalized`, `smooth_fraction`; asserts `smooth_fraction` in `[0,1]`.
  - `pair_weights(sorted_labels)`:
    - Computes `gain = gain_fn(labels)` and optionally normalizes by `inverse_max_dcg`.
    - `pair_gain = gain_i - gain_j` masked to valid pairs.
    - Computes discount `u` (relative rank diff) and `v` (absolute rank) per LambdaLoss/LambdaMART.
    - `pair_weight = |pair_gain| * ((1-smooth_fraction)*u + smooth_fraction*v)`.
    - If `topn` set, masks pairs where either i or j is within topn (OR mask).
  - `individual_weights(sorted_labels)`:
    - Cleans invalid labels to 0.
    - Computes `gain_fn(labels)` and optional normalization.
    - Returns `gain * rank_discount_fn(rank)`.
- `PrecisionLambdaWeight`:
  - `__init__(topn, positive_fn=label>=1.0)`.
  - `pair_weights`:
    - Computes binary labels via `positive_fn`.
    - `label_diff = |b_i - b_j|` for valid pairs.
    - Masks pairs where exactly one of i/j is in topn (xor).
    - Returns `label_diff * rank_mask`.
- `ListMLELambdaWeight`:
  - `pair_weights` returns `sorted_labels` (pass-through).
  - `individual_weights` returns `rank_discount_fn(rank)` broadcast to `[batch, list]`.
- `_sort_and_normalize(labels, logits, weights=None)`:
  - `logits` rank-2 and shape-compatible with `labels`.
  - `weights` can be scalar, `[batch,1]`, or `[batch, list]`; broadcast to labels.
  - Invalid labels (`<0`) get score `min(logits) - 1e-6` to force sorting to end.
  - Returns sorted `(labels, logits, weights)` using `sort_by_scores`.
- `_pairwise_comparison(sorted_labels, sorted_logits, sorted_weights, lambda_weight=None)`:
  - Computes pairwise label diffs, logits diffs, and `pairwise_labels = 1{l_i > l_j}`.
  - Invalid labels mask pairs; weights apply to `i` dimension (`w_i` only).
  - If `lambda_weight` provided, multiply by `lambda_weight.pair_weights`; else multiply by `|l_i - l_j|`.
  - Uses `stop_gradient` on pairwise weights.
- `_pairwise_loss(loss_fn, labels, logits, weights=None, lambda_weight=None, lambda_scale=True, reduction=SUM_BY_NONZERO_WEIGHTS)`:
  - Sorts and builds pairwise comparisons.
  - If `lambda_weight` and `lambda_scale`, scales weights by `list_size` to counteract shrinkage.
  - Uses `core_losses.compute_weighted_loss(loss_fn(pairwise_logits), weights=pairwise_weights, reduction=...)`.
- `_pairwise_hinge_loss(...)`:
  - Loss: `relu(1 - (s_i - s_j))` for pairs with `l_i > l_j`.
  - Default `lambda_scale=True`.
- `_pairwise_logistic_loss(...)`:
  - Loss: `log(1 + exp(-logits))` via stable formulation `relu(-x) + log1p(exp(-abs(x)))`.
- `_pairwise_soft_zero_one_loss(...)`:
  - Loss: `1 - sigmoid(logits)` if logits>0 else `sigmoid(-logits)` (smooth 0-1 loss).
- `_softmax_loss(labels, logits, weights=None, lambda_weight=None, reduction=SUM_BY_NONZERO_WEIGHTS)`:
  - Sorts labels/logits/weights, masks invalid labels to 0 and invalid logits to `log(_EPSILON)`.
  - If `lambda_weight` is `DCGLambdaWeight`, replaces labels with `lambda_weight.individual_weights`.
  - Multiplies labels by weights, computes `label_sum` per list.
  - Filters out lists with `label_sum == 0` using `boolean_mask`.
  - Computes `softmax_cross_entropy(labels/label_sum, logits, weights=label_sum, reduction=...)`.
- `_sigmoid_cross_entropy_loss(...)`:
  - Flattens and filters to valid labels (`>=0`) only.
  - Broadcasts weights to labels; passes vectors to `core_losses.sigmoid_cross_entropy`.
- `_mean_squared_loss(...)`:
  - Same filtering as sigmoid; uses `core_losses.mean_squared_error`.
- `_list_mle_loss(labels, logits, weights=None, lambda_weight=None, reduction=SUM_BY_NONZERO_WEIGHTS, seed=None)`:
  - Invalid labels -> 0; invalid logits -> `log(_EPSILON)`.
  - `weights` broadcast to labels, then `squeeze`d.
  - Shuffles valid entries per list using `shuffle_valid_indices(is_label_valid, seed)`; gather labels/logits.
  - Sorts by shuffled labels (descending) to form ground-truth permutation.
  - Applies max-shift for stability; `sums = log(cumsum(exp(sorted_logits), reverse=True)) - sorted_logits`.
  - If `lambda_weight` is `ListMLELambdaWeight`, multiplies `sums` by per-rank discounts.
  - Loss = `sum(sums, axis=1)`; reduced via `compute_weighted_loss`.
- `_approx_ndcg_loss(labels, logits, weights=None, reduction=SUM, alpha=10.)`:
  - Invalid labels -> 0; invalid logits -> `min(logits) - 1e3`.
  - `label_sum` per list; lists with `label_sum == 0` removed.
  - If weights is None, code uses `ones_like(label_sum)` (note: docstring says label sum; code uses 1.0).
  - Computes gains `2^label - 1`, ranks `approx_ranks(logits, alpha)`, discounts `1/log1p(ranks)`.
  - DCG = sum(gains * discounts); cost = `-dcg * inverse_max_dcg(labels)`.
  - Uses `compute_weighted_loss(cost, weights, reduction=SUM by default)`.
- Threading/concurrency: none; pure tensor ops.
- Determinism: only nondeterminism is in `shuffle_valid_indices` (seeded via op/global seed).
- Performance: pairwise and approx-ranks are O(batch * list_size^2) memory/compute; avoid large list_size or use batching.
- Logging/metrics: none.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (no ranking loss module yet). Likely add `monolith-rs/crates/monolith-training/src/losses/ltr.rs` or a new `monolith-losses` crate.
- Rust public API surface: loss functions and LambdaWeight structs mirroring Python names; a `make_loss_fn` factory for composing losses.
- Data model mapping:
  - Tensor ops required: sort-by-score, top-k, gather, boolean mask, pairwise diffs, softmax, sigmoid.
  - Shape handling must preserve `[batch, list]` contracts and broadcasted weights.
  - RNG seed input for ListMLE shuffling (explicit seed param; no global TF seed).
- Feature gating: none for Candle path; optional TF-runtime backend for parity with TF ops.
- Integration points: training loss computation in `monolith-training`, possibly `monolith-layers` heads or `monolith-native_training` equivalents.

**Implementation Steps (Detailed)**
1. Choose Rust location for ranking losses (training crate vs new `losses` crate) and define a public API mirroring Python names.
2. Implement helpers: `label_valid_fn`, `sort_by_scores`, `organize_valid_indices`, `reshape_first_ndims`, `approx_ranks`, `inverse_max_dcg`.
3. Implement LambdaWeight hierarchy (`DCGLambdaWeight`, `PrecisionLambdaWeight`, `ListMLELambdaWeight`) with the same default gains/discounts and smooth_fraction behavior.
4. Implement core loss kernels: `_pairwise_comparison`, `_pairwise_loss`, `_softmax_loss`, `_sigmoid_cross_entropy_loss`, `_mean_squared_loss`, `_list_mle_loss`, `_approx_ndcg_loss`.
5. Preserve exact masking semantics for invalid labels (<0) and the invalid-logit sentinel behavior (`log(_EPSILON)` and `min(logits)-1e3`).
6. Add `make_loss_fn` factory that accepts loss keys/weights, feature-based weights, `extra_args`, and `lambda_weight`/`seed`.
7. Decide how to represent reductions (`SUM_BY_NONZERO_WEIGHTS`, `SUM`) and match Python reductions for each loss.
8. Add tests for each loss with fixed small tensors; validate against Python outputs.
9. Document and resolve discrepancies (e.g., approx_ndcg docstring vs code for weights).

**Tests (Detailed)**
- Python tests: none in repo for `ltr_losses.py`.
- Rust tests: add new unit tests per loss in `monolith-rs/crates/monolith-training/tests/ltr_losses_test.rs`.
- Cross-language parity test:
  - Use fixed small tensors (e.g., batch=2, list=3) and compare loss values against Python TF output.
  - Include seeded ListMLE shuffling to keep deterministic comparisons.

**Gaps / Notes**
- No Rust implementation of TF-Ranking losses yet; all of these are missing.
- `get_batch_idx_size` is unused in Python file but must still be ported for parity.
- `_approx_ndcg_loss` docstring says weights default to label sum; code uses `ones_like(label_sum)`. Preserve code behavior.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/metric/cli.py`
<a id="monolith-native-training-metric-cli-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 28
- Purpose/role: Stub/no-op CLI client placeholder; provides a `Client` with no-op methods to satisfy callers.
- Key symbols/classes/functions: `Client`, `get_cli`.
- External dependencies: `absl.logging`, `threading` (imported but unused).
- Side effects: None.

**Required Behavior (Detailed)**
- `Client.__init__(*args, **kwargs)`:
  - No-op constructor; ignores all args/kwargs.
- `Client.__getattr__(name)`:
  - Returns a function `method(*args, **kwargs)` that does nothing and returns `None`.
  - Allows arbitrary attribute access without raising `AttributeError`.
- `get_cli(*args, **kwargs)`:
  - Returns a new `Client()`; ignores args/kwargs.
- No logging, no threads, no I/O.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (stub).
- Rust public API surface: optional `NoopClient` with methods that accept any inputs and do nothing.
- Data model mapping: none.
- Feature gating: none.
- Integration points: callers expecting a CLI client can receive a stub.

**Implementation Steps (Detailed)**
1. If Rust needs a CLI client, add a no-op struct with methods used by callers.
2. Ensure missing method calls do not panic (mirror Python `__getattr__` permissiveness).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: optional smoke test that unknown method calls are no-ops if implemented.
- Cross-language parity test: not needed (stub behavior only).

**Gaps / Notes**
- This is a pure stub; threading/logging imports are unused.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/metric/deep_insight_ops.py`
<a id="monolith-native-training-metric-deep-insight-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 134
- Purpose/role: Thin wrappers around custom Monolith Deep Insight TF ops (create client, write metrics).
- Key symbols/classes/functions: `deep_insight_client`, `write_deep_insight`, `write_deep_insight_v2`, `deep_insight_ops`.
- External dependencies: TensorFlow, `gen_monolith_ops` (custom ops), `socket.gethostname()`.
- Side effects: Custom ops create a Deep Insight client resource and emit metrics to a databus; may dump to file if `dump_filename` is provided by the op.

**Required Behavior (Detailed)**
- Constants: `_FEATURE_REQ_TIME`, `_SAMPLE_RATE`, `_UID` are defined but unused in this file.
- `deep_insight_client(enable_metrics_counter=False, is_fake=False, dump_filename=None, container=socket.gethostname())`:
  - Default `container` is evaluated at import time (module load), not at call time.
  - Calls `deep_insight_ops.monolith_create_deep_insight_client(enable_metrics_counter, is_fake, dump_filename, container)`.
  - Returns a `tf.Tensor` handle to the client resource.
- `write_deep_insight(...)`:
  - Args:
    - `deep_insight_client_tensor`: handle from `deep_insight_client`.
    - `uids`: 1-D int64 tensor.
    - `req_times`: 1-D int64 tensor.
    - `labels`, `preds`, `sample_rates`: 1-D float tensors.
    - `model_name`, `target`: strings.
    - `sample_ratio` float (default 0.01).
    - `return_msgs` bool: whether op returns serialized messages.
    - `use_zero_train_time` bool: if True uses 0 as train time (tests).
  - Calls `monolith_write_deep_insight` op with named args and returns 1-D string tensor.
- `write_deep_insight_v2(...)`:
  - Args:
    - `req_times`: 1-D int64 tensor (batch_size).
    - `labels`, `preds`, `sample_rates`: 2-D float tensors of shape `(num_targets, batch_size)`.
    - `extra_fields_values`: list of 1-D tensors (each batch_size).
    - `extra_fields_keys`: list of strings, same length as `extra_fields_values`.
    - `targets`: list of strings (num_targets).
  - Calls `monolith_write_deep_insight_v2` op with named args and returns 1-D string tensor.
- No Python-side validation of shapes/dtypes; relies on op validation.
- Threading/concurrency: op-level; Python wrapper is pure.
- Determinism: depends on external Deep Insight system; no RNG here.
- Logging/metrics: metrics emission happens inside the custom op.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (custom TF ops not bound in Rust).
- Rust public API surface: optional wrappers when TF-runtime backend is present.
- Data model mapping: Tensor handles and string vectors must map to TF runtime types.
- Feature gating: TF-runtime + custom ops only.
- Integration points: training metrics pipeline, databus output.

**Implementation Steps (Detailed)**
1. Expose `monolith_create_deep_insight_client` and write ops in Rust TF-runtime backend (FFI).
2. Mirror function signatures and defaults (especially container default semantics).
3. Add validation if desired, but preserve op behavior for parity.
4. Add tests using fake client mode (`is_fake=True`) to avoid external dependencies.

**Tests (Detailed)**
- Python tests: `monolith/native_training/metric/deep_insight_ops_test.py`.
- Rust tests: N/A until TF custom ops are bound.
- Cross-language parity test: compare returned messages (if `return_msgs=True`) and shape/dtype.

**Gaps / Notes**
- Requires custom TF ops; no Rust bindings exist today.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/metric/deep_insight_ops_test.py`
<a id="monolith-native-training-metric-deep-insight-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 33
- Purpose/role: Placeholder test module for Deep Insight ops; currently no assertions.
- Key symbols/classes/functions: `DeepInsightOpsTest.dummy_test`.
- External dependencies: TensorFlow test harness, `absl.logging`, `json`, `time` (unused).
- Side effects: None.

**Required Behavior (Detailed)**
- `DeepInsightOpsTest.dummy_test`: no-op; test does nothing.
- `__main__` block disables eager execution and runs `tf.test.main()`.
- No validation of deep insight ops; effectively a stub.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (empty test).
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. If Deep Insight ops are implemented in Rust, add real tests; otherwise keep as stub-equivalent.

**Tests (Detailed)**
- Python tests: `monolith/native_training/metric/deep_insight_ops_test.py` (no assertions).
- Rust tests: none.
- Cross-language parity test: not applicable until ops exist.

**Gaps / Notes**
- Tests are effectively empty; add real assertions when ops are implemented.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/metric/exit_hook.py`
<a id="monolith-native-training-metric-exit-hook-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 48
- Purpose/role: Installs signal handlers and an `atexit` hook that emits an "exit_hook" counter when the process exits due to a signal.
- Key symbols/classes/functions: `sig_no`, `sig_handler`, `exit_hook`.
- External dependencies: `atexit`, `signal`, `sys`, `monolith.native_training.utils`, `native_task_context`, `metric.cli`.
- Side effects: Registers signal handlers on import; registers an `atexit` handler; may call `sys.exit` on signal.

**Required Behavior (Detailed)**
- Module global `sig_no` initialized to `None`.
- `sig_handler(signo, frame)`:
  - Sets global `sig_no = signo`.
  - Calls `sys.exit(signo)` to terminate process.
- On import, installs handlers:
  - `signal.signal(signal.SIGHUP, sig_handler)`
  - `signal.signal(signal.SIGINT, sig_handler)`
  - `signal.signal(signal.SIGTERM, sig_handler)`
- `exit_hook()` (decorated with `@atexit.register`):
  - Fetches context via `native_task_context.get()`.
  - Builds metric client: `cli.get_cli(utils.get_metric_prefix())`.
  - `index = ctx.worker_index` if `ctx.server_type == 'worker'`, else `ctx.ps_index`.
  - Builds tags: `server_type`, `index` (string), `sig` (stringified `sig_no`).
  - Only emits counter if `sig_no is not None`: `mcli.emit_counter("exit_hook", 1, tags)`.
- No explicit error handling; depends on `native_task_context` and `cli` behavior.
- Determinism: signal arrival timing; otherwise deterministic.
- Logging/metrics: emits a counter metric when terminating due to signal.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (no Rust signal/exit hook yet).
- Rust public API surface: optional `exit_hook` module that registers signal handlers + exit hook.
- Data model mapping: tag map `server_type/index/sig` -> metrics client.
- Feature gating: none; only needed if Rust training/runtime needs parity.
- Integration points: metrics client (Rust equivalent of `cli.get_cli`), task context.

**Implementation Steps (Detailed)**
1. Implement signal handling in Rust (e.g., `signal-hook`) for HUP/INT/TERM.
2. Record the signal number in a global and trigger process exit.
3. Register an exit hook that emits `exit_hook` counter with identical tags.
4. Mirror `server_type`/index selection from `native_task_context`.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: optional integration test using signal simulation.
- Cross-language parity test: validate emitted tags and counter name.

**Gaps / Notes**
- Python `cli.get_cli` is a stub; metric emission may be a no-op unless replaced.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/metric/kafka_utils.py`
<a id="monolith-native-training-metric-kafka-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 119
- Purpose/role: Simple Kafka producer wrapper with a background thread + queue; tracks send counters.
- Key symbols/classes/functions: `KProducer`, `KProducer.send`, `KProducer.close`.
- External dependencies: `kafka.KafkaProducer`, `queue.Queue`, `threading.Thread/RLock`, `absl.logging`, `time`.
- Side effects: Starts a background thread on init; sends messages to Kafka.

**Required Behavior (Detailed)**
- `KProducer.__init__(brokers, topic)`:
  - Stores `brokers` and `topic`.
  - Creates `KafkaProducer(bootstrap_servers=brokers)`.
  - Initializes `_lock` (RLock), `_has_stopped=False`, `_msg_queue=Queue()`.
  - Initializes counters `_total`, `_success`, `_failed` to 0.
  - Spawns background thread targeting `_poll` and starts it.
- `send(msgs)`:
  - If `msgs` is `None` or empty, returns immediately.
  - If `msgs` is `str` or `bytes`, wraps into a list.
  - Else filters iterable to only non-`None` entries with `len(msg) > 0`.
  - If resulting list is non-empty:
    - Logs first message up to 10 times via `logging.log_first_n(INFO, msgs[0], n=10)`.
    - Increments `_total` by `len(msgs)`.
    - Enqueues the list into `_msg_queue`.
  - No encoding/conversion; message passed to KafkaProducer as-is.
- `_poll()` (background thread):
  - Loop: `msg_batch = _msg_queue.get(timeout=1)`.
  - On any exception (e.g., timeout), checks `_has_stopped` under lock:
    - If stopped: break; else continue.
  - If `msg_batch` non-empty: sends each message via `producer.send(topic, msg)` and attaches callbacks:
    - `_send_success` for success, `_send_failed` for error.
  - After processing a batch, exits if `_has_stopped` is True.
- `total()`, `success()`, `failed()`:
  - Return counters; not synchronized across threads (may race).
- `_flush()`:
  - Asserts `_has_stopped` is True.
  - Drains `_msg_queue` (timeout=1) until empty or exception.
  - Sends queued messages with callbacks (same as `_poll`).
- `close()`:
  - Sets `_has_stopped=True` under lock.
  - Joins background thread.
  - Calls `_flush()` and then `producer.close(timeout=1)`.
  - Logs warnings on any exception.
- `_send_success(...)`: increments `_success`.
- `_send_failed(...)`: sleeps 2 seconds, logs warning, increments `_failed`.
- Threading/concurrency: background thread + queue; counters are not locked.
- Determinism: none; dependent on Kafka/network.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (no Rust Kafka wrapper).
- Rust public API surface: optional Kafka producer wrapper with background worker.
- Data model mapping: messages as `Vec<u8>` or `String`; track counters.
- Feature gating: requires a Rust Kafka client (e.g., `rdkafka`).
- Integration points: metrics emission pipeline.

**Implementation Steps (Detailed)**
1. Choose Kafka client crate and implement a background worker with channel/queue.
2. Mirror `send` behavior (filtering, logging first message, counters).
3. Implement graceful shutdown (stop flag, join thread, flush queue).
4. Provide success/failure counters and expose them.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add unit tests with a mocked producer or test broker.
- Cross-language parity test: compare counters and send filtering behavior.

**Gaps / Notes**
- Python implementation is not thread-safe for counters; preserve semantics unless explicitly improved.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/metric/metric_hook.py`
<a id="monolith-native-training-metric-metric-hook-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 563
- Purpose/role: Collection of TensorFlow Estimator hooks for metrics, profiling, Kafka/file logging, and telemetry.
- Key symbols/classes/functions: `ThroughputMetricHook`, `StepLossMetricHook`, `CustomMetricHook`, `Tf2ProfilerHook`, `ByteCCLTelemetryHook`, `NVProfilerHook`, `KafkaMetricHook`, `FileMetricHook`, `WriteOnlyFileAndStat`, helper functions (`default_parse_fn`, `default_layout_fn`, `vepfs_layout_fn`, `vepfs_key_fn`).
- External dependencies: TensorFlow Estimator hooks, TF profiler, BytePS telemetry, Kafka (via `KProducer`), `tf.io.gfile`, `absl.flags/logging`, `alert_manager`, `alert_pb2`.
- Side effects: Registers exit hook via import (`exit_hook`), starts background threads, writes to Kafka/files, may start TF profiler server on port 6666.

**Required Behavior (Detailed)**
- Module globals:
  - `FLAGS = flags.FLAGS` used by `Tf2ProfilerHook`.
  - Importing `exit_hook` executes signal/atexit registration for metrics.
- `ThroughputMetricHook`:
  - `__init__(model_name, start_time_secs, cluster_type="stable", run_every_n_secs=30)`:
    - Initializes counters and `self._mcli = cli.get_cli(utils.get_metric_prefix())`.
    - If alert manager exists, creates `AlertProto` and registers rules with prefix.
  - `begin()`: sets `self._global_step_tensor = tf.compat.v1.train.get_global_step()`.
  - `before_run(run_context)`:
    - On first step, reads `global_step` via `session.run`.
    - Records `emit_time` (int seconds).
    - If `start_time_secs` provided, emits timer `run_start_elapsed_time.all` with tags `{model_name, cluster_type}`.
    - Returns `SessionRunArgs({"global_step": global_step_tensor})`.
  - `after_run(run_context, run_values)`:
    - If elapsed wall time >= `run_every_n_secs`, emits:
      - `run_steps.all` counter (step interval).
      - `run_steps_elapsed_time.all` timer (elapsed_time / step_interval).
    - Updates emit step/time. (No guard against `step_interval == 0`.)
- `StepLossMetricHook`:
  - `__init__(loss_tensor)` stores tensor and mcli.
  - `before_run`: requests loss tensor.
  - `after_run`: emits `step_loss` store with loss value.
- `CustomMetricHook`:
  - `__init__(metric_tensors)`:
    - Validates each tensor is scalar (rank 0) and dtype in `{tf.float32, tf.int32}`.
    - Raises `ValueError` if invalid or if metric list empty.
  - `before_run`: requests all metric tensors.
  - `after_run`: emits each metric as float via `emit_store`.
- `Tf2ProfilerHook`:
  - `__init__(logdir, init_step_range, save_steps=None, save_secs=None, options=None)`:
    - Validates `end_step > start_step` when provided.
    - Sets `delta = end_step - start_step` or default 10.
    - If `save_steps` provided and `<= delta`, raises `ValueError`.
    - Creates `SecondOrStepTimer(every_steps=save_steps, every_secs=save_secs)`.
  - `begin()`:
    - If `FLAGS.enable_sync_training` tries `tf.profiler.experimental.server.start(6666)`; logs warning on failure.
  - `before_run`:
    - If profiling, creates `_pywrap_traceme.TraceMe("TraceContext", graph_type="train", step_num=current_step)` for step-time graph fix.
    - Returns `SessionRunArgs(fetches=None)`.
  - `after_run`:
    - Increments `current_step`.
    - Stops TraceMe if active.
    - If `start_step` is None, defers profiling to `current_step + 500` with default delta.
    - Stops profiling when `current_step >= end_step`.
    - If timer triggers, starts profiling and sets new `[start_step, end_step)` window.
  - `end(sess)`: stops profiling if active.
  - `_start_profiling()`: `tf.profiler.experimental.start(logdir, options)`; ignores `AlreadyExistsError`.
  - `_stop_profiling()`: calls `tf.profiler.experimental.stop()`; ignores `UnavailableError`.
- `ByteCCLTelemetryHook`:
  - Requires global step tensor (`training_util._get_or_create_global_step_read()`), else `RuntimeError`.
  - Logs telemetry every `interval` steps by sampling BytePS ops on rank 0.
  - `_log_telemetry()` filters ops containing `alltoall` or first 3 `PushPull` entries.
- `NVProfilerHook`:
  - Subclass of `Tf2ProfilerHook` with `logdir=None`.
  - Loads `libcudart.so` and calls `cudaProfilerStart/Stop`.
- `KafkaMetricHook` (singleton):
  - Uses `KAFKA_BROKER_LIST` and `KAFKA_TOPIC_NAME` env vars to create `KProducer`.
  - `__init__`: loads `deep_insight_op` from TF collection if not provided; stores as tensor dict.
  - `after_run`: sends `deep_insight_op` messages to Kafka if any.
  - `end`: closes producer, logs success/failed counts.
- Helper functions:
  - `default_parse_fn`: JSON-decodes strings/bytes; otherwise returns input.
  - `default_layout_fn`: returns string or JSON dump; falls back to `repr` on error.
  - `vepfs_layout_fn`: formats deep insight record as `req_time;gid;uid;predict_scores;labels`.
  - `vepfs_key_fn`: builds path `base/model_name/date/worker_{id}`.
- `WriteOnlyFileAndStat`:
  - Holds buffered output; rotates partitions after `partition_size` lines (default 1e6).
  - Uses `tf.io.gfile` to write `part_XXXXXX.{file_ext}` under `key` directory.
  - `write()` buffers formatted strings; `flush()` writes and rotates; `close()` closes stream.
  - `is_available()` returns True if updated within last 24 hours.
  - Note: uses `List`/`Dict` typing annotations without importing them (potential NameError at runtime).
- `FileMetricHook` (singleton):
  - Initializes from `deep_insight_op` collection if not provided.
  - Requires `key_fn` for routing items; if `None`, `_send` will fail when called.
  - Spawns background thread on first `after_run`.
  - Enqueues messages (handles list/tuple/np.ndarray or scalar).
  - `_send` parses items, writes to per-key `WriteOnlyFileAndStat`, and cleans up inactive files every 10 minutes.
  - `end` waits for queue to drain, stops thread, closes open files.
- Threading/concurrency: multiple background threads; queue for metrics, RLock in file writer.
- Determinism: depends on timing, Kafka/network, filesystem.
- Logging/metrics: uses `mcli.emit_*`, absl logging, Kafka/file outputs.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF Estimator hooks and Kafka/file hooks not present in Rust).
- Rust public API surface: if needed, add a training hooks module with metrics, profiling, and output sinks.
- Data model mapping: map TF hooks to Rust training loop callbacks; map `deep_insight_op` outputs to Rust equivalents.
- Feature gating: Kafka and profiler hooks should be optional (feature flags).
- Integration points: training loop, metrics client, optional BytePS/collective telemetry.

**Implementation Steps (Detailed)**
1. Decide which hooks are needed in Rust training (throughput, loss, custom metrics).
2. Implement throughput/loss hooks as callbacks in Rust training loop.
3. Provide profiling hooks only if profiling support exists (TF2/NV profilers likely N/A).
4. Implement Kafka/File output sinks if required; reuse Rust Kafka + filesystem abstractions.
5. Match environment-variable configuration for Kafka (`KAFKA_BROKER_LIST`, `KAFKA_TOPIC_NAME`).
6. Preserve thread/queue behavior and file partitioning semantics.
7. Add tests for validation errors (CustomMetricHook), file rotation, and queue draining.

**Tests (Detailed)**
- Python tests: `monolith/native_training/metric/metric_hook_test.py`.
- Rust tests: N/A until hooks exist.
- Cross-language parity test: validate emitted metrics names/tags and file output formatting.

**Gaps / Notes**
- `List`/`Dict` are used in annotations without import; may require adding `from typing import List, Dict` in Python for runtime use.
- `FileMetricHook` will fail if `key_fn` is not provided; ensure callers pass `vepfs_key_fn` or a custom function.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/metric/metric_hook_test.py`
<a id="monolith-native-training-metric-metric-hook-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 189
- Purpose/role: Tests for `Tf2ProfilerHook` and `FileMetricHook` behaviors.
- Key symbols/classes/functions: `Tf2ProfilerHookTest`, `FileMetricHookTest`.
- External dependencies: TensorFlow, `os`, `time`, `json`, `random`, `datetime`.
- Side effects: Writes profiling data under `TEST_TMPDIR` and file metrics under `$HOME/tmp/file_metric_hook`.

**Required Behavior (Detailed)**
- `Tf2ProfilerHookTest`:
  - `setUp`:
    - `logdir = $TEST_TMPDIR/<test_name>`.
    - `filepattern = logdir/plugins/profile/*`.
    - Creates a graph with global_step and train_op (`assign_add` by 1).
  - `_count_files()` returns count of files matching pattern.
  - `test_steps`:
    - Hook: `Tf2ProfilerHook(logdir, init_step_range=[0,10], save_steps=50)`.
    - Runs one step in `SingularMonitoredSession`.
    - Expects exactly 1 profile file.
  - `test_multiple_steps_1`:
    - Hook with `save_steps=30`, runs 30 steps with 0.15s sleep.
    - Expects 1 file (profile only at 0~9).
  - `test_multiple_steps_2`:
    - Same hook, runs 31 steps with 0.15s sleep.
    - Expects 2 files (0~9 and step 30).
  - `test_secs_1`:
    - Hook with `save_secs=1`, runs 10 steps with 0.15s sleep.
    - Expects at least 1 file.
  - `test_secs_2`:
    - Hook with `save_secs=3`, runs 21 steps with 0.15s sleep.
    - Expects at least 2 files.
- `FileMetricHookTest`:
  - `setUpClass`:
    - `model_name='test_model'`, `base_name=$HOME/tmp/file_metric_hook`.
    - Creates `FileMetricHook(worker_id=0, key_fn=vepfs_key_fn, layout_fn=vepfs_layout_fn, batch_size=8, partition_size=32)`.
  - `tearDownClass`:
    - Calls `hook.end(None)` to flush/close.
    - For each of last 8 days, asserts:
      - date directory exists under `base_name/model_name/<YYYYMMDD>/worker_0/`.
      - exactly 2 files exist; each has 32 lines.
  - `test_vepfs_key_fn`:
    - Asserts path formatting for fixed data.
  - `test_vepfs_layout_fn`:
    - Asserts formatted string with predict/label JSON and fallback `gid`.
  - `test_after_run`:
    - Builds `RunValue` wrapper with `results={'deep_insight_op':[json.dumps(rv)]}`.
    - For last 8 days, sends 64 records/day with random predict/label values.
    - Calls `hook.after_run` to enqueue metrics; file writing validated in `tearDownClass`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF profiler and FileMetricHook not implemented in Rust).
- Rust public API surface: if implemented, provide equivalent tests for profiling triggers and file partitioning.
- Data model mapping: file output format must match `vepfs_layout_fn` and `vepfs_key_fn`.
- Feature gating: profiling/Kafka/file outputs should be optional.
- Integration points: Rust training hook system.

**Implementation Steps (Detailed)**
1. If Rust supports profiling hooks, add tests for step/second trigger behavior.
2. If file output hook is implemented, port these tests with deterministic data (no randomness).
3. Ensure file partitioning at 32 lines and 2 files per day for 64 records.

**Tests (Detailed)**
- Python tests: `monolith/native_training/metric/metric_hook_test.py`.
- Rust tests: N/A until hooks are implemented.
- Cross-language parity test: compare file outputs and profile dump counts if available.

**Gaps / Notes**
- Tests rely on filesystem and time sleeps; may be flaky or slow.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/metric/utils.py`
<a id="monolith-native-training-metric-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 104
- Purpose/role: Convenience wrapper to emit Deep Insight metrics using custom TF ops (v1 or v2).
- Key symbols/classes/functions: `write_deep_insight`.
- External dependencies: TensorFlow, `deep_insight_ops`, Python `logging`.
- Side effects: Calls custom ops that create clients and emit metrics; logs when disabling.

**Required Behavior (Detailed)**
- `write_deep_insight(features, sample_ratio, model_name, labels=None, preds=None, target=None, targets=None, labels_list=None, preds_list=None, sample_rates_list=None, extra_fields_keys=[], enable_deep_insight_metrics=True, enable_kafka_metrics=False, dump_filename=None)`:
  - Requires `features["req_time"]`; if missing:
    - Logs "Disabling deep_insight because req_time is absent".
    - Returns `tf.no_op()`.
  - `is_fake = enable_kafka_metrics or (dump_filename is not None and len(dump_filename) > 0)`.
  - Creates client: `deep_insight_ops.deep_insight_client(enable_deep_insight_metrics, is_fake, dump_filename)`.
  - `req_times = reshape(features["req_time"], [-1])`.
  - **Single-target path** (`not targets`):
    - `uids = reshape(features["uid"], [-1])`.
    - `sample_rates = reshape(features["sample_rate"], [-1])`.
    - Calls `deep_insight_ops.write_deep_insight` with `labels`, `preds`, `model_name`, `target`, `sample_ratio`, `return_msgs=is_fake`.
  - **Multi-target path** (`targets` truthy):
    - `labels = stack([label if rank==1 else reshape(label, (-1,)) for label in labels_list if label is not None])`.
    - `preds = stack([pred if rank==1 else reshape(pred, (-1,)) for pred in preds_list if pred is not None])`.
    - `sample_rates_list` handling:
      - If falsy: uses `features["sample_rate"]` reshaped to [-1] and repeats `len(targets)` times.
      - If list/tuple: reshapes each to rank 1; filters out None.
      - Else raises `Exception("sample_rates_list error!")`.
    - `sample_rates = stack(sample_rates_list)`.
    - Ensures `"uid"` in `extra_fields_keys` (mutates list default).
    - Builds `extra_fields_values` by reshaping each `features[key]` to [-1].
    - Calls `deep_insight_ops.write_deep_insight_v2` with `targets`, `extra_fields_*`, `return_msgs=is_fake`.
  - Returns the op tensor from deep_insight ops.
- Error cases:
  - Missing `uid`/`sample_rate`/extra fields -> `KeyError`.
  - Empty `labels_list`/`preds_list` -> `tf.stack` error.
  - `sample_rates_list` non-list and truthy -> raises generic `Exception`.
- Mutability note: `extra_fields_keys` default list is mutated when adding `"uid"`.
- No threading; deterministic aside from op behavior.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (custom TF ops not bound in Rust).
- Rust public API surface: optional wrapper around TF runtime deep insight ops.
- Data model mapping: feature tensors, string targets, list-of-tensors for multi-target.
- Feature gating: TF runtime + custom ops only.
- Integration points: metrics pipeline in training.

**Implementation Steps (Detailed)**
1. Add TF runtime bindings for deep insight ops if needed.
2. Mirror single-target vs multi-target branching.
3. Preserve `is_fake` semantics and `return_msgs`.
4. Avoid mutable default pitfalls if porting (but keep behavior if parity requires).

**Tests (Detailed)**
- Python tests: `monolith/native_training/metric/utils_test.py`.
- Rust tests: N/A until ops are bound.
- Cross-language parity test: verify that v1/v2 calls receive the same tensors/flags.

**Gaps / Notes**
- `extra_fields_keys` uses a mutable default list; repeated calls may accumulate `"uid"`.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/metric/utils_test.py`
<a id="monolith-native-training-metric-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 50
- Purpose/role: Tests basic call path for `utils.write_deep_insight` using a mocked op.
- Key symbols/classes/functions: `DeepInsightTest.test_basic`.
- External dependencies: TensorFlow, `unittest.mock`.
- Side effects: None (deep insight op is mocked).

**Required Behavior (Detailed)**
- `test_basic`:
  - Patches `deep_insight_ops.write_deep_insight` and sets a side-effect function.
  - `fake_call` evaluates `uids` tensor in a session and asserts it equals `[1,2,3]`.
  - Constructs `features` with `uid`, `req_time`, `sample_rate` tensors.
  - Creates `labels`, `preds`, `model_name`, `target`.
  - Calls `utils.write_deep_insight(...)`.
  - Note: Call uses positional arguments in a non-obvious order; still exercises `uids` extraction.
- `__main__` disables eager execution and runs `tf.test.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: TF runtime only.
- Integration points: deep insight wrapper tests if implemented.

**Implementation Steps (Detailed)**
1. If deep insight ops are bound in Rust, add a unit test to ensure `uid` extraction and reshape behavior.
2. Prefer explicit keyword arguments to avoid positional mis-ordering.

**Tests (Detailed)**
- Python tests: `monolith/native_training/metric/utils_test.py`.
- Rust tests: N/A until implementation exists.
- Cross-language parity test: verify tensor shapes/values passed to op.

**Gaps / Notes**
- The test passes arguments positionally in a confusing order relative to the function signature; keep behavior but consider fixing in Python if allowed.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/mlp_utils.py`
<a id="monolith-native-training-mlp-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 444
- Purpose/role: Utilities for MLP/YARN distributed training setup, tf.data service orchestration, MPI exception handling, and TF profiler control.
- Key symbols/classes/functions: `check_port`, `MLPEnv`, `add_mpi_exception_hook`, `mlp_pass`, `begin`, `after_create_session`, `EXTRA_DSWORKERS`.
- External dependencies: TensorFlow, tf.data service (`dsvc`), MPI/Horovod/BytePS, `yarn_runtime`, `distribution_utils`, `absl.flags/logging`, `socket`, `signal`, `subprocess`.
- Side effects: Opens sockets, starts TF DataService dispatcher/worker servers, starts TF profiler server, sets `sys.excepthook`, calls `os._exit(0)`, monkeypatches `_DatasetInitializerHook` methods.

**Required Behavior (Detailed)**
- `check_port(host, port, timeout=1)`:
  - Opens IPv4 or IPv6 socket based on host string (`':' in host.strip('[]')`).
  - Attempts to connect within `timeout` seconds; returns True if connected.
  - On timeout returns False.
  - On socket error, retries until timeout expires; returns False otherwise.
- `MLPEnv.__init__()`:
  - Collects env vars starting with `MLP_` or `MPI_`.
  - Reads `MLP_FRAMEWORK`, `MLP_SSH_PORT`, `MLP_LOG_PATH`, `MLP_DEBUG_PORT`, `MLP_ENTRYPOINT_DIR`, `MLP_TASK_CMD`, `MLP_ROLE`.
  - Builds `all_roles` from env vars like `MLP_<ROLE>_NUM`.
  - If `enable_mpi` (OMPI rank exists and role is WORKER):
    - `index = get_mpi_rank()`, `all_roles['WORKER'] = get_mpi_size()`.
    - `port = int(MLP_PORT) + index`.
  - Else uses `MLP_ROLE_INDEX` and `MLP_PORT`.
  - `avaiable` is True if MLP env and roles exist.
  - Records `cpu`, `gpu`, `gpu_type`, `mem` from env.
  - `host = yarn_runtime.get_local_host()`.
- `MLPEnv.enable_mpi`:
  - True when `OMPI_COMM_WORLD_RANK` exists and role is WORKER.
- `MLPEnv._get(name, default=None)`:
  - Reads from `_mlp_env`, strips quotes; returns default if missing.
- `num_replicas(role=None)`:
  - If MPI worker, returns `get_mpi_size()`; else reads `MLP_<ROLE>_NUM`.
- `get_all_host(role=None, is_primary=True)`:
  - Returns `MLP_<ROLE>_ALL_PRIMARY_HOSTS` or `MLP_<ROLE>_ALL_HOSTS` value.
- `get_all_addrs(role=None, is_primary=True)`:
  - Reads `MLP_<ROLE>_ALL_PRIMARY_ADDRS`/`ALL_ADDRS`, splits by comma, else [].
- `get_host(role=None, index=None, is_primary=True)`:
  - Uses MPI/local index logic; returns `MLP_<ROLE>_<index>_PRIMARY_HOST` or `_HOST`.
- `get_addr(role=None, index=None, is_primary=True)`:
  - Computes host and port:
    - MPI worker uses `MLP_<ROLE>_0_PORT` + index.
    - Otherwise `MLP_<ROLE>_<index>_PORT`.
  - Returns `"host:port"` or None.
- `get_port(role=None, index=None)`:
  - MPI worker: `MLP_<ROLE>_0_PORT` (default 2222) + index.
  - Else: `MLP_<ROLE>_<index>_PORT` (default 2222).
  - Note: `_get` returns strings; may require int conversion for correctness.
- `dispatcher_target()`:
  - Returns `grpc://{dispatcher_addr}` or `grpc://localhost:5050`.
- `dispatcher_addr(role=None)`:
  - Uses role default `'dispatcher'` and `get_addr`.
- `wait(role=None, index=0, timeout=-1, use_ssh=True)`:
  - Repeatedly calls `check_port` on `host:port`, sleeping 5s until ready or timeout.
  - Uses `ssh_port` when `use_ssh=True`.
- `join(role='worker', index=0, use_ssh=True)`:
  - Waits for host to come up, then loops until port stops responding (timeout=60 per check).
  - Stops TF profiler if started and calls `os._exit(0)`.
- `queue_device` property:
  - Returns `/job:ps/task:0/device:CPU:0` if PS exists; else worker CPU or local CPU.
- `start_profiler(port=6666)`:
  - Starts `tf.profiler.experimental.server` (port offset by MPI index).
- `profiler_trace(...)`:
  - Builds `ProfilerOptions` and calls `tf.profiler.experimental.client.trace`.
  - Uses all addresses if `index < 0`, otherwise specific address.
- `add_mpi_exception_hook()`:
  - If OMPI rank not set, returns.
  - Installs `sys.excepthook` that prints error details and calls `mpi4py.MPI.COMM_WORLD.Abort(1)`.
- `EXTRA_DSWORKERS`: global list of extra WorkerServer handles.
- `mlp_pass(dispatcher_role='dispatcher', dsworker_role='dsworker', worker_role='worker', ps_role='ps')`:
  - Uppercases roles; if `FLAGS.dataset_use_dataservice`, monkeypatches `_DatasetInitializerHook.begin` and `.after_create_session`.
  - Creates `MLPEnv`. If available:
    - Dispatcher role: starts `dsvc.DispatchServer` on `mlp_env.port`, then `mlp_env.join()`.
    - Dsworker role: waits for dispatcher, starts `dsvc.WorkerServer`, starts profiler, then join.
    - Worker role:
      - If dataset service enabled, waits for dispatcher and dsworkers, sets `FLAGS.data_service_dispatcher`.
      - Starts extra dsworkers on GPU worker based on `FLAGS.num_extra_dsworker_on_gpu_worker`.
      - Logs worker start info and roles.
- `begin(self)` (monkeypatched into `_DatasetInitializerHook`):
  - Stores iterator initializer; sets `_broadcast_dataset_id=None`.
  - If sync training enabled, tries to import BytePS or Horovod (based on `MONOLITH_WITH_BYTEPS`).
  - If `registed_dataset_id` collection exists, broadcasts dataset_id from rank 0 and stores `_broadcast_dataset_id`.
  - Calls `graph.clear_collection(...)` (graph is undefined; potential bug).
- `after_create_session(self, session, coord)`:
  - If `_broadcast_dataset_id` present, runs it to log dataset ids and clears it.
  - Runs iterator initializer.
- Threading/concurrency: uses tf.data service servers and background processes; no explicit threads here.
- Determinism: depends on environment, networking, MPI rank.
- Logging/metrics: extensive absl logging.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF data service + MPI orchestration not in Rust).
- Rust public API surface: if needed, separate runtime module for distributed training env/launcher.
- Data model mapping: env var parsing, host/port resolution, dataset service endpoints.
- Feature gating: MPI/BytePS/TF data service features optional.
- Integration points: training launcher, profiler integration.

**Implementation Steps (Detailed)**
1. Determine whether Rust training needs MLP/YARN orchestration features.
2. If yes, implement env parsing and host/port logic, plus dispatcher/worker lifecycle.
3. Implement MPI exception handling with `mpi` crate equivalents.
4. Provide profiler server/trace integration if applicable.
5. Mirror dataset initializer hook behavior in Rust data input pipeline.

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: add unit tests for env parsing and host/port computation.
- Cross-language parity test: compare computed addresses/ports for fixed env maps.

**Gaps / Notes**
- `begin()` references `graph.clear_collection` but `graph` is undefined (likely bug).
- `get_port` uses `_get` without int conversion; may return strings.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model.py`
<a id="monolith-native-training-model-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 182
- Purpose/role: Defines a small FFM (field-aware factorization machine) test model and input pipeline for native training.
- Key symbols/classes/functions: `_parse_example`, `TestFFMModel`, `FFMParams`, constants `_NUM_SLOTS`, `_FFM_SLOT`, `_VOCAB_SIZES`, `_NUM_EXAMPLES`.
- External dependencies: TensorFlow, NumPy, Monolith feature/entry APIs, `deep_insight_ops`.
- Side effects: Sets NumPy seed in input function; emits deep insight metrics in training; logs model info.

**Required Behavior (Detailed)**
- Constants:
  - `_NUM_SLOTS = 6`, `_VOCAB_SIZES = [5,5,5,5,5,5]`, `_NUM_EXAMPLES = 64`.
  - `_FFM_SLOT` defines tuple pairs and embedding dim for dot products.
- `_parse_example(example: str) -> Dict[str, tf.Tensor]`:
  - Builds feature map with fixed label and VarLenFeature for each slot.
  - Parses examples with `tf.io.parse_example`.
  - Converts any `SparseTensor` to `RaggedTensor`.
  - Returns feature dict including `"label"` and slot features.
- `TestFFMModel(NativeTask)`:
  - `create_input_fn(mode)` returns `input_fn()`:
    - Sets `np.random.seed(0)` for deterministic example generation.
    - Generates `_NUM_EXAMPLES` examples via `generate_ffm_example(_VOCAB_SIZES)`.
    - Builds dataset: batch to `per_replica_batch_size`, map `_parse_example`, cache, repeat, prefetch.
  - `create_model_fn()` returns `model_fn(features, mode, config)`:
    - Creates feature slots/columns for each slot with FTRL bias + SGD default optimizer.
    - Collects bias embeddings for first half of slots.
    - Computes FFM dot products for each `(user,item,dim)` in `_FFM_SLOT`.
    - `ffm_out = add_n(dot_res) + sum_bias`, `pred = sigmoid(ffm_out)`.
    - If `mode == PREDICT`: returns `EstimatorSpec(predictions=pred)`.
    - Loss: `reduce_sum(binary_crossentropy(features["label"], pred))`.
    - If deep insight metrics enabled and sample ratio > 0:
      - Creates deep insight client, builds uids/req_times/sample_rates, calls `write_deep_insight`.
      - Logs model_name/target.
    - Else uses `tf.no_op()`.
    - Increments global step and applies embedding gradients via `ctx.apply_embedding_gradients`.
    - Returns `EstimatorSpec(loss=loss, train_op=group(apply_grads, deep_insight_op))`.
  - `create_serving_input_receiver_fn()`:
    - Creates string placeholder `instances`, parses via `_parse_example`, returns `ServingInputReceiver`.
- `FFMParams(SingleTaskModelParams)`:
  - `task()` returns `TestFFMModel.params()` with `per_replica_batch_size = 64`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (test model; no Rust equivalent).
- Rust public API surface: none.
- Data model mapping: if ported, map to Rust embedding feature slots and FFM dot products.
- Feature gating: none.
- Integration points: training pipeline and deep insight metrics.

**Implementation Steps (Detailed)**
1. If needed for parity demos, implement a Rust FFM example model and parser.
2. Mirror dataset generation determinism (`np.random.seed(0)`) with fixed RNG.
3. Match bias/slot construction and FFM dot-product structure.
4. Add deep insight metrics only if TF runtime backend exists.

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: optional integration test for forward pass + loss.
- Cross-language parity test: compare forward outputs on fixed synthetic batch.

**Gaps / Notes**
- This is a test/sample model; may not be required for production parity.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_comp_test.py`
<a id="monolith-native-training-model-comp-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 183
- Purpose/role: Integration test comparing TF embedding updates vs Monolith embedding updates under sync training (Horovod).
- Key symbols/classes/functions: `EmbeddingUpdateTask`, `CpuSyncTrainTest`, `lookup_tf_embedding`.
- External dependencies: TensorFlow, Horovod, Monolith CPU training stack, Keras layers.
- Side effects: Sets environment variables at import; runs distributed training; writes model checkpoints under `/tmp/<user>/monolith_test/...`.

**Required Behavior (Detailed)**
- Module-level env vars (set before TF/Horovod import):
  - `MONOLITH_WITH_HOROVOD=True`, `HOROVOD_AUTOTUNE=1`, `HOROVOD_CYCLE_TIME=0.1`,
    `MONOLITH_SYNC_EMPTY_RANK0_PS_SHARD=0`, `MONOLITH_WITH_ALLREDUCE_FUSION=one`,
    `MONOLITH_ROOT_LOG_INTERVAL=10`.
- Sets TF v1 random seed to 42.
- Global constants: `num_features=17`, `batch_size=455`, `emb_dim=15`, `fid_max_val=100000`.
- `lookup_tf_embedding(features, f_name, dim)`:
  - Builds `RaggedTensor` from `tf_<f_name>_p1`/`p2`.
  - Embedding lookup on a zeros-initialized variable.
  - Returns `segment_sum` by row ids.
- `EmbeddingUpdateTask(MonolithModel)`:
  - `__init__`: sets `train.max_steps=50`, `train.per_replica_batch_size=batch_size`.
  - `input_fn`:
    - Generates random feature vectors with variable length per feature (1..24).
    - Uses `dense_to_ragged_batch` with batch_size and `advanced_parse`.
    - Adds `tf_feature{i}_p1/p2` to features for TF embedding lookup.
  - `model_fn`:
    - Creates embedding feature columns and Monolith embeddings.
    - Computes TF embeddings via `lookup_tf_embedding`.
    - Asserts Monolith embeddings equal TF embeddings.
    - Builds parallel Keras MLPs for both embedding sets, computes MSE losses.
    - Returns `EstimatorSpec` with combined loss, predictions, labels, head names, and optimizer.
  - `serving_input_receiver_fn`: unimplemented (`pass`).
- `CpuSyncTrainTest`:
  - `_create_config(gpu, multi_hash_table)` builds `DistributedCpuTrainingConfig` with sync training enabled.
  - `test_embedding_update`:
    - Initializes Horovod, runs distributed sync training in 2 configurations (cpu/multi-hash on/off).
    - If GPU available, repeats with GPU enabled.
- `__main__`: disables eager execution and runs `tf.test.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF/Horovod integration test only).
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: Horovod/TF runtime only.
- Integration points: training loop parity for embedding updates.

**Implementation Steps (Detailed)**
1. If Rust aims to match embedding update semantics, port the comparison into Rust unit tests using Candle/TF backend.
2. Provide deterministic random feature generation for repeatability.
3. Mirror embedding lookup + segment sum behavior.

**Tests (Detailed)**
- Python tests: `monolith/native_training/model_comp_test.py`.
- Rust tests: none.
- Cross-language parity test: compare embedding tensors and loss values on fixed seeds.

**Gaps / Notes**
- `serving_input_receiver_fn` is unimplemented.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_dump/dump_utils.py`
<a id="monolith-native-training-model-dump-dump-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 757
- Purpose/role: Central model dump utility that records feature/slice metadata, input/output tensors, signatures, and graph defs into `ModelDump` protobufs.
- Key symbols/classes/functions: `DumpUtils`, `parse_input_fn_result`, wrappers `record_feature`, `record_slice`, `record_receiver`.
- External dependencies: TensorFlow graph/ops internals, protobufs (`model_dump` protos), pickle, `tf.io.gfile`, export context, data parsers.
- Side effects: Monkeypatches `util.parse_input_fn_result`; stores `ProtoModel` on default graph; writes/reads dump files.

**Required Behavior (Detailed)**
- `DumpUtils` is a singleton (`_instance`), `__init__` initializes fields only once:
  - `enable`, `_params`, `_run_config`, `_user_params`, train/infer `ProtoModel` + graph defs, sub-model caches, table configs, slice dims, feature combiners.
- `model_dump` property:
  - Attaches `ProtoModel()` to default graph as `graph.monolith_model_dump`.
- `update_kwargs_with_default(func, kwargs)`:
  - Fills `kwargs` entries with function default values when `None`.
- `record_feature(func)` wrapper:
  - When `need_record`, appends to `model_dump.features`.
  - Copies args/kwargs into proto, converting integer `feature_name` via `get_feature_name_and_slot`.
  - Logs warnings if field missing in proto.
- `record_slice(func)` wrapper:
  - Forbids `learning_rate_fn` (raises `Exception`).
  - Records slice config into `model_dump.emb_slices` including initializer/optimizer/compressor protos.
  - Calls wrapped function and appends output tensor names.
- `record_receiver(func)` wrapper:
  - Records serving input receiver features and receiver tensors into `serving_input_receiver_fn`.
  - Stores ragged tensors as `{values,row_splits,is_ragged}`; dense tensors include dtype/last_dim.
- `record_params(model)`:
  - Captures non-callable, non-private attrs except a skip list into `_params`.
- `get_params_bytes(model)`:
  - Pickles model attributes (including deep-copied `p` and serialized `_layout_dict`).
  - Returns pickled bytes; used by `add_model_fn`.
- `add_signature` / `restore_signature`:
  - Syncs signatures with current export context, mapping tensor names.
- `add_model_fn(model, mode, features, label, loss, pred, head_name, is_classification)`:
  - Fills `model_fn` proto with labels, loss, predictions, head names, classification flags.
  - Adds user summaries from `GraphKeys.SUMMARIES`.
  - Records non-ragged features not already registered.
  - Records extra losses from graph `__losses`.
  - Records `export_outputs` as `extra_output.fetch_dict`.
  - Enforces that only `ItemPoolSaveRestoreHook` is allowed in `__training_hooks`.
  - Stores signatures and SaveSliceInfo for variables.
  - Snapshots graph_def for TRAIN vs INFER into `train_graph`/`infer_graph`.
- `add_input_fn(results)`:
  - Records input feature tensor names and ragged flags; records label if present.
  - Stores parser type and item pool name.
- `add_sub_model(sub_model_type, name, graph)` / `restore_sub_model(sub_model_type)`:
  - Stores/restore sub-graph defs for PS or dense submodels via export context subgraphs.
- `add_optimizer(optimizer)`:
  - Pickles optimizer into `model_dump.optimizer`.
- `dump(fname)`:
  - Builds `ModelDump` proto with run config, user params, train/infer graphs, sub-models, table configs, slice dims, combiners.
  - Writes serialized bytes to `fname` using `tf.io.gfile`.
- `load(fname)`:
  - Reads `ModelDump` and reconstructs train/infer graph defs, table configs, feature slices, combiners, user params, sub-models.
- `get_graph_helper(mode)`:
  - Builds `GraphDefHelper` with SaveSliceInfo from train/infer model dump.
  - Caches on graph as `graph.graph_def_helper`.
- `restore_params()`:
  - Unpickles model params; rebuilds `_layout_dict` from `OutConfig` proto bytes.
  - Deletes `_training_hooks` key if present; raises if layout_dict missing.
- `need_record`:
  - True when `enable` and graph does not have `DRY_RUN` attribute.
- `table_configs` property/setter:
  - Converts between proto configs and `entry.HashTableConfigInstance`.
  - Setter disallows non-numeric `learning_rate_fns`.
- `feature_slice_dims` property/setter:
  - Converts between proto list and dict of dims.
- `feature_combiners` property/setter:
  - Maps `ReduceSum/ReduceMean/FirstN` to/from proto enum `Combiner`.
- `get_slot_to_occurrence_threshold` / `get_slot_to_expire_time`:
  - Builds slot->value maps; warns if slot resolution fails.
- `has_collected`:
  - True if table configs, slice dims, combiners all non-empty; otherwise asserts they are empty.
- `parse_input_fn_result(result)`:
  - If `DatasetV2`, makes iterator + `_DatasetInitializerHook`.
  - Else uses `DatasetInitHook` from collection `mkiter`.
  - Calls `DumpUtils().add_input_fn` and returns parsed iterator result + input hooks.
  - Monkeypatches `util.parse_input_fn_result`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF graph/proto dump infrastructure not in Rust).
- Rust public API surface: if parity required, add a model-dump module capturing graph metadata and feature configs.
- Data model mapping: Protobuf `ModelDump` -> Rust structs; tensor names as strings.
- Feature gating: TF runtime only if applicable.
- Integration points: model export, serving input receivers, embedding table configs.

**Implementation Steps (Detailed)**
1. Decide whether Rust needs model dump/export parity; define equivalent data structures.
2. Implement feature/slice recording and tensor-name capture.
3. Implement save/load of graph metadata and signatures.
4. Mirror validation (learning_rate_fn disallow, training hooks restrictions).

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: add serialization/deserialization tests if implemented.
- Cross-language parity test: compare dumped proto fields for a simple model.

**Gaps / Notes**
- `export_outputs` branch uses `ts` variable that may be undefined when outputs are not dict (possible bug).
- `parse_input_fn_result` monkeypatch changes global TF estimator behavior.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_dump/graph_utils.py`
<a id="monolith-native-training-model-dump-graph-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 845
- Purpose/role: GraphDef utilities for reconstructing variables, importing subgraphs, and rebuilding input/model/receiver functions from dumped metadata.
- Key symbols/classes/functions: `DatasetInitHook`, `_node_name`, `_colocated_node_name`, `EchoInitializer`, `VariableDef`, `PartitionVariableDef`, `GraphDefHelper`.
- External dependencies: TensorFlow GraphDef/ops internals, protobufs (`LineId`, `FeatureConfigs`), `tf.keras`, `tf.io`, flags.
- Side effects: Mutates graph node attrs (`_class`), clears devices, adds to TF collections, imports graph defs into default graph.

**Required Behavior (Detailed)**
- Globals:
  - `DRY_RUN = 'dry_run'`, `FLAGS = flags.FLAGS`.
- `DatasetInitHook`: session hook that runs initializer in `after_create_session`.
- `_node_name(name)`:
  - Strips leading `^` and output suffix `:0`; returns node base name.
- `_colocated_node_name(name)`:
  - Decodes bytes to string; strips `loc:@` prefix if present.
- `EchoInitializer`:
  - Accepts an init value; if list/tuple uses first element.
  - Returns tensor directly if `tf.Tensor`, else returns op output (expects single output).
- `VariableDef`:
  - Wraps a `VarHandleOp` node and helper.
  - `initializer`: finds `<var>/Assign` node, determines initializer input, builds subgraph, imports it, and returns `EchoInitializer`.
  - `variable`: creates a `tf.get_variable` with dtype/shape/initializer on proper device, temporarily disabling partitioner; returns first partition if `PartitionedVariable`.
  - Tracks associated `ReadVariableOp` nodes via `add_read`.
- `PartitionVariableDef`:
  - Handles partitioned variables (`/part_N`); tracks partitions and read ops.
  - `get_base_name` extracts base variable name from VarHandleOp or ReadVariableOp inputs.
  - `initializer`: finds PartitionedInitializer slice nodes or Assign initializer nodes, imports subgraph, returns list of `EchoInitializer`.
  - `variable`: creates variables for each partition (uses first device as group_device), sets `save_slice_info`, and builds a `PartitionedVariable` for validation if multiple partitions.
- `GraphDefHelper.__init__(graph_def, save_slice_info)`:
  - Validates GraphDef type.
  - Clears node device and `_class` colocation hints (adds colocated names to input set).
  - Builds name-to-node, seq mapping, and tracks variables/readers into `VariableDef`/`PartitionVariableDef`.
  - Records PBDataset file_name const node if present.
- `_check_invalidate_node(graph_def, input_map)`:
  - Removes input_map entries not referenced by graph inputs; logs warnings.
- `_create_variables(variables)`:
  - Recreates variable read tensors using `read_variable_op` for all variable defs in subgraph.
  - Skips canonical `/Read/ReadVariableOp` nodes.
- `sub_graph(dest_nodes, source_nodes=None, with_library=True)`:
  - BFS from dest nodes through inputs; stops at source_nodes.
  - Builds a GraphDef containing non-variable nodes and collects variable names separately.
  - If `with_library`, copies required functions (including Dataset functions).
- `import_input_fn(input_conf, file_name)`:
  - Constructs dest_nodes from recorded output features and label; includes iterator ops unless DRY_RUN.
  - Updates PBDataset/file_name Const value to `file_name`.
  - Optionally updates PBDataset/input_pb_type based on `FLAGS.data_type`.
  - Imports subgraph; adds iterator/mkiter to collections.
  - Rebuilds features dict (ragged or dense) and adds label.
- `import_model_fn(input_map, proto_model)`:
  - Collects outputs from predict, extra outputs, loss, labels, extra_losses, signatures, summaries.
  - Builds subgraph, prunes input_map, recreates variable reads, imports graph_def.
  - Adds sparse feature names to collections by scanning ShardingSparseFids nodes.
  - Restores summaries to GraphKeys.SUMMARIES.
  - Validates signature inputs exist in graph.
  - Returns `(label, loss, predict, head_name, extra_output_dict, is_classification)`.
- `import_receiver_fn(receiver_conf)`:
  - Builds dest_nodes for ragged values/row_splits and dense features.
  - Populates collections: sparse_features, dense_features/types/shapes, extra_features/shapes, variant_type.
  - Imports subgraph and reconstructs feature tensors + receiver_tensors.
- `get_optimizer(proto_model)`:
  - Unpickles optimizer from proto bytes or returns None.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF GraphDef import is Python-specific).
- Rust public API surface: only relevant if a TF runtime backend is used for model import.
- Data model mapping: GraphDef + metadata + tensor names.
- Feature gating: TF runtime only.
- Integration points: model loader, serving input reconstruction.

**Implementation Steps (Detailed)**
1. If Rust needs TF graph import, wrap GraphDef parsing and node filtering.
2. Implement variable recreation and read-op mapping analogous to `_create_variables`.
3. Implement sub-graph extraction with function library filtering.
4. Recreate input/model/receiver functions using stored metadata and collections.

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: N/A unless TF GraphDef import is added.
- Cross-language parity test: verify imported outputs match original graph outputs.

**Gaps / Notes**
- Uses `eval` on serialized feature representations (security risk if untrusted).
- Clears node device assignments; placement is not preserved.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_dump/graph_utils_test.py`
<a id="monolith-native-training-model-dump-graph-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 86
- Purpose/role: Tests `GraphDefHelper` input/receiver reconstruction using a saved model dump.
- Key symbols/classes/functions: `GraphUtilsTest.test_load_input_fn`, `test_load_receiver`, `test_load_mode`.
- External dependencies: TensorFlow, `DumpUtils`, `GraphDefHelper`.
- Side effects: Loads model dump from `model_dump/test_data/model_dump`.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Sets `FLAGS.data_type = 'examplebatch'`.
  - Loads dump via `DumpUtils().load(...)`.
- `test_load_input_fn`:
  - Calls `import_input_fn` with `file_name`.
  - Verifies each output feature returns `tf.RaggedTensor` if flagged ragged, else `tf.Tensor`.
- `test_load_receiver`:
  - Calls `import_receiver_fn`.
  - Verifies feature tensor types and that receiver_tensors length is 1.
- `test_load_mode`:
  - `get_graph_helper` returns `GraphDefHelper` for TRAIN, TRAIN with `graph.dry_run=True`, and PREDICT.
- `__main__`: disables eager execution and runs tests.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: TF runtime only.
- Integration points: model dump loader.

**Implementation Steps (Detailed)**
1. If graph import is implemented in Rust, add tests for ragged/dense reconstruction.
2. Use fixed dump artifacts to avoid nondeterminism.

**Tests (Detailed)**
- Python tests: `monolith/native_training/model_dump/graph_utils_test.py`.
- Rust tests: none.
- Cross-language parity test: compare reconstructed tensors and types.

**Gaps / Notes**
- Test depends on external dump artifacts in the repo.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/__init__.py`
<a id="monolith-native-training-model-export-init-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 22
- Purpose/role: Re-exports model export modules under legacy module paths for backward compatibility.
- Key symbols/classes/functions: module aliasing via `sys.modules`.
- External dependencies: `export_context`, `saved_model_exporters`.
- Side effects: Inserts entries into `sys.modules` and deletes local `_sys`.

**Required Behavior (Detailed)**
- Imports `monolith.native_training.model_export.export_context` and `saved_model_exporters`.
- Registers aliases:
  - `'monolith.model_export.export_context'` → `export_context` module.
  - `'monolith.model_export.saved_model_exporters'` → `saved_model_exporters` module.
- Deletes `_sys` name after aliasing.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: Python import compatibility only.

**Implementation Steps (Detailed)**
1. If Rust wrappers need to mirror Python module paths, document the aliasing behavior in docs.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Pure import aliasing; no functional logic.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/data_gen_utils.py`
<a id="monolith-native-training-model-export-data-gen-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 732
- Purpose/role: Generates synthetic Example/Instance/ExampleBatch data and PredictionLogs for model export, warmup, and testing.
- Key symbols/classes/functions: `FeatureMeta`, `ParserArgs`, `gen_fids_v1`, `gen_fids_v2`, `fill_features`, `fill_line_id`, `gen_example`, `gen_instance`, `gen_example_batch`, `gen_prediction_log`, `gen_warmup_file`, `gen_random_data_file`.
- External dependencies: TensorFlow, TF Serving protos, Monolith feature list, proto types (`Example`, `Instance`, `LineId`).
- Side effects: Writes TFRecord warmup files and binary data files.

**Required Behavior (Detailed)**
- Constants: `MASK_V1/MAX_SLOT_V1`, `MASK_V2/MAX_SLOT_V2` control fid encoding.
- `FeatureMeta`:
  - Infers dtype from LineId field descriptors or slot (defaults to float32 for dense, int64 for sparse).
  - `shape` defaults to 1 for dense, -1 for sparse.
- `ParserArgs` dataclass:
  - Reads defaults from TF collections via `get_collection`.
  - Ensures `DEFAULT_SERVING_SIGNATURE_DEF_KEY` is present in `signature_name`.
  - Attempts `FeatureList.parse()` if no feature_list provided.
- `gen_fids_v1(slot, size)` / `gen_fids_v2(slot, size)`:
  - Encodes slot in high bits and random low bits; v1 logs when slot > max, v2 asserts slot range.
- `fill_features` (singledispatch):
  - For `EFeature` (Example):
    - Sparse: generates fid_v2_list with drop_rate logic and slot-specific handling.
    - Dense: fills float/double/int64 lists with random values.
  - For `IFeature` (Instance):
    - Similar logic; uses `feature.fid`, `float_value`, `int64_value`.
- `fill_line_id(line_id, features, hash_len=48, actions=None)`:
  - Fills LineId fields based on metadata or defaults; handles repeated vs scalar fields.
- `lg_header(source)` / `sort_header(sort_id, kafka_dump, kafka_dump_prefix)`:
  - Produce binary headers for data files, including Java hash computation.
- `gen_example(...)`:
  - Builds Example with named_feature entries; uses `FeatureList` or slot lookup; fills labels and LineId.
- `gen_instance(...)`:
  - Builds Instance proto using fidv1/fidv2 features; fills labels and LineId.
- `gen_example_batch(...)`:
  - Builds ExampleBatch with per-feature lists, LineId list (`__LINE_ID__`), and labels (`__LABEL__`).
- `gen_prediction_log(args)`:
  - Generates PredictRequest logs using the appropriate variant type.
  - Supports multiple signatures; may emit multiple requests for multi-head outputs.
- `gen_warmup_file(warmup_file, drop_rate)`:
  - Builds ParserArgs, removes dense label if present, writes PredictionLog TFRecord to file.
  - Creates directories if needed; returns file path or None.
- `gen_random_data_file(...)`:
  - Writes binary file with headers and serialized instances for `num_batch`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (Python proto generators).
- Rust public API surface: if needed, add a data-gen utility module for tests/warmup.
- Data model mapping: map proto types (Example/Instance/ExampleBatch) to Rust protobufs.
- Feature gating: TF Serving protos required for PredictionLog generation.
- Integration points: model export/warmup pipeline.

**Implementation Steps (Detailed)**
1. Implement fid encoding and feature filling logic in Rust.
2. Mirror ParserArgs collection-based defaults if Rust uses similar collections.
3. Implement PredictionLog generation and TFRecord writing for warmup data.
4. Port binary data file format (headers + length + payload).

**Tests (Detailed)**
- Python tests: none (`data_gen_utils_test.py` is empty).
- Rust tests: add unit tests for fid encoding and example generation.
- Cross-language parity test: compare serialized outputs for fixed RNG seed.

**Gaps / Notes**
- Uses `eval` on stored representations and random data generation; not deterministic without seeding.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/data_gen_utils_test.py`
<a id="monolith-native-training-model-export-data-gen-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 0
- Purpose/role: Empty test placeholder.
- Key symbols/classes/functions: none.
- External dependencies: none.
- Side effects: none.

**Required Behavior (Detailed)**
- File is empty; no tests executed.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. Add tests if/when data generation is ported to Rust.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: N/A.

**Gaps / Notes**
- No coverage for data generation utilities.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/demo_export.py`
<a id="monolith-native-training-model-export-demo-export-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 100
- Purpose/role: CLI demo that exports a saved model from the TestFFMModel using standalone or distributed exporter.
- Key symbols/classes/functions: `export_saved_model`, `main`.
- External dependencies: TensorFlow, Monolith CPU training, `parse_instances`, `StandaloneExporter`, `DistributedExporter`.
- Side effects: Writes SavedModel to disk under `export_base`; uses flags; disables eager execution.

**Required Behavior (Detailed)**
- Defines flags:
  - `num_ps` (default 5) for CPU training config.
  - `model_dir` and `export_base` default to `/tmp/<user>/monolith/native_training/demo/...`.
  - `export_mode` enum (Standalone or Distributed).
- `export_saved_model(model_dir, export_base, num_ps, export_mode)`:
  - Disables eager execution; sets TF logging verbosity to INFO.
  - Instantiates `TestFFMModel` params with name `"demo_export"` and batch size 64.
  - Creates `cpu_training.CpuTraining` with `CpuTrainingConfig(num_ps=num_ps)`.
  - Chooses exporter:
    - `StandaloneExporter` or `DistributedExporter` (with `shared_embedding=False`).
  - Defines `serving_input_receiver_fn`:
    - `instances` placeholder of dtype `tf.string` with shape `(None,)`.
    - Parses instances via `parse_instances`, with fidv1 features 0.._NUM_SLOTS-1.
    - Builds `features` dict with keys `feature_i` from `slot_i`.
    - Returns `tf.estimator.export.ServingInputReceiver`.
  - Calls `exporter.export_saved_model(serving_input_receiver_fn)`.
- `main(_)` calls `export_saved_model` with flags.
- `__main__` uses `absl.app.run`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (Python TF export demo).
- Rust public API surface: none.
- Data model mapping: if exporting in Rust, define equivalent serving input receiver.
- Feature gating: TF runtime only.
- Integration points: export pipeline.

**Implementation Steps (Detailed)**
1. If Rust export is desired, implement a demo exporter that mirrors TestFFMModel inputs.
2. Map parsing logic for FID v1 features to Rust serving inputs.
3. Preserve default paths and batch size for parity tests.

**Tests (Detailed)**
- Python tests: `monolith/native_training/model_export/demo_export_test.py`.
- Rust tests: none.
- Cross-language parity test: compare exported SavedModel signatures and input names.

**Gaps / Notes**
- Demo only; depends on TestFFMModel and CPU training stack.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/demo_export_test.py`
<a id="monolith-native-training-model-export-demo-export-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 48
- Purpose/role: Integration test that trains TestFFMModel and verifies standalone/distributed export paths run without error.
- Key symbols/classes/functions: `DemoExportTest.test_demo_export`.
- External dependencies: TensorFlow, Monolith CPU training, `demo_export.export_saved_model`.
- Side effects: Creates training checkpoints and two SavedModel export directories under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- Disables eager execution at import time.
- `test_demo_export`:
  - Creates `model_dir = $TEST_TMPDIR/test_ffm_model`.
  - Trains TestFFMModel with `cpu_training.local_train(params, num_ps=5, model_dir=...)`.
  - Calls `demo_export.export_saved_model` twice:
    - Standalone export to `$TEST_TMPDIR/standalone_saved_model`.
    - Distributed export to `$TEST_TMPDIR/distributed_saved_model`.
  - Uses `ExportMode.STANDALONE` and `ExportMode.DISTRIBUTED`.
- No explicit assertions on contents; success is absence of errors.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: TF runtime only.
- Integration points: export pipeline.

**Implementation Steps (Detailed)**
1. If Rust adds export support, add a smoke test for export outputs.
2. Ensure deterministic temp paths and cleanup.

**Tests (Detailed)**
- Python tests: `monolith/native_training/model_export/demo_export_test.py`.
- Rust tests: none.
- Cross-language parity test: compare exported SavedModel signatures.

**Gaps / Notes**
- Test is heavy (trains and exports); may be slow.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/demo_predictor.py`
<a id="monolith-native-training-model-export-demo-predictor-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 110
- Purpose/role: CLI demo to load a SavedModel and run prediction with randomly generated inputs.
- Key symbols/classes/functions: `make_fid_v1`, `generate_demo_instance`, `random_generate_instances`, `random_generate_int`, `random_generate_float`, `predict`, `main`.
- External dependencies: TensorFlow SavedModel, NumPy, `proto_parser_pb2.Instance`, TestFFMModel constants.
- Side effects: Loads SavedModel from disk; logs prediction outputs.

**Required Behavior (Detailed)**
- Flags:
  - `saved_model_path` (required path), `tag_set` (default "serve"), `signature` (default "serving_default"), `batch_size` (default 128).
- `make_fid_v1(slot_id, fid)`:
  - Encodes FID v1 as `(slot_id << 54) | fid`.
- `generate_demo_instance()`:
  - Creates `Instance` proto.
  - For each slot in `model._NUM_SLOTS`, generates 5 random fids in that slot based on `max_vocab`.
  - Returns serialized bytes.
- `random_generate_instances(bs)`:
  - Returns list of `bs` serialized Instance bytes.
- `random_generate_examples(bs)` (unused):
  - Returns list of serialized Example bytes using `model.generate_ffm_example`.
- `random_generate_int(shape)`:
  - Returns int64 array in `[0, max_vocab)` where `max_vocab = max(_VOCAB_SIZES) * _NUM_SLOTS`.
- `random_generate_float(shape)`:
  - Returns float array of `uniform(0,1)` values.
- `predict()`:
  - Loads SavedModel with `tf.compat.v1.saved_model.load`.
  - Reads signature inputs/outputs for `FLAGS.signature`.
  - For each input, builds a feed tensor based on dtype:
    - string -> list of serialized instances, shape length must be 1.
    - int64 -> random ints.
    - float32 -> random floats.
    - else raises `ValueError`.
  - Runs session and logs outputs.
- `main` calls `predict`; `__main__` sets logging verbosity to INFO and runs via absl.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: if implementing predictor in Rust, map SavedModel signature I/O to random data generation.
- Feature gating: TF runtime only.
- Integration points: serving validation / smoke tests.

**Implementation Steps (Detailed)**
1. If Rust can load SavedModels, implement a CLI to sample inputs and run predictions.
2. Mirror dtype-based generation (string -> serialized Instance, int64/float32 random arrays).
3. Match FID v1 encoding for feature IDs.

**Tests (Detailed)**
- Python tests: `demo_predictor_client.py` (manual) or none.
- Rust tests: none.
- Cross-language parity test: compare output shapes for identical inputs.

**Gaps / Notes**
- `random_generate_examples` is unused.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/demo_predictor_client.py`
<a id="monolith-native-training-model-export-demo-predictor-client-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 93
- Purpose/role: gRPC client for TensorFlow Serving PredictionService using random inputs derived from SavedModel signature.
- Key symbols/classes/functions: `get_signature_def`, `main`.
- External dependencies: gRPC, TensorFlow Serving protos, TensorFlow.
- Side effects: Sends Predict RPC to remote server.

**Required Behavior (Detailed)**
- Flags:
  - `server` (default "localhost:8500"), `model_name` ("default"), `signature_name` ("serving_default"), `use_example` (bool).
  - Note: code references `FLAGS.batch_size` but flag is not defined in this file (bug).
- `get_signature_def(stub)`:
  - Requests signature_def metadata via `GetModelMetadata`.
  - Unpacks `SignatureDefMap` and returns signature by `FLAGS.signature_name`.
  - Prints available signature names.
- `main`:
  - Creates insecure gRPC channel and PredictionService stub.
  - Builds PredictRequest with model spec.
  - For each input in signature:
    - Computes shape, substituting `FLAGS.batch_size` for -1 dims.
    - Generates example/instance bytes for string inputs.
    - Generates random ints/floats for int64/float32 inputs.
    - Raises `ValueError` for unsupported dtype.
  - Calls `stub.Predict(request, timeout=30)` and logs result.
- Logging verbosity set to INFO in `__main__`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: if implementing, use TF Serving gRPC protos in Rust.
- Feature gating: gRPC + TF Serving protos.
- Integration points: serving smoke tests.

**Implementation Steps (Detailed)**
1. Define missing `batch_size` flag or pass as CLI arg.
2. If Rust needs a client, implement signature discovery and random input generation.
3. Mirror example/instance encoding logic using demo_predictor helpers.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: compare request shapes and dtype handling.

**Gaps / Notes**
- `FLAGS.batch_size` is referenced but never defined (likely a bug).

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/export_context.py`
<a id="monolith-native-training-model-export-export-context-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 141
- Purpose/role: Manages export mode state and signatures for model export; provides context manager for export mode.
- Key symbols/classes/functions: `ExportMode`, `ExportContext`, `enter_export_mode`, `is_exporting*`, `get_current_export_ctx`, `is_dry_run_or_exporting`.
- External dependencies: TensorFlow, `tf_contextlib`, `monolith_export` decorator.
- Side effects: Global export mode state; stores signatures in TF collections.

**Required Behavior (Detailed)**
- `ExportMode` enum: `NONE`, `STANDALONE`, `DISTRIBUTED`.
- `SavedModelSignature` namedtuple (`name`, `inputs`, `outputs`).
- `ExportContext`:
  - Maintains `sub_graphs` and `dense_sub_graphs` as `defaultdict(tf.Graph)`.
  - Maintains `_signatures` keyed by graph id; each entry maps name -> SavedModelSignature.
  - `add_signature` adds to TF collection `signature_name` and stores signature.
  - `merge_signature` updates existing signature inputs/outputs or creates empty.
  - `signatures(graph)` returns signature values for given graph id.
  - `with_remote_gpu` property returns constructor flag.
  - `sub_graph_num` returns count of sub_graphs.
- Globals:
  - `EXPORT_MODE` starts as `NONE`.
  - `EXPORT_CTX` starts as `None`.
- `is_exporting` / `is_exporting_standalone` / `is_exporting_distributed`:
  - Compares `EXPORT_MODE` to enum values.
- `get_current_export_ctx`:
  - Returns `EXPORT_CTX`.
- `enter_export_mode(mode, export_ctx=None)`:
  - Asserts no nested export (`EXPORT_MODE is NONE` and `EXPORT_CTX is None`).
  - Creates new `ExportContext()` if not provided.
  - Sets globals, yields `export_ctx`, then resets globals to defaults in `finally`.
- `is_dry_run_or_exporting()`:
  - Returns True if export mode active or default graph has `dry_run` attribute.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: optional export context struct with thread-local state.
- Data model mapping: signatures map, subgraph registry.
- Feature gating: export-only.
- Integration points: model export pipeline.

**Implementation Steps (Detailed)**
1. Implement export context state in Rust (thread-local/global).
2. Provide RAII guard for entering/exiting export mode.
3. Mirror signature tracking and graph association logic if needed.

**Tests (Detailed)**
- Python tests: `export_context` is exercised by export hooks and demo exporters.
- Rust tests: add unit tests for mode nesting and signature registry.
- Cross-language parity test: ensure signature names collected match.

**Gaps / Notes**
- Uses global mutable state; not thread-safe for parallel exports.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/export_hooks.py`
<a id="monolith-native-training-model-export-export-hooks-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 137
- Purpose/role: Checkpoint saver listener that exports SavedModel on each checkpoint and prunes old exports.
- Key symbols/classes/functions: `get_global_step`, `ExportSaverListener`.
- External dependencies: TensorFlow, Monolith save_utils/export_state_utils, custom metrics client.
- Side effects: Writes export directories, deletes old ones, emits metrics.

**Required Behavior (Detailed)**
- `get_global_step(checkpoint_path)`:
  - Regex `model.ckpt-(\d+)`; asserts match; returns int.
- `ExportSaverListener`:
  - `__init__(save_path, serving_input_receiver_fn, exporter, exempt_checkpoint_paths=None, dense_only=False)`:
    - Stores serving_input_receiver_fn and exporter.
    - `self._helper = save_utils.SaveHelper(save_path)`.
    - Builds `self._exempt_checkpoint_steps` from `exempt_checkpoint_paths` by parsing global steps.
    - `dense_only` toggles special deletion logic.
    - Creates metric client via `cli.get_cli(utils.get_metric_prefix())`.
  - `after_save(session, global_step_value)`:
    - Uses SaveHelper to get checkpoint prefix for step.
    - Calls `exporter.export_saved_model(...)`.
    - Accepts export_dirs as bytes, list, or dict of values.
    - For each export_dir:
      - Adds entry to export state and prunes old entries.
  - `_add_entry_to_state(export_dir, global_step_value)`:
    - Decodes bytes; computes base/version.
    - Appends `ServingEntry(export_dir, global_step)` to state and overwrites state file.
    - Calls `_update_metrics`.
  - `_maybe_delete_old_entries(export_dir)`:
    - Loads existing state; computes `existing_steps` from current checkpoints plus exempt steps.
    - If `dense_only`, also loads full checkpoint state from model_dir and includes all steps.
    - Removes entries not in `existing_steps`, deleting directories via `tf.io.gfile.rmtree`.
  - `_update_metrics(export_dir_base, version)`:
    - Emits `export_models.latest_version` as int if version is numeric.
    - `version` uses `split(".")[0]` to handle float-like names.
    - Logs warning on exceptions every 1200 seconds.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: if implementing export hooks, add a checkpoint listener trait.
- Data model mapping: export state protobufs -> Rust structs.
- Feature gating: export-only.
- Integration points: checkpoint saving, export pipeline, metrics client.

**Implementation Steps (Detailed)**
1. Implement checkpoint listener in Rust training loop if needed.
2. Mirror export directory state tracking and pruning rules.
3. Add metrics emission for latest export version.

**Tests (Detailed)**
- Python tests: `export_hooks_test.py`.
- Rust tests: add filesystem tests for pruning behavior.
- Cross-language parity test: compare export state entries after simulated checkpoints.

**Gaps / Notes**
- `get_global_step` asserts regex match; invalid checkpoint paths will raise AssertionError.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/export_hooks_test.py`
<a id="monolith-native-training-model-export-export-hooks-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 141
- Purpose/role: Tests ExportSaverListener behavior (state updates, dict export outputs, deletion of old exports).
- Key symbols/classes/functions: `ExportHookTest.testBasic`, `testExporterReturnsDict`, `testDeleted`.
- External dependencies: TensorFlow, `save_utils`, `export_state_utils`, `unittest.mock`.
- Side effects: Creates model/export dirs under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `testBasic`:
  - Mocks exporter to return `export_dir` bytes and asserts checkpoint path format.
  - Runs `NoFirstSaveCheckpointSaverHook` and sets global_step to 10.
  - Verifies export state has one entry with correct dir and step.
- `testExporterReturnsDict`:
  - Mocks exporter to return dict of model names to export dirs.
  - Ensures no errors during export and state update.
- `testDeleted`:
  - Mocks exporter to create unique export dirs per step.
  - Uses `PartialRecoverySaver` with `max_to_keep=1`.
  - After two steps, verifies only latest export remains and old one deleted.
- Tests rely on `export_state_utils.get_export_saver_listener_state` and filesystem cleanup.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: export state proto to Rust struct if implemented.
- Feature gating: export-only.
- Integration points: export listener and checkpoint saver.

**Implementation Steps (Detailed)**
1. If Rust implements export hooks, add tests mirroring state entries and deletion.
2. Mock exporter to return bytes or dict.
3. Validate pruning when `max_to_keep` is 1.

**Tests (Detailed)**
- Python tests: `monolith/native_training/model_export/export_hooks_test.py`.
- Rust tests: none.
- Cross-language parity test: compare state entries after simulated checkpoints.

**Gaps / Notes**
- Uses real filesystem; may need cleanup to avoid test leakage.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/export_state_utils.py`
<a id="monolith-native-training-model-export-export-state-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 46
- Purpose/role: Reads/writes export state file (`ExportSaverListenerState`) containing `ServingModelState` proto.
- Key symbols/classes/functions: `get_export_saver_listener_state`, `overwrite_export_saver_listener_state`.
- External dependencies: TensorFlow gfile, protobuf text_format, `export_pb2`.
- Side effects: Reads and writes state files in export directory.

**Required Behavior (Detailed)**
- `_ExportSaverListenerStateFile = "ExportSaverListenerState"`.
- `get_export_saver_listener_state(export_dir_base)`:
  - Reads `<export_dir_base>/ExportSaverListenerState` as text proto.
  - If file missing, returns empty `ServingModelState`.
  - Parses using `text_format.Merge`.
- `overwrite_export_saver_listener_state(export_dir_base, state)`:
  - Ensures `export_dir_base` exists (`gfile.makedirs`).
  - Writes text proto to temp file `<filename>-tmp`.
  - Atomically renames temp to final file (overwrite=True).

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: if implemented, add export state read/write helpers.
- Data model mapping: `ServingModelState` proto in Rust.
- Feature gating: export-only.
- Integration points: export hooks.

**Implementation Steps (Detailed)**
1. Implement read/write of text-format proto in Rust (or switch to binary with parity notes).
2. Preserve temp-file rename semantics for atomic updates.

**Tests (Detailed)**
- Python tests: `export_state_utils_test.py`.
- Rust tests: add read/write round-trip tests.
- Cross-language parity test: compare serialized text output.

**Gaps / Notes**
- Uses text-format proto; any Rust implementation must match formatting if parity is required.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/export_state_utils_test.py`
<a id="monolith-native-training-model-export-export-state-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 36
- Purpose/role: Round-trip test for export state read/write.
- Key symbols/classes/functions: `ExportStateUtilsTest.test_basic`.
- External dependencies: `export_state_utils`, `export_pb2`, filesystem.
- Side effects: Writes state file under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `test_basic`:
  - Creates `ServingModelState` with one entry (`export_dir="a"`, `global_step=1`).
  - Writes state to temp dir via `overwrite_export_saver_listener_state`.
  - Reads state back and asserts equality with original.
- Uses `unittest.TestCase`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: ServingModelState proto.
- Feature gating: export-only.
- Integration points: export state utilities.

**Implementation Steps (Detailed)**
1. Implement Rust read/write helpers and add a round-trip test.
2. Ensure protobuf equality holds after text-format serialization.

**Tests (Detailed)**
- Python tests: `export_state_utils_test.py`.
- Rust tests: add round-trip test if implemented.
- Cross-language parity test: compare text serialization output.

**Gaps / Notes**
- Uses deprecated `assertEquals` (should be `assertEqual`).

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/export_utils.py`
<a id="monolith-native-training-model-export-export-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 98
- Purpose/role: Helper for defining and invoking remote prediction signatures via `distributed_serving_ops.remote_predict`.
- Key symbols/classes/functions: `RemotePredictHelper`, `_get_tensor_signature_name`.
- External dependencies: TensorFlow, `nested_tensors`, `export_context`, `distributed_serving_ops`.
- Side effects: Registers SavedModel signatures with `ExportContext`.

**Required Behavior (Detailed)**
- `_get_tensor_signature_name(t)`:
  - Returns tensor name with ":" replaced by "_" (e.g., "foo:0" -> "foo_0").
- `RemotePredictHelper.__init__(name, input_tensors, remote_func)`:
  - Wraps inputs in `NestedTensors`, stores remote func, calls `_define_remote_func`.
- `_define_remote_func()`:
  - Creates placeholders matching flat input tensors (dtype/shape) with suffix `_remote_input_ph`.
  - Builds nested input structure from placeholders and calls `remote_func`.
  - Wraps outputs in `NestedTensors`.
  - Builds signature input/output dicts keyed by `_get_tensor_signature_name`.
  - Asserts no name conflicts (lengths match).
  - Registers signature in current `ExportContext` via `add_signature`.
- `call_remote_predict(model_name, input_tensors=None, old_model_name=None, task=0)`:
  - Uses provided `input_tensors` or original input tensors.
  - Calls `distributed_serving_ops.remote_predict` with signature name and I/O names.
  - Passes `output_types` from output tensor dtypes and `signature_name=self._name`.
  - Returns outputs in original nested structure.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: if remote predict exists, implement a helper mirroring signature registration.
- Data model mapping: nested tensor structure + signature names.
- Feature gating: remote serving only.
- Integration points: distributed serving ops.

**Implementation Steps (Detailed)**
1. Implement nested tensor flattening and placeholder generation if Rust supports graph export.
2. Ensure signature name mapping replaces ":" with "_".
3. Provide remote predict wrapper matching arg ordering and output types.

**Tests (Detailed)**
- Python tests: `export_utils_test.py`.
- Rust tests: add unit test for signature name mapping and nested output reconstruction.
- Cross-language parity test: compare signature I/O names and remote_predict call args.

**Gaps / Notes**
- Relies on global export context; ensure one is active when constructing helper.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/export_utils_test.py`
<a id="monolith-native-training-model-export-export-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 43
- Purpose/role: Basic test for `RemotePredictHelper` signature definition and call path.
- Key symbols/classes/functions: `ExportUtilsTest.testBasic`.
- External dependencies: TensorFlow, `export_context`, `export_utils`.
- Side effects: Enters export mode (standalone).

**Required Behavior (Detailed)**
- `testBasic`:
  - Enters `export_context.enter_export_mode(EXPORT_MODE.STANDALONE)`.
  - Defines `remote_func(d)` returning `d["a"] * 3 + d["b"] * 4`.
  - Instantiates `RemotePredictHelper("test_func", {"a": tf.constant(1), "b": tf.constant(2)}, remote_func)`.
  - Calls `helper.call_remote_predict("model_name")`.
  - Asserts result is a `tf.Tensor`.
- Note: test intentionally only checks grammar due to missing TF Serving compilation.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: export-only.
- Integration points: remote predict.

**Implementation Steps (Detailed)**
1. If Rust implements RemotePredictHelper, add a similar smoke test.
2. Ensure export mode context is active for signature registration.

**Tests (Detailed)**
- Python tests: `export_utils_test.py`.
- Rust tests: none.
- Cross-language parity test: verify signature registration.

**Gaps / Notes**
- No actual remote serving is exercised.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/saved_model_exporters.py`
<a id="monolith-native-training-model-export-saved-model-exporters-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 739
- Purpose/role: Implements SavedModel exporters for standalone and distributed export, including hashtable restore/assign signatures and warmup assets.
- Key symbols/classes/functions: `BaseExporter`, `StandaloneExporter`, `DistributedExporter`.
- External dependencies: TensorFlow SavedModel internals, Monolith hash table ops, export_context, DumpUtils.
- Side effects: Writes SavedModel directories, copies assets, modifies TF collections, restores variables/hashtables.

**Required Behavior (Detailed)**
- `BaseExporter`:
  - Stores model_fn, model_dir, export_dir_base, shared_embedding, warmup_file, and optional export_context_list.
  - `create_asset_base()`:
    - Adds a `ASSET_BASE` tensor and AssetFileDef to assets collection if not already.
    - Returns tensor with value `"./"`.
  - `add_ckpt_to_assets(ckpt_to_export, pattern="*")`:
    - Adds all matching ckpt asset files to `ASSET_FILEPATHS` collection.
  - `build_signature(input_tensor_dict, output_tensor_dict)`:
    - Wraps tensors or TensorInfo into `SignatureDef` for PREDICT.
  - `_freeze_dense_graph(graph_def, signature_def_map, session)`:
    - Collects all input/output nodes from signatures and uses `convert_variables_to_constants`.
    - Restores device placement in frozen graph.
  - `_export_saved_model_from_graph(...)`:
    - Requires export_dir or export_dir_base.
    - Builds signatures from export_ctx.
    - Optionally adds hashtable assign signatures.
    - Creates Session with soft placement + GPU updates.
    - Restores variables and hashtables (restore ops and assign ops).
    - Writes SavedModel via `Builder` to temp dir then renames.
    - Copies `assets_extra` to `assets.extra` if provided.
  - `_export_frozen_saved_model_from_graph(...)`:
    - Similar but freezes graph and re-imports into a new graph before export.
  - `create_hashtable_restore_ops` / `create_multi_hashtable_restore_ops`:
    - For each (multi) hash table in graph collections, builds restore ops.
    - If not shared_embedding, adds ckpt files to assets and uses asset base; else uses ckpt asset dir.
  - `build_hashtable_assign_inputs_outputs`:
    - Creates placeholder-based assign tensors for hashtable update signature.
  - `add_multi_hashtable_assign_signatures`:
    - Adds raw_assign signatures for multi-hash tables (ragged id + flat values).
  - `_model_fn_with_input_reveiver`:
    - Runs model_fn in PREDICT mode and registers signatures in export_context.
  - `export_saved_model(...)`:
    - Abstract.
  - `gen_warmup_assets()`:
    - Generates warmup TFRecord via `gen_warmup_file` if not present and returns assets dict.
- `StandaloneExporter.export_saved_model(...)`:
  - Enters export mode STANDALONE; clears `TF_CONFIG` temporarily.
  - Builds graph, runs model_fn, exports SavedModel with warmup assets.
  - Restores `TF_CONFIG` on exit.
- `DistributedExporter.export_saved_model(...)`:
  - Creates ExportContext with `with_remote_gpu` flag.
  - Enters DISTRIBUTED export mode; clears `TF_CONFIG` temporarily.
  - Exports entry graph (optionally with GPU device placement).
  - Exports dense subgraphs and ps subgraphs stored in export_ctx.
  - Supports `dense_only`, `include_graphs`, `global_step_as_timestamp`, `freeze_variable`.
  - Skips exporting if target dir already exists.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: if implementing SavedModel export, mirror BaseExporter and specialized exporters.
- Data model mapping: SavedModel signatures, assets, and hashtable metadata.
- Feature gating: TF runtime + monolith hash table ops required.
- Integration points: export pipeline, checkpoint loader, hash table restore.

**Implementation Steps (Detailed)**
1. Implement export context signature collection in Rust if needed.
2. Add SavedModel builder wrappers and asset copying.
3. Implement hash table restore/assign signatures in Rust or document lack.
4. Mirror distributed export layout (entry + dense + ps submodels).

**Tests (Detailed)**
- Python tests: `saved_model_exporters_test.py`.
- Rust tests: add export smoke tests if implemented.
- Cross-language parity test: compare exported signature defs and asset layout.

**Gaps / Notes**
- `_model_fn_with_input_reveiver` typo in name (receiver misspelled) but used internally.
- Uses TF internal APIs; may be brittle across TF versions.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/saved_model_exporters_test.py`
<a id="monolith-native-training-model-export-saved-model-exporters-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 153
- Purpose/role: Tests StandaloneExporter with hash tables and multi-hash tables, including shared embedding mode.
- Key symbols/classes/functions: `ModelFnCreator`, `SavedModelExportersTest`.
- External dependencies: TensorFlow Estimator, Monolith hash table ops, SavedModel exporter.
- Side effects: Creates checkpoints and exports SavedModels under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `ModelFnCreator.create_model_fn()`:
  - Sets `_called_in_exported_mode` if `export_context.EXPORT_MODE != None`.
  - Builds hash table and multi-hash table.
  - In PREDICT mode:
    - Exports outputs for default signature, "table/lookup", and "mtable/lookup".
  - In TRAIN mode:
    - Adds assign_add ops for tables.
    - Adds `CheckpointSaverHook` with hash table saver listeners.
    - Returns `EstimatorSpec` with train_op and loss=0.
- `dummy_input_receiver_fn` returns empty features with a string placeholder.
- `SavedModelExportersTest`:
  - `run_pred(export_path, key=DEFAULT)` loads SavedModel and runs output tensor.
  - `testBasic`:
    - Trains one step to create checkpoint.
    - Exports SavedModel and asserts predictions for table and mtable lookups.
    - Asserts model_fn was called in export mode.
  - `testSharedEmebdding`:
    - Exports with `shared_embedding=True` and asserts predictions.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: hash table ops and SavedModel exports.
- Feature gating: TF runtime + hash table ops.
- Integration points: export pipeline and hash table checkpointing.

**Implementation Steps (Detailed)**
1. If Rust supports hash-table-backed exports, add equivalent tests.
2. Verify lookup outputs after export match expected values.
3. Cover shared embedding behavior if implemented.

**Tests (Detailed)**
- Python tests: `saved_model_exporters_test.py`.
- Rust tests: none.
- Cross-language parity test: compare exported predictions for fixed inputs.

**Gaps / Notes**
- Misspelling `testSharedEmebdding` in test name.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/saved_model_visulizer.py`
<a id="monolith-native-training-model-export-saved-model-visulizer-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 89
- Purpose/role: CLI utility to import a SavedModel protobuf and write it to TensorBoard for visualization.
- Key symbols/classes/functions: `import_to_tensorboard`, `main`.
- External dependencies: TensorFlow SavedModel proto, TensorBoard summary writer.
- Side effects: Reads a SavedModel file, writes TensorBoard logdir.

**Required Behavior (Detailed)**
- `import_to_tensorboard(model_dir, log_dir)`:
  - Opens SavedModel file at `model_dir` as bytes.
  - Parses `SavedModel` proto; if more than one meta_graph, prints message and exits with code 1.
  - Imports the first graph_def into a new graph.
  - Writes graph to `log_dir` using `summary.FileWriter`.
  - Prints TensorBoard command.
- `main` invokes `import_to_tensorboard` with CLI flags.
- CLI parsing via `argparse`, requires `--model_dir` and `--log_dir`.
- Uses `app.run` from TF platform with parsed args.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: SavedModel proto parsing.
- Feature gating: TF runtime only.
- Integration points: tooling/debugging.

**Implementation Steps (Detailed)**
1. If Rust needs similar tool, parse SavedModel protobuf and emit graph for visualization.
2. Provide CLI for model_dir/log_dir.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Uses TF internal APIs; depends on TensorFlow installation.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/warmup_data_decoder.py`
<a id="monolith-native-training-model-export-warmup-data-decoder-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 55
- Purpose/role: CLI tool to decode TF Serving warmup TFRecord files and print sanitized requests.
- Key symbols/classes/functions: `main`.
- External dependencies: TensorFlow, TF Serving PredictionLog proto, `env_utils`.
- Side effects: Reads TFRecord file, logs decoded model specs.

**Required Behavior (Detailed)**
- Flag `file_name` specifies input TFRecord path.
- `main`:
  - Attempts `env_utils.setup_hdfs_env()`; ignores errors.
  - Enables eager execution and sets TF logging verbosity.
  - Defines `decode_fn` to parse `PredictionLog` from record bytes.
  - Iterates TFRecordDataset over `file_name`, decodes each log.
  - Extracts PredictRequest, replaces `string_val:.*` with `string_val: ...` in printed output.
  - Logs index and sanitized request string.
- Uses `app.run(main)`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: TF Serving PredictionLog proto.
- Feature gating: TF Serving protos required.
- Integration points: tooling for warmup verification.

**Implementation Steps (Detailed)**
1. If Rust needs a decoder, parse TFRecord PredictionLogs and print sanitized requests.
2. Mirror regex sanitization for `string_val` fields.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Eager execution is required; script is for inspection only.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/warmup_data_gen.py`
<a id="monolith-native-training-model-export-warmup-data-gen-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 253
- Purpose/role: CLI tool to generate TF Serving warmup PredictionLog data from existing pb data or random generation.
- Key symbols/classes/functions: `PBReader`, `gen_prediction_log_from_file`, `tf_dtype`, `main`.
- External dependencies: TensorFlow, TF Serving protos, `data_gen_utils.gen_prediction_log`, `env_utils`.
- Side effects: Reads input files, writes TFRecord warmup data to `output_path`.

**Required Behavior (Detailed)**
- CLI flags cover file input, batch sizes, feature lists/types, and generation mode (`gen_type` = file/random).
- `PBReader`:
  - Iterates over binary input stream (stdin or file), reading size-prefixed records.
  - Supports lagrangex header, sort_id, kafka_dump_prefix/kafka_dump.
  - For `example_batch`, reads one record per batch; otherwise reads `batch_size` records.
  - `set_max_iter(max_records)` sets max iterations based on variant type.
- `gen_prediction_log_from_file(...)`:
  - Chooses input name based on variant_type (`instances`, `examples`, `example_batch`).
  - Ensures `serving_default` signature included.
  - Yields `PredictionLog` entries with `PredictRequest` containing the batch tensor.
- `tf_dtype(dtype: str)`:
  - Maps string/int aliases to TF dtypes; **bug**: returns `tf.int46` for int64 cases (invalid dtype).
- `main`:
  - Calls `env_utils.setup_hdfs_env()`.
  - Writes PredictionLog records to `FLAGS.output_path` using TFRecordWriter.
  - If `gen_type == 'file'`, uses `gen_prediction_log_from_file`.
  - Else constructs feature specs from CLI flags and calls `data_gen_utils.gen_prediction_log(...)`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: TF Serving PredictionLog proto and tensor encoding.
- Feature gating: TF Serving protos required.
- Integration points: warmup data generation tooling.

**Implementation Steps (Detailed)**
1. Fix `tf_dtype` mapping if porting (use `tf.int64` for int64/long).
2. Clarify `gen_prediction_log` API usage; current call signature appears outdated.
3. If porting to Rust, implement size-prefixed reader and PredictionLog writer.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add unit tests for PBReader and dtype mapping.
- Cross-language parity test: compare generated TFRecord entries for fixed inputs.

**Gaps / Notes**
- `tf_dtype` uses `tf.int46` (typo).
- `gen_prediction_log` call signature likely mismatched with current `data_gen_utils` (uses ParserArgs now).

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/warmup_example_batch.py`
<a id="monolith-native-training-model-export-warmup-example-batch-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 57
- Purpose/role: Converts saved example-batch files into TF Serving PredictionLog warmup records.
- Key symbols/classes/functions: `gen_prediction_log`, `main`.
- External dependencies: TensorFlow, TF Serving protos, `env_utils`.
- Side effects: Reads input folder, writes TFRecord output.

**Required Behavior (Detailed)**
- Flags: `input_folder`, `output_path` (both required for use).
- `gen_prediction_log(input_folder)`:
  - Iterates files in input folder.
  - Reads file bytes and parses into `PredictRequest`.
  - Sets `model_spec.name="default"` and `signature_name="serving_default"`.
  - Wraps in `PredictionLog` and yields.
  - Prints parse result (debug).
- `main`:
  - Writes logs to TFRecord at `output_path`.
- `__main__` calls `env_utils.setup_hdfs_env()` then runs app.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: TF Serving PredictionLog.
- Feature gating: TF Serving protos.
- Integration points: warmup data generation.

**Implementation Steps (Detailed)**
1. If porting, read binary example-batch files and wrap into PredictionLog.
2. Preserve default model/signature names.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: compare serialized output logs.

**Gaps / Notes**
- No validation of input file format; assumes each file is a serialized PredictRequest.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/monolith_export.py`
<a id="monolith-native-training-monolith-export-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 18
- Purpose/role: No-op decorator used to mark classes/functions for export.
- Key symbols/classes/functions: `monolith_export`.
- External dependencies: none.
- Side effects: Adds `__monolith_doc` attribute set to `None` on the object.

**Required Behavior (Detailed)**
- `monolith_export(obj)`:
  - Sets `obj.__monolith_doc = None`.
  - Returns the original object.
  - Used as decorator on classes/functions.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: documentation/export tooling only.

**Implementation Steps (Detailed)**
1. If Rust needs similar tagging, add a marker trait or attribute macro (optional).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Pure annotation; no runtime behavior beyond attribute set.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/multi_hash_table_ops.py`
<a id="monolith-native-training-multi-hash-table-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 695
- Purpose/role: Implements multi-hash-table ops wrapper around custom TF ops, including lookup/assign/optimize and checkpoint save/restore hooks.
- Key symbols/classes/functions: `CachedConfig`, `MultiHashTable`, `MultiHashTableCheckpointSaverListener`, `MultiHashTableCheckpointRestorerListener`, `MultiHashTableRestorerSaverListener`.
- External dependencies: TensorFlow custom ops (`gen_monolith_ops`), hash table protobufs, save_utils, distributed_serving_ops.
- Side effects: Registers proto functions, adds tables to TF collections, writes ckpt info files.

**Required Behavior (Detailed)**
- Constants: `_TIMEOUT_IN_MS` (1 hour), `_MULTI_HASH_TABLE_GRAPH_KEY`.
- `CachedConfig`:
  - Stores configs, table_names, serialized mconfig, tensor, dims, slot_expire_time_config.
- `infer_dims`/`convert_to_cached_config`:
  - Builds `MultiEmbeddingHashTableConfig`, sets entry_type=SERVING when exporting.
  - Serializes config and returns `CachedConfig`.
- `MultiHashTable`:
  - Creates/reads multi hash table handle via custom ops, registers resource, adds to collection.
  - `from_cached_config` sets device based on table type (gpucuco -> GPU).
  - Lookup/assign/add/optimize operations delegate to custom ops.
  - `raw_lookup`, `raw_assign`, `raw_apply_gradients` use ragged ids and flat values.
  - Provides fused lookup/optimize for sync training.
  - `save`/`restore` use custom ops with basename.
  - `to_proto`/`from_proto` allow graph serialization.
- Helpers: ragged concatenation and flattening utilities for input/outputs.
- Checkpoint listeners:
  - `MultiHashTableCheckpointSaverListener` saves tables before saver, optionally writes `ckpt.info-<step>` with feature counts.
  - `MultiHashTableCheckpointRestorerListener` restores tables before restore, with optional PS monitor skip.
  - `MultiHashTableRestorerSaverListener` triggers restore after save.
- Registers proto functions on `_MULTI_HASH_TABLE_GRAPH_KEY` and marks `IsHashTableInitialized` as not differentiable.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF custom ops).
- Rust public API surface: would require binding the custom multi-hash-table ops.
- Data model mapping: protobuf configs, table handles, ragged IDs.
- Feature gating: TF runtime + custom ops.
- Integration points: embedding tables, checkpointing, distributed serving.

**Implementation Steps (Detailed)**
1. Bind custom ops for multi-hash-table if TF runtime backend is enabled.
2. Mirror ragged-id flattening and value slicing for embeddings.
3. Implement checkpoint listeners or hook equivalents for save/restore.
4. Match proto serialization for export/import.

**Tests (Detailed)**
- Python tests: `multi_hash_table_ops_test.py`.
- Rust tests: none.
- Cross-language parity test: compare lookup/assign outputs and checkpoint restores.

**Gaps / Notes**
- Requires custom ops and protobuf definitions; no Rust equivalent today.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/multi_hash_table_ops_test.py`
<a id="monolith-native-training-multi-hash-table-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 249
- Purpose/role: Tests for MultiHashTable operations (lookup, assign_add, reinitialize, apply_gradients, save/restore, hooks).
- Key symbols/classes/functions: `MultiTypeHashTableTest.*`.
- External dependencies: TensorFlow, multi_hash_table_ops custom ops, save_utils.
- Side effects: Writes checkpoint and asset files under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `test_lookup_assign_add_reinitialize`:
  - Builds table with slots; assign_add values; lookup matches expected.
  - Reinitializes slot2 and slot3; slot3 returns status -1.
- `test_apply_gradients`:
  - Applies gradients and checks embeddings updated (negative values).
- `test_save_restore`:
  - Saves table to basename, restores into new graph with different slots.
  - Restored values for overlapping slots match expected.
- `test_save_restore_hook`:
  - Uses saver and restorer hooks; ensures restore overwrites sub_op updates.
  - Verifies values match initial add_op.
- `test_meta_graph_export`:
  - Ensures multi hash table collection appears in exported meta_graph.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: custom ops and checkpoint assets.
- Feature gating: TF runtime + custom ops.
- Integration points: embedding table subsystem.

**Implementation Steps (Detailed)**
1. If Rust binds multi hash table ops, add tests for assign/lookup/save/restore.
2. Mirror hook behavior in Rust checkpointing.

**Tests (Detailed)**
- Python tests: `multi_hash_table_ops_test.py`.
- Rust tests: none.
- Cross-language parity test: compare lookup outputs and restore behavior.

**Gaps / Notes**
- Heavily depends on custom ops; cannot run without TF runtime.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/multi_type_hash_table.py`
<a id="monolith-native-training-multi-type-hash-table-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 435
- Purpose/role: Abstractions for multi-type hash tables and a merged-table wrapper that deduplicates configs across slots.
- Key symbols/classes/functions: `BaseMultiTypeHashTable`, `MultiTypeHashTable`, `MergedMultiTypeHashTable`.
- External dependencies: TensorFlow, hash_table_ops, distribution_ops, prefetch_queue.
- Side effects: Uses device placement; may register queue hooks for pipelined execution.

**Required Behavior (Detailed)**
- `BaseMultiTypeHashTable`:
  - Abstract API: `lookup`, `assign`, `assign_add`, `reinitialize`, `apply_gradients`, `as_op`, `get_table_dim_sizes`.
  - Supports queue hook aggregation via `add_queue_hook` and `get_queue_hooks`.
- `MultiTypeHashTable`:
  - Builds per-slot hash tables using a factory; maintains resources and learning_rate tensors.
  - `lookup` returns per-slot embeddings.
  - `assign` / `assign_add` / `apply_gradients` delegate to per-slot tables and return updated copy.
  - `as_op` returns no-op dependent on all table ops.
  - Supports fused lookup/optimize via custom ops using flattened learning rate tensor.
  - `reinitialize` not supported (raises NotImplementedError).
- `_IndexedValues` dataclass: records merged slots, index, and value tensor for merged operations.
- `MergedMultiTypeHashTable`:
  - Deduplicates slots with identical config (stringified config as key).
  - Builds merged slot names using MD5; tracks slot->merged_slot mapping.
  - If old naming mismatch, adds `extra_restore_names`.
  - `lookup`:
    - Merges slot ids by merged slot, calls underlying table lookup.
    - Splits embeddings back to original slots using sizes.
    - Supports optional early reorder results via `auxiliary_bundle`.
  - `assign` / `assign_add` / `apply_gradients`:
    - Merges ids and values before delegating.
    - `skip_merge_id` option in `_update` to bypass merge for certain paths.
  - `reinitialize` not supported.
  - `get_table_dim_sizes` returns inferred sizes for merged configs.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF hash table ops).
- Rust public API surface: if embedding tables are implemented in Rust, add multi-type table abstraction and merged wrapper.
- Data model mapping: slot->embedding tensors, per-slot configs.
- Feature gating: embedding/hash table feature only.
- Integration points: embedding lookup and optimizer updates.

**Implementation Steps (Detailed)**
1. Implement BaseMultiTypeHashTable trait in Rust with lookup/assign APIs.
2. Add merged-table wrapper to reduce redundant configs.
3. Preserve slot ordering and size-based splitting for merged lookups.

**Tests (Detailed)**
- Python tests: `multi_type_hash_table_test.py`.
- Rust tests: add tests for merged slot mapping and lookup correctness.
- Cross-language parity test: compare embeddings for identical configs.

**Gaps / Notes**
- Merging uses `str(config)`; any changes in string representation alter merge behavior.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/multi_type_hash_table_test.py`
<a id="monolith-native-training-multi-type-hash-table-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 326
- Purpose/role: Tests MultiTypeHashTable and MergedMultiTypeHashTable behaviors, including fused ops and name stability.
- Key symbols/classes/functions: `MultiTypeHashTableTest.*`, `MergedMultiTypeHashTable.*`.
- External dependencies: TensorFlow, hash_table_ops custom ops.
- Side effects: None beyond TF session operations.

**Required Behavior (Detailed)**
- `test_basic`: assign_add + lookup values per slot.
- `test_apply_gradients`: applies gradients; expected negative embeddings.
- `test_apply_gradients_with_learning_rate_decay`: uses PolynomialDecay learning rate; checks scaled updates.
- `test_apply_gradients_without_lookup`: gradient updates without prior lookup.
- `test_fused_lookup` / `test_fused_lookup_multi_shards`:
  - Validate fused lookup outputs (embeddings, splits, offsets).
- `test_fused_apply_gradients` / `test_fused_apply_gradients_missing_tables`:
  - Validate fused optimize updates and resulting embeddings.
- `MergedMultiTypeHashTable.testBasic`:
  - Merges slots 1/2; verifies combined updates and gradients.
- `testNameStability`:
  - Ensures merged slot name (MD5) deterministic; factory called with single merged key.
- `testRestoreName`:
  - Verifies `extra_restore_names` for old naming convention `fc_slot_*`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: custom ops for hash tables.
- Feature gating: TF runtime + custom ops.
- Integration points: embedding table implementation.

**Implementation Steps (Detailed)**
1. If Rust binds custom ops, port tests for assign/lookup and fused ops.
2. Validate merged slot mapping and restore name behavior.

**Tests (Detailed)**
- Python tests: `multi_type_hash_table_test.py`.
- Rust tests: none.
- Cross-language parity test: compare embeddings and offsets for fused ops.

**Gaps / Notes**
- Tests rely on custom ops and fixed learning rate semantics.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/native_model.py`
<a id="monolith-native-training-native-model-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 1109
- Purpose/role: Core TF-native model base classes and helpers for loss/prediction, embedding slice management, device placement, file output, metrics (AUC/MSE + deep insight), and export hooks.
- Key symbols/classes/functions:
  - `get_sigmoid_loss_and_pred`, `get_softmax_loss_and_pred`
  - `DeviceCtxType`, `MonolithDeviceCtx`
  - `MonolithBaseModel`, `MonolithModel`
  - Key methods: `create_model_fn`, `create_input_fn`, `create_serving_input_receiver_fn`,
    `create_embedding_feature_column`, `lookup_embedding_slice`, `share_slot`, `add_extra_output`, `add_training_hook`
- External dependencies: TensorFlow (graph/Estimator APIs, metrics, summaries, SavedModel), `absl.flags/logging`,
  monolith internal modules (`feature`, `feature_utils`, `file_ops`, `metric_utils`, `export_context`,
  `layers.LogitCorrection`, `dump_utils`, `device_utils`, `distribution_utils`, `embedding_combiners`, etc.),
  `OutConfig/OutType/TensorShape` proto from `idl.matrix.proto.example_pb2`.
- Side effects:
  - Writes prediction/eval outputs to per-worker files.
  - Writes item embedding cache table files in resource-constrained roughsort predict mode.
  - Mutates TF graph collections and graph-attached lists (`__losses`, `__training_hooks`, `__export_outputs`).
  - Registers slots and switches feature slots (`register_slots`, `switch_slot`, `switch_slot_batch`).
  - Adds TF summary scalars and `tf.print` ops for metrics.
  - Adds custom hooks for Kafka/file metrics.
  - Enables TOB env (`enable_tob_env`) on init.

**Required Behavior (Detailed)**
- **`get_sigmoid_loss_and_pred(name, logits, label, batch_size, sample_rate=1.0, sample_bias=False, mode=TRAIN, instance_weight=None, mask=None, logit_clip_threshold=None, predict_before_correction=True)`**
  - Reshapes `logits` to `(-1,)` and **overrides** `batch_size` using `dim_size(logits, 0)` regardless of the caller-supplied `batch_size`.
  - If `mode != PREDICT`:
    - `sample_rate` handling:
      - If `float`, fill tensor of shape `(batch_size,)`.
      - If `None`, fill tensor with `1.0`.
    - Instantiate `LogitCorrection(activation=None, sample_bias=sample_bias, name='sample_rate_correction')`.
    - Compute `logits_biased = src((logits, sample_rate))`.
    - `pred` is sigmoid of **raw** `logits` when `predict_before_correction=True`, else sigmoid of `logits_biased`.
    - If `logit_clip_threshold` set:
      - Assert `0 < logit_clip_threshold < 1`.
      - Compute `threshold = log((1 - p) / p)` and clip `logits_biased` to `[-threshold, threshold]`.
    - Compute `losses = sigmoid_cross_entropy_with_logits(labels=label.reshape(-1), logits=logits_biased)`.
    - If `instance_weight` present, reshape to `(-1,)`.
    - If `mask` present, reshape to `(-1,)` and `boolean_mask` both `losses` and `instance_weight` (if present).
    - If `instance_weight` present, multiply `losses *= instance_weight`.
    - Final `loss = reduce_sum(losses)`.
  - If `mode == PREDICT`: `loss=None`, `pred = sigmoid(logits)` (no correction and no clipping).
  - Returns `(loss, pred)` with op names using `{name}_sigmoid_*`.
- **`get_softmax_loss_and_pred(name, logits, label, mode)`**
  - `pred = argmax(softmax(logits, name='{name}_softmax_pred'), axis=1)`.
  - If `mode != PREDICT`, `loss = softmax_cross_entropy_with_logits(labels=label, logits=logits, name='{name}_softmax_loss')`.
  - Else `loss=None`. Returns `(loss, pred)`.
- **`DeviceCtxType`**
  - Constants: `INPUT_FN`, `MODEL_FN`, `INPUT_RECEIVER_FN`, `OTHERS`.
  - `all_types()` returns set of all constants.
- **`MonolithDeviceCtx(ctx_type)`**
  - Context manager for device placement; asserts `ctx_type` in `DeviceCtxType.all_types()`.
  - `__enter__`:
    - No-op if `enable_sync_training()` is false or `export_context.is_exporting()` is true.
    - Selects device function:
      - `INPUT_FN` → `input_device_fn`
      - `MODEL_FN` → `model_device_fn`
      - `INPUT_RECEIVER_FN` → `serving_input_device_fn`
      - Otherwise no-op.
    - Calls `tf.compat.v1.device(self._device_fn)` and enters it.
  - `__exit__`:
    - If `ctx_type == MODEL_FN`, calls `ensure_variables_in_device()` before exiting device scope.
    - Resets `_current` and `_device_fn`.
  - `ensure_variables_in_device()`:
    - Iterates `graph.get_operations()` and for ops whose name starts with `global_step`, calls `graph._apply_device_functions(op)` (private TF API).
- **`MonolithBaseModel(NativeTask, ABC)`**
  - `params()` defines:
    - `output_path`, `output_fields`, `delimiter` (default `\t`), `file_name`,
      `enable_grads_and_vars_summary`, `dense_weight_decay`, `clip_norm` (default `1000.0`),
      `sparse_norm_warmup_steps`, `default_occurrence_threshold`.
  - `__init__`:
    - Calls `enable_tob_env()`.
    - Initializes dicts: `fs_dict`, `fc_dict`, `slice_dict`, `_layout_dict`, `_occurrence_threshold`, `_share_slot_mapping`.
    - `_use_dense_allreduce = FLAGS.enable_sync_training`.
  - `__getattr__`:
    - For attributes in `self.p`, returns param value.
    - Special case `batch_size`: returns `eval.per_replica_batch_size` if `p.mode == EVAL`, else `train.per_replica_batch_size`.
    - Falls back to property getters or base `__getattr__`.
  - `__setattr__`:
    - If `self.p` has attr, sets there.
    - Special case `batch_size`: sets both `train.per_replica_batch_size` and `eval.per_replica_batch_size`.
  - `__deepcopy__`:
    - Deep-copies all attributes except `dump_utils` (shared reference).
  - `_get_file_ops(features, pred)`:
    - Requires `p.output_fields` set.
    - Builds `output_path = p.output_path/part-{worker_index:05d}`; opens `file_ops.WritableFile`.
    - `op_fields` is `features[field]` for each `output_fields` (comma-separated).
    - Appends predictions:
      - list/tuple → extend
      - dict → sorted by key before extending
      - scalar → append
    - Attempts to `tf.squeeze` tensors where rank > 1 and last dim == 1.
    - Formats each row using `tf.strings.format` with delimiter-joined `{}` and `summarize=-1`.
    - Uses `tf.map_fn` with `fn_output_signature=tf.string` and `tf.stop_gradient`.
    - Returns `(op_file, write_op)` where `write_op = op_file.append(tf.strings.reduce_join(result))`.
  - `_dump_item_embedding_ops(features)`:
    - Assumes instance is `DeepRoughSortBaseModel` and features contain `item_id`, `item_bias`, `item_vec`.
    - Writes `MonolithHashTable_cached_item_embeddings-00000-of-00001` under `_cal_item_cache_table_path()`.
    - Uses `WritableFile.append_entry_dump` to write.
  - `_get_real_mode(mode)`:
    - If `mode == PREDICT`, returns `PREDICT`.
    - If `mode == TRAIN`, returns `self.mode` (not necessarily `TRAIN`).
    - Otherwise raises `ValueError('model error!')`.
  - `is_fused_layout()` returns `ctx.layout_factory is not None`.
  - `instantiate()` returns `self` (no cloning).
  - `add_loss(losses)` appends one or many losses to graph-level `__losses`.
  - `losses` property:
    - Stored on `tf.compat.v1.get_default_graph()` as `__losses` list.
  - `_global_step` property:
    - Inside `maybe_device_if_allowed('/device:GPU:0')`, returns `tf.compat.v1.train.get_or_create_global_step()`.
  - `_training_hooks` property:
    - Stored on graph as `__training_hooks` list.
  - `clean()` clears feature-slot caches: `fs_dict`, `fc_dict`, `slice_dict`, `_occurrence_threshold`.
  - `create_input_fn()`:
    - Returns closure that wraps `self.input_fn(mode)` in `MonolithDeviceCtx(INPUT_FN)`.
  - `create_model_fn()`:
    - Resets caches via `clean()`.
    - Defines `model_fn_internal(features, mode, config)`:
      - `global_step = _global_step`, `real_mode = _get_real_mode(mode)`.
      - Runs `self.model_fn(features, real_mode)` inside `MonolithDeviceCtx(MODEL_FN)`.
      - Accepts either `EstimatorSpec` or `(label, loss, pred)` tuple/list:
        - `EstimatorSpec`: extract `label`, `loss`, `pred`, optional `head_name`, `classification`.
          If `pred` is dict, `head_name` becomes keys (in insertion order).
        - Tuple/list: `label, loss, pred`; if `pred` dict, `head_name` from keys; else `head_name` from
          `self.metrics.deep_insight_target`; sets `is_classification=True` and emits a warning.
        - Otherwise raises `Exception("EstimatorSpec Error!")`.
      - Validates `head_name`, `label`, `pred` shapes and alignment.
      - Normalizes `label` to `tf.identity` (name `label_{_node_name(...)}` or dict keys).
      - Calls `dump_utils.add_model_fn(self, mode, features, label, loss, pred, head_name, is_classification)`.
      - Adds auxiliary losses: `loss += tf.add_n(self.losses)` when present.
      - **Resource-constrained roughsort predict path**:
        - When not exporting, `real_mode == PREDICT`, and `FLAGS.enable_resource_constrained_roughsort`,
          and `self` is `DeepRoughSortBaseModel`, it writes item cache table and returns
          `EstimatorSpec(PREDICT, loss=1, train_op=no_op, training_hooks=[FileCloseHook]+_training_hooks, predictions=identity(...))`.
      - **Predict path**:
        - `predictions = dict(zip(head_name, pred))` for list/tuple, else `pred`.
        - If exporting or `p.output_path` is `None`: returns `EstimatorSpec(PREDICT, predictions=..., training_hooks=_training_hooks)`.
        - Else writes per-worker file using `_get_file_ops` and wraps `predictions` with `tf.identity` under control deps.
        - If exporting and `_export_outputs` populated, merges via `spec._replace(export_outputs=...)`.
      - **Metrics accumulation for train/eval**:
        - Builds `targets`, `labels_list`, `preds_list` aligned with heads.
        - If `FLAGS.disable_native_metrics` is false:
          - Classification → `tf.compat.v1.metrics.auc`; regression → `tf.compat.v1.metrics.mean_squared_error`.
          - Adds `tf.print` of metric value to stderr and `tf.compat.v1.summary.scalar`.
          - Adds update op to `train_ops`.
      - **Deep insight metrics**:
        - If any of `metrics.enable_kafka_metrics`, `enable_file_metrics`, `enable_deep_insight` and
          `metrics.deep_insight_sample_ratio > 0`:
          - Calls `metric_utils.write_deep_insight` with features, labels, preds, model_name, target, etc.
          - Optionally uses `dump_filename = f\"{dump_filename}.part-{worker_index:05d}\"`.
          - Adds op to collection `"deep_insight_op"`.
          - Adds `KafkaMetricHook` or `FileMetricHook` (only one of each type allowed; see `add_training_hook`).
      - **Eval path**:
        - If exporting or no `output_path`: returns `EstimatorSpec(mode, loss=loss, train_op=tf.group(train_ops), training_hooks=_training_hooks)`,
          with `pred`/`preds` added into `train_ops`.
        - Else writes outputs to file (same as predict) and returns EstimatorSpec with close hook.
      - **Train path**:
        - Determines `dense_optimizer` from `local_spec.optimizer` or `self._default_dense_optimizer`, else raises `Exception("dense_optimizer not found!")`.
        - Calls `dump_utils.add_optimizer(dense_optimizer)`.
        - Adds `feature_utils.apply_gradients_with_var_optimizer` to `train_ops` with:
          - `clip_type=ClipByGlobalNorm`, `clip_norm=self.clip_norm`, `dense_weight_decay=self.dense_weight_decay`,
            `global_step=_global_step`, `grads_and_vars_summary`, `sparse_norm_warmup_steps`,
            `is_fused_layout`, `use_allreduce=_use_dense_allreduce`.
        - Calls `add_batch_norm_into_update_ops()` then groups `UPDATE_OPS` and returns EstimatorSpec with `train_op=tf.group(train_ops)`.
  - `create_serving_input_receiver_fn()`:
    - Wraps `self.serving_input_receiver_fn()` in `MonolithDeviceCtx(INPUT_RECEIVER_FN)` and
      passes through `dump_utils.record_receiver`.
  - Abstract methods:
    - `input_fn(mode) -> DatasetV2`, `model_fn(features, mode)`, `serving_input_receiver_fn() -> ServingInputReceiver`.
  - `_export_outputs` property:
    - Graph-attached dict `__export_outputs` (created lazily).
  - `add_extra_output(name, outputs, head_name=None, head_type=None)`:
    - Adds `name` to collection `'signature_name'`.
    - If exporting: inserts `PredictOutput(outputs)` into `_export_outputs`, else ignores.
    - Raises `KeyError` if `name` already exists.
  - `add_training_hook(hook)`:
    - Prevents multiple `KafkaMetricHook` or `FileMetricHook`.
  - `add_layout(name, slice_list, out_type, shape_list)`:
    - Builds `OutConfig` with `OutType` mapping (`concat/stack/addn/none`).
    - For each slice, adds `slice_configs` with `feature_name`, `start`, `end`.
    - For each shape, writes dims; first dim forced to `-1`, subsequent dims from int or `.value`.
    - Stores in `_layout_dict[name]`.
  - `layout_dict` property getter/setter.
- **`MonolithModel(MonolithBaseModel)`**
  - `params()` adds `feature_list` string path.
  - `__init__`:
    - Uses provided params or class params.
    - Sets `dump_utils.enable = FLAGS.enable_model_dump`.
  - `_get_fs_conf(shared_name, slot, occurrence_threshold, expire_time)`:
    - Returns `FeatureSlotConfig` with `has_bias=False`, `slot_id=slot`, `occurrence_threshold`, `expire_time`,
      and hash table config using `GpucucoHashTableConfig` if `self.p.train.use_gpu_emb_table` else `CuckooHashTableConfig`.
  - `_embedding_slice_lookup(fc, slice_name, slice_dim, initializer, optimizer, compressor, learning_rate_fn, slice_list)`:
    - Asserts non-fused layout.
    - Accepts `fc` as feature name or `FeatureColumn`.
    - Applies `_share_slot_mapping` for shared embedding names.
    - Creates or reuses `FeatureSlice` in `slice_dict[feature_name][slice_name]`.
    - Appends `(fc.feature_name, fc_slice)` to `slice_list`.
    - Returns `fc.embedding_lookup(fc_slice)`.
  - `create_embedding_feature_column(feature_name, occurrence_threshold=None, expire_time=36500, max_seq_length=0, shared_name=None, combiner=None)`:
    - Converts `combiner` string to `FeatureColumn.reduce_sum/reduce_mean/first_n`.
    - Resolves `feature_name` and `slot` via `get_feature_name_and_slot`.
    - If `feature_name` exists in `fc_dict`, returns it.
    - If `shared_name` provided:
      - Stores `_share_slot_mapping[feature_name] = shared_name`.
      - Reuses existing `fs_dict` or `fc_dict` for shared slot if present.
      - Else creates new `FeatureSlot` for `shared_name` and stores in `fs_dict`.
      - If shared slot not created first and `get_feature_name_and_slot` fails, raises exception with explicit message.
    - If not shared: creates new `FeatureSlot` via `ctx.create_feature_slot`.
    - Default `combiner`: `first_n(max_seq_length)` for sequence features, else `reduce_sum`.
    - Creates `FeatureColumn`, stores in `fc_dict`, returns it.
  - `lookup_embedding_slice(features, slice_name, slice_dim=None, initializer=None, optimizer=None, compressor=None, learning_rate_fn=None, group_out_type='add_n', out_type=None)`:
    - Computes `layout_name = f'{slice_name}_{md5(sorted(features)).hexdigest()}'`.
    - If fused layout:
      - If `features` is list/tuple and `slice_dim` int and contains group tuples/lists: raises `ValueError("group pool is not support when fused_layout")`.
      - Returns `ctx.layout_factory.get_layout(layout_name)`.
    - Otherwise builds `feature_embeddings` and `slice_list` from `features` in these cases:
      - `dict`: feature name → slice dim.
      - `list/tuple` + `slice_dim` int:
        - if all elements are (str|int|FeatureColumn): fixed-dim list.
        - if all elements are list/tuple groups: `group_out_type` must be `concat` or `add_n`;
          each group is a list of feature names; group embeddings are summed or concatenated.
        - else raises `ValueError("ValueError for features")`.
      - `list/tuple` of `(feature, dim)` pairs: variable dims.
      - Otherwise raises `ValueError("ValueError for features")`.
    - If `out_type is None`: records layout with `shape_list` from embeddings and returns list of embeddings.
    - Else `out_type` in `{concat, stack, add_n, addn}`:
      - `concat`: `tf.concat(axis=1)`
      - `stack`: `tf.stack(axis=1)`
      - `add_n/addn`: `tf.add_n`
      - Records layout and returns tensor.
  - `share_slot(features=None, share_meta=None, variant_type='example', suffix='share')`:
    - For each `name -> (inplace, slot)` in `share_meta`:
      - Registers slot mapping via `register_slots`, using `shared_name = f'{name}_{suffix}'` when not inplace.
    - If `features` is dict:
      - `inplace=True`: `features[name] = switch_slot(features[name], slot)`
      - Else: `features[shared_name] = switch_slot(features[name], slot)`
      - Returns modified dict.
    - Else returns `map_fn = lambda tensor: switch_slot_batch(tensor, share_meta, variant_type=variant_type, suffix=suffix)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` for model base + training loop glue; `monolith-rs/crates/monolith-layers` for logit correction; `monolith-rs/crates/monolith-data` for feature slots; `monolith-rs/crates/monolith-hash-table` for embedding tables; `monolith-rs/crates/monolith-serving` for export signatures.
- Rust public API surface:
  - `get_sigmoid_loss_and_pred` / `get_softmax_loss_and_pred` equivalents in a `losses` or `metrics` module.
  - `DeviceCtxType` + `MonolithDeviceCtx` analog for device placement (no-op in pure Candle; required for TF runtime backend).
  - `MonolithBaseModel` trait + `MonolithModel` struct that mirror param plumbing, embedding-slice helpers,
    training/eval/predict flow, file output, and metrics hooks.
  - Export signature registry mirroring `add_extra_output`.
- Data model mapping:
  - Feature slots/slices map to Rust `FeatureSlot`, `FeatureColumn`, `FeatureSlice` types.
  - Layout dictionary maps to Rust `OutConfig` protobufs (via `monolith-proto`) with consistent shape semantics (`-1` batch dim).
- Feature gating:
  - **Default**: Candle-native backend; `MonolithDeviceCtx` becomes a no-op.
  - **Optional**: TF runtime backend only when `saved_model.pb` + `libtensorflow` + custom ops present.
  - Metrics hooks (Kafka/file/deep_insight) should be feature-gated with optional dependencies.
- Integration points:
  - Training entrypoints (Estimator analog) must call into `create_model_fn` flow equivalents.
  - Export flow must honor `export_context`-like state to control output signatures and device placement.

**Implementation Steps (Detailed)**
1. Define Rust equivalents for loss helpers with identical shape handling, sample-rate correction, clipping, and mask/weight semantics.
2. Implement `MonolithDeviceCtx` abstraction; in Candle, no-op; in TF runtime, map to device placement APIs.
3. Build Rust `MonolithBaseModel` trait:
   - Store per-graph/per-run `losses`, `training_hooks`, and `export_outputs`.
   - Implement `create_input_fn`, `create_model_fn`, `create_serving_input_receiver_fn` analogs.
4. Implement file output writer with exact formatting and ordering (output_fields order, dict pred sorted by key, delimiter handling).
5. Port metrics collection:
   - AUC/MSE metrics with logging + summary behavior (or compatible replacements).
   - Deep insight pipeline with Kafka/file hooks and per-worker filename suffixing.
6. Implement embedding feature slot + slice machinery with shared slots and layout tracking.
7. Add export signature registry and merge semantics for extra outputs.
8. Add feature flags and error handling parity for unsupported paths (e.g., fused layout group pooling).
9. Add cross-language tests: compare output file contents, metric logging events, and embedding slice layouts.

**Tests (Detailed)**
- Python tests: `monolith/native_training/model_comp_test.py` (uses `MonolithModel`).
- Rust tests: none yet (needs new parity tests for loss helpers, file output formatting, and embedding slice layouts).
- Cross-language parity test:
  - Golden test that runs a minimal TF model with two heads and compares loss/pred outputs + output file formatting.
  - Embedding slice layout test comparing `OutConfig` shape/slice configs between Python and Rust.

**Gaps / Notes**
- `_get_real_mode` rejects `EVAL`; this is likely intentional in their training flow (only TRAIN/PREDICT). If Rust adds eval mode, define parity behavior explicitly.
- Prediction file output uses **sorted dict keys** only when `pred` is dict; list/tuple order is preserved.
- `create_model_fn` treats tuple/list return as classification and emits warning; exact warning text should be preserved if exposed.
- Uses private TF API `graph._apply_device_functions` for `global_step` ops.
- Metrics include `tf.print(..., output_stream=sys.stderr)` side effects.
- Deep insight hooks only attach when sample ratio > 0.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/native_task.py`
<a id="monolith-native-training-native-task-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 213
- Purpose/role: Defines the `NativeTask` base class (TF-native training/eval/serving) and the `NativeContext` helper for feature slot creation, embedding gradient application, and async functions.
- Key symbols/classes/functions: `NativeContext`, `NativeTask`.
- External dependencies: TensorFlow, `monolith.core.base_task.BaseTask`, `monolith.core.hyperparams`,
  `monolith.native_training.feature`, `monolith.native_training.prefetch_queue`,
  `monolith.native_training.model_export.export_context.ExportMode`.
- Side effects:
  - Raises `ValueError` if both `feature_factory` and `layout_factory` are provided in `NativeContext`.
  - Delegates to async function manager (may enqueue TF ops).

**Required Behavior (Detailed)**
- **`NativeContext(feature_factory=None, async_function_mgr=None, layout_factory=None)`**
  - Stores `feature_factory`, `async_function_mgr`, `layout_factory`.
  - If both `feature_factory` and `layout_factory` are set, raises:
    - `ValueError("Cannot set feature_factory and layout_factory in the same time")`.
- **`NativeContext.create_feature_slot(config)`**
  - If `layout_factory` present, delegates to `layout_factory.create_feature_slot(config)`.
  - Else uses `feature_factory.create_feature_slot(config)`.
  - No TF ops are created by this call (per docstring).
- **`NativeContext.apply_embedding_gradients(grads_and_vars, scale=1)`**
  - If `layout_factory` present, delegates to `layout_factory.apply_gradients(grads_and_vars)`.
  - Else delegates to `feature_factory.apply_gradients(grads_and_vars, scale=scale)`.
  - Expects `grads_and_vars` from `FeatureColumn.get_all_embeddings_concatenated`.
- **`NativeContext.add_async_function(target, args=None, kwargs=None, is_async=None, queue_name="async_queue")`**
  - Delegates to `async_function_mgr.add_async_function(...)`.
  - Returns enqueue op if async, else result of `target`.
  - Semantic contract (documented): tensors used by async function should be passed via args/kwargs only.
- **`NativeTask(BaseTask, abc.ABC)`**
  - `params()` extends BaseTask params with:
    - `metrics.*`:
      - `enable_deep_insight` (default False), `deep_insight_target` ("ctr_head"),
        `deep_insight_name` (None), `deep_insight_sample_ratio` (0.01),
        `extra_fields_keys` (list),
        `enable_throughput_hook` (True),
        `enable_kafka_metrics` (False),
        `enable_tf2_profiler_hook` (True),
        `enable_file_metrics` (False),
        `file_base_name` ("/vepfs/jaguar_deepinsight_results"),
        `file_ext` ("txt"),
        `parse_fn`/`key_fn`/`layout_fn` (None),
        `dump_filename` (""),
        `use_data_service` (False).
    - `mode`: `tf.estimator.ModeKeys.TRAIN` (temporary; doc says will be removed).
    - `train.*`:
      - `max_pending_seconds_for_barrier` (30),
      - `slow_start_steps` (0),
      - `sample_bias` (0.0),
      - `use_gpu_emb_table` (False),
      - `use_fountain` (False),
      - `fountain_zk_host` (""), `fountain_model_name` (""), `fountain_parse_on_server` (False),
        `fountain_precompute_value_rowids` (False).
    - `serving.*`:
      - `export_with_gpu_allowed` (False),
      - `export_with_cleared_entry_devices` (False),
      - `export_when_saving` (False),
      - `export_dir_base` ("exported_models"),
      - `export_mode` (`ExportMode.DISTRIBUTED`),
      - `shared_embedding` (True),
      - `with_remote_gpu` (False).
  - `__init__(params)`:
    - Calls `BaseTask.__init__` and sets `self._ctx = NativeContext()` and `self.p = params`.
  - `ctx` property returns `self._ctx`.
  - Abstract methods:
    - `create_input_fn(self, mode)`
    - `create_model_fn(self)`
  - `create_serving_input_receiver_fn()`:
    - Returns `None` by default; callers must override when `serving.export_when_saving` is enabled.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (task base + context), `monolith-rs/crates/monolith-data` (feature factories/layouts).
- Rust public API surface:
  - `NativeContext` struct with `create_feature_slot`, `apply_embedding_gradients`, `add_async_function`.
  - `NativeTask` trait with `params()`, `create_input_fn`, `create_model_fn`, `create_serving_input_receiver_fn`.
- Data model mapping:
  - `feature_factory` ↔ `FeatureFactory` trait.
  - `layout_factory` ↔ `EmbeddingLayoutFactory` trait (optional, mutually exclusive).
- Feature gating:
  - Async function manager (prefetch queue) behind feature flag if not supported by backend.
  - Export-related params gated by serving feature.
- Integration points:
  - `NativeTask` becomes base for `MonolithBaseModel` in Rust; param defaults must match Python.

**Implementation Steps (Detailed)**
1. Port `NativeContext` with mutual exclusion validation and delegation to feature/layout factories.
2. Implement an async function manager interface in Rust (or stubs if backend lacks it).
3. Port `NativeTask.params()` with identical defaults and nested param groups.
4. Implement `NativeTask` base that stores `params` and `ctx`.
5. Add validation hooks so `create_serving_input_receiver_fn` must be overridden when export is enabled.
6. Add unit tests for parameter defaults and mutual exclusion errors.

**Tests (Detailed)**
- Python tests: none dedicated; covered indirectly by model/task usage.
- Rust tests: add unit tests for `NativeContext` validation and param defaults.
- Cross-language parity test: compare serialized defaults from Python vs Rust `params()` output.

**Gaps / Notes**
- `OutConfig`, `OutType`, `TensorShape` are imported but unused in this file (no runtime effect).
- `create_serving_input_receiver_fn` returning `None` is explicitly invalid when export is enabled; Rust should enforce or document this.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/native_task_context.py`
<a id="monolith-native-training-native-task-context-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 58
- Purpose/role: Defines a global per-process task context (`NativeTaskContext`) and helper functions to set/get it.
- Key symbols/classes/functions: `NativeTaskContext`, `with_ctx`, `get`.
- External dependencies: `contextlib`, `typing.NamedTuple`, `monolith.agent_service.backends.SyncBackend`.
- Side effects: Mutates module-level `_CTX` global.

**Required Behavior (Detailed)**
- **`NativeTaskContext(NamedTuple)`** with fields:
  - `num_ps: int`
  - `ps_index: int`
  - `num_workers: int`
  - `worker_index: int`
  - `model_name: str`
  - `sync_backend: SyncBackend`
  - `server_type: str`
- **`with_ctx(ctx)`** (context manager):
  - Stores previous `_CTX` as `old_ctx`, sets `_CTX = ctx`, yields.
  - On exit:
    - If `old_ctx` is not `None`, restores `_CTX = old_ctx`.
    - If `old_ctx` is `None`, leaves `_CTX` set to `ctx` (no reset to `None`).
- **`get()`**:
  - If `_CTX is None`, returns a new `NativeTaskContext` with defaults:
    - `num_ps=0`, `ps_index=0`, `num_workers=1`, `worker_index=0`,
      `server_type=""`, `model_name=""`, `sync_backend=None`.
  - Else returns `_CTX` object as-is (no copy).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (task context).
- Rust public API surface:
  - `struct NativeTaskContext { ... }`
  - `with_ctx(ctx, |...| { ... })` or RAII guard to set/restore.
  - `get_ctx()` returns current context or default.
- Data model mapping:
  - `sync_backend` should be an enum or trait object mirroring `SyncBackend`.
- Feature gating: none.
- Integration points:
  - Training/serving flows should call `get_ctx()` for worker index and model name.

**Implementation Steps (Detailed)**
1. Implement a thread-local or global context storage in Rust (match Python semantics).
2. Implement context guard that **only** restores prior context if it existed.
3. Provide `get_ctx()` returning default values when unset.
4. Add unit tests for default context and nesting behavior.

**Tests (Detailed)**
- Python tests: none dedicated.
- Rust tests: add unit tests for `with_ctx` nesting and default values.
- Cross-language parity test: compare defaults and nesting semantics.

**Gaps / Notes**
- The context manager does **not** clear `_CTX` when exiting outermost scope; Rust should match this behavior exactly.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed
