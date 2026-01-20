<!--
Source: task/request.md
Lines: 10442-10668 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/feature_utils.py`
<a id="monolith-native-training-feature-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 419
- Purpose/role: Applies gradients to dense variables and embedding tables with optional clipping, Horovod/BytePS allreduce, and async embedding updates.
- Key symbols/classes/functions: `allreduce_cond`, `GradClipType`, `_gen_norm_warmup`, `apply_gradients_with_var_optimizer`, `apply_gradients`.
- External dependencies: TensorFlow, `clip_ops`, `distribution_ops.gen_distribution_ops`, `device_utils`, `feature`, `NativeContext`, Horovod/BytePS (optional).
- Side effects: Reads env vars, performs allreduce, writes TF summaries, updates global step, mutates globals `control_ops` and `dense_opt_ops`.

**Required Behavior (Detailed)**
- Env flags read at import time:
  - `MONOLITH_WITH_HOROVOD`, `MONOLITH_WITH_BYTEPS`, `MONOLITH_WITH_BYTEPS_ALLREDUCE`, `MONOLITH_WITH_ALLREDUCE_FUSION`, `MONOLITH_WITH_ALLREDUCE_FP16`, `MONOLITH_SKIP_ALLREDUCE`.
  - If Horovod enabled, imports `horovod.tensorflow` and compression classes.
- `allreduce_cond(grads, scale=1)`:
  - Selects BytePS or Horovod compression (FP16 vs None) based on envs.
  - Filters `None` grads, allreduces only non-None grads, then maps results back into original positions.
  - Fusion modes:
    - `one`: uses `monolith_aligned_flat_concat` + allreduce + `monolith_aligned_flat_split`.
    - `grouped`: uses `hvd.grouped_allreduce` (not supported with BytePS).
    - `multi`: raises `RuntimeError` (dropped).
    - default: allreduces each grad individually with Average op.
- `GradClipType` enum: `ClipByNorm`, `ClipByGlobalNorm`, `ClipByValue`, `ClipByDenseAndSparse`, `NoClip`.
- `_gen_norm_warmup(clip_norm, global_step_var, warmup_step)`:
  - Returns `clip_norm` scaled linearly from 0 to 1 over `warmup_step` using `tf.cond`.
- `apply_gradients_with_var_optimizer(...)`:
  - Computes grads for dense variables + embedding tensors.
  - For fused layout, replaces missing grads with zeros.
  - Splits dense vs sparse grads and optionally applies UE conditional gradient check.
  - Supports clip by global norm (dense/sparse), value, or per-tensor norm; optional sparse warmup.
  - Defers global norm clipping to a scale factor when using GPU + allreduce (fused with later kernels).
  - Optionally writes gradient/variable histograms and norms to summaries.
  - Dense grads optionally allreduced and L2 weight-decayed.
  - Applies dense grads via custom per-variable optimizer or shared `var_opt` (async via `ctx.add_async_function`).
  - Applies embedding grads via `ctx.apply_embedding_gradients` (on CPU) with optional scale.
  - Increments `global_step` after optimize ops with control dependencies.
- `apply_gradients(...)`:
  - Similar flow for layout-based embeddings (`ctx.layout_factory.flattened_layout()`) with simpler clipping logic.
  - If no dense variables, still increments global_step.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/feature_utils.rs` (new) and `monolith-rs/crates/monolith-optimizer` for optimizer integration.
- Rust public API surface: gradient application helpers for dense + embedding params, clip modes, and allreduce hooks.
- Data model mapping: Candle tensors for dense grads; embedding grads routed through hash table/update API.
- Feature gating: Horovod/BytePS allreduce under `tf-runtime` or `distributed` feature; default backend uses local grads only.
- Integration points: `NativeContext`, `EmbeddingLayoutFactory`, async function manager, and training loop.

**Implementation Steps (Detailed)**
1. Implement `GradClipType` enum and clipping helpers in Rust.
2. Implement global norm computation and optional warmup scaling.
3. Implement dense vs sparse gradient separation (embedding tensors tracked separately).
4. Add optional allreduce hooks (no-op when disabled) and fusion strategy `one` if TF runtime is enabled.
5. Add weight decay for dense grads.
6. Wire into `NativeContext.apply_embedding_gradients` equivalent and async scheduling.
7. Add summary/logging equivalents where possible.

**Tests (Detailed)**
- Python tests: `monolith/native_training/feature_utils_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/feature_utils_test.rs` (new).
- Cross-language parity test: verify gradient updates and global_step increments on identical toy graphs.

**Gaps / Notes**
- Fusion path depends on custom TF ops (`monolith_aligned_flat_concat/split`).
- UE gradient check logic depends on feature tensors and names; requires parity in Rust model representation.

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

### `monolith/native_training/feature_utils_test.py`
<a id="monolith-native-training-feature-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 144
- Purpose/role: Tests gradient application for dense vars and embeddings, including fused allreduce path and async embedding push.
- Key symbols/classes/functions: `_setup_test_embedding`, `FeatureUtilsTest` cases.
- External dependencies: TensorFlow, `feature_utils`, `feature`, `embedding_combiners`, `NativeContext`, `prefetch_queue`.
- Side effects: Sets env `MONOLITH_WITH_ALLREDUCE_FUSION=one`.

**Required Behavior (Detailed)**
- `_setup_test_embedding(is_async=False)`:
  - Builds embedding var and ragged ids, creates embedding slices via `create_embedding_slices`.
  - Mocks `feature_factory.apply_gradients` to subtract gradients from embedding vars.
  - Returns `(ctx, fc, emb_var, emb)`.
- `test_apply_gradients_with_dense_optimizer`:
  - Loss includes dense var and embedding sum; clip_norm=1.0.
  - After one step: dense var becomes 0.5; embedding var becomes `[0.5,0.5,0.5,1.0]`; global_step=1.
- `test_apply_gradients_with_dense_optimizer_gpu` (GPU-only):
  - Same expectations with `use_allreduce=True` and no summary; tests deferred clip fusion path.
- `test_apply_gradients_with_dense_optimizer_post_push`:
  - Async embedding push enabled; running op three times triggers two async pushes.
  - Dense var becomes -1.0; embedding var becomes `[-2.0,-2.0,-2.0,1.0]`.
- `test_apply_gradients_without_dense_optimizer`:
  - Loss uses embeddings only; after step, embedding var becomes `[0.0,0.0,0.0,1.0]` and global_step=1.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/feature_utils_test.rs` (new).
- Rust public API surface: gradient application helpers and async embedding push hooks.
- Feature gating: GPU tests behind `cuda`/`tf-runtime` feature; skip if unavailable.
- Integration points: `NativeContext` equivalent and embedding update interface.

**Implementation Steps (Detailed)**
1. Port `_setup_test_embedding` logic to create a small embedding table and feature column in Rust.
2. Implement tests for dense+embedding gradients with clipping and global_step increments.
3. Add GPU test for deferred clip + allreduce path (skip if no GPU).
4. Add async embedding push test verifying delayed updates.

**Tests (Detailed)**
- Python tests: `monolith/native_training/feature_utils_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/feature_utils_test.rs`.
- Cross-language parity test: compare updated dense var and embedding values after a single step.

**Gaps / Notes**
- Tests assume deterministic gradients and initial values; Rust must mirror initialization and scaling exactly.

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

### `monolith/native_training/gen_seq_mask.py`
<a id="monolith-native-training-gen-seq-mask-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 26
- Purpose/role: Wrapper around custom op to generate sequence masks from row splits.
- Key symbols/classes/functions: `gen_seq_mask`.
- External dependencies: TensorFlow, `gen_monolith_ops`.
- Side effects: None.

**Required Behavior (Detailed)**
- Accepts `splits` as Tensor or RaggedTensor; uses `row_splits()` when ragged.
- Calls `ops.gen_seq_mask(splits=..., max_seq_length=...)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src/gen_seq_mask.rs` (new).
- Rust public API surface: `gen_seq_mask` wrapper.
- Feature gating: TF runtime + custom ops.

**Implementation Steps (Detailed)**
1. Add binding for `gen_seq_mask` custom op.
2. Accept either row_splits tensor or ragged wrapper.

**Tests (Detailed)**
- Python tests: `gen_seq_mask_test.py`.
- Rust tests: `monolith-rs/crates/monolith-tf/tests/gen_seq_mask_test.rs`.
- Cross-language parity test: compare masks for fixed splits.

**Gaps / Notes**
- Requires custom ops library.

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

### `monolith/native_training/gen_seq_mask_test.py`
<a id="monolith-native-training-gen-seq-mask-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 42
- Purpose/role: Tests gen_seq_mask for int32 and int64 splits.
- Key symbols/classes/functions: `GenSeqMaskTest`.
- External dependencies: TensorFlow, `gen_seq_mask`.
- Side effects: None.

**Required Behavior (Detailed)**
- For splits `[0,5,7,9,13]` and `max_seq_length=6`, mask equals expected matrix for both int32 and int64.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/tests/gen_seq_mask_test.rs`.
- Rust public API surface: gen_seq_mask wrapper.
- Feature gating: TF runtime + custom ops.

**Implementation Steps (Detailed)**
1. Add tests for int32 and int64 splits with expected outputs.

**Tests (Detailed)**
- Python tests: `gen_seq_mask_test.py`.
- Rust tests: `monolith-rs/crates/monolith-tf/tests/gen_seq_mask_test.rs`.
- Cross-language parity test: compare masks.

**Gaps / Notes**
- None.

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
