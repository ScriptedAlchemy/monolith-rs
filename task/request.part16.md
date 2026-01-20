<!--
Source: task/request.md
Lines: 4120-4351 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/core/mixed_emb_op_comb_nws.py`
<a id="monolith-core-mixed-emb-op-comb-nws-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 421
- Purpose/role: Keras layer for neural width search (NWS) over embedding sizes, with optional teacher distillation transform.
- Key symbols/classes/functions: `TeacherEmbeddingTransform`, `MixedEmbedOpComb`.
- External dependencies: `numpy`, `tensorflow`, `tensorflow.keras.layers.Layer/InputSpec`.
- Side effects: creates trainable TF variables, prints to stdout, uses TF random sampling.

**Required Behavior (Detailed)**
- `TeacherEmbeddingTransform(max_choice_per_embedding, teacher_embedding_sizes_list)`:
  - Stores arrays; asserts same length.
  - `build(input_shape)`:
    - Input is 2D; last dim equals sum(teacher sizes).
    - Creates `teacher_embedding_transform_weight` with shape `[sum(max_choice * size), 1]` initialized with `TruncatedNormal(stddev=0.15)`.
    - Creates `teacher_embedding_transform_bias` with shape `[sum(max_choice)]` initialized with zeros.
    - Calls `_snapshot_for_serving` on both (method not defined in this class).
  - `call(inputs)`:
    - Slices teacher embedding per size; for each slice, reshapes weight to `[size, max_choice]` and matmul.
    - Concats results along axis 1 and adds bias.
  - `compute_output_shape` raises `NotImplementedError`.
  - `get_config` returns max choices + teacher sizes.
- `MixedEmbedOpComb(slot_names, embedding_size_choices_list, warmup_steps, pretraining_steps, teacher_embedding_sizes_list=None, distillation_mask=False)`:
  - Asserts slot names length matches choices list; prints lengths.
  - Computes per-slot num choices, per-slot max embedding choice (sum of sizes), total embedding size, max num choices.
  - **Note:** `teacher_embedding_sizes_list` is ignored (set to `None`), so teacher path is disabled.
  - `build(input_shape)`:
    - Asserts input is 2D; verifies total input dim matches total embedding size.
    - Creates `arch_embedding_weights` variable of length sum(num_choices), init uniform(-1e-3, 1e-3).
    - For each slot:
      - Softmax over weights slice; compute entropy (softmax_cross_entropy_with_logits_v2).
      - Compute expected embedding dims as weighted sum of choice sizes.
      - If first choice size is 0, scale its prob by warmup schedule based on global step.
      - Create per-choice masks (ranges over max_emb_choice), pad to max_num_choices.
      - Pretraining: if global_step < pretraining_steps, probability hard-coded to `[0.5, 0.5]` (assumes 2 choices).
      - Sample choice via `tf.random.categorical`, one-hot, select mask.
      - Apply straight-through trick: mask * (1 + w - stop_gradient(w)).
    - Concatenates per-slot masks into `_arch_embedding_masks_multipler`.
    - Stores `_arch_entropy`, `_expected_emb_dims`, `_expected_zero_embedding_size_weights`,
      `_arch_embedding_weights_after_softmax_list`.
  - `call(inputs)`:
    - Computes `masked_embedding = embedding * _arch_embedding_masks_multipler`.
    - If teacher path active:
      - Builds `TeacherEmbeddingTransform`, transforms teacher embedding.
      - Computes distillation MSE against masked embedding (optionally with mask).
      - Returns `(mixed_embedding, distillation_loss, teacher_embedding_transform.name)` but `mixed_embedding` is undefined (bug).
    - Else returns `masked_embedding`.
  - `get_config` includes `slot_names`, `embedding_size_choices_list`, `warmup_steps`, `teacher_embedding_sizes_list` (pretraining_steps and distillation flag omitted).
  - `get_arch_embedding_weights()` returns variable; `get_summaries()` returns entropy, expected dims, and weight list.
- Determinism: sampling uses TF RNG; dependent on global step and random categorical.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src` (no existing equivalent).
- Rust public API surface: none; would need a layer module for NWS + distillation.
- Data model mapping: Keras layers + TF ops → Rust tensor ops (Candle/TF runtime).
- Feature gating: requires TF runtime or compatible tensor ops.
- Integration points: embedding layer selection / search in training pipeline.

**Implementation Steps (Detailed)**
1. Decide whether to support NWS (sampling-based) in Rust backend.
2. If yes, implement mask sampling, expected dims, and entropy summaries.
3. Add teacher distillation transform and MSE loss path.
4. Ensure global step and warmup/pretraining schedules are wired.
5. Fix Python bugs when porting (teacher path, mixed_embedding).

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: add unit tests for mask construction, sampling shapes, and expected dims.
- Cross-language parity test: compare mask selection behavior under fixed RNG seeds.

**Gaps / Notes**
- `teacher_embedding_sizes_list` is ignored in `__init__` (always `None`).
- `call()` returns `mixed_embedding` in teacher path, but it is never defined.
- `pretraining_steps` only supports 2 choices (`[0.5, 0.5]` hard-coded).
- `_snapshot_for_serving` is called but not defined on `Layer` (likely missing mixin).

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

### `monolith/core/model.py`
<a id="monolith-core-model-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 320
- Purpose/role: Deprecated Sail-like TPU Model wrapper for configuring TPU embedding tables, input pipelines, and pooling.
- Key symbols/classes/functions: `Model`, `create_input_fn`, `create_feature_and_table_config_dict`, `sum_pooling`, `init_slot_to_dims`.
- External dependencies: `tensorflow`, `tpu_embedding`, `absl.logging`, `math`, `FeatureSlot/FeatureColumnV1/Env`.
- Side effects: reads vocab file, logs size info, builds TF datasets/TPU table configs.

**Required Behavior (Detailed)**
- `Model.__init__(params)`:
  - Reads `vocab_size_per_slot`, logs if fixed.
  - Builds vocab dict from file; constructs `Env` and runs `init_slot_to_dims()`.
- `_create_vocab_dict(file_path, vocab_size_per_slot=None)`:
  - Reads TSV with `slot_id<TAB>count`.
  - Skips non-digit slot IDs; uses fixed vocab size if provided.
  - Returns `{slot_id: vocab_size}`.
- `_get_feature_map()`: abstract; must return TF parse feature map.
- `_post_process_example(example)`:
  - For each slot in `env.slot_to_dims`:
    - If `vocab_size_per_slot` set, mod values in `slot_{id}_0`.
    - Duplicates `slot_{id}_0` into `slot_{id}_{i}` for each additional dim.
- `create_input_fn(file_pattern, repeat=True)`:
  - Returns `input_fn(params)` using TF Dataset:
    - `list_files(shuffle=False)`, shard by `context.current_input_fn_deployment()`.
    - Optional per-shard skip from `params["shard_skip_file_number"]`.
    - `interleave(TFRecordDataset, cycle_length, num_parallel_calls, deterministic=False)`.
    - `batch(drop_remainder=True).map(parse_example, num_parallel_calls=AUTOTUNE, deterministic=False)`.
    - `repeat()` if requested; `prefetch(AUTOTUNE)`.
- `_padding_8(dim)`:
  - Rounds up to multiple of 8.
- `_get_slot_number(optimizer, use_gradient_accumulation)`:
  - Maps TPU optimizer class → slot count:
    - FTRL: 3, Adagrad: 2, Adam: 3, SGD: 1; adds 1 if gradient accumulation.
  - Else asserts unsupported optimizer (assert uses truthy string; ineffective).
- `_get_max_slot_number()`:
  - Iterates env slots and dims; chooses bias vs vec optimizer per dim; returns max slot count.
- `create_feature_and_table_config_dict()`:
  - Requires `env.is_finalized()`.
  - For each slot/dim: builds `tpu_embedding.TableConfig` and `FeatureConfig`.
  - Computes and logs embedding table sizes (raw, padded-to-8, padded+max-slot).
  - Returns `(feature_to_config_dict, table_to_config_dict)`.
- `sum_pooling(fc_dict, input_map, features, dim, total_embeddings, add_into_embeddings=True)`:
  - For each slot in `features`, calls `fc_dict[slot].add_vector(dim)`, appends to totals, populates `input_map` with unique keys.
  - Returns single embedding if one feature; else `tf.add_n`.
- `logits_fn()`, `create_model_fn()` are abstract.
- `init_slot_to_dims()` calls `logits_fn()`, `env.finalize()`, logs slot dims.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (no direct equivalent).
- Rust public API surface: none; core training uses different abstractions.
- Data model mapping: TF TPU embedding configs have no Rust analog.
- Feature gating: requires TF runtime backend.
- Integration points: input pipelines, embedding table config, TPU training loop.

**Implementation Steps (Detailed)**
1. Decide whether to port deprecated Model API or replace with `base_embedding_task` parity.
2. If ported, implement vocab dict parsing, dataset sharding, and TFRecord pipeline equivalents.
3. Add TPU embedding config generator or document unsupported feature in Rust.
4. Implement pooling helper and slot/embedding table sizing logs.

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: none (integration tests would be required with TF runtime).
- Cross-language parity test: compare table config dicts and size calculations.

**Gaps / Notes**
- File is marked deprecated; likely superseded by `base_embedding_task.py`.
- `Env` constructor here omits `params` argument (signature mismatch with current `Env`).
- `_get_slot_number` uses `assert("message")` which never fails; should raise.
- Depends on `env.slot_to_dims` and `env.slot_to_config` which do not exist in current `Env` implementation.

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

### `monolith/core/model_imports.py`
<a id="monolith-core-model-imports-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 104
- Purpose/role: Dynamic import helpers for task/model parameter modules.
- Key symbols/classes/functions: `_Import`, `ImportAllParams`, `ImportParams`.
- External dependencies: `importlib`, `absl.logging`.
- Side effects: imports Python modules dynamically; logs import results.

**Required Behavior (Detailed)**
- `_Import(name)`:
  - Logs attempt; imports module; logs success; returns True on success.
  - On ImportError, logs error and returns False.
- `ImportAllParams(task_root=_ROOT, task_dirs=_DIRS, require_success=False)`:
  - For each `task` in `task_dirs`, attempts to import `{task_root}.{task}.params.params`.
  - If `require_success` and nothing imported, raises `LookupError`.
  - Note: code defines `module_str` with `path` but `path` is undefined (bug); actual import uses `.params.params`.
- `ImportParams(model_name, task_root, task_dirs, require_success=True)`:
  - Expects `model_name` to contain a dot; else ValueError.
  - Extracts `model_module` and attempts to import it directly.
  - For built-in tasks, if `model_module` starts with `{task}.`, builds module path `{task_root}.{task}.params.{path}` and attempts import.
  - If `require_success` and no import succeeded, raises LookupError with guidance.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/model_imports.rs`.
- Rust public API surface: helper functions to load/register model param types (likely via registry rather than dynamic import).

**Implementation Steps (Detailed)**
1. Provide a Rust registry-based import mechanism (explicit registration, no dynamic import).
2. Mirror error messages and logging at call sites.
3. Document that Python dynamic import is replaced by static registration.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: verify registry lookup error messages when missing.

**Gaps / Notes**
- Python uses dynamic import; Rust should use explicit registration or plugin loading if required.

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
