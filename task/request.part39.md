<!--
Source: task/request.md
Lines: 9243-9449 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/distribution_ops_fused_benchmark.py`
<a id="monolith-native-training-distribution-ops-fused-benchmark-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 61
- Purpose/role: Benchmarks `fused_reorder_by_indices` performance on large random IDs.
- Key symbols/classes/functions: `run_fused_reorder_by_indicies`.
- External dependencies: numpy, TensorFlow, `distribution_ops`.
- Side effects: none; prints average wall time.

**Required Behavior (Detailed)**
- Generates ~1e6 unique int64 IDs, 30 slots, 256 shards.
- For each slot, duplicates IDs to force duplicates and shuffles.
- Runs `distribution_ops.fused_reorder_by_indices(ids_list, num_of_shards=256)` in a session and times execution.
- Main prints average wall time over 5 runs.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/benches`.
- Rust public API surface: fused reorder benchmark.
- Data model mapping: list of id tensors, shard count.
- Feature gating: fused reorder op implementation.
- Integration points: `distribution_ops` fused reorder.

**Implementation Steps (Detailed)**
1. Implement Rust bench that generates similar random IDs and runs fused reorder.
2. Use consistent shard count and slot count for comparability.

**Tests (Detailed)**
- Python tests: this file (benchmark).
- Rust tests: bench only.
- Cross-language parity test: not required beyond output correctness.

**Gaps / Notes**
- Pure benchmark; no correctness assertions.

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

### `monolith/native_training/distribution_ops_fused_test.py`
<a id="monolith-native-training-distribution-ops-fused-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 148
- Purpose/role: Tests for `fused_reorder_by_indices` correctness and embedding offset outputs.
- Key symbols/classes/functions: `_test_fused_reorder_by_indices`, `test_fused_reorder_by_indices`, `test_ragged_tensor_workflow`.
- External dependencies: TensorFlow, `distribution_ops`.
- Side effects: none.

**Required Behavior (Detailed)**
- `test_benchmark`: runs a large random `fused_reorder_by_indices` to smoke test.
- `_test_fused_reorder_by_indices`:
  - Calls `fused_reorder_by_indices(ids_list, num_of_shards, dim_sizes)`.
  - Asserts output order, split sizes, and sharded slot sizes; optionally checks embedding offsets.
- `test_fused_reorder_by_indices`:
  - Multiple cases: single slot, extra empty slot, plus offset ids, empty slots, different shard counts, and dim_sizes for offsets.
- `test_ragged_tensor_workflow`:
  - Builds merged slot values from ragged tensors and validates fused reorder output.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: `fused_reorder_by_indices` op.
- Data model mapping: list of id tensors → reordered ids + split sizes.
- Feature gating: fused distribution ops.
- Integration points: partitioned hash table lookup pipeline.

**Implementation Steps (Detailed)**
1. Implement Rust tests mirroring each expected output case.
2. Validate embedding offsets for dim_sizes cases.
3. Include ragged workflow test for merged slots.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distribution_ops_fused_test.rs`.
- Cross-language parity test: compare outputs and offsets for fixed inputs.

**Gaps / Notes**
- Uses Python list inputs; Rust tests should use deterministic tensors.

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

### `monolith/native_training/distribution_ops_test.py`

<a id="monolith-native-training-distribution-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 536
- Purpose/role: TensorFlow test coverage for custom distribution ops (split/reorder, ragged routing, embedding gather, reduction, fused GPU ops).
- Key symbols/classes/functions: `DistributionOpsTest` test cases.
- External dependencies: `numpy`, `tensorflow`, `tensorflow.python.framework.test_util`, `random`, `monolith.native_training.distribution_ops`.
- Side effects: Requires GPU for `@test_util.run_gpu_only` tests; uses TF v1 sessions and graph mode.

**Required Behavior (Detailed)**
- `test_split_by_indices`:
  - `ids=[0,1,2,2,3]`, `indices=ids % 3`, `split_by_indices(..., num_splits=3)` -> `[[0,3],[1],[2,2]]`.
- `test_reorder_by_indices`:
  - `ids=[0,1,2,2,3,5]`, `indices=ids % 3`, `reorder_by_indices(..., num_of_shards=3)` -> `output=[3,0,1,5,2]`, `split_sizes=[2,1,2]`.
- `test_split_by_indices_gradient`:
  - Gradient of split over `tensor=[[0,0],[1,1],[2,2]]` returns all ones.
- `test_split_by_indices_empty_gradient`:
  - Empty inputs return empty gradient `[]`.
- `test_ragged_split_by_indices`:
  - Ragged `num=[[],[],[4,3,2],[1],[],[]]`, `indices=[0,1,0,1]` -> `splits` and `pos` arrays match expected nested ragged values.
- `test_unique_key_with_value_and_offset_and_fill_with_offset_map`:
  - `unique_key_with_value_and_offset` over ragged keys returns:
    - `unique_key=[[],[0,1,2],[0,1],[]]`
    - `value_offset=[[],[[0,8],[2,6],[4]],[[10,16],[13]],[]]`
  - `fill_with_offset_map` + `finalize_shared_tensor` yields `buffer=[0,1,2,3,4,5,2,3,0,1,6,7,8,9,10,11,6,7,8]`.
  - Gradient of `buffer` wrt `value` equals `[8,10,8,10,4,5,26,28,30,13,14,15]`.
- `test_fill_with_offset_map_error_case`:
  - When `value` length too small (10 vs expected 12), evaluating `filled_tensor` raises `InvalidArgumentError`.
- `test_unique_key_with_value_and_offset_empty`:
  - Empty ragged keys -> empty `unique_key`/`value_offset`.
- `test_map_id_to_embedding`:
  - Map ids `[1]`,`[2]` to embeddings `[[1,1]]`,`[[2,2]]` and input `[[1],[2]]` -> output `[[[1,1]],[[2,2]]]`.
- `test_map_id_to_embedding_multi_threads`:
  - 1k ids, 16-dim embeddings, `ps_num=10` -> multi-threaded mapping returns exact original embeddings.
- `test_map_id_to_embedding_gradient`:
  - Loss vs target `[[2,2],[2,2],[2,2]]` yields gradients `embeddings1=[[-2,-2]]`, `embeddings2=[[-1,-1]]`.
- `test_gather_embeddings_by_ids`:
  - `ids=[1,2,3]`, `embeddings=[[1,1],[2,2],[3,3]]`, input `[[2],[1],[2]]` -> output `[[[2,2]],[[1,1]],[[2,2]]]`, `index_mapping=[[1],[0],[1]]`.
- `test_gather_embeddings_by_ids_gradient`:
  - Gradient wrt embeddings equals `[[-2,-2],[-1,-1],[0,0]]`.
- `test_gather_embeddings_by_ids_gradient_back_prop`:
  - `ids=[2,3,1]`, `grads` + `index_mapping=[1,0,1,2]` -> output `[[2,2],[5,5],[8,8]]`.
- `test_fused_gather_embeddings_by_input` (GPU only):
  - Uses fused embeddings + offsets with large SCALE; expects exact outputs per slot (repeated SCALE times).
- `test_fused_gather_embeddings_by_input_gradient` (GPU only):
  - `fused_embeddings_size=22`, `embedding_dims=[3,2]`, SCALE=888 -> output length 22 and expected sums scaled; tolerance `rtol=1e-7 * SCALE`.
- `test_reduce_mean` and `test_reduce_mean_gradient`:
  - Mean reductions produce expected values; gradients are `[-1,-1]` per row.
- `test_reduce_sum` and `test_reduce_sum_gradient`:
  - Sum reductions produce expected values; gradients are `[-1,-1]` per row.
- `test_reduce_sqrtn`, `test_reduce_sqrtn_gradient`, `test_reduce_sqrtn_gradient_zero`:
  - Sqrt-N reductions and gradients match expected numeric values; zero inputs yield zero gradients.
- `test_fused_reduce_sum_and_split`:
  - CPU-only; verifies split sizes `[2,1]` and `[1,2]` for consecutive/non-consecutive indices, with zero-filled rows for gaps.
- `test_fused_reduce_sum_and_split_grad`:
  - Gradient wrt id_values is all ones.
- `test_fused_reduce_scatter` (GPU only):
  - `fused_sorted_segment_sum` matches `scatter_nd` output and gradient across multiple shapes (includes empty tensor case).
- `test_fused_reduce_and_split_gpu` (GPU only):
  - For ragged rows and many embedding lengths, outputs match scatter+split and gradients match for all outputs.
- `test_aligned_concat_split` (GPU only):
  - Random tensors round-trip through `monolith_aligned_flat_concat`/`monolith_aligned_flat_split`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf` (TF runtime adapter) + new tests.
- Rust public API surface: wrappers in `monolith-rs/crates/monolith-tf/src/distribution_ops.rs` (new) and tests in `monolith-rs/crates/monolith-tf/tests/distribution_ops_test.rs` and `monolith-rs/crates/monolith-tf/tests/distribution_ops_gpu_test.rs`.
- Data model mapping: TF tensors (dense and ragged), custom op handles, gradient support for TF runtime.
- Feature gating: `tf-runtime` feature for these tests; GPU-only tests gated on CUDA availability.
- Integration points: custom op library load (libmonolith_ops) before creating the TF graph/session.

**Implementation Steps (Detailed)**
1. Add TF runtime harness in Rust: build graph/session, load `libmonolith_ops`, wrap op invocation.
2. Implement CPU tests for `split_by_indices`, `reorder_by_indices`, `ragged_split_by_indices`, `unique_key_with_value_and_offset`, `fill_with_offset_map`, `finalize_shared_tensor`, `map_id_to_embedding`, `gather_embeddings_by_input`, and `reduce_*` ops.
3. Add gradient checks using TF gradient API; if Rust bindings do not expose gradients, run parity via Python harness and document skip in Rust.
4. Add GPU-only tests for fused gather, fused reduce scatter, fused reduce+split GPU, and aligned concat/split; skip when CUDA/custom ops missing.
5. Seed RNG or replace random tensors with deterministic values to avoid flakiness (especially `test_aligned_concat_split`).
6. Validate error handling for `fill_with_offset_map` with invalid input sizes (InvalidArgumentError).
7. Document Candle backend deviations: these ops require TF custom kernels and are only supported under the TF runtime feature.

**Tests (Detailed)**
- Python tests: `monolith/native_training/distribution_ops_test.py`.
- Rust tests: `monolith-rs/crates/monolith-tf/tests/distribution_ops_test.rs` (CPU) and `monolith-rs/crates/monolith-tf/tests/distribution_ops_gpu_test.rs` (GPU).
- Cross-language parity test: run Python and Rust TF tests on identical inputs; compare tensors within tolerance and verify gradients.

**Gaps / Notes**
- Requires TF custom ops build + dynamic loading; without this, Rust tests must be skipped.
- GPU tests are sensitive to CUDA availability and may need CI skips.

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
