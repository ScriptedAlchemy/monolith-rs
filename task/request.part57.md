<!--
Source: task/request.md
Lines: 12987-13151 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/layers/feature_cross.py`
<a id="monolith-native-training-layers-feature-cross-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 805
- Purpose/role: Collection of feature-crossing layers: GroupInt/FFM, AllInt, CDot, CAN, DCN, CIN.
- Key symbols/classes/functions: `GroupInt` (alias `FFM`), `AllInt`, `CDot`, `CAN`, `DCN`, `CIN`.
- External dependencies: TensorFlow/Keras (`Layer`, `Conv1D`, activations/initializers/regularizers), Monolith layers (`MLP`), layer utils (`merge_tensor_list`, `DCNType`, `check_dim`, `dim_size`), `layer_ops.ffm`, TF internals (`variable_ops.PartitionedVariable`, `base_layer_utils`, `K.track_variable`).
- Side effects: Creates multiple trainable weights and nested Keras layers (MLP, Conv1D); uses TF variable tracking for split variables; logs via `absl.logging` (imported).

**Required Behavior (Detailed)**
- `GroupInt` (aka `FFM`):
  - Inputs: tuple `(left_fields, right_fields)` where each is list of tensors.
  - Concats left/right along axis=1, then calls `ffm(left, right, dim_size, int_type)`.
  - `interaction_type` in `{'multiply', 'dot'}`; `use_attention` only valid for `multiply`.
  - If `use_attention`: reshape to `(bs, num_feature, emb_dim)`, run MLP to get attention `(bs, num_feature, 1)`, apply elementwise weighting; output reshaped to `(bs, num_feature * emb_dim)`.
  - Returns list `[ffm_embeddings]` if `keep_list` else tensor.
  - Config includes interaction_type, attention_units, activation, initializer, regularizer, out_type, keep_list.
- `AllInt`:
  - Inputs: `embeddings` shape `[batch, num_feat, emb_size]`.
  - Builds kernel shape `(num_feat, cmp_dim)` and optional bias `(cmp_dim,)`.
  - Call: transposes embeddings to `[batch, emb_size, num_feat]`, computes `feature_comp = transposed @ kernel` (+bias), then `interaction = embeddings @ feature_comp` to get `[batch, num_feat, cmp_dim]`.
  - Returns `merge_tensor_list(interaction, merge_type=out_type, keep_list=keep_list)`.
- `CDot`:
  - Build: stores `_num_feature`, `_emd_size`, creates `project_weight` `(num_feature, project_dim)` and `compress_tower` MLP with output dims `compress_units + [emd_size * project_dim]`.
  - Call:
    - Project input: `(bs, emb_size, num_feature) @ project_weight` → `(bs, emb_size, project_dim)`.
    - Flatten and run compress MLP → `compressed` `(bs, emb_size * project_dim)`.
    - Cross: `inputs @ reshape(compressed, (bs, emb_size, project_dim))` → `(bs, num_feature, project_dim)`, flatten to `(bs, num_feature * project_dim)`.
    - Output: `concat([crossed, compressed], axis=1)`.
- `CAN`:
  - Inputs: `(user_emb, item_emb)`.
  - `item_emb` is split into alternating weight/bias tensors for `layer_num` layers; expects size `u_emb_size*(u_emb_size+1) * layer_num`.
  - Handles four shape cases based on `is_seq` and `is_stacked`, reshaping weights/bias accordingly.
  - Applies `layer_num` iterations of `user_emb = activation(user_emb @ weight + bias)` (or linear if activation is None).
  - Output reduces/squeezes based on `is_seq/is_stacked`.
- `DCN`:
  - Supports types: `Vector`, `Matrix`, `Mixed` (from `DCNType`).
  - `Vector`: kernel shape `(dim,1)` per layer; update `xl = x0 * (xl @ w) + b + xl`.
  - `Matrix`: kernel shape `(dim,dim)` per layer; update `xl = x0 * (xl @ W + b) + xl`.
  - `Mixed`: per layer, per-expert low-rank factors `U,V,C` (dims `dim x low_rank`), gating `G` (`dim x 1`), bias; computes expert outputs and softmax-gated mixture; adds residual `+ xl`.
  - Optional `allow_kernel_norm` in `get_variable`: normalizes var (axis=0, eps=1e-6) and multiplies by trainable norm initialized with `tf.norm(var_init, axis=0)`.
  - Optional dropout during `TRAIN` mode: `tf.nn.dropout(xl, rate=1-keep_prob)`.
- `CIN`:
  - Inputs: `[batch, num_feat, emb_size]`, uses `Conv1D` per layer (`hidden_uints`).
  - For each layer: compute `zl = einsum('bdh,bdm->bdhm', xl, x0)`, reshape to `(bs, emb_size, last_hidden_dim * num_feat)`, apply Conv1D.
  - Concatenate `reduce_sum` of each layer output along emb_size: `concat([sum(hi, axis=1) ...], axis=1)`.

**Rust Mapping (Detailed)**
- Target crate/module:
  - `monolith-rs/crates/monolith-layers/src/feature_cross.rs` (GroupInt/AllInt/CDot/CAN/CIN).
  - `monolith-rs/crates/monolith-layers/src/dcn.rs` (DCN).
- Rust public API surface:
  - `GroupInt`, `AllInt`, `CDot`, `CAN`, `CIN` in `feature_cross.rs`.
  - `CrossNetwork`/`CrossLayer` in `dcn.rs` for DCN variants.
- Data model mapping:
  - Python `DCNType` → Rust `DCNMode` (vector/matrix/mixed).
  - `interaction_type` → Rust `GroupIntType`.
  - `merge_tensor_list` → Rust `merge_tensor_list` / `MergeType`.
- Feature gating: GPU-accelerated paths in Rust when `cuda`/`metal` features enabled (if used).
- Integration points: MLP, merge utils, embedding and pooling layers.

**Implementation Steps (Detailed)**
1. Verify each Python layer has a Rust counterpart with matching defaults and shapes.
2. Align `GroupInt` attention path with Python’s MLP attention (last dim must be 1).
3. Ensure `AllInt` and `CDot` matmul/reshape orderings match Python exactly.
4. Implement CAN’s `is_seq` / `is_stacked` shape logic and weight/bias splitting.
5. Map DCN modes and kernel norm behavior; match gating and mixture logic for `Mixed`.
6. Ensure CIN’s einsum/reshape/Conv1D logic matches; if Conv1D is missing, emulate with 1x1 conv via matmul.
7. Add config serialization parity for each layer (activation, initializer, regularizer, units, dims).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/feature_cross_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/feature_cross_test.rs` (new).
- Cross-language parity test:
  - Fix small inputs and compare outputs for each layer variant (GroupInt, AllInt, CDot, CAN, DCN, CIN).

**Gaps / Notes**
- Python uses TF internals for split variable tracking in DCN kernel_norm path; Rust does not have a direct analogue.
- Some layers (e.g., CDot/CIN) depend on Keras `Conv1D`; ensure Rust kernel shapes/stride/activation match.

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

### `monolith/native_training/layers/feature_cross_test.py`
<a id="monolith-native-training-layers-feature-cross-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 286
- Purpose/role: Smoke tests for feature crossing layers (GroupInt/AllInt/CDot/CAN/DCN/CIN).
- Key symbols/classes/functions: `FeatureCrossTest` methods for instantiate/serde/call per layer.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Disables v2 behavior in main guard; runs TF sessions.

**Required Behavior (Detailed)**
- GroupInt:
  - Instantiate via params and direct constructor.
  - `test_groupint_call`: left list of 5 tensors `(100,10)`, right list of 3 tensors `(100,10)`.
  - `test_groupint_attention_call`: same shapes with attention MLP.
- AllInt:
  - Instantiate/serde with `cmp_dim=4`.
  - Call on input `(100,10,10)`.
- CDot:
  - Instantiate/serde with `project_dim=8`, `compress_units=[128,256]`, `activation='tanh'`.
  - Call on input `(100,10,10)`.
- CAN:
  - Instantiate/serde with `layer_num=8`.
  - `test_can_seq_call`: user `(128,10,12,10)`, item `(128,220)`.
  - `test_can_call`: user `(128,10,10)`, item `(128,220)`.
- DCN:
  - Instantiate/serde for `dcn_type='matrix'`, `use_dropout=True`, `keep_prob=0.5`.
  - Call for vector/matrix/mixed modes; input `(128,10,10)`, kernel_norm enabled.
- CIN:
  - Instantiate/serde with `hidden_uints=[10,5]`, activation configured.
  - Call on input `(128,10,10)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/feature_cross_test.rs`.
- Rust public API surface: `GroupInt`, `AllInt`, `CDot`, `CAN`, `CIN`, `DCN` equivalents.
- Data model mapping:
  - Params-based instantiation ↔ Rust config/builder.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::feature_cross` and `monolith_layers::dcn`.

**Implementation Steps (Detailed)**
1. Add Rust tests for each layer’s constructor and config serialization.
2. Add forward tests with same input shapes as Python.
3. For DCN, include tests for vector/matrix/mixed modes with kernel_norm on.
4. For CAN, enforce item size consistency with `layer_num`/`u_emb_size`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/feature_cross_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/feature_cross_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare output sums per layer.

**Gaps / Notes**
- Python tests are smoke tests; Rust should add deterministic assertions.

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
