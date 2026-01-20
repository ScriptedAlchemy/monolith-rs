<!--
Source: task/request.md
Lines: 13152-13435 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/layers/feature_seq.py`
<a id="monolith-native-training-layers-feature-seq-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 361
- Purpose/role: Sequence feature models: DIN attention, DIEN interest evolution, DMR_U2I sequence matching.
- Key symbols/classes/functions: `DIN`, `DIEN`, `DMR_U2I`.
- External dependencies: TensorFlow/Keras (`Layer`, `Dense`, `GRUCell`, activations/initializers/regularizers), Monolith layers (`MLP`, `AGRUCell`, `dynamic_rnn_with_attention`), `monolith_export`, `with_params`.
- Side effects: Adds nested layer weights/regularization losses; uses TF summary in DIN when mask provided.

**Required Behavior (Detailed)**
- `DIN`:
  - Inputs: `queries` `[B,H]`, `keys` `[B,T,H]`; optional `mask` in kwargs.
  - Builds MLP (`dense_tower`) with `hidden_units` (last dim must be 1).
  - Call:
    - Tile `queries` to `[B,T,H]`.
    - `din_all = concat([q, k, q-k, q*k], axis=-1)` -> `[B,T,4H]`.
    - `attention_weight = dense_tower(din_all)` -> `[B,T,1]`.
    - If `decay`, divide by `sqrt(H)`.
    - If `mask`: zero out masked positions and emit summary histogram `{name}_attention_outputs`.
    - If `mode == 'sum'`: `attention_out = matmul(attention_weight, keys, transpose_a=True)` -> `[B,1,H]`, squeeze to `[B,H]`.
    - Else: elementwise `keys * attention_weight` -> `[B,T,H]`.
- `DIEN`:
  - Builds:
    - GRUCell for interest extraction (`gru_cell`).
    - AGRUCell (`augru_cell`) for interest evolution (note: `att_type` argument exists but build hard-codes `att_type='AGRU'`).
    - Attention weight matrix `weight` `(num_units, num_units)`.
  - `_attention(queries, keys)`:
    - `query_weight = matmul(queries, weight, transpose_b=True)` reshape to `[B, H, 1]`.
    - `logit = squeeze(matmul(keys, query_weight))` -> `[B,T]`.
    - `softmax(logit)` returns attention scores.
  - `call`:
    - Accepts `queries` and `keys` from args/kwargs (mask optional but unused).
    - `outputs = dynamic_rnn(gru_cell, keys)` -> `[B,T,H]`.
    - `attn_scores = _attention(queries, outputs)` -> `[B,T]`.
    - `_, final_state = dynamic_rnn_with_attention(augru_cell, outputs, attn_scores)`.
    - Returns `final_state` `[B,H]`.
- `DMR_U2I`:
  - Build: `pos_emb (seq_len, cmp_dim)`, `emb_weight (ue_size, cmp_dim)`, `z_weight (cmp_dim,1)`, `bias (cmp_dim)`, `linear Dense` to `ie_size`.
  - Call:
    - `emb_cmp = user_seq @ emb_weight`.
    - `comped = pos_emb + emb_cmp + bias`.
    - `alpha = softmax(comped @ z_weight, axis=1)` -> `[B,seq_len,1]`.
    - `user_seq_merged = squeeze(transpose(user_seq) @ alpha)` -> `[B, ue_size]`.
    - `user_seq_merged = linear(user_seq_merged)` -> `[B, ie_size]`.
    - Output `user_seq_merged * items` (elementwise).

**Rust Mapping (Detailed)**
- Target crate/module:
  - `monolith-rs/crates/monolith-layers/src/din.rs` (DIN).
  - `monolith-rs/crates/monolith-layers/src/dien.rs` (DIEN).
  - `monolith-rs/crates/monolith-layers/src/dmr.rs` (DMR_U2I).
- Rust public API surface:
  - `DINAttention`/`DINConfig`, `DIENLayer`/`DIENConfig`, `DMRU2I`.
- Data model mapping:
  - Python `mode` (`sum` vs elementwise) → Rust `DINOutputMode`.
  - Activation strings → Rust `ActivationType`.
  - AGRU/GRU cell configs → Rust GRU/AUGRU implementations.
- Feature gating: None.
- Integration points: AGRU, MLP, Dense, activation registry.

**Implementation Steps (Detailed)**
1. Verify DIN attention math and mask handling match Python (including decay scaling and summary logging).
2. Ensure DIEN uses the same attention formula; decide whether to respect Python’s `att_type` parameter (currently ignored).
3. Align DIEN to use AGRU vs AUGRU to match Python behavior.
4. Implement DMR_U2I using Dense + activation as in Python, including position embeddings.
5. Add config serialization for all three layers.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/feature_seq_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/feature_seq_test.rs` (new).
- Cross-language parity test:
  - Fixed inputs and weights; compare outputs for DIN (sum/elementwise), DIEN, and DMR_U2I.

**Gaps / Notes**
- DIEN’s `att_type` argument is not used in build (hard-coded `'AGRU'`); decide whether to mirror this bug or fix with a flag.

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

### `monolith/native_training/layers/feature_seq_test.py`
<a id="monolith-native-training-layers-feature-seq-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 126
- Purpose/role: Smoke tests for DIN, DIEN, and DMR_U2I layers.
- Key symbols/classes/functions: `FeatureSeqTest` methods for instantiate/serde/call.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Runs TF sessions; eager disabled in main guard.

**Required Behavior (Detailed)**
- DIN:
  - Instantiate via params and direct constructor (`hidden_units=[10,1]`).
  - `test_din_call`: query `(100,10)`, keys `(100,15,10)`.
- DIEN:
  - Instantiate via params and direct constructor (`num_units=10`).
  - `test_dien_call`: query `(100,10)`, keys `(100,15,10)`.
- DMR_U2I:
  - Instantiate via params and direct constructor (`cmp_dim=10`, `activation='relu'`).
  - `test_dmr_call`: query `(100,10)`, keys `(100,15,10)`.
- All tests compute sum of outputs and run session initialization.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/feature_seq_test.rs`.
- Rust public API surface: `DINAttention`/`DINConfig`, `DIENLayer`, `DMRU2I`.
- Data model mapping:
  - Params-based instantiation ↔ Rust config/builder.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::din`, `monolith_layers::dien`, `monolith_layers::dmr`.

**Implementation Steps (Detailed)**
1. Add Rust tests for constructor and config serialization for each layer.
2. Add forward tests with the same input shapes.
3. Add deterministic assertions (output shapes and sums) for parity.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/feature_seq_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/feature_seq_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare output sums for DIN, DIEN, DMR_U2I.

**Gaps / Notes**
- Python tests do not assert numeric values; Rust should add explicit assertions.

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

### `monolith/native_training/layers/feature_trans.py`
<a id="monolith-native-training-layers-feature-trans-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 340
- Purpose/role: Feature transformation layers: AutoInt (self-attention), iRazor (feature/embedding dimension selection), SeNet (feature re-weighting).
- Key symbols/classes/functions: `AutoInt`, `iRazor`, `SeNet`.
- External dependencies: TensorFlow/Keras (`Layer`, `InputSpec`, initializers/regularizers), Monolith (`MLP`, `add_layer_loss`, `merge_tensor_list`, `with_params`, `check_dim`/`dim_size`).
- Side effects: Emits TF summary histogram for iRazor NAS weights; adds auxiliary loss via `add_layer_loss`.

**Required Behavior (Detailed)**
- `AutoInt`:
  - Input shape `[B, num_feat, emb_dim]` (3D).
  - For `layer_num` iterations:
    - `attn = softmax(autoint_input @ autoint_input^T)` → `[B, num_feat, num_feat]`.
    - `autoint_input = attn @ autoint_input` → `[B, num_feat, emb_dim]`.
  - Output via `merge_tensor_list` with `out_type` and `keep_list`.
- `iRazor`:
  - `nas_space` defines embedding dimension groups; `rigid_masks` is a constant `[nas_len, emb_size]` with grouped 1s.
  - Build: `nas_logits` weight shape `(num_feat, nas_len)`.
  - Call:
    - `nas_weight = softmax(nas_logits / t)`; histogram summary `"nas_weight"`.
    - `soft_masks = nas_weight @ rigid_masks` → `[num_feat, emb_size]`.
    - If `feature_weight` provided, compute `nas_loss = feature_weight @ sum(soft_masks, axis=1)` and call `add_layer_loss`.
    - `out_embeds = embeds * soft_masks`.
    - Return `merge_tensor_list(out_embeds, out_type, keep_list)`.
- `SeNet`:
  - If inputs is a tensor `[B, num_feat, emb_dim]`: `sequeeze_embedding = reduce_mean(inputs, axis=2)`.
  - If inputs is list of tensors:
    - `on_gpu=True`: use `segment_sum` on concatenated embeddings and lens to compute means.
    - Else: `sequeeze_embedding = concat([reduce_mean(embed, axis=1)] ...)`.
  - `cmp_tower` MLP outputs feature weights of shape `[B, num_feat]`.
  - For tensor input: reshape weights to `[B, num_feat, 1]` and multiply.
  - For list input: split weights and multiply per tensor.
  - Output via `merge_tensor_list` with `num_feature`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/feature_trans.rs` (AutoInt, IRazor, SeNet).
- Rust public API surface:
  - `AutoInt` + config, `IRazor`, `SENetLayer` (if present) or equivalent.
- Data model mapping:
  - `out_type` → `MergeType`.
  - `feature_weight` auxiliary loss → Rust loss registry or explicit return.
- Feature gating: None; GPU optimization optional.
- Integration points: merge utils, MLP, loss aggregation.

**Implementation Steps (Detailed)**
1. Align AutoInt attention math and merge semantics.
2. Implement iRazor rigid/soft mask logic and optional feature-weighted auxiliary loss.
3. Implement SeNet for both tensor and list inputs; include `on_gpu` optimization if possible.
4. Add config serialization for each layer (including `nas_space`, `t`, `cmp_dim`, `num_feature`).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/feature_trans_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/feature_trans_test.rs` (new).
- Cross-language parity test:
  - Fix embeddings and weights; compare outputs for AutoInt, iRazor, SeNet.

**Gaps / Notes**
- Python iRazor uses `add_layer_loss` to register aux loss; Rust needs an equivalent or explicit API.

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

### `monolith/native_training/layers/feature_trans_test.py`
<a id="monolith-native-training-layers-feature-trans-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 140
- Purpose/role: Smoke tests for AutoInt, SeNet, and iRazor layers.
- Key symbols/classes/functions: `FeatureTransTest` methods for instantiate/serde/call.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Runs TF session after variable init.

**Required Behavior (Detailed)**
- AutoInt:
  - Instantiate via params and direct constructor (`layer_num=1`).
  - `test_autoint_call`: input `(100,10,10)`, `layer_num=2`.
- SeNet:
  - Instantiate/serde with `num_feature=10`, `cmp_dim=4`, custom initializers.
  - `test_senet_call`: input `(100,10,10)`.
- iRazor:
  - Instantiate/serde with `nas_space=[0,2,5,7,10]`, `t=0.08`.
  - `test_irazor_call`: input `(100,10,10)`.
- All tests sum outputs and run session initialization.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/feature_trans_test.rs`.
- Rust public API surface: `AutoInt`, `IRazor`, `SeNet` equivalents.
- Data model mapping:
  - Params-based instantiation ↔ Rust config/builder.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::feature_trans`.

**Implementation Steps (Detailed)**
1. Add Rust tests for constructor and config serialization for each layer.
2. Add forward tests with the same input shapes.
3. Add deterministic assertions on output shapes/sums.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/feature_trans_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/feature_trans_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare output sums for AutoInt, SeNet, iRazor.

**Gaps / Notes**
- Python tests are smoke tests without numeric assertions; Rust should add explicit checks.

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
