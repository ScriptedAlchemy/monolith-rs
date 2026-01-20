<!--
Source: task/request.md
Lines: 14846-14947 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
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
