<!--
Source: task/request.md
Lines: 14948-15144 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
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
