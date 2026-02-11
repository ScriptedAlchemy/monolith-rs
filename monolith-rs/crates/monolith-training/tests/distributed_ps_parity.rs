use monolith_training::{aggregate_gradients, dedup_ids, get_shard_for_id, route_to_shards};

#[test]
fn lookup_dedup_mapping_matches_python_tf_unique_semantics() {
    // Mirrors `distributed_ps_test.py::test_lookup_dedup` (dedup + remap).
    let ids = vec![1_i64, 1, 3];
    let (unique, mapping) = dedup_ids(&ids);
    assert_eq!(unique, vec![1, 3]);
    assert_eq!(mapping, vec![0, 0, 1]);
}

#[test]
fn apply_gradients_aggregates_duplicates_like_python_map_backprop() {
    // Mirrors `distributed_ps_test.py::test_apply_gradients_with_duplicates`.
    // Python path dedups ids then sums grads for duplicates.
    let dim = 1;
    let ids = vec![0_i64, 3, 0, 1];
    // grads are ones (as produced by loss = 2 * values with initial zeros): [2,2,2,2] here,
    // but only relative sums matter.
    let grads = vec![2.0_f32, 2.0, 2.0, 2.0];
    let (unique_ids, agg) = aggregate_gradients(&ids, &grads, dim);
    assert_eq!(unique_ids, vec![0, 3, 1]);
    assert_eq!(agg, vec![4.0, 2.0, 2.0]);
}

#[test]
fn shard_routing_uses_floormod_for_negatives() {
    // Python uses `tf.math.floormod`. This differs from `%` for negatives.
    assert_eq!(get_shard_for_id(-1, 3), 2);

    let routed = route_to_shards(&[-1_i64, 0, 1, 2], 3);
    assert_eq!(routed[0], vec![0]);
    assert_eq!(routed[1], vec![1]);
    assert_eq!(routed[2], vec![-1, 2]);
}
