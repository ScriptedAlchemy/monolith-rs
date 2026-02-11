use monolith_proto::monolith::native_training::monolith_checkpoint_state::HashTableType;
use monolith_proto::monolith::native_training::MonolithCheckpointState;
use monolith_training::native_training::save_utils::{
    get_monolith_checkpoint_state, write_monolith_checkpoint_state,
};
use std::fs;
use tempfile::tempdir;

#[test]
fn monolith_checkpoint_state_roundtrip_pbtxt() {
    // Mirrors the pbtxt read/write behavior in:
    // - monolith/native_training/save_utils.py::get_monolith_checkpoint_state
    // - monolith/native_training/monolith_checkpoint_state.proto
    let dir = tempdir().unwrap();
    let mut st = MonolithCheckpointState::default();
    st.exempt_model_checkpoint_paths = vec!["a".to_string()];
    st.last_checkpoint_save_timestamp = Some(123);
    st.builtin_hash_table_type = Some(HashTableType::MultiCuckooHashMap as i32);

    let p = write_monolith_checkpoint_state(dir.path(), &st, true).unwrap();
    assert!(p.exists());

    let parsed = get_monolith_checkpoint_state(dir.path(), None, false)
        .unwrap()
        .unwrap();
    assert_eq!(parsed.exempt_model_checkpoint_paths, vec!["a".to_string()]);
    assert_eq!(parsed.last_checkpoint_save_timestamp, Some(123));
    assert_eq!(
        parsed.builtin_hash_table_type,
        Some(HashTableType::MultiCuckooHashMap as i32)
    );
}

#[test]
fn monolith_checkpoint_state_remove_invalid_paths() {
    let dir = tempdir().unwrap();

    // Create one "existing" path and one missing path.
    let existing = dir.path().join("ckpt1");
    fs::write(&existing, "x").unwrap();

    let pbtxt = r#"
exempt_model_checkpoint_paths: "ckpt1"
exempt_model_checkpoint_paths: "does_not_exist"
builtin_hash_table_type: CUCKOO_HASH_MAP
"#;
    fs::write(dir.path().join("monolith_checkpoint"), pbtxt).unwrap();

    let parsed = get_monolith_checkpoint_state(dir.path(), None, true)
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.exempt_model_checkpoint_paths,
        vec![existing.to_string_lossy().to_string()]
    );
}
