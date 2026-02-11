use monolith_serving::ModelManager;
use std::fs;
use std::path::Path;
use std::time::Duration;
use tempfile::tempdir;

// Mirrors `monolith/agent_service/model_manager_test.py`.

fn create_file(model_name: &str, timestamp: &str, p2p_data_path: &Path) {
    // p2p/test_model@1234567/test_model/ps_item_embedding_0/1234567
    let p = p2p_data_path
        .join(format!("{model_name}@{timestamp}"))
        .join(model_name);
    fs::create_dir_all(p.join("ps_item_embedding_0").join(timestamp)).unwrap();
    fs::create_dir_all(p.join("ps_item_embedding_1").join(timestamp)).unwrap();

    // p2p/test_model@1234567.write.done
    fs::write(
        p2p_data_path.join(format!("{model_name}@{timestamp}.write.done")),
        b"",
    )
    .unwrap();
}

#[test]
fn model_manager_start_copies_model() {
    let base = tempdir().unwrap();
    let p2p_data_path = base.path().join("p2p");
    let model_data_path = base.path().join("model_data");

    let model_name = "test_model";
    let timestamp = "1234567";
    create_file(model_name, timestamp, &p2p_data_path);

    let mm = ModelManager::new(
        Some(model_name.to_string()),
        p2p_data_path.clone(),
        model_data_path.clone(),
        false,
    );
    mm.set_wait_timeout(Duration::from_secs(5));
    mm.set_loop_interval(Duration::from_secs(5));
    assert!(mm.start());

    let ready1 = model_data_path
        .join(model_name)
        .join("ps_item_embedding_0")
        .join(timestamp);
    let ready2 = model_data_path
        .join(model_name)
        .join("ps_item_embedding_1")
        .join(timestamp);
    assert!(ready1.exists());
    assert!(ready2.exists());

    mm.stop();
}

#[test]
fn model_manager_ignore_old() {
    let base = tempdir().unwrap();
    let p2p_data_path = base.path().join("p2p");
    let model_data_path = base.path().join("model_data");

    let model_name = "test_model";
    let timestamp = "1234567";
    let timestamp_old = "1234566";

    create_file(model_name, timestamp, &p2p_data_path);

    let mm = ModelManager::new(
        Some(model_name.to_string()),
        p2p_data_path.clone(),
        model_data_path.clone(),
        false,
    );
    mm.set_wait_timeout(Duration::from_secs(5));
    mm.set_loop_interval(Duration::from_secs(5));
    assert!(mm.start());

    // Add an older version after the manager started; it should be ignored.
    create_file(model_name, timestamp_old, &p2p_data_path);
    std::thread::sleep(Duration::from_secs(11));

    let old1 = model_data_path
        .join(model_name)
        .join("ps_item_embedding_0")
        .join(timestamp_old);
    let old2 = model_data_path
        .join(model_name)
        .join("ps_item_embedding_1")
        .join(timestamp_old);
    assert!(!old1.exists());
    assert!(!old2.exists());

    mm.stop();
}
