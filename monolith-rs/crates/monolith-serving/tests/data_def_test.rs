use monolith_serving::{Event, EventType, ModelMeta, ReplicaMeta, ResourceSpec};

#[test]
fn test_model_info() {
    let obj = ModelMeta {
        model_name: Some("monolith".to_string()),
        num_shard: 3,
        model_dir: Some("/tmp/opt".to_string()),
        ckpt: Some("model.ckpt-1234".to_string()),
        ..Default::default()
    };
    let bytes = obj.serialize();
    let recom = ModelMeta::deserialize(&bytes).unwrap();
    assert_eq!(obj, recom);
}

#[test]
fn test_event_bytes_wire_format_matches_python_dataclasses_json() {
    // Python dataclasses_json encodes bytes as a JSON list of integers.
    // Ensure we serialize/deserialize the same "on-wire" format.
    let json = r#"{"path":"p","data":[104,105],"etype":3}"#;
    let ev: Event = serde_json::from_str(json).unwrap();
    assert_eq!(ev.path.as_deref(), Some("p"));
    assert_eq!(ev.data, b"hi".to_vec());
    assert_eq!(i32::from(ev.etype), 3);

    let out = String::from_utf8(ev.serialize()).unwrap();
    assert!(out.contains("\"data\":[104,105]"), "serialized: {out}");
}

#[test]
fn test_event_deserialize_legacy_base64_bytes() {
    // Older Rust implementations used base64 strings for `data`; we still accept them on read.
    let json = r#"{"path":"p","data":"aGk=","etype":3}"#;
    let ev: Event = serde_json::from_str(json).unwrap();
    assert_eq!(ev.data, b"hi".to_vec());
    assert_eq!(ev.etype, EventType::Publish);
}

#[test]
fn test_resource() {
    let obj = ResourceSpec {
        address: Some("localhost:123".to_string()),
        shard_id: Some(10),
        replica_id: Some(2),
        memory: Some(12345),
        cpu: 3.5,
        ..Default::default()
    };
    let bytes = obj.serialize();
    let recom = ResourceSpec::deserialize(&bytes).unwrap();
    assert_eq!(obj, recom);
}

#[test]
fn test_replica_meta() {
    let obj = ReplicaMeta {
        address: Some("localhost:123".to_string()),
        model_name: Some("monolith".to_string()),
        server_type: Some("ps".to_string()),
        task: 0,
        replica: 0,
        ..Default::default()
    };
    let bytes = obj.serialize();
    let recom = ReplicaMeta::deserialize(&bytes).unwrap();
    assert_eq!(obj, recom);
}
