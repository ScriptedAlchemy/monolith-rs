use monolith_proto::tensorflow_serving::apis::model_version_status::State as ModelState;
use monolith_proto::tensorflow_serving::error::Code as ErrorCode;
use monolith_serving::utils::{AgentConfig, InstanceFormatter, ZkPath};
use monolith_serving::{
    gen_model_config, gen_model_spec, gen_model_version_status, gen_status_proto,
};
use std::collections::HashMap;
use std::path::PathBuf;

// Mirrors `monolith/agent_service/utils_test.py` (subset exercised by Rust).

fn repo_root() -> PathBuf {
    // tests run with CWD = crate dir; the monorepo root is 4 levels up:
    // monolith-rs/crates/monolith-serving -> monolith
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repo root must exist")
}

#[test]
fn test_gen_model_spec() {
    let name = "model";
    let version = 1;
    let signature_name = "predict";
    let model_spec = gen_model_spec(name, Some(version), Some(signature_name));
    assert_eq!(model_spec.name, name);
    match model_spec.version_choice {
        Some(monolith_proto::tensorflow_serving::apis::model_spec::VersionChoice::Version(v)) => {
            assert_eq!(v, version);
        }
        other => panic!("unexpected version_choice: {other:?}"),
    }
    assert_eq!(model_spec.signature_name, signature_name);
}

#[test]
fn test_gen_model_config_latest() {
    let name = "model";
    let base_path = "/tmp/model/saved_model";
    let num_versions = 2;
    let mut version_labels: HashMap<String, i64> = HashMap::new();
    version_labels.insert("v0".to_string(), 0);
    version_labels.insert("v1".to_string(), 1);

    let model_config = gen_model_config(name, base_path, num_versions, Some(version_labels));
    assert_eq!(model_config.name, name);
    assert_eq!(model_config.base_path, base_path);

    let policy = model_config.model_version_policy.unwrap();
    match policy.policy_choice.unwrap() {
        monolith_proto::tensorflow_serving::apis::file_system_storage_path_source_config::servable_version_policy::PolicyChoice::Latest(latest) => {
            assert_eq!(latest.num_versions, num_versions as u32);
        }
        other => panic!("unexpected policy: {other:?}"),
    }
}

#[test]
fn test_gen_status_proto() {
    let sp = gen_status_proto(ErrorCode::Cancelled, Some("CANCELLED"));
    assert_eq!(sp.error_code, ErrorCode::Cancelled as i32);
    assert_eq!(sp.error_message, "CANCELLED");
}

#[test]
fn test_gen_model_version_status() {
    let mvs =
        gen_model_version_status(1, ModelState::Start, ErrorCode::NotFound, Some("NOT_FOUND"));
    assert_eq!(mvs.version, 1);
    assert_eq!(mvs.state, ModelState::Start as i32);
}

#[test]
fn test_agent_config_from_file_and_list_fields() {
    let conf_path = repo_root().join("monolith/agent_service/agent.conf");
    let conf = AgentConfig::from_file(&conf_path).unwrap();
    // Python test expects this to be true for the checked-in `agent.conf`.
    assert!(conf.stand_alone_serving);
    assert_eq!(
        conf.layout_filters,
        vec!["ps_0".to_string(), "ps_1".to_string()]
    );
}

#[test]
fn test_instance_formatter_from_json() {
    let p = repo_root().join("monolith/agent_service/test_data/inst.json");
    let iw = InstanceFormatter::from_json(&p).unwrap();
    let tp = iw.to_tensor_proto(5);
    assert_eq!(
        tp.dtype,
        monolith_proto::tensorflow_core::DataType::DtString as i32
    );
    assert_eq!(tp.tensor_shape.as_ref().unwrap().dim[0].size, 5);
}

#[test]
fn test_instance_formatter_from_pbtext() {
    let p = repo_root().join("monolith/agent_service/test_data/inst.pbtext");
    let iw = InstanceFormatter::from_pb_text(&p).unwrap();
    let tp = iw.to_tensor_proto(5);
    assert_eq!(
        tp.dtype,
        monolith_proto::tensorflow_core::DataType::DtString as i32
    );
    assert_eq!(tp.tensor_shape.as_ref().unwrap().dim[0].size, 5);
}

#[test]
fn test_instance_formatter_from_dump() {
    let p = repo_root().join("monolith/agent_service/test_data/inst.dump");
    let iw = InstanceFormatter::from_dump(&p).unwrap();
    let tp = iw.to_tensor_proto(5);
    assert_eq!(
        tp.dtype,
        monolith_proto::tensorflow_core::DataType::DtString as i32
    );
    assert_eq!(tp.tensor_shape.as_ref().unwrap().dim[0].size, 5);
}

#[test]
fn test_get_cmd_and_port_agent_v2_includes_poll_wait_seconds_flag() {
    let conf_path = repo_root().join("monolith/agent_service/agent.conf");
    let mut conf = AgentConfig::from_file(&conf_path).unwrap();
    conf.agent_version = 2;
    let (cmd, _port) = conf
        .get_cmd_and_port("tensorflow_model_server", Some("ps"), None)
        .unwrap();
    assert!(cmd.contains("model_config_file_poll_wait_seconds"));
}

#[test]
fn test_zk_path_full() {
    let p = ZkPath::new("/bzid/service/base_name/idc:cluster/server_type:0/1");
    assert_eq!(p.bzid.as_deref(), Some("bzid"));
    assert_eq!(p.base_name.as_deref(), Some("base_name"));
    assert_eq!(p.idc.as_deref(), Some("idc"));
    assert_eq!(p.cluster.as_deref(), Some("cluster"));
    assert_eq!(p.server_type.as_deref(), Some("server_type"));
    assert_eq!(p.index.as_deref(), Some("0"));
    assert_eq!(p.replica_id.as_deref(), Some("1"));
    assert_eq!(p.location().as_deref(), Some("idc:cluster"));
    assert_eq!(p.task().as_deref(), Some("server_type:0"));
    assert!(p.ship_in(None, None));
}

#[test]
fn test_zk_path_partial() {
    let p = ZkPath::new("/bzid/service/base_name/idc:cluster/server_type:0");
    assert_eq!(p.replica_id.as_deref(), None);
    assert!(p.ship_in(Some("idc"), Some("cluster")));
}

#[test]
fn test_zk_path_old_full() {
    let p = ZkPath::new("/bzid/service/base_name/server_type:0/1");
    assert_eq!(p.idc.as_deref(), None);
    assert_eq!(p.cluster.as_deref(), None);
    assert_eq!(p.replica_id.as_deref(), Some("1"));
    assert!(p.ship_in(None, None));
}

#[test]
fn test_zk_path_old_partial() {
    let p = ZkPath::new("/bzid/service/base_name/server_type:0");
    assert_eq!(p.replica_id.as_deref(), None);
    assert!(p.ship_in(None, None));
}

#[test]
fn test_zk_path_old_partial2() {
    let p = ZkPath::new("/1_20001223_44ce735e-d05c-11ec-ba29-00163e356637/service/20001223_zm_test_realtime_training_1328_v4_r982567_0/ps:1");
    assert_eq!(
        p.bzid.as_deref(),
        Some("1_20001223_44ce735e-d05c-11ec-ba29-00163e356637")
    );
    assert_eq!(
        p.base_name.as_deref(),
        Some("20001223_zm_test_realtime_training_1328_v4_r982567_0")
    );
    assert_eq!(p.server_type.as_deref(), Some("ps"));
    assert_eq!(p.index.as_deref(), Some("1"));
    assert_eq!(p.replica_id.as_deref(), None);
    assert_eq!(p.task().as_deref(), Some("ps:1"));
    assert!(p.ship_in(None, None));
}
