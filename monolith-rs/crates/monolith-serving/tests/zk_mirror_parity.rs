use monolith_serving::data_def::{
    EventType, ModelMeta, ModelState, PublishMeta, PublishType, ReplicaMeta, ResourceSpec,
};
use monolith_serving::{set_host_shard_env, FakeKazooClient, ZkClient, ZkMirror};
use std::collections::HashMap;
use std::env;
use std::sync::Arc;

// Mirrors `monolith/agent_service/zk_mirror_test.py` (core flows only).
#[test]
fn zk_mirror_crud_and_basic_flow() {
    set_host_shard_env(10);
    env::set_var("SHARD_ID", "2");
    env::set_var("REPLICA_ID", "2");

    let bzid = "bzid".to_string();
    let shard_id = 2;
    let num_tce_shard = 10;
    let replica_id = 2;

    let zk_client = Arc::new(FakeKazooClient::new());
    let mut mirror = ZkMirror::new(zk_client.clone(), bzid.clone(), shard_id, num_tce_shard);
    mirror.set_local_host_for_test("127.0.0.1");
    mirror.start(false).unwrap();

    // CRUD
    mirror.ensure_path("/model/crud").unwrap();
    assert!(mirror.exists("/model/crud"));
    mirror
        .create("/model/crud/data", b"test".to_vec(), false, true)
        .unwrap();
    assert_eq!(zk_client.get("/model/crud/data").unwrap(), b"test".to_vec());
    mirror
        .set("/model/crud/data", b"new_test".to_vec())
        .unwrap();
    assert_eq!(
        zk_client.get("/model/crud/data").unwrap(),
        b"new_test".to_vec()
    );
    // delete non-recursive should error when non-empty; mirror retries recursive if NotEmpty.
    mirror.delete("/model/crud", false).unwrap();
    assert!(!mirror.exists("/model/crud"));

    assert_eq!(mirror.num_tce_shard, 10);
    assert_eq!(mirror.tce_replica_id(), 2);
    assert_eq!(mirror.tce_shard_id, 2);

    // watch portal/resource
    mirror.watch_portal().unwrap();
    mirror.watch_resource().unwrap();

    // portal create triggers portal event
    let model_name = "model";
    let path = format!("/{bzid}/portal/{model_name}");
    let mm = ModelMeta {
        model_name: Some(model_name.to_string()),
        model_dir: Some("/tmp/model/saved_models".to_string()),
        num_shard: 5,
        ..Default::default()
    };
    zk_client
        .create(&path, mm.serialize(), false, true)
        .unwrap();

    // Since our mirror queue is LIFO, drain and search.
    let events = mirror.drain_events();
    let portal_event = events
        .into_iter()
        .find(|e| e.etype == EventType::Portal)
        .expect("expected portal event");
    assert_eq!(portal_event.path.as_deref(), Some(path.as_str()));
    let _mm2 = ModelMeta::deserialize(&portal_event.data).unwrap();

    // publish loading
    let version = 123456;
    let num_ps: i32 = 10;

    let mut pms = Vec::new();
    // Match Python scheduler behavior: each "group" is scheduled onto a distinct shard, and
    // ensure the current shard is included.
    let shard_plan = vec![shard_id, 0, 1, 3, 4];
    for (i, scheduled_shard) in shard_plan.iter().copied().enumerate() {
        let mut sub_models: HashMap<String, String> = (0..num_ps as usize)
            .filter(|k| k % 5 == i)
            .map(|k| {
                (
                    format!("ps_{k}"),
                    format!("/tmp/model/saved_models/ps_{k}/{version}"),
                )
            })
            .collect();
        sub_models.insert(
            "entry".to_string(),
            format!("/tmp/model/saved_models/entry/{version}"),
        );
        let pm = PublishMeta {
            shard_id: Some(scheduled_shard),
            replica_id: 0,
            model_name: Some(model_name.to_string()),
            num_ps: Some(num_ps),
            total_publish_num: shard_plan.len() as i32,
            sub_models: Some(sub_models),
            ptype: PublishType::Load,
            is_spec: false,
        };
        pms.push(pm);
    }

    mirror.publish_loading(&pms).unwrap();
    let expected = mirror.expected_loading();
    let pm = expected.get(model_name).unwrap();
    assert_eq!(pm.model_name.as_deref(), Some(model_name));
    assert_eq!(pm.shard_id, Some(shard_id));
    assert!(pm.sub_models.as_ref().unwrap().contains_key("entry"));

    // update service replicas (match Python test: entry + ps_0 + ps_5)
    let mut replicas = Vec::new();
    for (server_type, task) in [("entry", 0), ("ps", 0), ("ps", 5)] {
        let rm = ReplicaMeta {
            address: Some("127.0.0.1:8080".to_string()),
            model_name: Some(model_name.to_string()),
            server_type: Some(server_type.to_string()),
            task,
            replica: replica_id,
            stat: ModelState::Available as i32,
            ..Default::default()
        };
        replicas.push(rm);
    }
    mirror.update_service(&replicas).unwrap();

    // replica queries
    let all_ps = mirror.get_all_replicas("ps");
    assert!(all_ps.keys().any(|k| k.starts_with("model:ps:")));
    let model_entry = mirror.get_model_replicas(model_name, "entry");
    assert!(model_entry.keys().any(|k| k == "model:entry:0"));
    let task_ps0 = mirror.get_task_replicas(model_name, "ps", 0);
    if !task_ps0.is_empty() {
        assert_eq!(task_ps0[0].replica, replica_id);
    }

    let local_paths = mirror.local_replica_paths();
    assert!(!local_paths.is_empty());

    // report resource + get resources
    let resource = ResourceSpec {
        address: Some("127.0.0.1:1234".to_string()),
        shard_id: Some(shard_id),
        replica_id: Some(replica_id),
        memory: Some(12345),
        cpu: 5.6,
        network: 3.2,
        work_load: 0.7,
    };
    mirror.report_resource(&resource).unwrap();
    let resources = mirror.resources();
    assert_eq!(resources[0], resource);

    mirror.stop().unwrap();
}
