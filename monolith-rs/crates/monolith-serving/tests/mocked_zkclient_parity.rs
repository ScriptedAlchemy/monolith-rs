use monolith_serving::backends::{WatchEventType, WatchedEvent};
use monolith_serving::{FakeKazooClient, ZkClient};
use std::sync::{Arc, Mutex};

// Mirrors `monolith/agent_service/mocked_zkclient_test.py`.
#[test]
fn fake_kazoo_client_create_set_get_delete_and_watches() {
    let zk = Arc::new(FakeKazooClient::new());
    zk.start().unwrap();

    // create
    let path = "/monolith/zk/data";
    zk.create(path, Vec::new(), false, true).unwrap();
    assert!(zk.exists(path));

    // set/get
    let data = b"hi, I am Fitz!".to_vec();
    // setting a non-existent node should error for the raw client.
    zk.set(&format!("{path}/error"), data.clone())
        .expect_err("setting non-existent child path should fail");
    zk.set(path, data.clone()).unwrap();
    let got = zk.get(path).unwrap();
    assert_eq!(got, data);

    // delete and delete parent recursively.
    zk.delete(path, true).unwrap();
    zk.delete("/monolith", true).unwrap();
    assert!(!zk.exists("/monolith"));

    // data watch: should fire initial callback (event None) when node exists.
    zk.create(path, b"init".to_vec(), false, true).unwrap();
    let record: Arc<Mutex<Vec<(Option<Vec<u8>>, Option<WatchEventType>)>>> =
        Arc::new(Mutex::new(Vec::new()));
    let record_cb = record.clone();
    zk.data_watch(
        path,
        Arc::new(move |data, _stat, event| {
            record_cb
                .lock()
                .unwrap()
                .push((data, event.map(|e| e.event_type)));
            true
        }),
    )
    .unwrap();
    assert!(!record.lock().unwrap().is_empty());

    // children watch with event: should receive CHILD event when adding a sibling.
    let children_events: Arc<Mutex<Vec<Option<WatchedEvent>>>> = Arc::new(Mutex::new(Vec::new()));
    let children_events_cb = children_events.clone();
    zk.ensure_path("/monolith/zk").unwrap();
    zk.children_watch_event(
        "/monolith/zk",
        Arc::new(move |_children, event| {
            children_events_cb.lock().unwrap().push(event);
            true
        }),
    )
    .unwrap();
    zk.create("/monolith/zk/test", b"123".to_vec(), false, false)
        .unwrap();
    let evs = children_events.lock().unwrap().clone();
    assert!(evs.iter().any(|e| matches!(
        e,
        Some(WatchedEvent {
            event_type: WatchEventType::Child,
            ..
        })
    )));
}
