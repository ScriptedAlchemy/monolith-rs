use monolith_training::py_discovery::{
    MlpServiceDiscovery, PyServiceDiscovery, TfConfigServiceDiscovery,
};
use std::collections::HashMap;

#[test]
fn tf_config_service_discovery_matches_python_semantics() {
    // Mirrors `TfConfigServiceDiscoveryTest.test_tf_conf_sd` in Python.
    let tf_conf = r#"
    {
      "cluster": {
        "chief": ["host0:2222"],
        "ps": ["host1:2222", "host2:2222"],
        "worker": ["host3:2222", "host4:2222", "host5:2222"]
      },
      "task": {"type": "worker", "index": 1}
    }
    "#;

    let discovery = TfConfigServiceDiscovery::new(tf_conf).unwrap();
    let ps = discovery.query("ps").unwrap();
    assert_eq!(
        ps,
        HashMap::from([(0, "host1:2222".to_string()), (1, "host2:2222".to_string())])
    );

    let worker = discovery.query("worker").unwrap();
    assert_eq!(
        worker,
        HashMap::from([
            (0, "host0:2222".to_string()),
            (1, "host3:2222".to_string()),
            (2, "host4:2222".to_string()),
            (3, "host5:2222".to_string())
        ])
    );

    assert_eq!(discovery.addr().unwrap(), "host4:2222");
    assert_eq!(discovery.server_type(), "worker");
    assert_eq!(discovery.index(), 2);
}

#[test]
fn mlp_service_discovery_matches_filtering_semantics() {
    // Minimal environment slice matching what Python consumes.
    std::env::set_var("MLP_ROLE", "WORKER");
    std::env::set_var("MLP_PS_NUM", "2");
    std::env::set_var("MLP_PS_0_PRIMARY_HOST", "ps0");
    std::env::set_var("MLP_PS_0_PORT", "1001");
    std::env::set_var("MLP_PS_1_PRIMARY_HOST", "ps1");
    std::env::set_var("MLP_PS_1_PORT", "1002");

    let discovery = MlpServiceDiscovery::new();

    // Initially both PS endpoints should appear.
    let all = discovery.query("ps").unwrap();
    assert_eq!(
        all,
        HashMap::from([(0, "ps0:1001".to_string()), (1, "ps1:1002".to_string())])
    );

    // Deregister filters out an entry.
    discovery.deregister("ps", 0, "ps0:1001").unwrap();
    let filtered = discovery.query("ps").unwrap();
    assert_eq!(filtered, HashMap::from([(1, "ps1:1002".to_string())]));

    // Register removes filter.
    discovery.register("ps", 0, "ps0:1001").unwrap();
    let restored = discovery.query("ps").unwrap();
    assert_eq!(
        restored,
        HashMap::from([(0, "ps0:1001".to_string()), (1, "ps1:1002".to_string())])
    );
}
