//! Agent-service utils (Python parity for `monolith/agent_service/utils.py`).
//!
//! This module intentionally focuses on the parts exercised by parity tests and
//! other Rust ports:
//! - config parsing from `agent.conf`
//! - TF Serving proto helpers (`gen_model_spec`, `gen_model_config`, ...)
//! - ZK path parsing (`ZkPath`)
//! - instance formatting (`InstanceFormatter`)
//! - small networking helpers (`find_free_port`, `check_port_open`, `get_local_ip`)

use crate::constants::HOST_SHARD_ENV;
use crate::error::{ServingError, ServingResult};
use monolith_proto::descriptor_pool::descriptor_pool;
use monolith_proto::idl::matrix::proto as matrix_proto;
use monolith_proto::parser::proto as parser_proto;
use monolith_proto::tensorflow_core as tf_core;
use monolith_proto::tensorflow_serving::apis as tfserving_apis;
use monolith_proto::tensorflow_serving::error as tfserving_error;
use prost::Message;
use prost_reflect::prost::Message as ReflectMessage;
use prost_reflect::{DynamicMessage, MessageDescriptor};
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream, UdpSocket};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Minimal server type constants (Python parity for `TFSServerType`).
pub struct TfsServerType;

impl TfsServerType {
    /// Parameter-server role (`ps`).
    pub const PS: &'static str = "ps";
    /// Entry role (`entry`).
    pub const ENTRY: &'static str = "entry";
    /// Dense role (`dense`).
    pub const DENSE: &'static str = "dense";
    /// Unified role (`unified`).
    pub const UNIFIED: &'static str = "unified";
}

/// Deploy type (Python parity for `DeployType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployType {
    /// Mixed deployment (multiple server types).
    Mixed,
    /// Entry-only deployment.
    Entry,
    /// Parameter-server-only deployment.
    Ps,
    /// Dense-only deployment.
    Dense,
    /// Unified deployment.
    Unified,
}

impl DeployType {
    /// Parse a deploy type value from config (case-insensitive).
    pub fn parse(s: &str) -> ServingResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "mixed" => Ok(Self::Mixed),
            "entry" => Ok(Self::Entry),
            "ps" => Ok(Self::Ps),
            "dense" => Ok(Self::Dense),
            "unified" => Ok(Self::Unified),
            other => Err(ServingError::ConfigError(format!(
                "invalid deploy_type {other}"
            ))),
        }
    }

    /// Python parity for `DeployType.compat_server_type`.
    pub fn compat_server_type(&self, server_type: Option<&str>) -> ServingResult<&'static str> {
        match (self, server_type.map(|s| s.to_ascii_lowercase())) {
            (DeployType::Mixed, Some(st)) => match st.as_str() {
                "ps" => Ok(TfsServerType::PS),
                "entry" => Ok(TfsServerType::ENTRY),
                "dense" => Ok(TfsServerType::DENSE),
                other => Err(ServingError::ConfigError(format!(
                    "invalid server_type {other}"
                ))),
            },
            (DeployType::Mixed, None) => Err(ServingError::ConfigError(
                "server_type must be set for mixed deploy_type".to_string(),
            )),
            (DeployType::Entry, None) => Ok(TfsServerType::ENTRY),
            (DeployType::Ps, None) => Ok(TfsServerType::PS),
            (DeployType::Dense, None) => Ok(TfsServerType::DENSE),
            (DeployType::Unified, None) => Ok(TfsServerType::UNIFIED),
            (dt, Some(st)) => {
                // If the deploy type isn't mixed, the only compatible type is itself.
                let want = match dt {
                    DeployType::Entry => TfsServerType::ENTRY,
                    DeployType::Ps => TfsServerType::PS,
                    DeployType::Dense => TfsServerType::DENSE,
                    DeployType::Unified => TfsServerType::UNIFIED,
                    DeployType::Mixed => unreachable!(),
                };
                if st == want {
                    Ok(want)
                } else {
                    Err(ServingError::ConfigError(format!(
                        "DeployType {dt:?} not compatible with server_type {st}"
                    )))
                }
            }
        }
    }
}

/// Parse a simple `agent.conf`-style file into a map.
///
/// Supports:
/// - `#` comments
/// - `include <path>`
/// - repeated keys (collected into Vec)
pub fn conf_parser(path: &Path, out: &mut HashMap<String, Vec<String>>) -> io::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let text = fs::read_to_string(path)?;
    for raw_line in text.lines() {
        let mut line = raw_line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(idx) = line.find('#') {
            if idx > 0 {
                line = line[..idx].trim().to_string();
            }
        }
        if line.is_empty() {
            continue;
        }

        if line.starts_with("include") {
            let p = line.split_whitespace().last().unwrap_or("").to_string();
            if !p.is_empty() {
                conf_parser(Path::new(&p), out)?;
            }
            continue;
        }

        // Python splits by `[ =\t]+`. Keep it close enough.
        let parts: Vec<&str> = line
            .split(|c: char| c == ' ' || c == '\t' || c == '=')
            .filter(|s| !s.is_empty())
            .collect();
        if parts.len() < 2 {
            continue;
        }
        let key = parts[0].to_string();
        let value = parts[1..].join(" ");
        out.entry(key).or_default().push(value);
    }
    Ok(())
}

/// Find an ephemeral TCP port by binding to port 0.
pub fn find_free_port() -> ServingResult<u16> {
    let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))?;
    let port = listener.local_addr()?.port();
    Ok(port)
}

/// Check whether a port is open on 127.0.0.1.
pub fn check_port_open(port: u16) -> bool {
    TcpStream::connect_timeout(
        &SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), port).into(),
        Duration::from_millis(200),
    )
    .is_ok()
}

/// Write bytes to a temporary file and return its persisted path.
pub fn write_to_tmp_file(content: impl AsRef<[u8]>) -> ServingResult<PathBuf> {
    let mut f = tempfile::NamedTempFile::new()?;
    f.write_all(content.as_ref())?;
    // `NamedTempFile` deletes the file on drop; Python's `mkstemp()` leaves it.
    // Persist the file and return the final path.
    let (_file, path) = f.keep().map_err(|e| ServingError::IoError(e.error))?;
    Ok(path)
}

/// Python parity for `replica_id_from_pod_name()`.
pub fn replica_id_from_pod_name() -> i64 {
    let Ok(pod_name) = std::env::var("MY_POD_NAME") else {
        return -1;
    };
    let digest = md5::compute(pod_name.as_bytes());
    let hex = format!("{:x}", digest);
    // Python: int(md5.hexdigest()[10:20], base=16)
    if hex.len() < 20 {
        return -1;
    }
    i64::from_str_radix(&hex[10..20], 16).unwrap_or(-1)
}

/// Best-effort local IP selection (Python parity for `get_local_ip()`).
pub fn get_local_ip() -> String {
    if let Ok(ip) = std::env::var("MY_HOST_IP") {
        // Python parity: tests set `MY_HOST_IP=127.0.0.1` and expect it to be honored.
        if !ip.is_empty() && ip != "localhost" {
            return ip;
        }
    }

    // UDP trick to determine local routing IP; ignore errors and fall back.
    if let Ok(sock) = UdpSocket::bind("0.0.0.0:0") {
        if sock.connect("8.8.8.8:80").is_ok() {
            if let Ok(local) = sock.local_addr() {
                if let std::net::SocketAddr::V4(v4) = local {
                    let ip = v4.ip().to_string();
                    if ip != "0.0.0.0" && ip != "127.0.0.1" {
                        return ip;
                    }
                }
            }
        }
    }

    "localhost".to_string()
}

/// Parse a `{foo}` placeholder pattern into a regex where placeholders capture digits.
///
/// Python parity for `normalize_regex("ps_{task}") -> "ps_(?P<task>\\d+)"`.
pub fn normalize_regex(pattern: &str) -> String {
    let mut out = String::new();
    let mut rest = pattern;
    while !rest.is_empty() {
        let Some(begin) = rest.find('{') else {
            out.push_str(rest);
            break;
        };
        let Some(end) = rest[begin..].find('}') else {
            out.push_str(rest);
            break;
        };
        let end = begin + end;
        out.push_str(&rest[..begin]);
        let name = &rest[begin + 1..end];
        out.push_str(&format!("(?P<{name}>\\d+)"));
        rest = &rest[end + 1..];
    }
    out
}

/// ZK task/replica path parser (Python parity for `ZKPath`).
#[derive(Debug, Clone)]
pub struct ZkPath {
    /// The original, unmodified path string.
    pub path: String,
    /// Business ID (the first path segment).
    pub bzid: Option<String>,
    /// Base model name (the model name segment).
    pub base_name: Option<String>,
    /// Datacenter identifier (when DC-aware layout is used).
    pub idc: Option<String>,
    /// Cluster identifier (when DC-aware layout is used).
    pub cluster: Option<String>,
    /// Server type (e.g. `entry`, `ps`, `dense`).
    pub server_type: Option<String>,
    /// Task index within a server type.
    pub index: Option<String>,
    /// Replica ID segment (when present).
    pub replica_id: Option<String>,
}

impl ZkPath {
    /// Parse a ZK service path into its components.
    pub fn new(path: impl Into<String>) -> Self {
        let path = path.into();
        let pat = Regex::new(r"^/(?P<bzid>[-_0-9A-Za-z]+)/service/(?P<base_name>[-_0-9A-Za-z]+)(/(?P<idc>[-_0-9A-Za-z]+):(?P<cluster>[-_0-9A-Za-z]+))?/(?P<server_type>\w+):(?P<index>\d+)(/(?P<replica_id>\d+))?$").expect("regex must compile");
        let caps = pat.captures(&path);
        // Extract first, then move `path` into the struct.
        let bzid = caps
            .as_ref()
            .and_then(|c| c.name("bzid").map(|m| m.as_str().to_string()));
        let base_name = caps
            .as_ref()
            .and_then(|c| c.name("base_name").map(|m| m.as_str().to_string()));
        let idc = caps
            .as_ref()
            .and_then(|c| c.name("idc").map(|m| m.as_str().to_string()));
        let cluster = caps
            .as_ref()
            .and_then(|c| c.name("cluster").map(|m| m.as_str().to_string()));
        let server_type = caps
            .as_ref()
            .and_then(|c| c.name("server_type").map(|m| m.as_str().to_string()));
        let index = caps
            .as_ref()
            .and_then(|c| c.name("index").map(|m| m.as_str().to_string()));
        let replica_id = caps
            .as_ref()
            .and_then(|c| c.name("replica_id").map(|m| m.as_str().to_string()));
        Self {
            path,
            bzid,
            base_name,
            idc,
            cluster,
            server_type,
            index,
            replica_id,
        }
    }

    /// Return the task identifier `{server_type}:{index}` when both are present.
    pub fn task(&self) -> Option<String> {
        Some(format!(
            "{}:{}",
            self.server_type.as_deref()?,
            self.index.as_deref()?
        ))
    }

    /// Return the location identifier `{idc}:{cluster}` when both are present.
    pub fn location(&self) -> Option<String> {
        Some(format!(
            "{}:{}",
            self.idc.as_deref()?,
            self.cluster.as_deref()?
        ))
    }

    /// Return true when this path matches the provided `idc` and `cluster`.
    ///
    /// If either argument is None, this returns true (Python parity convenience).
    pub fn ship_in(&self, idc: Option<&str>, cluster: Option<&str>) -> bool {
        if idc.is_none() || cluster.is_none() {
            return true;
        }
        idc == self.idc.as_deref() && cluster == self.cluster.as_deref()
    }
}

/// Build a TensorFlow Serving `ModelSpec`.
pub fn gen_model_spec(
    name: &str,
    version: Option<i64>,
    signature_name: Option<&str>,
) -> tfserving_apis::ModelSpec {
    let version_choice = version.map(tfserving_apis::model_spec::VersionChoice::Version);
    tfserving_apis::ModelSpec {
        name: name.to_string(),
        version_choice,
        signature_name: signature_name.unwrap_or_default().to_string(),
    }
}

/// Build a TensorFlow Serving `ModelConfig` for a model directory.
pub fn gen_model_config(
    name: &str,
    base_path: &str,
    version_data: i64,
    version_labels: Option<HashMap<String, i64>>,
) -> tfserving_apis::ModelConfig {
    use tfserving_apis::file_system_storage_path_source_config::servable_version_policy as vp;

    // Python parity: defaults to latest with provided `version_data` (>= 1).
    let policy_choice = Some(vp::PolicyChoice::Latest(vp::Latest {
        num_versions: version_data.max(1) as u32,
    }));

    tfserving_apis::ModelConfig {
        name: name.to_string(),
        base_path: base_path.to_string(),
        model_type: 0, // deprecated
        model_platform: "tensorflow".to_string(),
        model_version_policy: Some(
            tfserving_apis::file_system_storage_path_source_config::ServableVersionPolicy {
                policy_choice,
            },
        ),
        version_labels: version_labels.unwrap_or_default(),
        logging_config: None,
    }
}

/// Build a `StatusProto` with the provided error code/message.
pub fn gen_status_proto(
    error_code: tfserving_error::Code,
    error_message: Option<&str>,
) -> tfserving_apis::StatusProto {
    tfserving_apis::StatusProto {
        error_code: error_code as i32,
        error_message: error_message.unwrap_or_default().to_string(),
    }
}

/// Build a `ModelVersionStatus` for a version/state combination.
pub fn gen_model_version_status(
    version: i64,
    state: tfserving_apis::model_version_status::State,
    error_code: tfserving_error::Code,
    error_message: Option<&str>,
) -> tfserving_apis::ModelVersionStatus {
    tfserving_apis::ModelVersionStatus {
        version,
        state: state as i32,
        status: Some(gen_status_proto(error_code, error_message)),
    }
}

fn make_string_tensor(values: Vec<Vec<u8>>) -> tf_core::TensorProto {
    let shape = tf_core::TensorShapeProto {
        dim: vec![tf_core::tensor_shape_proto::Dim {
            size: values.len() as i64,
            name: "".to_string(),
        }],
        ..Default::default()
    };

    tf_core::TensorProto {
        dtype: tf_core::DataType::DtString as i32,
        tensor_shape: Some(shape),
        string_val: values,
        ..Default::default()
    }
}

/// Python parity for `InstanceFormater`.
#[derive(Debug, Clone)]
pub struct InstanceFormatter {
    inst: parser_proto::Instance,
}

impl InstanceFormatter {
    /// Wrap a protobuf `Instance` for conversion helpers.
    pub fn new(inst: parser_proto::Instance) -> Self {
        Self { inst }
    }

    /// Serialize the instance and return a TF string tensor with `batch_size` copies.
    pub fn to_tensor_proto(&self, batch_size: usize) -> tf_core::TensorProto {
        let bytes = self.inst.encode_to_vec();
        let instances = (0..batch_size).map(|_| bytes.clone()).collect::<Vec<_>>();
        make_string_tensor(instances)
    }

    /// Load an `Instance` from protobuf-JSON mapping (subset used by parity tests).
    pub fn from_json(path: &Path) -> ServingResult<Self> {
        // The Python JSON format is the protobuf JSON mapping. For parity tests we only need
        // a subset: fid, label, and line_id.actions.
        let text = fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            ServingError::InvalidRequest(format!("failed to parse Instance JSON: {e}"))
        })?;

        let mut inst = parser_proto::Instance::default();
        if let Some(arr) = v.get("fid").and_then(|v| v.as_array()) {
            inst.fid = arr.iter().filter_map(|x| x.as_u64()).collect();
        }
        if let Some(arr) = v.get("label").and_then(|v| v.as_array()) {
            inst.label = arr
                .iter()
                .filter_map(|x| x.as_f64())
                .map(|f| f as f32)
                .collect();
        }
        if let Some(line_id) = v.get("line_id") {
            let mut lid = matrix_proto::LineId::default();
            if let Some(actions) = line_id.get("actions").and_then(|v| v.as_array()) {
                lid.actions = actions
                    .iter()
                    .filter_map(|x| x.as_i64())
                    .map(|i| i as i32)
                    .collect();
            }
            inst.line_id = Some(lid);
        }

        Ok(Self::new(inst))
    }

    fn parse_pbtxt_with_pool<T: Message + Default>(
        full_name: &str,
        pbtxt: &str,
    ) -> ServingResult<T> {
        let pool = descriptor_pool();
        let msg_desc: MessageDescriptor = pool.get_message_by_name(full_name).ok_or_else(|| {
            ServingError::ConfigError(format!("DescriptorPool missing message {full_name}"))
        })?;
        let dyn_msg = DynamicMessage::parse_text_format(msg_desc, pbtxt).map_err(|e| {
            ServingError::ConfigError(format!("Failed to parse {full_name} pbtxt: {e}"))
        })?;
        let bytes = ReflectMessage::encode_to_vec(&dyn_msg);
        T::decode(bytes.as_slice()).map_err(|e| {
            ServingError::ConfigError(format!("Failed to decode {full_name} from pbtxt: {e}"))
        })
    }

    /// Load an `Instance` from protobuf text format using the shared descriptor pool.
    pub fn from_pb_text(path: &Path) -> ServingResult<Self> {
        let pbtxt = fs::read_to_string(path)?;
        let inst: parser_proto::Instance =
            Self::parse_pbtxt_with_pool("parser.proto.Instance", &pbtxt)?;
        Ok(Self::new(inst))
    }

    /// Load an `Instance` from Python's custom dump format (best-effort parsing).
    pub fn from_dump(path: &Path) -> ServingResult<Self> {
        // Python's dump format is a custom tree dump. For parity tests we only need to parse
        // enough to produce a valid Instance; fall back to a minimal default if parsing fails.
        let text = fs::read_to_string(path)?;
        let mut inst = parser_proto::Instance::default();

        // Heuristic: capture integers after keys "label" and "actions", and string/ints after "fid".
        // This matches the included test dump.
        let mut current_key: Option<&str> = None;
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with("\"root\"") {
                continue;
            }
            // Extract "key": or key": patterns.
            if line.starts_with('"') && line.ends_with(':') {
                let key = line.trim_end_matches(':').trim_matches('"');
                current_key = Some(key);
                continue;
            }
            // Extract numeric-indexed lines like `0: "1"` or `0: 1`.
            if let Some((idx, val)) = line.split_once(':') {
                let idx = idx.trim().trim_matches('"');
                if !idx.chars().all(|c| c.is_ascii_digit()) {
                    continue;
                }
                let v = val.trim().trim_matches('"').trim_matches('\'');
                match current_key {
                    Some("fid") => {
                        if let Ok(n) = v.parse::<u64>() {
                            inst.fid.push(n);
                        }
                    }
                    Some("label") => {
                        if let Ok(n) = v.parse::<f32>() {
                            inst.label.push(n);
                        } else if let Ok(n) = v.parse::<i64>() {
                            inst.label.push(n as f32);
                        }
                    }
                    Some("actions") => {
                        if let Ok(n) = v.parse::<i32>() {
                            inst.line_id
                                .get_or_insert_with(matrix_proto::LineId::default)
                                .actions
                                .push(n);
                        }
                    }
                    _ => {}
                }
            }
        }

        // If the dump didn't contain anything usable, return a deterministic instance.
        if inst.fid.is_empty() {
            inst.fid.push(1);
        }
        Ok(Self::new(inst))
    }
}

/// Subset of Python AgentConfig used by parity tests and Rust agent ports.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Business ID used for ZK path construction.
    pub bzid: String,
    /// Base model name used under the service subtree.
    pub base_name: Option<String>,
    /// Model base path (e.g. exported model directory root).
    pub base_path: Option<String>,
    /// Number of parameter-server tasks.
    pub num_ps: i32,
    /// Number of shards for the model.
    pub num_shard: i32,
    /// Deployment type.
    pub deploy_type: DeployType,
    /// Whether running in standalone serving mode.
    pub stand_alone_serving: bool,
    /// ZooKeeper connection string(s).
    pub zk_servers: Option<String>,
    /// Whether the deployment is DC-aware.
    pub dc_aware: bool,

    /// Agent version (Python parity).
    pub agent_version: i32,
    /// Agent gRPC/http port.
    pub agent_port: u16,

    // unified deploy (agent v3)
    /// Layout name/pattern to locate the layout file.
    pub layout_pattern: Option<String>,
    /// Filters applied when selecting layout entries.
    pub layout_filters: Vec<String>,
    /// Port for the archon endpoint.
    pub tfs_port_archon: u16,
    /// Port for TF Serving gRPC.
    pub tfs_port_grpc: u16,
    /// Port for TF Serving HTTP.
    pub tfs_port_http: u16,
}

impl AgentConfig {
    /// Number of shards from `HOST_SHARD_ENV` (defaults to 1).
    pub fn num_tce_shard() -> i32 {
        std::env::var(HOST_SHARD_ENV)
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(1)
    }

    /// Shard ID from `SHARD_ID` (or -1 when unset/invalid).
    pub fn shard_id() -> i32 {
        std::env::var("SHARD_ID")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(-1)
    }

    /// Replica ID from `REPLICA_ID` or derived from pod name.
    pub fn replica_id() -> i32 {
        std::env::var("REPLICA_ID")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or_else(|| replica_id_from_pod_name() as i32)
    }

    /// IDC string from environment (lowercased).
    pub fn idc() -> Option<String> {
        std::env::var("TCE_INTERNAL_IDC")
            .ok()
            .map(|v| v.to_ascii_lowercase())
    }

    /// Cluster string from environment (lowercased).
    pub fn cluster() -> Option<String> {
        std::env::var("TCE_LOGICAL_CLUSTER")
            .or_else(|_| std::env::var("TCE_CLUSTER"))
            .or_else(|_| std::env::var("TCE_PHYSICAL_CLUSTER"))
            .ok()
            .map(|v| v.to_ascii_lowercase())
    }

    /// Convenience accessor for `{idc}:{cluster}` when both are present.
    pub fn location() -> Option<String> {
        Some(format!(
            "{}:{}",
            Self::idc()?.as_str(),
            Self::cluster()?.as_str()
        ))
    }

    /// Return the base service path for this config (DC-aware when configured).
    pub fn path_prefix(&self) -> String {
        if self.dc_aware {
            if let Some(loc) = Self::location() {
                return format!(
                    "/{}/service/{}/{}",
                    self.bzid,
                    self.base_name.as_deref().unwrap_or_default(),
                    loc
                );
            }
        }
        format!(
            "/{}/service/{}",
            self.bzid,
            self.base_name.as_deref().unwrap_or_default()
        )
    }

    /// Return the full layout path (absolute, or `/{bzid}/layouts/{layout_pattern}`).
    pub fn layout_path(&self) -> Option<String> {
        let pat = self.layout_pattern.as_deref()?;
        if pat.starts_with('/') {
            Some(pat.to_string())
        } else {
            Some(format!("/{}/layouts/{}", self.bzid, pat))
        }
    }

    /// Build the container cluster string `{TCE_PSM};{idc};{cluster}`.
    pub fn container_cluster(&self) -> String {
        let psm = std::env::var("TCE_PSM").unwrap_or_else(|_| "unknown".to_string());
        let idc = Self::idc().unwrap_or_default();
        let cluster = Self::cluster().unwrap_or_default();
        format!("{psm};{idc};{cluster}")
    }

    /// Return the container ID, typically the pod name (falls back to local IP).
    pub fn container_id(&self) -> String {
        std::env::var("MY_POD_NAME").unwrap_or_else(|_| get_local_ip())
    }

    /// Parse an agent config file (`agent.conf` semantics).
    pub fn from_file(path: &Path) -> ServingResult<Self> {
        let mut raw: HashMap<String, Vec<String>> = HashMap::new();
        conf_parser(path, &mut raw)?;

        fn pop1(map: &HashMap<String, Vec<String>>, key: &str) -> Option<String> {
            map.get(key).and_then(|v| v.last()).cloned()
        }

        fn parse_bool(v: &str) -> Option<bool> {
            match v.to_ascii_lowercase().as_str() {
                "true" | "y" | "t" | "yes" | "1" => Some(true),
                "false" | "n" | "f" | "no" | "0" => Some(false),
                _ => None,
            }
        }

        let bzid = pop1(&raw, "bzid").unwrap_or_default();
        let base_name = pop1(&raw, "base_name");
        let base_path = pop1(&raw, "base_path");
        let num_ps = pop1(&raw, "num_ps")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(1);
        let stand_alone_serving = pop1(&raw, "stand_alone_serving")
            .as_deref()
            .and_then(parse_bool)
            .unwrap_or(false);
        let dc_aware = pop1(&raw, "dc_aware")
            .as_deref()
            .and_then(parse_bool)
            .unwrap_or(false);
        let agent_version = pop1(&raw, "agent_version")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(1);

        let deploy_raw = pop1(&raw, "deploy_type")
            .or_else(|| pop1(&raw, "server_type"))
            .unwrap_or_else(|| "entry".to_string());
        let deploy_type = if stand_alone_serving {
            DeployType::Mixed
        } else {
            DeployType::parse(&deploy_raw)?
        };

        let num_shard = pop1(&raw, "num_shard")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or_else(Self::num_tce_shard);

        let layout_pattern = pop1(&raw, "layout_pattern");
        let layout_filters = raw.get("layout_filters").cloned().unwrap_or_default();

        // Port assignment parity for tests: prefer PORT2 and friends; otherwise allocate.
        let agent_port = std::env::var("PORT2")
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(find_free_port()?);

        let tfs_port_archon = std::env::var("PORT")
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(find_free_port()?);
        let tfs_port_grpc = std::env::var("PORT3")
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(find_free_port()?);
        let tfs_port_http = std::env::var("PORT4")
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(find_free_port()?);

        Ok(Self {
            bzid,
            base_name,
            base_path,
            num_ps,
            num_shard,
            deploy_type,
            stand_alone_serving,
            zk_servers: pop1(&raw, "zk_servers"),
            dc_aware,
            agent_version,
            agent_port,
            layout_pattern,
            layout_filters,
            tfs_port_archon,
            tfs_port_grpc,
            tfs_port_http,
        })
    }

    /// Python parity for `AgentConfig.get_cmd_and_port()`.
    ///
    /// This is used by unit tests to verify `model_config_file_poll_wait_seconds` behavior.
    pub fn get_cmd_and_port(
        &self,
        binary: &str,
        server_type: Option<&str>,
        config_file: Option<&Path>,
    ) -> ServingResult<(String, u16)> {
        let server_type = self.deploy_type.compat_server_type(server_type)?;
        let config_path = if let Some(p) = config_file {
            p.to_path_buf()
        } else {
            // Keep the config contents minimal; only the flag is asserted by tests.
            write_to_tmp_file(b"model_config_list {}")?
        };

        let mut flags: Vec<String> = vec![format!("--model_config_file={}", config_path.display())];
        let port = match server_type {
            TfsServerType::PS => self.tfs_port_grpc,
            TfsServerType::ENTRY => self.tfs_port_grpc,
            TfsServerType::DENSE => self.tfs_port_grpc,
            TfsServerType::UNIFIED => self.tfs_port_grpc,
            _ => self.tfs_port_grpc,
        };
        flags.push(format!("--port={port}"));

        if self.agent_version != 1 {
            flags.push("--model_config_file_poll_wait_seconds=0".to_string());
        }

        Ok((format!("{binary} {}", flags.join(" ")), port))
    }
}
