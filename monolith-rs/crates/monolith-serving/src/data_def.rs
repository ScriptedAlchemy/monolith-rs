//! Agent-service data definitions (Python parity).
//!
//! These structs mirror the JSON-on-ZK wire format used by the Python agent-service.
//!
//! Python `dataclasses_json` encodes `bytes` fields as a JSON array of integers
//! (e.g. `b"hi" -> [104, 105]`). We serialize `Vec<u8>` the same way.
//!
//! For backwards compatibility with older Rust implementations, we still accept base64 strings
//! when deserializing those byte fields.

use base64::Engine;
use monolith_proto::tensorflow_serving::apis as tfserving_apis;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// TensorFlow Serving model state enum (protobuf).
pub type ModelState = tfserving_apis::model_version_status::State;

/// Simple model name.
pub type ModelName = String;
/// Sub-model name (e.g. `entry`, `ps_{num}`, `dense`).
pub type SubModelName = String;
/// Sub-model size in bytes.
pub type SubModelSize = i64;
/// TensorFlow Serving model name: `{model_name}:{sub_model_name}`.
pub type TfsModelName = String;
/// Version path: `.../exported_models/{sub_model_name}/{version}`.
pub type VersionPath = String;

/// Python parity for `StatusProto()`.
pub fn empty_status() -> tfserving_apis::StatusProto {
    tfserving_apis::StatusProto::default()
}

fn default_action() -> String {
    "NONE".to_string()
}

fn default_i32_minus_one() -> i32 {
    -1
}

fn default_f64_minus_one() -> f64 {
    -1.0
}

fn default_total_publish_num() -> i32 {
    1
}

/// Model metadata stored under `/portal`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMeta {
    #[serde(default)]
    /// Model name (ZK node name).
    pub model_name: Option<String>,
    #[serde(default)]
    /// Model directory path.
    pub model_dir: Option<String>,
    #[serde(default)]
    /// Checkpoint identifier (when applicable).
    pub ckpt: Option<String>,
    #[serde(default = "default_i32_minus_one")]
    /// Number of shards for this model (or -1 when unset).
    pub num_shard: i32,
    #[serde(default = "default_action")]
    /// Action string used by the Python agent-service (`NONE`, `CREATED`, `DELETED`, ...).
    pub action: String,
    #[serde(default)]
    /// Replica IDs explicitly targeted by this model meta.
    pub spec_replicas: Vec<i32>,
}

impl Default for ModelMeta {
    fn default() -> Self {
        Self {
            model_name: None,
            model_dir: None,
            ckpt: None,
            num_shard: -1,
            action: default_action(),
            spec_replicas: Vec::new(),
        }
    }
}

impl ModelMeta {
    /// Build the ZK path for this model under `base_path` (typically `/{bzid}/portal`).
    pub fn get_path(&self, base_path: &str) -> String {
        let name = self.model_name.as_deref().unwrap_or_default();
        std::path::Path::new(base_path)
            .join(name)
            .to_string_lossy()
            .to_string()
    }

    /// Serialize to JSON bytes (Python `dataclasses_json` parity).
    pub fn serialize(&self) -> Vec<u8> {
        // `dataclasses_json` produces UTF-8 JSON bytes.
        serde_json::to_vec(self).expect("ModelMeta must be JSON-serializable")
    }

    /// Deserialize from JSON bytes.
    pub fn deserialize(serialized: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(serialized)
    }
}

/// Resource info stored under `/{bzid}/resource/{shard_id}:{replica_id}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSpec {
    /// host:port
    #[serde(default)]
    pub address: Option<String>,
    #[serde(default)]
    /// Shard ID reported by the instance.
    pub shard_id: Option<i32>,
    #[serde(default)]
    /// Replica ID reported by the instance.
    pub replica_id: Option<i32>,
    #[serde(default)]
    /// Memory capacity in bytes (when reported).
    pub memory: Option<i64>,
    #[serde(default = "default_f64_minus_one")]
    /// CPU capacity (or -1 when unknown).
    pub cpu: f64,
    #[serde(default = "default_f64_minus_one")]
    /// Network capacity (or -1 when unknown).
    pub network: f64,
    #[serde(default = "default_f64_minus_one")]
    /// Workload score (or -1 when unknown).
    pub work_load: f64,
}

impl Default for ResourceSpec {
    fn default() -> Self {
        Self {
            address: None,
            shard_id: None,
            replica_id: None,
            memory: None,
            cpu: -1.0,
            network: -1.0,
            work_load: -1.0,
        }
    }
}

impl ResourceSpec {
    /// Build the ZK path for this instance under `base_path` (typically `/{bzid}/resource`).
    pub fn get_path(&self, base_path: &str) -> String {
        let shard = self.shard_id.unwrap_or_default();
        let replica = self.replica_id.unwrap_or_default();
        std::path::Path::new(base_path)
            .join(format!("{shard}:{replica}"))
            .to_string_lossy()
            .to_string()
    }

    /// Serialize to JSON bytes.
    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("ResourceSpec must be JSON-serializable")
    }

    /// Deserialize from JSON bytes.
    pub fn deserialize(serialized: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(serialized)
    }
}

/// Publish type for a publish meta.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PublishType {
    /// Indicates a model should be loaded.
    Load = 1,
    /// Indicates a model should be unloaded.
    Unload = 2,
}

impl Default for PublishType {
    fn default() -> Self {
        Self::Load
    }
}

impl From<PublishType> for i32 {
    fn from(v: PublishType) -> Self {
        v as i32
    }
}

impl TryFrom<i32> for PublishType {
    type Error = String;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Load),
            2 => Ok(Self::Unload),
            v => Err(format!("invalid PublishType: {v}")),
        }
    }
}

impl Serialize for PublishType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_i32((*self).into())
    }
}

impl<'de> Deserialize<'de> for PublishType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = i32::deserialize(deserializer)?;
        PublishType::try_from(v).map_err(serde::de::Error::custom)
    }
}

/// Publish metadata stored under `/{bzid}/publish/{shard_id}:{replica_id}:{model_name}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublishMeta {
    #[serde(default)]
    /// Shard ID for this publish entry.
    pub shard_id: Option<i32>,
    #[serde(default = "default_i32_minus_one")]
    /// Replica ID for this publish entry.
    pub replica_id: i32,
    #[serde(default)]
    /// Model name.
    pub model_name: Option<String>,
    #[serde(default)]
    /// Number of PS tasks (when applicable).
    pub num_ps: Option<i32>,
    #[serde(default = "default_total_publish_num")]
    /// Total number of publish nodes expected for the model.
    pub total_publish_num: i32,
    #[serde(default)]
    /// Mapping from sub-model name to version path.
    pub sub_models: Option<HashMap<SubModelName, VersionPath>>,
    #[serde(default)]
    /// Publish operation type (load/unload).
    pub ptype: PublishType,
    #[serde(default)]
    /// Whether this publish entry is replica-specific.
    pub is_spec: bool,
}

impl Default for PublishMeta {
    fn default() -> Self {
        Self {
            shard_id: None,
            replica_id: -1,
            model_name: None,
            num_ps: None,
            total_publish_num: 1,
            sub_models: None,
            ptype: PublishType::Load,
            is_spec: false,
        }
    }
}

impl PublishMeta {
    /// Build the ZK path for this publish entry under `base_path` (typically `/{bzid}/publish`).
    pub fn get_path(&self, base_path: &str) -> String {
        let shard = self.shard_id.unwrap_or_default();
        let name = self.model_name.as_deref().unwrap_or_default();
        std::path::Path::new(base_path)
            .join(format!("{shard}:{}:{name}", self.replica_id))
            .to_string_lossy()
            .to_string()
    }

    /// Serialize to JSON bytes.
    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("PublishMeta must be JSON-serializable")
    }

    /// Deserialize from JSON bytes.
    pub fn deserialize(serialized: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(serialized)
    }
}

/// Address family constants (Python parity for `monolith.native_training.net_utils.AddressFamily`).
pub struct AddressFamily;

impl AddressFamily {
    /// Prefer IPv4 addresses.
    pub const IPV4: &'static str = "ipv4";
    /// Prefer IPv6 addresses.
    pub const IPV6: &'static str = "ipv6";
}

/// Replica metadata stored under `/{bzid}/service/{model_name}/{server_type}:{task}/{replica}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicaMeta {
    /// host:port
    #[serde(default)]
    pub address: Option<String>,
    /// [host]:port
    #[serde(default)]
    pub address_ipv6: Option<String>,
    #[serde(default)]
    /// Replica model state (`ModelState` as i32).
    pub stat: i32,
    #[serde(default)]
    /// Model name.
    pub model_name: Option<String>,
    #[serde(default)]
    /// Server type (e.g. `entry`, `ps`, `dense`, `unified`).
    pub server_type: Option<String>,
    #[serde(default = "default_i32_minus_one")]
    /// Task ID (or -1 when unset).
    pub task: i32,
    #[serde(default = "default_i32_minus_one")]
    /// Replica ID (or -1 when unset).
    pub replica: i32,
    /// host:port
    #[serde(default)]
    pub archon_address: Option<String>,
    /// [host]:port
    #[serde(default)]
    pub archon_address_ipv6: Option<String>,
}

impl Default for ReplicaMeta {
    fn default() -> Self {
        Self {
            address: None,
            address_ipv6: None,
            stat: ModelState::Unknown as i32,
            model_name: None,
            server_type: None,
            task: -1,
            replica: -1,
            archon_address: None,
            archon_address_ipv6: None,
        }
    }
}

impl ReplicaMeta {
    /// Serialize to JSON bytes.
    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("ReplicaMeta must be JSON-serializable")
    }

    /// Deserialize from JSON bytes.
    pub fn deserialize(serialized: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(serialized)
    }

    /// Build a service path for this replica under `/{bzid}/service`.
    pub fn get_path(&self, bzid: &str, sep: &str) -> String {
        // Python parity:
        // paths = ['', bzid, 'service', model_name, f'{server_type}:{task}', str(replica)]
        let model = self.model_name.as_deref().unwrap_or_default();
        let st = self.server_type.as_deref().unwrap_or_default();
        [
            "",
            bzid,
            "service",
            model,
            &format!("{st}:{}", self.task),
            &self.replica.to_string(),
        ]
        .join(sep)
    }

    /// Return the best address by family preference with IPv4/IPv6 fallback.
    ///
    /// This matches Python logic, including filtering out wildcard binds (`0.0.0.0:*` and
    /// `[::]:*`), which are not reachable by peers.
    pub fn get_address(&self, use_archon: bool, address_family: &str) -> Option<String> {
        assert!(
            address_family == AddressFamily::IPV4 || address_family == AddressFamily::IPV6,
            "address_family must be ipv4/ipv6"
        );

        let mut ipv4_address = if use_archon {
            self.archon_address.clone()
        } else {
            self.address.clone()
        };
        if matches!(ipv4_address.as_deref(), Some(a) if a.starts_with("0.0.0.0")) {
            ipv4_address = None;
        }

        let mut ipv6_address = if use_archon {
            self.archon_address_ipv6.clone()
        } else {
            self.address_ipv6.clone()
        };
        if matches!(ipv6_address.as_deref(), Some(a) if a.starts_with("[::]")) {
            ipv6_address = None;
        }

        if address_family == AddressFamily::IPV4 {
            ipv4_address.or(ipv6_address)
        } else {
            ipv6_address.or(ipv4_address)
        }
    }
}

/// Event type used internally by the Python agent-service.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum EventType {
    /// Scheduler/ZK-watch trigger.
    Portal = 1,
    /// StatusReportHandler/time trigger.
    Service = 2,
    /// ModelLoaderHandler/ZK-watch trigger.
    Publish = 3,
    /// ResourceReportHandler/time trigger.
    Resource = 4,
}

impl EventType {
    /// Python parity: `EventType.UNKNOWN = 1` is an alias of `PORTAL`.
    pub const UNKNOWN: EventType = EventType::Portal;
}

impl Default for EventType {
    fn default() -> Self {
        EventType::UNKNOWN
    }
}

impl From<EventType> for i32 {
    fn from(v: EventType) -> Self {
        v as i32
    }
}

impl TryFrom<i32> for EventType {
    type Error = String;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Portal),
            2 => Ok(Self::Service),
            3 => Ok(Self::Publish),
            4 => Ok(Self::Resource),
            v => Err(format!("invalid EventType: {v}")),
        }
    }
}

impl Serialize for EventType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_i32((*self).into())
    }
}

impl<'de> Deserialize<'de> for EventType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = i32::deserialize(deserializer)?;
        EventType::try_from(v).map_err(serde::de::Error::custom)
    }
}

fn deserialize_bytes_list_or_base64<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct BytesVisitor;

    impl<'de> serde::de::Visitor<'de> for BytesVisitor {
        type Value = Vec<u8>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "array of integers (Python `dataclasses_json`), or base64 string (legacy)")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            // Legacy compatibility: older Rust versions used base64 strings on the wire.
            base64::engine::general_purpose::STANDARD
                .decode(v)
                .map_err(E::custom)
        }

        fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(v.to_vec())
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut out = Vec::new();
            while let Some(b) = seq.next_element::<u8>()? {
                out.push(b);
            }
            Ok(out)
        }
    }

    deserializer.deserialize_any(BytesVisitor)
}

/// An internal event carrying opaque payload bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    #[serde(default)]
    /// Path that triggered the event.
    pub path: Option<String>,
    #[serde(default, deserialize_with = "deserialize_bytes_list_or_base64")]
    /// Opaque, event-specific payload.
    pub data: Vec<u8>,
    #[serde(default)]
    /// Event type discriminator.
    pub etype: EventType,
}

impl Default for Event {
    fn default() -> Self {
        Self {
            path: None,
            data: Vec::new(),
            etype: EventType::UNKNOWN,
        }
    }
}

impl Event {
    /// Serialize to JSON bytes.
    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("Event must be JSON-serializable")
    }

    /// Deserialize from JSON bytes.
    pub fn deserialize(serialized: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(serialized)
    }
}
