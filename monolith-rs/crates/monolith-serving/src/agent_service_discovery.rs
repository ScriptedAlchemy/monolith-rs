//! Agent-service discovery gRPC implementation (Python parity for `monolith/agent_service/agent_service.py`).
//!
//! The upstream Python `AgentService` is a small gRPC server used for service discovery:
//! - `HeartBeat`: return all replica addresses keyed by task
//! - `GetReplicas`: return addresses for a specific task
//! - `GetResource`: return resource info for scheduling/monitoring (v2+)
//!
//! Rust parity reuses existing watcher/mirror components:
//! - v1: [`crate::replica_manager::ReplicaWatcher`]
//! - v2: [`crate::zk_mirror::ZkMirror`]
//! - v3: an injected address provider callback (used by unified container agent)

#![cfg(feature = "grpc")]

use crate::error::{ServingError, ServingResult};
use crate::resource_utils::cal_available_memory_v2;
use crate::utils::{get_local_ip, AgentConfig};
use crate::zk_mirror::ZkMirror;
use crate::{ReplicaWatcher, ServerType as LocalServerType};
use monolith_proto::monolith::serving::agent_service::agent_service_server::AgentService;
use monolith_proto::monolith::serving::agent_service::{
    AddressList, GetReplicasRequest, GetReplicasResponse, GetResourceRequest, GetResourceResponse,
    HeartBeatRequest, HeartBeatResponse, ServerType,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};

/// Python parity for `AgentDataProvider` (v3 address callback provider).
#[derive(Clone)]
pub struct AgentDataProvider {
    addrs_fn: Arc<dyn Fn() -> HashMap<String, Vec<String>> + Send + Sync + 'static>,
}

impl AgentDataProvider {
    /// Create a provider backed by a callback returning `{task -> addrs}`.
    pub fn new(
        addrs_fn: impl Fn() -> HashMap<String, Vec<String>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            addrs_fn: Arc::new(addrs_fn),
        }
    }

    fn get(&self) -> HashMap<String, Vec<String>> {
        (self.addrs_fn)()
    }
}

#[derive(Clone)]
enum Mode {
    Watcher {
        watcher: Arc<ReplicaWatcher>,
        conf: Option<AgentConfig>,
    },
    ZkMirror {
        zk: Arc<ZkMirror>,
        conf: AgentConfig,
    },
    Provider {
        provider: AgentDataProvider,
        conf: AgentConfig,
    },
}

/// A tonic-compatible AgentService implementation following the Python branching logic.
#[derive(Clone)]
pub struct AgentServiceDiscoveryImpl {
    mode: Arc<RwLock<Mode>>,
}

impl AgentServiceDiscoveryImpl {
    /// Python parity: `AgentServiceImpl(watcher, conf=None)`.
    pub fn from_watcher(watcher: Arc<ReplicaWatcher>, conf: Option<AgentConfig>) -> Self {
        Self {
            mode: Arc::new(RwLock::new(Mode::Watcher { watcher, conf })),
        }
    }

    /// Python parity: `AgentServiceImpl(zk, conf)` for agent_version=2.
    pub fn from_zk_mirror(zk: Arc<ZkMirror>, conf: AgentConfig) -> Self {
        Self {
            mode: Arc::new(RwLock::new(Mode::ZkMirror { zk, conf })),
        }
    }

    /// Python parity: `AgentServiceImpl(data_provider, conf)` for agent_version=3.
    pub fn from_provider(provider: AgentDataProvider, conf: AgentConfig) -> Self {
        Self {
            mode: Arc::new(RwLock::new(Mode::Provider { provider, conf })),
        }
    }

    fn server_type_to_local(st: i32) -> LocalServerType {
        match ServerType::try_from(st) {
            Ok(ServerType::Ps) => LocalServerType::Ps,
            Ok(ServerType::Entry) => LocalServerType::Entry,
            Ok(ServerType::Dense) => LocalServerType::Dense,
            _ => LocalServerType::Ps,
        }
    }

    fn server_type_to_str(st: i32) -> &'static str {
        match ServerType::try_from(st) {
            Ok(ServerType::Ps) => "ps",
            Ok(ServerType::Entry) => "entry",
            Ok(ServerType::Dense) => "dense",
            _ => "ps",
        }
    }
}

#[tonic::async_trait]
impl AgentService for AgentServiceDiscoveryImpl {
    async fn get_replicas(
        &self,
        request: Request<GetReplicasRequest>,
    ) -> Result<Response<GetReplicasResponse>, Status> {
        let req = request.into_inner();
        let mode = self.mode.read().clone();

        let mut out: Vec<String> = Vec::new();
        match mode {
            Mode::Watcher { watcher, conf } => {
                // Python: `if conf is None or conf.agent_version == 1:`
                if conf.as_ref().map(|c| c.agent_version).unwrap_or(1) == 1 {
                    let idc = AgentConfig::idc();
                    let cluster = AgentConfig::cluster();
                    out = watcher.get_replicas(
                        Self::server_type_to_local(req.server_type),
                        req.task,
                        idc.as_deref(),
                        cluster.as_deref(),
                    );
                } else {
                    return Err(Status::unimplemented("watcher mode for agent_version != 1"));
                }
            }
            Mode::ZkMirror { zk, conf } => {
                if conf.agent_version == 2 {
                    let rms = zk.get_task_replicas(
                        &req.model_name,
                        Self::server_type_to_str(req.server_type),
                        req.task,
                    );
                    out.extend(rms.into_iter().filter_map(|rm| rm.address));
                } else {
                    return Err(Status::unimplemented(
                        "zk mirror mode only supports agent_version=2",
                    ));
                }
            }
            Mode::Provider { .. } => {
                return Err(Status::unimplemented(
                    "GetReplicas not implemented for agent v3",
                ));
            }
        }

        Ok(Response::new(GetReplicasResponse {
            address_list: Some(AddressList { address: out }),
        }))
    }

    async fn heart_beat(
        &self,
        request: Request<HeartBeatRequest>,
    ) -> Result<Response<HeartBeatResponse>, Status> {
        let req = request.into_inner();
        let mode = self.mode.read().clone();

        let mut addresses: HashMap<String, AddressList> = HashMap::new();

        match mode {
            Mode::Watcher { watcher, conf } => {
                if conf.as_ref().map(|c| c.agent_version).unwrap_or(1) == 1 {
                    let idc = AgentConfig::idc();
                    let cluster = AgentConfig::cluster();
                    let dc_aware = conf.as_ref().map(|c| c.dc_aware).unwrap_or(true);

                    let addrs = watcher.get_all_replicas(
                        Self::server_type_to_local(req.server_type),
                        idc.as_deref(),
                        cluster.as_deref(),
                    );

                    for (k, v) in addrs {
                        let key = if dc_aware {
                            k.split('/').last().unwrap_or(&k).to_string()
                        } else {
                            k
                        };
                        addresses.insert(key, AddressList { address: v });
                    }
                } else {
                    return Err(Status::unimplemented("watcher mode for agent_version != 1"));
                }
            }
            Mode::ZkMirror { zk, conf } => {
                if conf.agent_version == 2 {
                    let rm_dict = zk.get_all_replicas(Self::server_type_to_str(req.server_type));
                    for (k, rms) in rm_dict {
                        addresses.insert(
                            k,
                            AddressList {
                                address: rms.into_iter().filter_map(|rm| rm.address).collect(),
                            },
                        );
                    }
                } else {
                    return Err(Status::unimplemented(
                        "zk mirror mode only supports agent_version=2",
                    ));
                }
            }
            Mode::Provider { provider, .. } => {
                for (k, v) in provider.get() {
                    addresses.insert(k, AddressList { address: v });
                }
            }
        }

        Ok(Response::new(HeartBeatResponse { addresses }))
    }

    async fn get_resource(
        &self,
        _request: Request<GetResourceRequest>,
    ) -> Result<Response<GetResourceResponse>, Status> {
        let mode = self.mode.read().clone();
        match mode {
            Mode::Watcher { conf, .. } => {
                // Python parity: v1 returns empty response.
                if conf.as_ref().map(|c| c.agent_version).unwrap_or(1) == 1 {
                    return Ok(Response::new(GetResourceResponse::default()));
                }
                // fallthrough to v2+ response shape
                let conf = conf.ok_or_else(|| Status::internal("missing AgentConfig"))?;
                Ok(Response::new(GetResourceResponse {
                    address: format!("{}:{}", get_local_ip(), conf.agent_port),
                    shard_id: AgentConfig::shard_id(),
                    replica_id: AgentConfig::replica_id(),
                    memory: cal_available_memory_v2() as i64,
                    cpu: -1.0,
                    network: -1.0,
                    work_load: -1.0,
                }))
            }
            Mode::ZkMirror { conf, .. } | Mode::Provider { conf, .. } => {
                Ok(Response::new(GetResourceResponse {
                    address: format!("{}:{}", get_local_ip(), conf.agent_port),
                    shard_id: AgentConfig::shard_id(),
                    replica_id: AgentConfig::replica_id(),
                    memory: cal_available_memory_v2() as i64,
                    cpu: -1.0,
                    network: -1.0,
                    work_load: -1.0,
                }))
            }
        }
    }
}

/// Running discovery server handle (for tests / embedding into apps).
#[derive(Debug)]
pub struct AgentDiscoveryServer {
    shutdown_tx: Arc<RwLock<Option<oneshot::Sender<()>>>>,
}

impl AgentDiscoveryServer {
    /// Start serving AgentService at the given address.
    pub async fn serve(addr: SocketAddr, svc: AgentServiceDiscoveryImpl) -> ServingResult<Self> {
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let shutdown_tx = Arc::new(RwLock::new(Some(shutdown_tx)));

        tokio::spawn(async move {
            let server = tonic::transport::Server::builder()
                .add_service(monolith_proto::monolith::serving::agent_service::agent_service_server::AgentServiceServer::new(svc))
                .serve_with_shutdown(addr, async {
                    let _ = shutdown_rx.await;
                });

            if let Err(e) = server.await {
                tracing::error!("AgentService discovery server error: {}", e);
            }
        });

        Ok(Self { shutdown_tx })
    }

    /// Signal the server to stop.
    pub fn shutdown(&self) {
        if let Some(tx) = self.shutdown_tx.write().take() {
            let _ = tx.send(());
        }
    }
}

/// Convenience: connect a real tonic AgentService client to `host:port`.
pub async fn connect_agent_service_client(
    addr: impl AsRef<str>,
) -> ServingResult<
    monolith_proto::monolith::serving::agent_service::agent_service_client::AgentServiceClient<
        tonic::transport::Channel,
    >,
> {
    let endpoint = format!("http://{}", addr.as_ref());
    monolith_proto::monolith::serving::agent_service::agent_service_client::AgentServiceClient::connect(
        endpoint,
    )
    .await
    .map_err(|e| ServingError::server(format!("connect failed: {}", e)))
}
