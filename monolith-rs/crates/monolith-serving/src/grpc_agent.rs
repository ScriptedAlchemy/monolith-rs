//! Real gRPC server/client bindings for the upstream AgentService proto.
//!
//! This module wires `monolith_proto::monolith::serving::agent_service` into an
//! actual tonic server. It is the “real” gRPC boundary compatible with the
//! Python implementation which uses the same `agent_service.proto`.

use crate::error::{ServingError, ServingResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

use monolith_proto::monolith::serving::agent_service::agent_service_server::{
    AgentService, AgentServiceServer,
};
use monolith_proto::monolith::serving::agent_service::{
    AddressList, GetReplicasRequest, GetReplicasResponse, GetResourceRequest, GetResourceResponse,
    HeartBeatRequest, HeartBeatResponse, ServerType,
};

/// Minimal in-memory AgentService implementation (replica registry + heartbeat).
///
/// This matches the proto surface area. Higher-level Monolith serving (model
/// inference) lives elsewhere; the upstream `AgentService` is primarily for
/// discovery/coordination.
#[derive(Debug)]
pub struct AgentServiceRealImpl {
    start: Instant,
    replicas: Arc<RwLock<HashMap<i32, Vec<String>>>>,
}

impl AgentServiceRealImpl {
    /// Create a new service.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            replicas: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a replica address for a given server type.
    pub fn register_replica(&self, server_type: ServerType, address: impl Into<String>) {
        let mut map = self.replicas.write();
        let key = server_type as i32;
        let addrs = map.entry(key).or_insert_with(Vec::new);
        let addr = address.into();
        if !addrs.contains(&addr) {
            addrs.push(addr);
        }
    }
}

#[tonic::async_trait]
impl AgentService for AgentServiceRealImpl {
    async fn get_replicas(
        &self,
        request: Request<GetReplicasRequest>,
    ) -> Result<Response<GetReplicasResponse>, Status> {
        let req = request.into_inner();
        debug!(
            "GetReplicas: server_type={} task={} model_name={}",
            req.server_type, req.task, req.model_name
        );

        let replicas = self.replicas.read();
        let addrs = replicas
            .get(&req.server_type)
            .cloned()
            .unwrap_or_else(Vec::new);

        Ok(Response::new(GetReplicasResponse {
            address_list: Some(AddressList { address: addrs }),
        }))
    }

    async fn get_resource(
        &self,
        _request: Request<GetResourceRequest>,
    ) -> Result<Response<GetResourceResponse>, Status> {
        // Keep this intentionally lightweight; it is used for monitoring and
        // scheduling in upstream deployments. We return placeholders for now.
        Ok(Response::new(GetResourceResponse {
            address: String::new(),
            shard_id: 0,
            replica_id: 0,
            memory: 0,
            cpu: -1.0,
            network: -1.0,
            work_load: -1.0,
        }))
    }

    async fn heart_beat(
        &self,
        request: Request<HeartBeatRequest>,
    ) -> Result<Response<HeartBeatResponse>, Status> {
        let req = request.into_inner();
        debug!("HeartBeat: server_type={}", req.server_type);

        let replicas = self.replicas.read();
        let mut addresses: HashMap<String, AddressList> = HashMap::new();

        // Upstream response is keyed by string; we’ll use the enum name.
        for (server_type_i32, addrs) in replicas.iter() {
            let key = match ServerType::try_from(*server_type_i32) {
                Ok(ServerType::Ps) => "PS",
                Ok(ServerType::Entry) => "ENTRY",
                Ok(ServerType::Dense) => "DENSE",
                _ => "UNKNOWN",
            };
            addresses.insert(
                key.to_string(),
                AddressList {
                    address: addrs.clone(),
                },
            );
        }

        Ok(Response::new(HeartBeatResponse { addresses }))
    }
}

/// Running server handle (for tests / embedding into apps).
#[derive(Debug)]
pub struct AgentGrpcServer {
    shutdown_tx: Arc<RwLock<Option<oneshot::Sender<()>>>>,
}

impl AgentGrpcServer {
    /// Start serving AgentService at the given address.
    pub async fn serve(addr: SocketAddr, svc: AgentServiceRealImpl) -> ServingResult<Self> {
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let shutdown_tx = Arc::new(RwLock::new(Some(shutdown_tx)));

        tokio::spawn(async move {
            info!("AgentService (real tonic) listening on {}", addr);
            let server = tonic::transport::Server::builder()
                .add_service(AgentServiceServer::new(svc))
                .serve_with_shutdown(addr, async {
                    let _ = shutdown_rx.await;
                    info!("AgentService shutdown signal received");
                });

            if let Err(e) = server.await {
                error!("AgentService server error: {}", e);
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

/// Helper: connect a real tonic client.
pub async fn connect_agent_client(
    addr: impl AsRef<str>,
) -> ServingResult<
    monolith_proto::monolith::serving::agent_service::agent_service_client::AgentServiceClient<
        tonic::transport::Channel,
    >,
> {
    let endpoint = format!("http://{}", addr.as_ref());
    let client = monolith_proto::monolith::serving::agent_service::agent_service_client::AgentServiceClient::connect(
        endpoint,
    )
    .await
    .map_err(|e| ServingError::server(format!("connect failed: {}", e)))?;
    Ok(client)
}
