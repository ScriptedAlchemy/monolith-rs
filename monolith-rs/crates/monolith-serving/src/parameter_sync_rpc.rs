//! Real gRPC implementation for parameter sync.
//!
//! Upstream Monolith defines the wire-format protos (`PushRequest`, `PushResponse`, ...)
//! but does not define a gRPC service in `parameter_sync.proto` (it is used via C++/TF ops).
//!
//! For Rust parity we define a small gRPC service in `proto/parameter_sync_rpc.proto`
//! and implement it with tonic here.

use crate::error::{ServingError, ServingResult};
use monolith_proto::monolith::parameter_sync::{PushRequest, PushResponse};
use monolith_proto::monolith::parameter_sync_rpc::parameter_sync_rpc_client::ParameterSyncRpcClient as TonicClient;
use monolith_proto::monolith::parameter_sync_rpc::parameter_sync_rpc_server::{
    ParameterSyncRpc, ParameterSyncRpcServer,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::{Request, Response, Status};

/// Sink for incoming parameter-sync pushes.
pub trait PushSink: Send + Sync {
    /// Apply an incoming delta update and return a `PushResponse`.
    fn handle_push(&self, req: PushRequest) -> ServingResult<PushResponse>;
}

/// Default push sink that just acknowledges the request.
#[derive(Debug, Default)]
pub struct NoopPushSink;

impl PushSink for NoopPushSink {
    fn handle_push(&self, _req: PushRequest) -> ServingResult<PushResponse> {
        Ok(PushResponse {
            target: None,
            status_code: Some(0),
            error_message: None,
            update_num: Some(0),
        })
    }
}

/// Tonic server implementation for `ParameterSyncService`.
#[derive(Clone)]
pub struct ParameterSyncGrpcServer {
    sink: Arc<dyn PushSink>,
}

impl ParameterSyncGrpcServer {
    /// Create a new gRPC server wrapper using the provided push sink.
    pub fn new(sink: Arc<dyn PushSink>) -> Self {
        Self { sink }
    }

    /// Convert this wrapper into a tonic service.
    pub fn into_service(self) -> ParameterSyncRpcServer<Self> {
        ParameterSyncRpcServer::new(self)
    }

    /// Serve the ParameterSync gRPC service at the given address.
    pub async fn serve(self, addr: SocketAddr) -> Result<(), tonic::transport::Error> {
        tonic::transport::Server::builder()
            .add_service(self.into_service())
            .serve(addr)
            .await
    }
}

#[tonic::async_trait]
impl ParameterSyncRpc for ParameterSyncGrpcServer {
    async fn push(&self, request: Request<PushRequest>) -> Result<Response<PushResponse>, Status> {
        let req = request.into_inner();
        let resp = self
            .sink
            .handle_push(req)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(resp))
    }
}

/// Real networked ParameterSync client.
#[derive(Debug, Clone)]
pub struct ParameterSyncRpcClient {
    inner: TonicClient<tonic::transport::Channel>,
}

impl ParameterSyncRpcClient {
    /// Connect to a ParameterSync gRPC server at `host:port`.
    pub async fn connect(target: impl AsRef<str>) -> ServingResult<Self> {
        let target = target.as_ref();
        let endpoint = if target.starts_with("http://") || target.starts_with("https://") {
            target.to_string()
        } else {
            format!("http://{}", target)
        };

        let inner = TonicClient::connect(endpoint)
            .await
            .map_err(|e| ServingError::server(format!("ParameterSync connect failed: {e}")))?;

        Ok(Self { inner })
    }

    /// Push delta embeddings to the server.
    pub async fn push(&mut self, req: PushRequest) -> ServingResult<PushResponse> {
        let resp = self
            .inner
            .push(Request::new(req))
            .await
            .map_err(|e| ServingError::server(format!("ParameterSync push failed: {e}")))?
            .into_inner();
        Ok(resp)
    }
}
