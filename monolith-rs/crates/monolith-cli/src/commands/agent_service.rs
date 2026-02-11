//! Agent-service related CLI commands (Python parity).
//!
//! The upstream repository exposes several small Python CLIs under
//! `monolith/agent_service/` (agent runner, controller, and discovery clients).
//! This module ports the flag surface and basic wiring to Rust.

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use monolith_proto::monolith::serving::agent_service as agent_proto;
use monolith_serving::agent_controller::{bzid_info, declare_saved_model, map_model_to_layout};
use monolith_serving::agent_service_discovery::connect_agent_service_client;
use monolith_serving::agent_v3::AgentV3;
use monolith_serving::{AgentConfig, DeployType, FakeKazooClient, ZkBackend, ZkClient};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Args, Debug, Clone)]
pub struct AgentServiceCommand {
    #[command(subcommand)]
    pub command: AgentServiceSubcommand,
}

#[derive(Subcommand, Debug, Clone)]
pub enum AgentServiceSubcommand {
    /// Run the agent process (Python parity: `monolith/agent_service/agent.py`).
    Agent(AgentCommand),

    /// Manage saved_model declarations and layouts (Python parity: `agent_controller.py`).
    Controller(AgentControllerCommand),

    /// Query an AgentService server (Python parity: `svr_client.py` / subset of `agent_client.py`).
    Client(AgentClientCommand),
}

impl AgentServiceCommand {
    pub async fn run(&self) -> Result<()> {
        match &self.command {
            AgentServiceSubcommand::Agent(cmd) => cmd.run().await,
            AgentServiceSubcommand::Controller(cmd) => cmd.run().await,
            AgentServiceSubcommand::Client(cmd) => cmd.run().await,
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct AgentCommand {
    /// Agent conf file (Python parity: global `--conf` flag in `utils.py`).
    #[arg(long, env = "MONOLITH_AGENT_CONF")]
    pub conf: PathBuf,

    /// The tfs log file path (Python parity: `--tfs_log`).
    #[arg(long, default_value = "/var/log/tfs.std.log")]
    pub tfs_log: String,

    /// Use an in-memory fake ZooKeeper client (useful for local testing).
    ///
    /// The upstream Python agent connects to real ZooKeeper via `MonolithKazooClient`.
    /// Rust parity tests use `FakeKazooClient`; full production ZK wiring is out of scope here.
    #[arg(long, default_value_t = false)]
    pub fake_zk: bool,
}

impl AgentCommand {
    pub async fn run(&self) -> Result<()> {
        let conf = AgentConfig::from_file(&self.conf)
            .with_context(|| format!("failed to read agent config: {}", self.conf.display()))?;

        // Only AgentV3 is wired here; v1/v2 lifecycle requires real TFServing + ZK integration.
        if conf.agent_version != 3 || conf.deploy_type != DeployType::Unified {
            anyhow::bail!(
                "only agent_version=3 with deploy_type=unified is supported by this CLI (got agent_version={} deploy_type={:?})",
                conf.agent_version,
                conf.deploy_type
            );
        }

        let zk: Arc<dyn ZkClient> = if self.fake_zk {
            let zk = Arc::new(FakeKazooClient::new());
            zk.start().context("failed to start fake zk")?;
            zk
        } else {
            anyhow::bail!(
                "real ZooKeeper client is not wired in Rust yet; rerun with --fake-zk for local testing"
            );
        };

        // The Rust agent uses a file-based TFServing wrapper; in CLI mode we don't spawn a TFServing process.
        let agent = AgentV3::new(conf, zk)?;

        // Keep the agent alive until Ctrl-C.
        agent.start().await?;
        tracing::info!(
            "agent started (tfs_log flag is accepted for parity: {})",
            self.tfs_log
        );
        tokio::signal::ctrl_c().await?;
        agent.stop().await;
        Ok(())
    }
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentControllerCmd {
    Decl,
    Pub,
    Unpub,
    BzidInfo,
}

#[derive(Args, Debug, Clone)]
pub struct AgentControllerCommand {
    /// zk connection string (Python parity: `--zk_servers`).
    #[arg(long, default_value = "")]
    pub zk_servers: String,

    /// namespace (Python parity: `--bzid`).
    #[arg(long, default_value = "test")]
    pub bzid: String,

    /// exported model base path (Python parity: `--export_base`).
    #[arg(long)]
    pub export_base: Option<PathBuf>,

    /// overwrite existing saved_model configs (Python parity: `--overwrite` int flag).
    #[arg(long, default_value_t = 0)]
    pub overwrite: i32,

    /// model_name or model_pattern (Python parity: `--model_name`).
    #[arg(long, default_value = "")]
    pub model_name: String,

    /// layout base (Python parity: `--layout`).
    #[arg(long, default_value = "")]
    pub layout: String,

    /// serving architecture (Python parity: `--arch`, default entry_ps).
    #[arg(long, default_value = "entry_ps")]
    pub arch: String,

    /// controller command (Python parity: `--cmd`).
    #[arg(long, value_enum, default_value_t = AgentControllerCmd::BzidInfo)]
    pub cmd: AgentControllerCmd,

    /// Use an in-memory fake ZooKeeper client (local testing).
    #[arg(long, default_value_t = false)]
    pub fake_zk: bool,
}

impl AgentControllerCommand {
    pub async fn run(&self) -> Result<()> {
        let zk: Arc<dyn ZkClient> = if self.fake_zk {
            let zk = Arc::new(FakeKazooClient::new());
            zk.start().context("failed to start fake zk")?;
            zk
        } else {
            anyhow::bail!(
                "real ZooKeeper client is not wired in Rust yet; rerun with --fake-zk for local testing"
            );
        };

        let backend = ZkBackend::new(self.bzid.clone(), zk);
        backend.start().context("backend start failed")?;

        match self.cmd {
            AgentControllerCmd::Decl => {
                let export_base = self
                    .export_base
                    .as_ref()
                    .context("--export-base is required for --cmd decl")?;
                let model_name = if self.model_name.is_empty() {
                    None
                } else {
                    Some(self.model_name.as_str())
                };
                let model_name = declare_saved_model(
                    &backend,
                    export_base,
                    model_name,
                    self.overwrite != 0,
                    &self.arch,
                )
                .context("declare_saved_model failed")?;
                println!("{model_name}");
            }
            AgentControllerCmd::Pub | AgentControllerCmd::Unpub => {
                if self.layout.is_empty() || self.model_name.is_empty() {
                    anyhow::bail!("--layout and --model-name are required for pub/unpub");
                }
                let layout_path = format!("/{}/layouts/{}", self.bzid, self.layout);
                let action = match self.cmd {
                    AgentControllerCmd::Pub => "pub",
                    AgentControllerCmd::Unpub => "unpub",
                    _ => unreachable!(),
                };
                map_model_to_layout(&backend, &self.model_name, &layout_path, action)
                    .context("map_model_to_layout failed")?;
            }
            AgentControllerCmd::BzidInfo => {
                let v = bzid_info(&backend).context("bzid_info failed")?;
                println!("{}", serde_json::to_string_pretty(&v)?);
            }
        }

        backend.stop().ok();
        Ok(())
    }
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientCmdType {
    Hb,
    Gr,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientServerType {
    Ps,
    Entry,
    Dense,
}

#[derive(Args, Debug, Clone)]
pub struct AgentClientCommand {
    /// Agent conf file (Python parity: `--conf`).
    #[arg(long, env = "MONOLITH_AGENT_CONF")]
    pub conf: Option<PathBuf>,

    /// Override agent port (Python parity: `agent_client.py --port`).
    #[arg(long, default_value_t = 0)]
    pub port: u16,

    /// Override target address (`host:port`). When set, it overrides config + env resolution.
    #[arg(long)]
    pub target: Option<String>,

    /// Command type: heartbeat or get_replicas.
    #[arg(long, value_enum, default_value_t = ClientCmdType::Hb)]
    pub cmd_type: ClientCmdType,

    /// Server type to query.
    #[arg(long, value_enum, default_value_t = ClientServerType::Ps)]
    pub server_type: ClientServerType,

    /// Task id for get_replicas.
    #[arg(long, default_value_t = 0)]
    pub task: i32,

    /// Model name for GetReplicas (proto requires it but v1 ignores it).
    #[arg(long, default_value = "")]
    pub model_name: String,
}

impl AgentClientCommand {
    pub async fn run(&self) -> Result<()> {
        let target = if let Some(t) = &self.target {
            t.clone()
        } else {
            // Python parity: use MY_HOST_IP if present, else local host.
            let host = std::env::var("MY_HOST_IP").unwrap_or_else(|_| "127.0.0.1".to_string());
            let mut port = self.port;
            if port == 0 {
                let conf_path = self
                    .conf
                    .as_ref()
                    .context("--conf is required when --target is not provided")?;
                let conf = AgentConfig::from_file(conf_path)?;
                port = conf.agent_port;
            }
            format!("{host}:{port}")
        };

        let mut client = connect_agent_service_client(&target).await?;
        let st = match self.server_type {
            ClientServerType::Ps => agent_proto::ServerType::Ps as i32,
            ClientServerType::Entry => agent_proto::ServerType::Entry as i32,
            ClientServerType::Dense => agent_proto::ServerType::Dense as i32,
        };

        match self.cmd_type {
            ClientCmdType::Hb => {
                let resp = client
                    .heart_beat(agent_proto::HeartBeatRequest { server_type: st })
                    .await?
                    .into_inner();
                println!("{:?}", resp.addresses);
            }
            ClientCmdType::Gr => {
                let resp = client
                    .get_replicas(agent_proto::GetReplicasRequest {
                        server_type: st,
                        task: self.task,
                        model_name: self.model_name.clone(),
                    })
                    .await?
                    .into_inner();
                let addrs = resp.address_list.map(|l| l.address).unwrap_or_default();
                println!("{:?}", addrs);
            }
        }

        Ok(())
    }
}
