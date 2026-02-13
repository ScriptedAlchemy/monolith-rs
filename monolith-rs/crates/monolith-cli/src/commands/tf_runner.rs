//! Python runner parity entrypoints.
//!
//! Python provides `gpu_runner.py` and `tpu_runner.py` which are thin wrappers around
//! TF Estimator execution. Rust does not embed TensorFlow runtime, but we can
//! preserve CLI/env/config semantics and filesystem side-effects:
//! - resolves `task` via model_registry (where available)
//! - creates `model_dir/eval/` on eval modes
//! - writes TensorBoard event files with scalar summaries (compatible with TF)
//! - preserves "overwrite_end_date" behavior for TPU runner params
//!
//! This lets tooling and integration scripts that expect the runner interface
//! continue working in Rust-only environments.

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use monolith_core::model_registry;
use monolith_proto::tensorflow_core as tf;
use prost::Message;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Args, Debug, Clone)]
pub struct TfRunnerCommand {
    #[command(subcommand)]
    pub cmd: TfRunnerSubcommand,
}

#[derive(Subcommand, Debug, Clone)]
pub enum TfRunnerSubcommand {
    /// GPU runner parity (`monolith/gpu_runner.py`).
    Gpu(GpuRunnerArgs),
    /// TPU runner parity (`monolith/tpu_runner.py`).
    Tpu(TpuRunnerArgs),
}

impl TfRunnerCommand {
    pub async fn run(&self) -> Result<()> {
        match &self.cmd {
            TfRunnerSubcommand::Gpu(args) => args.run().await,
            TfRunnerSubcommand::Tpu(args) => args.run().await,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum RunnerMode {
    TrainAndEval,
    Train,
    Eval,
}

impl RunnerMode {
    fn as_python_str(&self) -> &'static str {
        match self {
            RunnerMode::TrainAndEval => "train_and_eval",
            RunnerMode::Train => "train",
            RunnerMode::Eval => "eval",
        }
    }
}

/// Minimal TF event writer that writes a single Event record containing a Summary.
///
/// We use TFRecord encoding for event files. This is a strict subset of what the
/// Python FileWriter produces, but TensorBoard can still read scalar summaries.
struct TfEventWriter {
    writer: monolith_data::TFRecordWriter<File>,
}

impl TfEventWriter {
    fn new(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir).context("Failed to create eval output dir")?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let pid = std::process::id();
        let host = hostname::get()
            .ok()
            .and_then(|h| h.into_string().ok())
            .unwrap_or_else(|| "unknown".to_string());

        // Match typical TF naming: events.out.tfevents.<secs>.<host>.<pid>
        let fname = format!("events.out.tfevents.{}.{}.{}", now, host, pid);
        let path = dir.join(fname);
        let f = File::create(&path).context("Failed to create tfevents file")?;
        Ok(Self {
            writer: monolith_data::TFRecordWriter::new(f),
        })
    }

    fn write_scalars(&mut self, logs: &[(String, f32)], step: i64) -> Result<()> {
        let values: Vec<tf::summary::Value> = logs
            .iter()
            .map(|(tag, v)| tf::summary::Value {
                node_name: String::new(),
                tag: tag.clone(),
                metadata: None,
                value: Some(tf::summary::value::Value::SimpleValue(*v)),
            })
            .collect();
        let summary = tf::Summary { value: values };

        let wall_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let ev = tf::Event {
            wall_time,
            step,
            source_metadata: None,
            what: Some(tf::event::What::Summary(summary)),
        };

        let bytes = ev.encode_to_vec();
        self.writer
            .write_record(&bytes)
            .context("Failed to write TF event record")?;
        self.writer.flush().ok(); // best effort
        Ok(())
    }
}

#[derive(Args, Debug, Clone)]
pub struct GpuRunnerArgs {
    /// Name of the task class to run.
    #[arg(long)]
    pub task: String,

    /// The directory where the model and summaries are stored.
    #[arg(long)]
    pub model_dir: PathBuf,

    /// Save checkpoint every N steps. (Ignored by Rust runner; preserved for parity.)
    #[arg(long)]
    pub save_checkpoints_steps: Option<i64>,

    /// Job mode.
    #[arg(long, value_enum, default_value_t = RunnerMode::Train)]
    pub mode: RunnerMode,
}

impl GpuRunnerArgs {
    pub async fn run(&self) -> Result<()> {
        // Preserve registry lookup semantics: error if task isn't registered.
        let _task_params = model_registry::get_params(&self.task)
            .with_context(|| format!("Failed to resolve --task {}", self.task))?;

        tracing::info!(
            task = %self.task,
            model_dir = %self.model_dir.display(),
            mode = %self.mode.as_python_str(),
            "Starting GPU runner (Rust parity stub)"
        );

        match self.mode {
            RunnerMode::Train => {
                // No-op training stub: create model_dir and return.
                fs::create_dir_all(&self.model_dir).context("Failed to create model_dir")?;
                Ok(())
            }
            RunnerMode::Eval | RunnerMode::TrainAndEval => {
                // Emit a minimal eval summary so downstream tooling sees the same structure.
                let eval_dir = self.model_dir.join("eval");
                let mut w = TfEventWriter::new(&eval_dir)?;
                w.write_scalars(&[("dummy_loss".to_string(), 0.0)], 0)?;
                Ok(())
            }
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct TpuRunnerArgs {
    /// TensorFlow version (Python flag). Preserved for parity.
    #[arg(long, default_value = "nightly")]
    pub tf_version: String,

    /// The Cloud TPU to use for training.
    #[arg(long)]
    pub tpu: Option<String>,

    /// GCP project.
    #[arg(long)]
    pub gcp_project: Option<String>,

    /// TPU zone.
    #[arg(long)]
    pub tpu_zone: Option<String>,

    /// Name of the task class to run.
    #[arg(long)]
    pub task: String,

    /// The directory where the model and summaries are stored.
    #[arg(long)]
    pub model_dir: PathBuf,

    /// Job mode.
    #[arg(long, value_enum, default_value_t = RunnerMode::Train)]
    pub mode: RunnerMode,

    /// Save checkpoint every N steps.
    #[arg(long)]
    pub save_checkpoints_steps: Option<i64>,

    /// Train iterations per loop.
    #[arg(long, default_value_t = 10000)]
    pub iterations_per_loop: i64,

    /// TPU Embedding flags.
    #[arg(long, default_value_t = false)]
    pub pipeline_execution: bool,

    #[arg(long, default_value_t = true)]
    pub enable_tpu_version_config: bool,

    #[arg(long, default_value_t = 500)]
    pub host_call_every_n_steps: i64,

    #[arg(long, default_value_t = false)]
    pub enable_stopping_signals: bool,

    #[arg(long, default_value_t = false)]
    pub cpu_test: bool,

    #[arg(long, default_value = "mod")]
    pub partition_strategy: String,

    #[arg(long, default_value = "")]
    pub overwrite_end_date: String,
}

impl TpuRunnerArgs {
    pub async fn run(&self) -> Result<()> {
        let mut params = model_registry::get_params(&self.task)
            .with_context(|| format!("Failed to resolve --task {}", self.task))?;

        // Mirror Python: force accelerator="tpu".
        let _ = params.set("accelerator", "tpu");

        // Mirror overwrite_end_date behavior if present and train.end_date exists.
        if !self.overwrite_end_date.is_empty() {
            if let Ok(train) = params.get_params("train") {
                if train.contains("end_date") {
                    let _ = params.set("train.end_date", self.overwrite_end_date.clone());
                }
            }
        }

        tracing::info!(
            task = %self.task,
            model_dir = %self.model_dir.display(),
            mode = %self.mode.as_python_str(),
            cpu_test = self.cpu_test,
            tpu = ?self.tpu,
            "Starting TPU runner (Rust parity stub)"
        );

        if self.cpu_test && self.mode != RunnerMode::Train {
            anyhow::bail!("Cpu test can only work with train mode.");
        }

        match self.mode {
            RunnerMode::Train => {
                fs::create_dir_all(&self.model_dir).context("Failed to create model_dir")?;
                Ok(())
            }
            RunnerMode::Eval => {
                let eval_dir = self.model_dir.join("eval");
                let mut w = TfEventWriter::new(&eval_dir)?;
                w.write_scalars(&[("dummy_eval_loss".to_string(), 0.0)], 0)?;
                Ok(())
            }
            RunnerMode::TrainAndEval => {
                // Python raises TypeError for TPU train_and_eval.
                anyhow::bail!("train_and_eval has not been supported.");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use monolith_core::base_model_params::SingleTaskModelParams;
    use monolith_core::hyperparams::Params;
    use monolith_core::{model_registry as reg, register_single_task_model};
    use std::sync::Mutex;
    use tempfile::tempdir;

    // The model registry is a global singleton. These tests register/clear keys and
    // are not safe under parallel execution.
    static REGISTRY_TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[derive(Default)]
    struct Dummy;
    impl SingleTaskModelParams for Dummy {
        fn task(&self) -> Params {
            let mut p = monolith_core::base_task::base_task_params();
            // Add train.end_date to test overwrite behavior.
            let _ = p
                .get_params_mut("train")
                .unwrap()
                .define("end_date", "", "end date");
            p
        }
    }

    #[test]
    fn test_gpu_runner_eval_writes_events_file() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        reg::clear_registry_for_test();
        // Tests run in parallel; use a unique key to avoid cross-test registry collisions.
        register_single_task_model!("dummy.eval_writes_events_file", Dummy).unwrap();

        let tmp = tempdir().unwrap();
        let args = GpuRunnerArgs {
            task: "dummy.eval_writes_events_file".to_string(),
            model_dir: tmp.path().join("model"),
            save_checkpoints_steps: None,
            mode: RunnerMode::Eval,
        };
        futures::executor::block_on(args.run()).unwrap();

        let eval_dir = tmp.path().join("model").join("eval");
        let entries: Vec<_> = fs::read_dir(&eval_dir).unwrap().collect();
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_tpu_overwrite_end_date() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        reg::clear_registry_for_test();
        // Tests run in parallel; use a unique key to avoid cross-test registry collisions.
        register_single_task_model!("dummy.tpu_overwrite_end_date", Dummy).unwrap();

        let tmp = tempdir().unwrap();
        let args = TpuRunnerArgs {
            tf_version: "nightly".to_string(),
            tpu: None,
            gcp_project: None,
            tpu_zone: None,
            task: "dummy.tpu_overwrite_end_date".to_string(),
            model_dir: tmp.path().join("model"),
            mode: RunnerMode::Train,
            save_checkpoints_steps: None,
            iterations_per_loop: 1,
            pipeline_execution: false,
            enable_tpu_version_config: true,
            host_call_every_n_steps: 500,
            enable_stopping_signals: false,
            cpu_test: false,
            partition_strategy: "mod".to_string(),
            overwrite_end_date: "20250101".to_string(),
        };
        futures::executor::block_on(args.run()).unwrap();
    }
}
