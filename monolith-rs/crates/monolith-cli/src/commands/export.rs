//! Export Command Implementation
//!
//! Provides checkpoint export functionality for model deployment.
//! Supports various export formats for different serving environments.

use anyhow::{Context, Result};
use clap::Args;
use monolith_checkpoint::{
    Checkpointer, ExportConfig as CkptExportConfig, ExportFormat as CkptExportFormat,
    JsonCheckpointer, ModelExporter,
};
use std::fs;
use std::path::PathBuf;
use tracing::{info, warn};

/// Export format for the model
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum ExportFormat {
    /// SavedModel format (TensorFlow Serving compatible)
    #[default]
    SavedModel,
    /// ONNX format for cross-platform deployment
    Onnx,
    /// TorchScript format for PyTorch Serving
    TorchScript,
    /// Native Monolith format
    Native,
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportFormat::SavedModel => write!(f, "saved_model"),
            ExportFormat::Onnx => write!(f, "onnx"),
            ExportFormat::TorchScript => write!(f, "torchscript"),
            ExportFormat::Native => write!(f, "native"),
        }
    }
}

/// Export a checkpoint for deployment
///
/// This command exports a trained model checkpoint to a format
/// suitable for production serving. Supports multiple output formats
/// and optimization options.
///
/// # Example
///
/// ```bash
/// monolith export \
///     --checkpoint-path /path/to/checkpoint \
///     --output-path /path/to/export \
///     --format saved-model
/// ```
#[derive(Args, Debug, Clone)]
pub struct ExportCommand {
    /// Path to the checkpoint to export
    #[arg(long, short = 'c', env = "MONOLITH_CHECKPOINT_PATH")]
    pub checkpoint_path: PathBuf,

    /// Output path for the exported model
    #[arg(long, short = 'o', env = "MONOLITH_EXPORT_PATH")]
    pub output_path: PathBuf,

    /// Export format
    #[arg(long, short = 'f', default_value = "saved-model")]
    pub format: ExportFormat,

    /// Optimize the model for inference
    #[arg(long, default_value = "true")]
    pub optimize: bool,

    /// Enable quantization (reduces model size)
    #[arg(long)]
    pub quantize: bool,

    /// Quantization precision (8 or 16 bits)
    #[arg(long, default_value = "8")]
    pub quantize_bits: u8,

    /// Include embedding tables in export
    #[arg(long, default_value = "true")]
    pub include_embeddings: bool,

    /// Model signature name for serving
    #[arg(long, default_value = "serving_default")]
    pub signature_name: String,

    /// Optional exported model version label.
    #[arg(long)]
    pub model_version: Option<String>,

    /// Overwrite existing export directory
    #[arg(long)]
    pub overwrite: bool,

    /// Add warmup data for inference optimization
    #[arg(long)]
    pub warmup_data_path: Option<PathBuf>,
}

impl ExportCommand {
    fn resolve_checkpoint_path(&self, checkpointer: &JsonCheckpointer) -> Result<PathBuf> {
        if self.checkpoint_path.is_file() {
            return Ok(self.checkpoint_path.clone());
        }

        if self.checkpoint_path.is_dir() {
            let latest = checkpointer
                .latest(&self.checkpoint_path)
                .with_context(|| {
                    format!(
                        "No checkpoint-*.json found under directory: {:?}",
                        self.checkpoint_path
                    )
                })?;
            return Ok(latest);
        }

        anyhow::bail!(
            "Checkpoint path must be a file or directory: {:?}",
            self.checkpoint_path
        )
    }

    fn quantize_state_in_place(
        state: &mut monolith_checkpoint::ModelState,
        bits: u8,
    ) -> Result<()> {
        if bits != 8 && bits != 16 {
            anyhow::bail!("Unsupported quantize bits: {} (expected 8 or 16)", bits);
        }

        fn quantize_slice(values: &mut [f32], bits: u8) {
            let max_abs = values
                .iter()
                .copied()
                .fold(0.0_f32, |acc, v| acc.max(v.abs()));
            if max_abs == 0.0 {
                return;
            }

            let qmax = ((1_i32 << (bits - 1)) - 1) as f32;
            let scale = qmax / max_abs;
            for v in values.iter_mut() {
                let q = (*v * scale).round().clamp(-qmax, qmax);
                *v = q / scale;
            }
        }

        for values in state.dense_params.values_mut() {
            quantize_slice(values, bits);
        }
        for table in &mut state.hash_tables {
            for values in table.entries.values_mut() {
                quantize_slice(values, bits);
            }
        }
        Ok(())
    }

    fn maybe_copy_warmup_assets(&self) -> Result<()> {
        let Some(warmup_path) = &self.warmup_data_path else {
            return Ok(());
        };

        if !warmup_path.exists() {
            warn!("Warmup data path does not exist: {:?}", warmup_path);
            return Ok(());
        }

        let warmup_dir = self.output_path.join("warmup");
        fs::create_dir_all(&warmup_dir).context("Failed to create warmup output directory")?;
        if warmup_path.is_dir() {
            let entries = fs::read_dir(warmup_path)
                .with_context(|| format!("Failed to read warmup directory {:?}", warmup_path))?;
            for entry in entries {
                let entry = entry.context("Failed to read warmup dir entry")?;
                let src = entry.path();
                let dst = warmup_dir.join(entry.file_name());
                if src.is_file() {
                    fs::copy(&src, &dst).with_context(|| {
                        format!("Failed to copy warmup file {:?} -> {:?}", src, dst)
                    })?;
                }
            }
        } else {
            let filename = warmup_path
                .file_name()
                .with_context(|| format!("Invalid warmup file name for path {:?}", warmup_path))?;
            let dst = warmup_dir.join(filename);
            fs::copy(warmup_path, &dst).with_context(|| {
                format!("Failed to copy warmup file {:?} -> {:?}", warmup_path, dst)
            })?;
        }
        Ok(())
    }

    /// Execute the export command
    pub async fn run(&self) -> Result<()> {
        info!("Starting model export...");
        info!("Checkpoint path: {:?}", self.checkpoint_path);
        info!("Output path: {:?}", self.output_path);
        info!("Export format: {}", self.format);

        // Validate checkpoint exists
        if !self.checkpoint_path.exists() {
            anyhow::bail!("Checkpoint path does not exist: {:?}", self.checkpoint_path);
        }

        // Check if output path exists
        if self.output_path.exists() {
            if self.overwrite {
                warn!("Output path exists, overwriting: {:?}", self.output_path);
                std::fs::remove_dir_all(&self.output_path)
                    .context("Failed to remove existing output directory")?;
            } else {
                anyhow::bail!(
                    "Output path already exists: {:?}. Use --overwrite to replace.",
                    self.output_path
                );
            }
        }

        // Create output directory
        std::fs::create_dir_all(&self.output_path).context("Failed to create output directory")?;

        // Log export configuration
        info!("Export configuration:");
        info!("  - Optimize: {}", self.optimize);
        info!("  - Quantize: {}", self.quantize);
        if self.quantize {
            info!("  - Quantize bits: {}", self.quantize_bits);
        }
        info!("  - Include embeddings: {}", self.include_embeddings);
        info!("  - Signature: {}", self.signature_name);

        // Resolve checkpoint path and load model state.
        let checkpointer = JsonCheckpointer::new();
        let resolved_checkpoint = self.resolve_checkpoint_path(&checkpointer)?;
        info!("Resolved checkpoint to: {:?}", resolved_checkpoint);

        let mut state = checkpointer
            .restore(&resolved_checkpoint)
            .with_context(|| format!("Failed to restore checkpoint {:?}", resolved_checkpoint))?;

        // Optionally strip embedding tables for dense-only exports.
        if !self.include_embeddings {
            state.hash_tables.clear();
        }

        // Optional quantization pass.
        if self.quantize {
            Self::quantize_state_in_place(&mut state, self.quantize_bits)?;
        }

        // Keep a small metadata trail for exported artifacts.
        state.set_metadata(
            "export_source_checkpoint",
            resolved_checkpoint.to_string_lossy().to_string(),
        );
        state.set_metadata("export_signature_name", self.signature_name.clone());
        state.set_metadata("export_optimized", self.optimize.to_string());
        state.set_metadata("export_quantized", self.quantize.to_string());
        if self.quantize {
            state.set_metadata("export_quantize_bits", self.quantize_bits.to_string());
        }

        // Map CLI format to checkpoint exporter format.
        let ckpt_format = match self.format {
            ExportFormat::SavedModel => CkptExportFormat::SavedModel,
            ExportFormat::Native => CkptExportFormat::Binary,
            ExportFormat::Onnx => {
                anyhow::bail!("ONNX export is not implemented yet");
            }
            ExportFormat::TorchScript => {
                anyhow::bail!("TorchScript export is not implemented yet");
            }
        };

        let mut export_cfg = CkptExportConfig::new(&self.output_path).with_format(ckpt_format);
        // For serving exports we default to inference-only optimizer stripping.
        export_cfg.include_optimizer = false;
        export_cfg.metadata.insert(
            "signature_name".to_string(),
            self.signature_name.to_string(),
        );
        if let Some(version) = &self.model_version {
            export_cfg = export_cfg.with_version(version.clone());
        }

        let exporter = ModelExporter::new(export_cfg);
        exporter
            .export(&state)
            .with_context(|| format!("Failed to export model to {:?}", self.output_path))?;

        // Handle warmup assets, if provided.
        self.maybe_copy_warmup_assets()?;

        info!("Model exported successfully to: {:?}", self.output_path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use monolith_checkpoint::{Checkpointer, JsonCheckpointer, ModelExporter, ModelState};
    use tempfile::tempdir;

    #[test]
    fn test_export_format_display() {
        assert_eq!(ExportFormat::SavedModel.to_string(), "saved_model");
        assert_eq!(ExportFormat::Onnx.to_string(), "onnx");
        assert_eq!(ExportFormat::TorchScript.to_string(), "torchscript");
        assert_eq!(ExportFormat::Native.to_string(), "native");
    }

    #[test]
    fn test_export_command_defaults() {
        let cmd = ExportCommand {
            checkpoint_path: PathBuf::from("/tmp/checkpoint"),
            output_path: PathBuf::from("/tmp/export"),
            format: ExportFormat::SavedModel,
            optimize: true,
            quantize: false,
            quantize_bits: 8,
            include_embeddings: true,
            signature_name: "serving_default".to_string(),
            model_version: None,
            overwrite: false,
            warmup_data_path: None,
        };

        assert!(cmd.optimize);
        assert!(!cmd.quantize);
        assert_eq!(cmd.quantize_bits, 8);
        assert!(cmd.include_embeddings);
    }

    #[tokio::test]
    async fn test_export_saved_model_from_checkpoint_file() {
        let dir = tempdir().unwrap();
        let ckpt_path = dir.path().join("checkpoint-12.json");
        let out = dir.path().join("export");

        let mut state = ModelState::new(12);
        state.add_dense_param("linear.weight", vec![1.0, 2.0, 3.0]);
        JsonCheckpointer::new().save(&ckpt_path, &state).unwrap();

        let cmd = ExportCommand {
            checkpoint_path: ckpt_path,
            output_path: out.clone(),
            format: ExportFormat::SavedModel,
            optimize: true,
            quantize: false,
            quantize_bits: 8,
            include_embeddings: true,
            signature_name: "serving_default".to_string(),
            overwrite: false,
            warmup_data_path: None,
            model_version: Some("1.2.3".to_string()),
        };

        cmd.run().await.unwrap();
        assert!(out.join("manifest.json").exists());
        assert!(out.join("dense/params.json").exists());

        let manifest = ModelExporter::load_manifest(&out).unwrap();
        assert_eq!(manifest.global_step, 12);
        assert_eq!(manifest.version, "1.2.3");
    }

    #[tokio::test]
    async fn test_export_uses_latest_checkpoint_from_directory() {
        let dir = tempdir().unwrap();
        let ckpt_dir = dir.path().join("checkpoints");
        let out = dir.path().join("export");
        std::fs::create_dir_all(&ckpt_dir).unwrap();

        let cp = JsonCheckpointer::new();
        cp.save(&ckpt_dir.join("checkpoint-1.json"), &ModelState::new(1))
            .unwrap();
        cp.save(&ckpt_dir.join("checkpoint-9.json"), &ModelState::new(9))
            .unwrap();

        let cmd = ExportCommand {
            checkpoint_path: ckpt_dir,
            output_path: out.clone(),
            format: ExportFormat::SavedModel,
            optimize: true,
            quantize: false,
            quantize_bits: 8,
            include_embeddings: true,
            signature_name: "serving_default".to_string(),
            overwrite: false,
            warmup_data_path: None,
            model_version: None,
        };

        cmd.run().await.unwrap();
        let manifest = ModelExporter::load_manifest(&out).unwrap();
        assert_eq!(manifest.global_step, 9);
    }

    #[tokio::test]
    async fn test_export_unsupported_format_errors() {
        let dir = tempdir().unwrap();
        let ckpt_path = dir.path().join("checkpoint-1.json");
        JsonCheckpointer::new()
            .save(&ckpt_path, &ModelState::new(1))
            .unwrap();

        let cmd = ExportCommand {
            checkpoint_path: ckpt_path,
            output_path: dir.path().join("export"),
            format: ExportFormat::Onnx,
            optimize: true,
            quantize: false,
            quantize_bits: 8,
            include_embeddings: true,
            signature_name: "serving_default".to_string(),
            overwrite: false,
            warmup_data_path: None,
            model_version: None,
        };

        let err = cmd.run().await.unwrap_err().to_string();
        assert!(err.contains("ONNX export is not implemented yet"));
    }

    #[tokio::test]
    async fn test_export_invalid_quantize_bits_errors() {
        let dir = tempdir().unwrap();
        let ckpt_path = dir.path().join("checkpoint-1.json");
        JsonCheckpointer::new()
            .save(&ckpt_path, &ModelState::new(1))
            .unwrap();

        let cmd = ExportCommand {
            checkpoint_path: ckpt_path,
            output_path: dir.path().join("export"),
            format: ExportFormat::SavedModel,
            optimize: true,
            quantize: true,
            quantize_bits: 4,
            include_embeddings: true,
            signature_name: "serving_default".to_string(),
            overwrite: false,
            warmup_data_path: None,
            model_version: None,
        };

        let err = cmd.run().await.unwrap_err().to_string();
        assert!(err.contains("Unsupported quantize bits"));
    }
}
