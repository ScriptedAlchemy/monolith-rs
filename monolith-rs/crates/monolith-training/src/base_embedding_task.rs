//! Base embedding task utilities mirroring Python `BaseEmbeddingTask`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use chrono::NaiveDate;
use glob::glob;
use monolith_core::base_task::{base_task_params, TaskMode};
use monolith_core::env::EnvBuilder;
use monolith_core::error::{MonolithError, Result};
use monolith_core::fid::SlotId;
use monolith_core::hyperparams::{ParamValue, Params};
use monolith_data::batch::BatchedDataset;
use monolith_data::dataset::Dataset;
use monolith_data::interleave::InterleavedDataset;
use monolith_data::tfrecord::TFRecordDataset;
use monolith_proto::Example;

/// Embedding task configuration matching Python defaults.
#[derive(Debug, Clone)]
pub struct BaseEmbeddingTaskConfig {
    pub base: BaseTaskConfig,
    pub vocab_size_per_slot: Option<usize>,
    pub custom_vocab_size_mapping: Option<HashMap<SlotId, usize>>,
    pub vocab_size_offset: Option<isize>,
    pub qr_multi_hashing: bool,
    pub qr_hashing_threshold: usize,
    pub qr_collision_rate: usize,
    pub vocab_file_path: Option<PathBuf>,
    pub enable_deepinsight: bool,
    pub enable_host_call_scalar_metrics: bool,
    pub enable_host_call_norm_metrics: bool,
    pub files_interleave_cycle_length: usize,
    pub deterministic: bool,
    pub gradient_multiplier: f32,
    pub enable_caching_with_tpu_var_mode: bool,
    pub top_k_sampling_num_per_core: usize,
    pub use_random_init_embedding_for_oov: bool,
    pub merge_vector: bool,
}

impl Default for BaseEmbeddingTaskConfig {
    fn default() -> Self {
        Self {
            base: BaseTaskConfig::default(),
            vocab_size_per_slot: None,
            custom_vocab_size_mapping: None,
            vocab_size_offset: None,
            qr_multi_hashing: false,
            qr_hashing_threshold: 100_000_000,
            qr_collision_rate: 4,
            vocab_file_path: None,
            enable_deepinsight: false,
            enable_host_call_scalar_metrics: false,
            enable_host_call_norm_metrics: false,
            files_interleave_cycle_length: 4,
            deterministic: false,
            gradient_multiplier: 1.0,
            enable_caching_with_tpu_var_mode: false,
            top_k_sampling_num_per_core: 6,
            use_random_init_embedding_for_oov: false,
            merge_vector: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BaseTaskConfig {
    pub input: InputConfig,
    pub eval: EvalConfig,
    pub train: TrainConfig,
}

impl Default for BaseTaskConfig {
    fn default() -> Self {
        Self {
            input: InputConfig::default(),
            eval: EvalConfig::default(),
            train: TrainConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InputConfig {
    pub eval_examples: Option<u64>,
    pub train_examples: Option<u64>,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            eval_examples: None,
            train_examples: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub per_replica_batch_size: Option<usize>,
    pub steps_per_eval: u64,
    pub steps: Option<u64>,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            per_replica_batch_size: None,
            steps_per_eval: 10_000,
            steps: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub steps: Option<u64>,
    pub max_steps: Option<u64>,
    pub per_replica_batch_size: Option<usize>,
    pub file_pattern: Option<String>,
    pub repeat: bool,
    pub label_key: String,
    pub save_checkpoints_steps: Option<u64>,
    pub save_checkpoints_secs: Option<u64>,
    pub dense_only_save_checkpoints_secs: Option<u64>,
    pub dense_only_save_checkpoints_steps: Option<u64>,
    pub file_folder: Option<PathBuf>,
    pub date_and_file_name_format: String,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
    pub vocab_file_folder_prefix: Option<PathBuf>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: None,
            max_steps: None,
            per_replica_batch_size: None,
            file_pattern: None,
            repeat: false,
            label_key: "label".to_string(),
            save_checkpoints_steps: None,
            save_checkpoints_secs: None,
            dense_only_save_checkpoints_secs: None,
            dense_only_save_checkpoints_steps: None,
            file_folder: None,
            date_and_file_name_format: "*/*/part*".to_string(),
            start_date: None,
            end_date: None,
            vocab_file_folder_prefix: None,
        }
    }
}

/// Dataset wrapper for embedding tasks.
#[derive(Clone)]
pub enum EmbeddingDataset {
    Tf(TFRecordDataset),
    Interleaved(InterleavedDataset),
}

impl Dataset for EmbeddingDataset {
    type Iter = Box<dyn Iterator<Item = Example> + Send>;

    fn iter(self) -> Self::Iter {
        match self {
            EmbeddingDataset::Tf(ds) => Box::new(ds.iter()),
            EmbeddingDataset::Interleaved(ds) => Box::new(ds.iter()),
        }
    }
}

/// Base embedding task with vocab and dataset handling.
pub struct BaseEmbeddingTask {
    pub config: BaseEmbeddingTaskConfig,
    pub env: monolith_core::env::Env,
    pub vocab_size_dict: HashMap<SlotId, usize>,
}

impl BaseEmbeddingTask {
    /// Constructs a new BaseEmbeddingTask.
    pub fn new(mut config: BaseEmbeddingTaskConfig) -> Result<Self> {
        let vocab_size_dict = Self::create_vocab_dict(&mut config)?;
        let mut env_builder = EnvBuilder::new();
        for (slot_id, vocab_size) in &vocab_size_dict {
            env_builder = env_builder.with_vocab_size(*slot_id, *vocab_size);
        }
        let env = env_builder.build();
        Ok(Self {
            config,
            env,
            vocab_size_dict,
        })
    }

    /// Builds default params tree for embedding task.
    pub fn params() -> Params {
        let mut p = base_task_params();
        let _ = p.define("vocab_size_per_slot", ParamValue::None, "Fixed vocab size.");
        let _ = p.define(
            "custom_vocab_size_mapping",
            ParamValue::None,
            "Fixed vocab size for some slots.",
        );
        let _ = p.define("vocab_size_offset", ParamValue::None, "Vocab size offset.");
        let _ = p.define("qr_multi_hashing", false, "Enable QR multi hashing.");
        let _ = p.define(
            "qr_hashing_threshold",
            100_000_000_i64,
            "QR hashing threshold.",
        );
        let _ = p.define("qr_collision_rate", 4_i64, "QR collision rate.");
        let _ = p.define("vocab_file_path", ParamValue::None, "Vocab file path.");
        let _ = p.define("enable_deepinsight", false, "Enable deepinsight.");
        let _ = p.define(
            "enable_host_call_scalar_metrics",
            false,
            "Enable host call scalar metrics.",
        );
        let _ = p.define(
            "enable_host_call_norm_metrics",
            false,
            "Enable host call norm metrics.",
        );
        let _ = p.define(
            "files_interleave_cycle_length",
            4_i64,
            "Interleave cycle length.",
        );
        let _ = p.define("deterministic", false, "Deterministic training.");
        let _ = p.define("gradient_multiplier", 1.0_f64, "Gradient multiplier.");
        let _ = p.define(
            "enable_caching_with_tpu_var_mode",
            false,
            "Enable caching with TPU var mode.",
        );
        let _ = p.define(
            "top_k_sampling_num_per_core",
            6_i64,
            "Top K sampling per core.",
        );
        let _ = p.define(
            "use_random_init_embedding_for_oov",
            false,
            "Random init for OOV.",
        );
        let _ = p.define("merge_vector", false, "Merge vector tables.");

        // Extend train params
        if let Ok(ParamValue::Params(mut train)) = p.get("train").cloned() {
            let _ = train.define("file_folder", ParamValue::None, "Training file folder.");
            let _ = train.define(
                "date_and_file_name_format",
                "*/*/part*",
                "File name format.",
            );
            let _ = train.define("start_date", ParamValue::None, "Start date.");
            let _ = train.define("end_date", ParamValue::None, "End date.");
            let _ = train.define(
                "vocab_file_folder_prefix",
                ParamValue::None,
                "Vocab folder prefix.",
            );
            let _ = p.set("train", ParamValue::Params(train));
        }
        p
    }

    /// Creates vocab dict based on config.
    fn create_vocab_dict(config: &mut BaseEmbeddingTaskConfig) -> Result<HashMap<SlotId, usize>> {
        if config.base.train.end_date.is_some()
            && config.base.train.vocab_file_folder_prefix.is_some()
        {
            config.vocab_file_path = Self::download_vocab_size_file_from_hdfs(
                config.base.train.vocab_file_folder_prefix.as_ref().unwrap(),
                config.base.train.end_date.as_ref().unwrap(),
            )
            .ok()
            .or_else(|| config.vocab_file_path.clone());
        }

        let vocab_path =
            config
                .vocab_file_path
                .clone()
                .ok_or_else(|| MonolithError::ConfigError {
                    message:
                        "Either vocab_file_path or vocab_file_folder_prefix + end_date required"
                            .to_string(),
                })?;

        let mut vocab_size_dict = HashMap::new();
        let content =
            std::fs::read_to_string(&vocab_path).map_err(|e| MonolithError::InternalError {
                message: e.to_string(),
            })?;
        for line in content.lines() {
            let fields: Vec<&str> = line.trim().split('\t').collect();
            if fields.len() != 2 {
                return Err(MonolithError::ConfigError {
                    message: format!("each line in {:?} must have 2 fields", vocab_path),
                });
            }
            if !fields[0].chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let slot_id: SlotId = fields[0].parse().map_err(|_| MonolithError::ConfigError {
                message: format!("Invalid slot id {}", fields[0]),
            })?;
            let mut distinct = if let Some(v) = config.vocab_size_per_slot {
                v
            } else {
                fields[1]
                    .parse::<usize>()
                    .map_err(|_| MonolithError::ConfigError {
                        message: format!("Invalid vocab size {}", fields[1]),
                    })?
            };
            // Python only applies `custom_vocab_size_mapping` when `vocab_size_per_slot` is not set.
            if config.vocab_size_per_slot.is_none() {
                if let Some(custom) = &config.custom_vocab_size_mapping {
                    if let Some(val) = custom.get(&slot_id) {
                        distinct = *val;
                    }
                }
            }
            if let Some(offset) = config.vocab_size_offset {
                // Python applies the offset directly.
                let updated = (distinct as isize) + offset;
                if updated <= 0 {
                    return Err(MonolithError::ConfigError {
                        message: format!(
                            "vocab_size_offset results in non-positive vocab size for slot {}",
                            slot_id
                        ),
                    });
                }
                distinct = updated as usize;
            }
            vocab_size_dict.insert(slot_id, distinct);
        }

        Ok(vocab_size_dict)
    }

    #[cfg(test)]
    fn create_vocab_dict_for_test(
        config: &mut BaseEmbeddingTaskConfig,
    ) -> Result<HashMap<SlotId, usize>> {
        Self::create_vocab_dict(config)
    }

    /// Downloads vocab size file from HDFS and returns local path.
    fn download_vocab_size_file_from_hdfs(prefix: &Path, end_date: &str) -> Result<PathBuf> {
        let tmp_folder = PathBuf::from("temp");
        if tmp_folder.exists() {
            std::fs::remove_dir_all(&tmp_folder).map_err(|e| MonolithError::InternalError {
                message: e.to_string(),
            })?;
        }
        std::fs::create_dir_all(&tmp_folder).map_err(|e| MonolithError::InternalError {
            message: e.to_string(),
        })?;

        let hdfs_path = format!("{}/{}{}", prefix.display(), end_date, "/part*.csv");
        let cmd = format!(
            "hadoop fs -copyToLocal {} {}",
            hdfs_path,
            tmp_folder.display()
        );
        let status = Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .status()
            .map_err(|e| MonolithError::InternalError {
                message: e.to_string(),
            })?;
        let entries: Vec<PathBuf> = std::fs::read_dir(&tmp_folder)
            .map_err(|e| MonolithError::InternalError {
                message: e.to_string(),
            })?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        if status.success() && entries.len() == 1 {
            Ok(entries[0].clone())
        } else {
            Err(MonolithError::ConfigError {
                message: format!("Failed to download vocab file from {}", hdfs_path),
            })
        }
    }

    /// Resolves training file paths.
    fn resolve_training_files(&self) -> Result<Vec<PathBuf>> {
        if let Some(pattern) = &self.config.base.train.file_pattern {
            return expand_patterns(pattern);
        }
        let folder = self.config.base.train.file_folder.as_ref().ok_or_else(|| {
            MonolithError::ConfigError {
                message: "file_pattern or file_folder must be provided".to_string(),
            }
        })?;
        let start = self.config.base.train.start_date.as_ref().ok_or_else(|| {
            MonolithError::ConfigError {
                message: "start_date is required when using file_folder".to_string(),
            }
        })?;
        let end = self.config.base.train.end_date.as_ref().unwrap_or(start);

        let start_date =
            NaiveDate::parse_from_str(start, "%Y%m%d").map_err(|e| MonolithError::ConfigError {
                message: format!("Invalid start_date {}: {}", start, e),
            })?;
        let end_date =
            NaiveDate::parse_from_str(end, "%Y%m%d").map_err(|e| MonolithError::ConfigError {
                message: format!("Invalid end_date {}: {}", end, e),
            })?;

        let mut patterns = Vec::new();
        let mut date = start_date;
        while date <= end_date {
            let date_str = date.format("%Y%m%d").to_string();
            let pattern = format!(
                "{}/{}/{}",
                folder.display(),
                date_str,
                self.config.base.train.date_and_file_name_format
            );
            patterns.push(pattern);
            date = date.succ_opt().ok_or_else(|| MonolithError::ConfigError {
                message: "Date overflow".to_string(),
            })?;
        }

        let mut paths = Vec::new();
        for pattern in patterns {
            paths.extend(expand_patterns(&pattern)?);
        }
        if paths.is_empty() {
            return Err(MonolithError::ConfigError {
                message: "No training files found".to_string(),
            });
        }
        Ok(paths)
    }

    /// Creates a dataset for training.
    pub fn create_train_dataset(&self) -> Result<EmbeddingDataset> {
        let paths = self.resolve_training_files()?;
        let dataset =
            TFRecordDataset::open_multiple(&paths).map_err(|e| MonolithError::InternalError {
                message: e.to_string(),
            })?;
        if self.config.files_interleave_cycle_length > 1 {
            Ok(EmbeddingDataset::Interleaved(
                InterleavedDataset::from_tfrecord(
                    dataset,
                    self.config.files_interleave_cycle_length,
                ),
            ))
        } else {
            Ok(EmbeddingDataset::Tf(dataset))
        }
    }

    /// Creates batched input iterator for the given mode.
    pub fn create_input_batches(
        &self,
        mode: TaskMode,
    ) -> Result<Box<dyn Iterator<Item = Vec<Example>> + Send>> {
        let batch_size = match mode {
            TaskMode::Train => self.config.base.train.per_replica_batch_size.unwrap_or(1),
            TaskMode::Eval => self.config.base.eval.per_replica_batch_size.unwrap_or(1),
            TaskMode::Predict => 1,
        };

        let dataset = match mode {
            TaskMode::Train => self.create_train_dataset()?,
            _ => {
                return Err(MonolithError::ConfigError {
                    message: "Only training dataset is implemented".to_string(),
                })
            }
        };

        let batch_iter: Box<dyn Iterator<Item = Vec<Example>> + Send> =
            if self.config.base.train.repeat {
                Box::new(
                    BatchedDataset::new(dataset.repeat().iter(), batch_size)
                        .iter()
                        .map(|b| b.into_examples()),
                )
            } else {
                Box::new(
                    BatchedDataset::new(dataset.iter(), batch_size)
                        .iter()
                        .map(|b| b.into_examples()),
                )
            };

        Ok(batch_iter)
    }
}

fn expand_patterns(pattern: &str) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in glob(pattern).map_err(|e| MonolithError::ConfigError {
        message: e.to_string(),
    })? {
        let path = entry.map_err(|e| MonolithError::ConfigError {
            message: e.to_string(),
        })?;
        if path.exists() {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    #[test]
    fn test_create_vocab_dict_parity_fixed_and_custom_and_offset() {
        let mut f = NamedTempFile::new().unwrap();
        // Include a non-digit header line which should be ignored (Python behavior).
        std::io::Write::write_all(&mut f, b"slot\tvocab\n1\t10\n2\t20\n").unwrap();

        let mut cfg = BaseEmbeddingTaskConfig::default();
        cfg.vocab_file_path = Some(f.path().to_path_buf());

        let mut custom = HashMap::new();
        custom.insert(2, 99usize);
        cfg.custom_vocab_size_mapping = Some(custom);
        cfg.vocab_size_offset = Some(1);

        let vocab = BaseEmbeddingTask::create_vocab_dict_for_test(&mut cfg).unwrap();
        assert_eq!(vocab.get(&1).copied(), Some(11));
        assert_eq!(vocab.get(&2).copied(), Some(100));
    }

    #[test]
    fn test_create_vocab_dict_fixed_vocab_size_per_slot() {
        let mut f = NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut f, b"1\t10\n2\t20\n").unwrap();

        let mut cfg = BaseEmbeddingTaskConfig::default();
        cfg.vocab_file_path = Some(f.path().to_path_buf());
        cfg.vocab_size_per_slot = Some(7);
        // If `vocab_size_per_slot` is set, custom mapping should be ignored (Python behavior).
        let mut custom = HashMap::new();
        custom.insert(2, 99usize);
        cfg.custom_vocab_size_mapping = Some(custom);

        let vocab = BaseEmbeddingTask::create_vocab_dict_for_test(&mut cfg).unwrap();
        assert_eq!(vocab.get(&1).copied(), Some(7));
        assert_eq!(vocab.get(&2).copied(), Some(7));
    }

    #[test]
    fn test_create_vocab_dict_invalid_line_errors() {
        let mut f = NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut f, b"1\t10\textra\n").unwrap();

        let mut cfg = BaseEmbeddingTaskConfig::default();
        cfg.vocab_file_path = Some(f.path().to_path_buf());

        let err = BaseEmbeddingTask::create_vocab_dict_for_test(&mut cfg).unwrap_err();
        assert!(err.to_string().contains("must have 2 fields"));
    }
}
