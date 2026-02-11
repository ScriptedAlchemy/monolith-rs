//! Core traits, types, and feature abstractions for Monolith.
//!
//! This crate provides the foundational types and abstractions used throughout
//! the Monolith machine learning system. It includes:
//!
//! - **Feature ID (Fid) utilities**: Types and functions for working with feature IDs
//!   that encode both slot and feature information.
//! - **Feature abstractions**: Traits and structs for feature slots, slices, and columns.
//! - **Configuration types**: Structures for embedding configurations, initializers,
//!   and hyperparameters.
//! - **Error types**: Structured error handling with detailed context.
//!
//! # Example
//!
//! ```
//! use monolith_core::fid::{make_fid, extract_slot, extract_feature};
//! use monolith_core::feature::{FeatureSlot, SparseFeatureColumn, FeatureColumn};
//!
//! // Create a feature ID from slot and feature hash
//! let fid = make_fid(1, 12345).unwrap();
//! assert_eq!(extract_slot(fid), 1);
//! assert_eq!(extract_feature(fid), 12345);
//!
//! // Create a feature slot
//! let slot = FeatureSlot::new(1, "user_id", 64);
//!
//! // Create a sparse feature column
//! let mut column = SparseFeatureColumn::new(1);
//! column.push_example(&[fid], &[1.0]);
//! assert_eq!(column.batch_size(), 1);
//! ```
//!
//! # Modules
//!
//! - [`env`]: Environment configuration for feature management.
//! - [`fid`]: Feature ID types and utilities.
//! - [`feature`]: Feature slots, slices, and columns.
//! - [`params`]: Configuration and hyperparameter types.
//! - [`error`]: Error types for the library.

pub mod base_embedding_host_call;
pub mod base_host_call;
pub mod base_layer;
pub mod base_model_params;
pub mod base_task;
pub mod dyn_value;
pub mod env;
pub mod error;
pub mod feature;
pub mod fid;
pub mod hyperparams;
pub mod mem_profiling;
pub mod model_imports;
pub mod model_registry;
pub mod nested_map;
pub mod optimizers;
pub mod params;
pub mod path_utils;
pub mod util;
pub mod utils;

// Re-export commonly used types at the crate root for convenience
pub use base_host_call::{BaseHostCall, HostTensor, HostTensor1D, HostTensor2D};
pub use base_layer::{add_layer_loss, get_layer_loss, get_uname, BaseLayerCore};
pub use base_model_params::SingleTaskModelParams;
pub use base_task::{base_task_params, Accelerator, BaseTask, TaskMode};
pub use dyn_value::DynValue;
pub use env::{Env, EnvBuilder};
pub use error::{MonolithError, Result};
pub use feature::{
    DenseFeatureColumn, EmbeddingTensor, FeatureColumn, FeatureColumnV1, FeatureSlice, FeatureSlot,
    SailEnv, SailFeatureSlice, SailFeatureSlot, SparseFeatureColumn,
};
pub use fid::{extract_feature, extract_slot, make_fid, Fid, SlotId};
pub use hyperparams::{
    copy_params_to, update_params, InstantiableParams, ParamValue, Params as HyperParams,
};
pub use mem_profiling::{enable_tcmalloc, setup_heap_profile};
pub use model_imports::{import_all_params, import_params};
pub use model_registry::{get_all_registered, get_class, get_params, register_single_task_model};
pub use nested_map::{NestedMap, NestedValue};
pub use optimizers::OptimizerName;
pub use params::{EmbeddingConfig, InitializerConfig, Params, TrainingParams};
pub use path_utils::{find_main, get_libops_path};
pub use util::{
    calculate_shard_skip_file_number, get_bucket_name_and_relavite_path,
    parse_example_number_meta_file, range_dateset,
};
pub use utils::{copy_file, copy_recursively, enable_monkey_patch, monkey_patch_enabled};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_reexports() {
        // Test that all re-exports are accessible
        let fid: Fid = make_fid(1, 100).unwrap();
        let slot: SlotId = extract_slot(fid);
        assert_eq!(slot, 1);

        let _slot = FeatureSlot::new(1, "test", 64);
        let _slice = FeatureSlice::new(0, 10, 1);
        let _sparse = SparseFeatureColumn::new(1);
        let _dense = DenseFeatureColumn::new(1, 64);

        let _config = EmbeddingConfig::new(1, 64);
        let _init = InitializerConfig::zeros();
        let _params = TrainingParams::default();

        let _err: Result<()> = Ok(());
    }

    #[test]
    fn test_integration_workflow() {
        // Test a typical workflow using the crate

        // 1. Create feature slots
        let user_slot = FeatureSlot::new(1, "user_id", 64);
        let item_slot = FeatureSlot::with_pooling(2, "item_tags", 32, Some(100));

        assert_eq!(user_slot.dim(), 64);
        assert!(item_slot.pooling());

        // 2. Create embedding configs for the slots
        let user_config = EmbeddingConfig::builder(user_slot.slot_id(), user_slot.dim())
            .learning_rate(0.01)
            .initializer(InitializerConfig::xavier_uniform(1.0))
            .build()
            .unwrap();

        let item_config = EmbeddingConfig::builder(item_slot.slot_id(), item_slot.dim())
            .learning_rate(0.001)
            .initializer(InitializerConfig::normal(0.0, 0.01))
            .build()
            .unwrap();

        assert_eq!(user_config.dim(), 64);
        assert_eq!(item_config.dim(), 32);

        // 3. Create feature columns and add data
        let mut user_column = SparseFeatureColumn::new(user_slot.slot_id());
        let mut item_column = SparseFeatureColumn::new(item_slot.slot_id());

        // Add features for 2 examples
        let user_fid1 = make_fid(1, 1001).unwrap();
        let user_fid2 = make_fid(1, 1002).unwrap();
        let item_fid1 = make_fid(2, 2001).unwrap();
        let item_fid2 = make_fid(2, 2002).unwrap();
        let item_fid3 = make_fid(2, 2003).unwrap();

        user_column.push_example(&[user_fid1], &[1.0]);
        user_column.push_example(&[user_fid2], &[1.0]);

        item_column.push_example(&[item_fid1, item_fid2], &[1.0, 0.5]);
        item_column.push_example(&[item_fid3], &[1.0]);

        assert_eq!(user_column.batch_size(), 2);
        assert_eq!(item_column.batch_size(), 2);

        // 4. Access individual example features
        let (user_fids, user_values) = user_column.example_features(0).unwrap();
        assert_eq!(user_fids.len(), 1);
        assert_eq!(user_values, &[1.0]);

        let (item_fids, item_values) = item_column.example_features(0).unwrap();
        assert_eq!(item_fids.len(), 2);
        assert_eq!(item_values, &[1.0, 0.5]);

        // 5. Verify slot extraction from fids
        for &fid in user_fids {
            assert_eq!(extract_slot(fid), 1);
        }
        for &fid in item_fids {
            assert_eq!(extract_slot(fid), 2);
        }
    }
}
