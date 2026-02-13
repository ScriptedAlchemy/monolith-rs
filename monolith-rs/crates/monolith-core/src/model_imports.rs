//! Model import helpers (Python parity).
//!
//! Python `monolith/core/model_imports.py` is responsible for importing task
//! modules so that their `ModelParams` classes register into the global model
//! registry. In Rust we do not have runtime imports in the same way, but we
//! keep a compatible API surface:
//! - `import_all_params` is a no-op hook (returns whether anything happened)
//! - `import_params` validates model names and returns deterministic errors
//!
//! This is primarily used by CLI stubs and for parity with error messages.

use crate::error::{MonolithError, Result};

/// Default task root prefix from Python.
pub const DEFAULT_TASK_ROOT: &str = "monolith.tasks";

/// Attempts to import/register all task params.
///
/// In Rust this is a no-op by default; registries are populated by linking code
/// that calls `register_single_task_model!` at init/test time.
pub fn import_all_params(_task_root: &str, _task_dirs: &[&str], require_success: bool) -> Result<bool> {
    let success = false;
    if require_success && !success {
        return Err(MonolithError::PyLookupError {
            message: "Could not import any task params. Make sure task params are linked into the binary."
                .to_string(),
        });
    }
    Ok(success)
}

/// Attempts to import the params for the requested model.
///
/// Since Rust does not dynamically import modules, this function only validates
/// the model name shape and returns a Python-parity error when `require_success`
/// is requested.
pub fn import_params(
    model_name: &str,
    _task_root: &str,
    _task_dirs: &[&str],
    require_success: bool,
) -> Result<bool> {
    // Python: if '.' not in model_name: raise ValueError('Invalid model name %s' % model_name)
    if !model_name.contains('.') {
        return Err(MonolithError::PyValueError {
            message: format!("Invalid model name {}", model_name),
        });
    }

    // In Python this would try `_Import(model_module)` and then built-in task imports.
    // We preserve the failure message when required.
    if require_success {
        let model_module = model_name.rsplit_once('.').map(|(m, _)| m).unwrap_or(model_name);
        return Err(MonolithError::PyLookupError {
            message: format!(
                "Could not find any valid import paths for module {}. Check the logs above to see if there were errors importing the module, and make sure the relevant params files are linked into the binary.",
                model_module
            ),
        });
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_params_invalid_name_message() {
        let err = import_params("NoDots", DEFAULT_TASK_ROOT, &[], true)
            .expect_err("invalid model name should fail import_params");
        assert_eq!(err.to_string(), "Invalid model name NoDots");
    }

    #[test]
    fn test_import_params_require_success_message() {
        let err = import_params("a.b.C", DEFAULT_TASK_ROOT, &[], true)
            .expect_err("require_success import should fail when no valid module paths exist");
        assert_eq!(
            err.to_string(),
            "Could not find any valid import paths for module a.b. Check the logs above to see if there were errors importing the module, and make sure the relevant params files are linked into the binary."
        );
    }

    #[test]
    fn test_import_all_params_require_success_message() {
        let err = import_all_params(DEFAULT_TASK_ROOT, &[], true)
            .expect_err("require_success import_all_params should fail when no task params import");
        assert_eq!(
            err.to_string(),
            "Could not import any task params. Make sure task params are linked into the binary."
        );
    }
}

