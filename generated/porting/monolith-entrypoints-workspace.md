# Monolith entrypoints/workspace (Rust)

This port implements parity-critical utilities and runner entrypoints used by the
Python `monolith/` tree, without embedding TensorFlow runtime.

Implemented:
- `monolith/path_utils.py`: `monolith-core::path_utils::{find_main,get_libops_path}`
  with a test-friendly `MONOLITH_MAIN_DIR` override.
- `monolith/utils.py`: `monolith-core::utils::{enable_monkey_patch,copy_file,copy_recursively}`
  (TF monkey patch is a Rust-visible marker; file copy mirrors local filesystem semantics).
- `monolith/common/python/mem_profiling.py`: `monolith-core::mem_profiling::{enable_tcmalloc,setup_heap_profile}`
  (sets gperftools env vars, does not require bundling `libtcmalloc`).
- `monolith/gpu_runner.py` + `monolith/tpu_runner.py`: `monolith-cli` adds
  `monolith tf-runner gpu|tpu` subcommands to preserve flag/env behavior and to emit
  TensorBoard-compatible event files in eval modes.

Notes:
- TensorFlow runtime remains optional/out-of-process; Rust does not link or vendor TF.
- TF event writing uses TFRecord-encoded `tensorflow.Event` protos compiled in `monolith-proto`.

