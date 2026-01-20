<!--
Source: task/request.md
Lines: 16210-16416 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/model_dump/graph_utils.py`
<a id="monolith-native-training-model-dump-graph-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 845
- Purpose/role: GraphDef utilities for reconstructing variables, importing subgraphs, and rebuilding input/model/receiver functions from dumped metadata.
- Key symbols/classes/functions: `DatasetInitHook`, `_node_name`, `_colocated_node_name`, `EchoInitializer`, `VariableDef`, `PartitionVariableDef`, `GraphDefHelper`.
- External dependencies: TensorFlow GraphDef/ops internals, protobufs (`LineId`, `FeatureConfigs`), `tf.keras`, `tf.io`, flags.
- Side effects: Mutates graph node attrs (`_class`), clears devices, adds to TF collections, imports graph defs into default graph.

**Required Behavior (Detailed)**
- Globals:
  - `DRY_RUN = 'dry_run'`, `FLAGS = flags.FLAGS`.
- `DatasetInitHook`: session hook that runs initializer in `after_create_session`.
- `_node_name(name)`:
  - Strips leading `^` and output suffix `:0`; returns node base name.
- `_colocated_node_name(name)`:
  - Decodes bytes to string; strips `loc:@` prefix if present.
- `EchoInitializer`:
  - Accepts an init value; if list/tuple uses first element.
  - Returns tensor directly if `tf.Tensor`, else returns op output (expects single output).
- `VariableDef`:
  - Wraps a `VarHandleOp` node and helper.
  - `initializer`: finds `<var>/Assign` node, determines initializer input, builds subgraph, imports it, and returns `EchoInitializer`.
  - `variable`: creates a `tf.get_variable` with dtype/shape/initializer on proper device, temporarily disabling partitioner; returns first partition if `PartitionedVariable`.
  - Tracks associated `ReadVariableOp` nodes via `add_read`.
- `PartitionVariableDef`:
  - Handles partitioned variables (`/part_N`); tracks partitions and read ops.
  - `get_base_name` extracts base variable name from VarHandleOp or ReadVariableOp inputs.
  - `initializer`: finds PartitionedInitializer slice nodes or Assign initializer nodes, imports subgraph, returns list of `EchoInitializer`.
  - `variable`: creates variables for each partition (uses first device as group_device), sets `save_slice_info`, and builds a `PartitionedVariable` for validation if multiple partitions.
- `GraphDefHelper.__init__(graph_def, save_slice_info)`:
  - Validates GraphDef type.
  - Clears node device and `_class` colocation hints (adds colocated names to input set).
  - Builds name-to-node, seq mapping, and tracks variables/readers into `VariableDef`/`PartitionVariableDef`.
  - Records PBDataset file_name const node if present.
- `_check_invalidate_node(graph_def, input_map)`:
  - Removes input_map entries not referenced by graph inputs; logs warnings.
- `_create_variables(variables)`:
  - Recreates variable read tensors using `read_variable_op` for all variable defs in subgraph.
  - Skips canonical `/Read/ReadVariableOp` nodes.
- `sub_graph(dest_nodes, source_nodes=None, with_library=True)`:
  - BFS from dest nodes through inputs; stops at source_nodes.
  - Builds a GraphDef containing non-variable nodes and collects variable names separately.
  - If `with_library`, copies required functions (including Dataset functions).
- `import_input_fn(input_conf, file_name)`:
  - Constructs dest_nodes from recorded output features and label; includes iterator ops unless DRY_RUN.
  - Updates PBDataset/file_name Const value to `file_name`.
  - Optionally updates PBDataset/input_pb_type based on `FLAGS.data_type`.
  - Imports subgraph; adds iterator/mkiter to collections.
  - Rebuilds features dict (ragged or dense) and adds label.
- `import_model_fn(input_map, proto_model)`:
  - Collects outputs from predict, extra outputs, loss, labels, extra_losses, signatures, summaries.
  - Builds subgraph, prunes input_map, recreates variable reads, imports graph_def.
  - Adds sparse feature names to collections by scanning ShardingSparseFids nodes.
  - Restores summaries to GraphKeys.SUMMARIES.
  - Validates signature inputs exist in graph.
  - Returns `(label, loss, predict, head_name, extra_output_dict, is_classification)`.
- `import_receiver_fn(receiver_conf)`:
  - Builds dest_nodes for ragged values/row_splits and dense features.
  - Populates collections: sparse_features, dense_features/types/shapes, extra_features/shapes, variant_type.
  - Imports subgraph and reconstructs feature tensors + receiver_tensors.
- `get_optimizer(proto_model)`:
  - Unpickles optimizer from proto bytes or returns None.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF GraphDef import is Python-specific).
- Rust public API surface: only relevant if a TF runtime backend is used for model import.
- Data model mapping: GraphDef + metadata + tensor names.
- Feature gating: TF runtime only.
- Integration points: model loader, serving input reconstruction.

**Implementation Steps (Detailed)**
1. If Rust needs TF graph import, wrap GraphDef parsing and node filtering.
2. Implement variable recreation and read-op mapping analogous to `_create_variables`.
3. Implement sub-graph extraction with function library filtering.
4. Recreate input/model/receiver functions using stored metadata and collections.

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: N/A unless TF GraphDef import is added.
- Cross-language parity test: verify imported outputs match original graph outputs.

**Gaps / Notes**
- Uses `eval` on serialized feature representations (security risk if untrusted).
- Clears node device assignments; placement is not preserved.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_dump/graph_utils_test.py`
<a id="monolith-native-training-model-dump-graph-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 86
- Purpose/role: Tests `GraphDefHelper` input/receiver reconstruction using a saved model dump.
- Key symbols/classes/functions: `GraphUtilsTest.test_load_input_fn`, `test_load_receiver`, `test_load_mode`.
- External dependencies: TensorFlow, `DumpUtils`, `GraphDefHelper`.
- Side effects: Loads model dump from `model_dump/test_data/model_dump`.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Sets `FLAGS.data_type = 'examplebatch'`.
  - Loads dump via `DumpUtils().load(...)`.
- `test_load_input_fn`:
  - Calls `import_input_fn` with `file_name`.
  - Verifies each output feature returns `tf.RaggedTensor` if flagged ragged, else `tf.Tensor`.
- `test_load_receiver`:
  - Calls `import_receiver_fn`.
  - Verifies feature tensor types and that receiver_tensors length is 1.
- `test_load_mode`:
  - `get_graph_helper` returns `GraphDefHelper` for TRAIN, TRAIN with `graph.dry_run=True`, and PREDICT.
- `__main__`: disables eager execution and runs tests.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: TF runtime only.
- Integration points: model dump loader.

**Implementation Steps (Detailed)**
1. If graph import is implemented in Rust, add tests for ragged/dense reconstruction.
2. Use fixed dump artifacts to avoid nondeterminism.

**Tests (Detailed)**
- Python tests: `monolith/native_training/model_dump/graph_utils_test.py`.
- Rust tests: none.
- Cross-language parity test: compare reconstructed tensors and types.

**Gaps / Notes**
- Test depends on external dump artifacts in the repo.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/model_export/__init__.py`
<a id="monolith-native-training-model-export-init-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 22
- Purpose/role: Re-exports model export modules under legacy module paths for backward compatibility.
- Key symbols/classes/functions: module aliasing via `sys.modules`.
- External dependencies: `export_context`, `saved_model_exporters`.
- Side effects: Inserts entries into `sys.modules` and deletes local `_sys`.

**Required Behavior (Detailed)**
- Imports `monolith.native_training.model_export.export_context` and `saved_model_exporters`.
- Registers aliases:
  - `'monolith.model_export.export_context'` → `export_context` module.
  - `'monolith.model_export.saved_model_exporters'` → `saved_model_exporters` module.
- Deletes `_sys` name after aliasing.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: Python import compatibility only.

**Implementation Steps (Detailed)**
1. If Rust wrappers need to mirror Python module paths, document the aliasing behavior in docs.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Pure import aliasing; no functional logic.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed
