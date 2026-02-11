import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="monolith-entrypoints-workspace"
    target={{ language: 'rust' }}
    description="Port top-level runners/utils and workspace config into monolith-cli/build tooling."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/monolith-entrypoints-workspace.md" />
    <Agent id="apply-monolith-entrypoints-workspace">
      <Prompt>
        <System>
          You are porting top-level monolith entrypoints and workspace config to
          Rust tooling. Edit the Rust codebase directly; do not write plans,
          mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/base_runner.py
- monolith/gpu_runner.py
- monolith/tpu_runner.py
- monolith/utils.py
- monolith/utils_test.py
- monolith/path_utils.py
- monolith/common/python/mem_profiling.py

Workspace/build inputs to map:
- monolith/monolith_workspace.bzl
- monolith/tf_serving_workspace.bzl
- third_party/org_tensorflow/**
- third_party/org_tensorflow_serving/**
- third_party/org_apache_zookeeper/**
- third_party/repo.bzl

Rust targets:
- monolith-rs/crates/monolith-cli
- monolith-rs/crates/monolith-core (shared utils)
- monolith-rs build/config files as needed

Requirements:
- Match CLI/env/config behavior from Python runners.
- Preserve path resolution and filesystem semantics.
- Add or update Rust tests that mirror the Python tests listed above.
- Keep TensorFlow runtime optional; do not vendor TF binaries.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
