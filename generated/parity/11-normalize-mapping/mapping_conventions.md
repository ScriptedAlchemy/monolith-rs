# Mapping Conventions (Python -> Rust Parity)

Normalization takes the upstream Python->Rust mapping table and rewrites it into a stable, machine-friendly format used downstream to generate parity plans, checklists, and test/fixture scaffolding.

## Status Vocabulary

The normalizer only permits these statuses:

- `TODO`: Planned but not started; mapping exists, implementation does not.
- `IN_PROGRESS`: Actively being ported/validated.
- `DONE`: Ported/validated; parity behavior is expected to match.
- `N/A`: No Rust port required; must include an explicit justification (see N/A policy).
- `BLOCKED`: Cannot proceed due to an external dependency (tooling, upstream API, missing crate, etc.); notes should include the blocker.

### N/A Policy

When a Python file does not need a Rust port:

- Set `status` to `N/A`.
- Set `rustTargets` to an empty list.
- Set `notes` to a concrete justification starting with `N/A:`.

Example:

- `N/A: Pure re-export __init__.py; no runtime behavior.`

## Canonical Rust Target Format

Rust targets should be expressed using a canonical, crate-rooted path:

`monolith-rs/crates/<crate>/(src|tests|examples|benches)/<path>`

Examples:

- `monolith-rs/crates/monolith-core/src/lib.rs`
- `monolith-rs/crates/monolith-serving/tests/parity/agent_service/foo.rs`

Notes:

- Prefer `path` relative to the crate root (e.g. `src/feature.rs`) alongside an explicit `crate` field.
- Avoid targets that only say `src`/`tests` without crate context, since those are ambiguous across crates.

## Test Mapping Conventions

Python test modules map to Rust tests under a crate-local parity tests namespace:

- Prefer: `monolith-rs/crates/<crate>/tests/parity/...`

Example:

- `monolith/tests/foo_test.py -> monolith-rs/crates/monolith-core/tests/parity/tests/foo_test.rs`

Guidelines:

- Keep test file names stable and recognizable; only adjust when necessary for Rust module naming rules.
- Prefer mirroring the Python directory structure under `tests/parity/` to keep navigation predictable.

## Fixture Placement (Golden/Parity Fixtures)

Golden fixtures used for parity should live under a stable shared root (usable by both Python and Rust harnesses).

Suggested roots:

- `monolith-rs/fixtures/parity`
- `monolith-rs/testdata/parity`

Guidelines:

- Keep fixtures deterministic (no timestamps, randomized IDs, or environment-dependent output).
- Prefer small, composable fixtures that are easy to inspect and update.

## Proto / Op Boundary Conventions

Keep schema generation and TF op/runtime integration separated from core logic:

- Protobuf schemas belong in: `monolith-rs/crates/monolith-proto`
- TensorFlow custom ops belong in dedicated op/runtime crates (proposed direction):
  - `monolith/native_training/**/ops -> monolith-rs/crates/monolith-tensor-ops (proposed)`

Rule of thumb:

- `*.proto` and generated types live in `monolith-proto`.
- TF op loading, registration, and runtime glue live in dedicated crates.
- Core algorithmic/data logic stays in the most specific functional crate (e.g. `monolith-core`, `monolith-data`, `monolith-training`).

## Normalization Checks

The normalizer emits issues (and/or warnings) to make cleanup work actionable and repeatable. Common issue IDs:

- `missing_upstream_mapping_json`: Upstream mapping JSON is missing; the normalization pass cannot populate records.
- `unknown_crate`: A target references a Rust crate name that does not exist under `monolith-rs/crates/`.
- `missing_rust_target_or_na`: A record has no Rust targets but is not marked `N/A` with an explicit `N/A:` justification.
- `invalid_status`: Status is not one of `TODO`, `IN_PROGRESS`, `DONE`, `N/A`, `BLOCKED`.
- `invalid_na_notes`: `status` is `N/A` but `notes` is missing or does not start with `N/A:`.
- `invalid_rust_target_format`: A Rust target does not match the canonical crate-rooted format.
- `python_test_should_map_to_rust_tests`: A Python test file is mapped outside of crate `tests/parity/...` conventions (or is mapped to `src` instead of `tests`).

## Next Normalization Improvements (Prioritized)

1) Automatically reconcile crate sets using `monolith-rs/crates/*/Cargo.toml` and surface actionable `unknown_crate` diffs (e.g. consistent handling for `monolith-tf` references).
2) Normalize Rust targets to `{crate, path}` consistently and strip incidental annotations like ` (new)` from `raw`.
3) Improve deterministic rendering for reports by using an unambiguous `crate:path` string instead of bare `src`/`tests`.
4) Add structured grouping in reports (by crate, by status, by folder prefix) while preserving lexicographic stability.
