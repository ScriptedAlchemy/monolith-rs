import "./11-normalize-mapping.ai.tsx";

import {
  Action,
  Agent,
  Asset,
  Context,
  Instructions,
  Program,
  Prompt,
  System,
  action,
  assetRef,
  ctx,
} from "@unpack/ai";

import {
  mappingConventionsDoc,
  normalizedMappingJson,
} from "./11-normalize-mapping.ai.tsx";

export const parityTestHarnessPlanDoc = assetRef("parity_test_harness_plan_doc");
export const parityTestHarnessPlanJson = assetRef(
  "parity_test_harness_plan_json",
);

type ParityHarnessKind =
  | "golden_fixture"
  | "differential"
  | "grpc_proto_compat"
  | "snapshot_state"
  | "integration"
  | "conformance";

type ParitySignal = {
  id: string;
  description: string;
  evidence: string[];
};

type HarnessStep = {
  id: string;
  purpose: string;
  commandExamples: string[];
  inputs: string[];
  outputs: string[];
  notes: string[];
};

type ParityHarness = {
  id: string;
  kind: ParityHarnessKind;
  scope: "unit" | "component" | "service" | "workspace";
  ownerDomain:
    | "agent_service"
    | "core"
    | "native_training"
    | "tf_runtime"
    | "third_party"
    | "global";
  description: string;
  gating: {
    requiredByDefault: boolean;
    tfRuntimeOptional: boolean;
    candleDefault: boolean;
    requiresNetwork: boolean;
  };
  fixtures: {
    rule: string;
    recommendedRoots: string[];
    examples: string[];
  };
  comparisons: {
    rule: string;
    tolerances: string[];
    nonDeterminismNotes: string[];
  };
  steps: HarnessStep[];
  risks: string[];
};

type ParityTestHarnessPlan = {
  version: "1";
  inputs: {
    normalizedMappingPath: string;
    mappingConventionsPath: string;
    pythonTestGlobs: string[];
    pythonFixtureGlobs: string[];
    rustCratesTomlGlob: string;
    rustTestGlob: string;
  };
  repoScan: {
    pythonTests: { total: number; sample: string[]; byTopDir: Record<string, number> };
    pythonFixtures: { total: number; sample: string[] };
    rustCrates: { total: number; crates: string[] };
    rustTests: { total: number; sample: string[] };
  };
  requiredHarnesses: ParityHarness[];
  recommendedRepoLayout: {
    rule: string;
    paths: Record<string, string>;
  };
  signals: ParitySignal[];
  warnings: string[];
};

function uniqSorted(xs: string[]): string[] {
  return Array.from(new Set(xs.filter((x) => typeof x === "string"))).sort();
}

function topDir(p: string): string {
  const parts = (p ?? "").split("/").filter(Boolean);
  if (parts.length === 0) return "unknown";
  if (parts[0] === "monolith" && parts.length >= 2) return `monolith/${parts[1]}`;
  return parts[0];
}

function byTopDir(paths: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const p of paths) {
    const d = topDir(p);
    out[d] = (out[d] ?? 0) + 1;
  }
  return Object.fromEntries(
    Object.entries(out).sort((a, b) => a[0].localeCompare(b[0])),
  );
}

function toSample(paths: string[], n: number): string[] {
  return paths.slice().sort().slice(0, n);
}

function parseCrateNamesFromCargoTomlPaths(paths: string[]): string[] {
  const crates: string[] = [];
  for (const p of paths) {
    const m = p.match(/monolith-rs\/crates\/([^/]+)\/Cargo\.toml$/);
    if (m) crates.push(m[1]);
  }
  return uniqSorted(crates);
}

export const computeParityTestHarnessPlanInputs = action(
  async (actx): Promise<ParityTestHarnessPlan> => {
    const normalizedMappingPath =
      "generated/parity/11-normalize-mapping/normalized_mapping.json";
    const mappingConventionsPath =
      "generated/parity/11-normalize-mapping/mapping_conventions.md";

    const pythonTestGlobs = [
      "monolith/**/*_test.py",
      "monolith/**/test_*.py",
      "monolith/**/tests/**/*.py",
    ];
    const pythonFixtureGlobs = [
      "monolith/**/testdata/**",
      "monolith/**/fixtures/**",
      "monolith/**/golden/**",
    ];

    const rustCratesTomlGlob = "monolith-rs/crates/*/Cargo.toml";
    const rustTestGlob = "monolith-rs/crates/*/tests/**/*.rs";

    const warnings: string[] = [];

    const pythonTests = uniqSorted(
      (
        await Promise.all(
          pythonTestGlobs.map(async (g) => (await actx.fs.glob(g)) as string[]),
        )
      ).flat(),
    ).filter((p) => !p.endsWith("/"));

    const pythonFixtures = uniqSorted(
      (
        await Promise.all(
          pythonFixtureGlobs.map(
            async (g) => (await actx.fs.glob(g)) as string[],
          ),
        )
      ).flat(),
    )
      .filter((p) => !p.endsWith("/"))
      .slice()
      .sort();

    const cargoTomls = uniqSorted(
      ((await actx.fs.glob(rustCratesTomlGlob)) as string[]).filter(
        (p) => !p.endsWith("/"),
      ),
    );
    const rustCrates = parseCrateNamesFromCargoTomlPaths(cargoTomls);

    const rustTests = uniqSorted(
      ((await actx.fs.glob(rustTestGlob)) as string[]).filter(
        (p) => !p.endsWith("/"),
      ),
    );

    if (rustCrates.length === 0) {
      warnings.push(
        "No Rust crates found under monolith-rs/crates/*; harness plan includes layout/commands but repo scan appears empty.",
      );
    }

    const recommendedRepoLayout = {
      rule: "Keep parity fixtures canonical and shareable: Python generates/normalizes fixtures, Rust consumes them. Store fixtures under generated/ for reproducible runs, and allow checked-in minimal fixtures under a stable directory for CI.",
      paths: {
        parityRoot: "generated/parity",
        fixturesRoot: "generated/parity/fixtures",
        fixturesCheckedInRoot: "parity_fixtures",
        reportsRoot: "generated/parity/reports",
        protoGoldensRoot: "generated/parity/proto-goldens",
      },
    };

    const requiredHarnesses: ParityHarness[] = [
      {
        id: "harness_golden_fixtures_core",
        kind: "golden_fixture",
        scope: "component",
        ownerDomain: "global",
        description:
          "Golden fixtures for deterministic, TF-optional core behaviors (serialization, feature transformations, config parsing, deterministic math). Python produces canonical outputs; Rust must match byte-for-byte or with explicit tolerances.",
        gating: {
          requiredByDefault: true,
          tfRuntimeOptional: true,
          candleDefault: true,
          requiresNetwork: false,
        },
        fixtures: {
          rule: "Fixtures must be small, stable, and versioned. Prefer JSON (or msgpack) for structured tensors + metadata; store any large binary blobs behind a separate opt-in.",
          recommendedRoots: [
            recommendedRepoLayout.paths.fixturesCheckedInRoot,
            recommendedRepoLayout.paths.fixturesRoot,
          ],
          examples: [
            "parity_fixtures/core/hparams/v1/case_001.json",
            "parity_fixtures/native_training/batch_norm/v1/case_003.json",
          ],
        },
        comparisons: {
          rule: "Prefer exact equality for shapes/dtypes/IDs; allow numeric tolerances for floats. Document any non-deterministic fields and normalize them.",
          tolerances: [
            "Float comparisons: abs<=1e-6 OR rel<=1e-5 unless otherwise specified per test",
            "Timestamps/UUIDs: normalize to sentinel values before comparing",
          ],
          nonDeterminismNotes: [
            "TF runtime kernels and multithread scheduling can change floating point reduction order.",
            "Random seeding differs across frameworks; require explicit seed inputs in fixtures.",
          ],
        },
        steps: [
          {
            id: "python_generate",
            purpose:
              "Generate or refresh golden fixtures from Python reference implementation.",
            commandExamples: [
              "python -m monolith.tools.parity.generate_fixtures --out parity_fixtures",
              "pytest -q monolith/... -k parity_fixture -- --update-fixtures",
            ],
            inputs: ["Python reference code", "Seeds + input cases"],
            outputs: ["Fixture JSON files", "Optional binary blobs"],
            notes: [
              "All fixture generation must be deterministic (seeded) and stable across machines.",
              "Store a manifest per fixture group (schema version, normalization rules).",
            ],
          },
          {
            id: "rust_validate",
            purpose:
              "Run Rust-side readers/implementations against the fixtures and compare to expectations.",
            commandExamples: [
              "cargo test -p monolith-parity -- test_golden --nocapture",
              "cargo test --workspace --features parity_fixtures",
            ],
            inputs: ["Fixture files", "Rust implementation"],
            outputs: ["Test pass/fail", "Diff reports under generated/parity/reports"],
            notes: [
              "Emit rich diffs (first mismatch path, numeric deltas, shape/dtype changes).",
              "Keep fixture parsing separate from model/runtime code to reduce coupling.",
            ],
          },
        ],
        risks: [
          "Overfitting to fixtures can miss behavior outside curated cases; pair with differential testing.",
          "Fixture drift if schema isn't versioned; require per-group schema versioning.",
        ],
      },
      {
        id: "harness_differential_python_vs_rust",
        kind: "differential",
        scope: "component",
        ownerDomain: "global",
        description:
          "Differential tests that run the same operation (or small pipeline) in Python and Rust and compare outputs. Used for behaviors that are hard to represent as static goldens or evolve quickly.",
        gating: {
          requiredByDefault: false,
          tfRuntimeOptional: true,
          candleDefault: true,
          requiresNetwork: false,
        },
        fixtures: {
          rule: "Use generated fixtures as the shared input corpus; differential runs consume the same inputs and compare results.",
          recommendedRoots: [recommendedRepoLayout.paths.fixturesRoot],
          examples: ["generated/parity/fixtures/native_training/*"],
        },
        comparisons: {
          rule: "Run Python and Rust side-by-side under a small runner contract; compare structured outputs using a stable normalization function.",
          tolerances: [
            "Float comparisons: configurable per op; default abs<=1e-6 rel<=1e-5",
          ],
          nonDeterminismNotes: [
            "Require deterministic seeds; record and replay seeds in fixture metadata.",
          ],
        },
        steps: [
          {
            id: "runner_contract",
            purpose:
              "Define a minimal runner protocol (stdin/stdout JSON) to invoke Python and Rust implementations for the same case.",
            commandExamples: [
              "python -m monolith.tools.parity.run_case --case case_001.json",
              "cargo run -p monolith-parity -- run-case --case case_001.json",
            ],
            inputs: ["Case JSON", "Runner binaries/modules"],
            outputs: ["Normalized output JSON", "Diff JSON"],
            notes: [
              "Prefer local process invocation; avoid network unless needed for service-level parity.",
              "Keep the protocol stable and versioned to prevent churn.",
            ],
          },
        ],
        risks: [
          "Differential tests can be slow; use sharding and allow a small curated CI set.",
          "Python environment setup can be complex; keep a hermetic runner wrapper and document it.",
        ],
      },
      {
        id: "harness_grpc_proto_compat",
        kind: "grpc_proto_compat",
        scope: "workspace",
        ownerDomain: "third_party",
        description:
          "gRPC/proto compatibility tests: ensure Python and Rust agree on serialized messages, service definitions, and any TFServing-related protos used by agent_service or runtime integration.",
        gating: {
          requiredByDefault: true,
          tfRuntimeOptional: true,
          candleDefault: true,
          requiresNetwork: false,
        },
        fixtures: {
          rule: "Generate canonical 'wire' fixtures: encoded proto bytes + decoded JSON view (for debugging).",
          recommendedRoots: [recommendedRepoLayout.paths.protoGoldensRoot],
          examples: [
            "generated/parity/proto-goldens/tf_serving/predict_request/v1/case_001.bin",
          ],
        },
        comparisons: {
          rule: "Roundtrip in both languages: bytes->message->bytes stability, and message field parity using deterministic JSON rendering.",
          tolerances: [
            "Bytes: exact match when using deterministic serialization settings",
          ],
          nonDeterminismNotes: [
            "Some protobuf runtimes do not guarantee deterministic map ordering unless configured; require deterministic modes.",
          ],
        },
        steps: [
          {
            id: "compile_protos",
            purpose:
              "Compile protos and ensure the generated code matches expected package/service names on both sides.",
            commandExamples: [
              "bazel build //...:all_protos",
              "cargo test -p monolith-proto -- proto_compile_smoke",
            ],
            inputs: ["Proto sources", "Build/toolchain configuration"],
            outputs: ["Generated code", "Compile/test report"],
            notes: [
              "Treat proto package names as API; changes require explicit mapping and compatibility notes.",
            ],
          },
          {
            id: "wire_fixture_roundtrip",
            purpose: "Roundtrip and cross-validate golden wire fixtures.",
            commandExamples: [
              "python -m monolith.tools.parity.proto_roundtrip --in generated/parity/proto-goldens",
              "cargo test -p monolith-proto -- proto_roundtrip --nocapture",
            ],
            inputs: ["Golden proto bytes fixtures"],
            outputs: ["Pass/fail + mismatch diffs"],
            notes: [
              "Always include a human-friendly decoded view for debugging mismatches.",
            ],
          },
        ],
        risks: [
          "Proto generation flags can differ between toolchains; pin them and test in CI.",
        ],
      },
      {
        id: "harness_service_integration_agent_service",
        kind: "integration",
        scope: "service",
        ownerDomain: "agent_service",
        description:
          "Service-level parity harness for monolith/agent_service/**: run Rust service with fake ZK and fake TFServing (or a local stub) and validate request/response semantics against Python reference behaviors where applicable.",
        gating: {
          requiredByDefault: false,
          tfRuntimeOptional: true,
          candleDefault: true,
          requiresNetwork: true,
        },
        fixtures: {
          rule: "Use recorded RPC interactions (requests + expected responses) and deterministic fake backends.",
          recommendedRoots: [recommendedRepoLayout.paths.fixturesCheckedInRoot],
          examples: ["parity_fixtures/agent_service/rpc/v1/predict_case_001.json"],
        },
        comparisons: {
          rule: "Compare gRPC status codes, error details, and normalized response payloads. Permit ordering-insensitive comparisons where appropriate.",
          tolerances: ["Float comparisons: abs<=1e-6 rel<=1e-5 in model outputs"],
          nonDeterminismNotes: [
            "When backends are faked, enforce deterministic ordering and timeouts.",
          ],
        },
        steps: [
          {
            id: "stand_up_fakes",
            purpose:
              "Start fake ZK and fake TFServing (or stub) and run smoke tests for agent_service RPCs.",
            commandExamples: [
              "cargo test -p monolith-serving --features fake_zk,fake_tfserving -- agent_service_smoke",
            ],
            inputs: ["Fake backend configs", "Recorded interactions fixtures"],
            outputs: ["RPC transcript logs", "Test report"],
            notes: [
              "Use explicit timeouts; record and surface retry behavior and backoff policy.",
            ],
          },
        ],
        risks: [
          "Networked tests are flaky; keep a hermetic, single-process fake by default.",
        ],
      },
      {
        id: "harness_tf_optional_conformance",
        kind: "conformance",
        scope: "component",
        ownerDomain: "tf_runtime",
        description:
          "TF-runtime-optional conformance suite: tests that validate behavior with Candle default, and additionally (best-effort) validate TF runtime integration when available (dynamic libtensorflow + custom op loading).",
        gating: {
          requiredByDefault: false,
          tfRuntimeOptional: true,
          candleDefault: true,
          requiresNetwork: false,
        },
        fixtures: {
          rule: "Reuse fixture corpus; tag cases requiring TF runtime (SavedModel signatures, custom ops) separately.",
          recommendedRoots: [
            recommendedRepoLayout.paths.fixturesCheckedInRoot,
            recommendedRepoLayout.paths.fixturesRoot,
          ],
          examples: [
            "parity_fixtures/tf_runtime/saved_model_signatures/v1/case_002.json",
          ],
        },
        comparisons: {
          rule: "On Candle: validate functional behavior against goldens. On TF runtime: validate the same set plus TF-specific cases; record known gaps explicitly.",
          tolerances: [
            "SavedModel outputs: apply numeric tolerances and stable normalization of output maps",
          ],
          nonDeterminismNotes: [
            "TF runtime availability differs per machine; suite must skip gracefully when missing.",
          ],
        },
        steps: [
          {
            id: "candle_default",
            purpose: "Run the default (TF-free) conformance set in CI.",
            commandExamples: [
              "cargo test --workspace --features candle_default -- conformance",
            ],
            inputs: ["Fixtures", "Rust implementation"],
            outputs: ["Pass/fail", "Diff reports"],
            notes: [
              "CI should always run this lane; keep it fast and deterministic.",
            ],
          },
          {
            id: "tf_best_effort",
            purpose:
              "Run TF runtime lane when libtensorflow is available and custom ops are loadable.",
            commandExamples: [
              "cargo test -p monolith-tf-runtime --features tf_runtime -- conformance_tf",
            ],
            inputs: ["libtensorflow", "custom ops", "Fixtures"],
            outputs: ["Pass/fail + explicit skip reasons"],
            notes: [
              "Always record skip reasons and missing symbols to aid setup/debugging.",
            ],
          },
        ],
        risks: [
          "TF runtime tests can be non-hermetic and platform-dependent; keep them opt-in and well-instrumented.",
        ],
      },
    ];

    const signals: ParitySignal[] = [
      {
        id: "signal_mapping_suggested_rust_test_locations",
        description:
          "Normalized mapping includes suggested Rust test locations for Python test modules; use this to drive harness placement and missing-test detection.",
        evidence: [
          normalizedMappingPath,
          "normalized_mapping.records[].suggestedRustTestLocation",
        ],
      },
      {
        id: "signal_proto_boundary",
        description:
          "Conventions doc records proto/op boundaries; use this to prioritize proto compatibility harnesses and TF runtime opt-in tests.",
        evidence: [mappingConventionsPath],
      },
    ];

    return {
      version: "1",
      inputs: {
        normalizedMappingPath,
        mappingConventionsPath,
        pythonTestGlobs,
        pythonFixtureGlobs,
        rustCratesTomlGlob,
        rustTestGlob,
      },
      repoScan: {
        pythonTests: {
          total: pythonTests.length,
          sample: toSample(pythonTests, 25),
          byTopDir: byTopDir(pythonTests),
        },
        pythonFixtures: {
          total: pythonFixtures.length,
          sample: toSample(pythonFixtures, 25),
        },
        rustCrates: { total: rustCrates.length, crates: rustCrates },
        rustTests: { total: rustTests.length, sample: toSample(rustTests, 25) },
      },
      requiredHarnesses,
      recommendedRepoLayout,
      signals,
      warnings,
    };
  },
);

export default (
  <Program
    id="parity-test-harness-plan"
    target={{ language: "md" }}
    description="Define repeatable parity validation harnesses (golden fixtures, differential tests, proto compatibility, and TF-optional gating) for Python->Rust monolith parity."
  >
    <Asset
      id="parity_test_harness_plan_doc"
      kind="doc"
      path="../generated/parity/80-parity-test-harness-plan/parity_test_harness_plan.md"
    />
    <Asset
      id="parity_test_harness_plan_json"
      kind="json"
      path="../generated/parity/80-parity-test-harness-plan/parity_test_harness_plan.json"
    />
    <Action
      id="compute-parity-test-harness-plan"
      export="computeParityTestHarnessPlanInputs"
      cache
    />
    <Agent
      id="write-parity-test-harness-plan-json"
      produces={["parity_test_harness_plan_json"]}
      external_needs={[
        { alias: "normalizedMappingJson", agent: "write-normalized-mapping-json" },
        { alias: "mappingConventionsDoc", agent: "write-mapping-conventions-doc" },
      ]}
    >
      <Prompt>
        <System>
          You maintain a parity planning pipeline. You output strictly valid JSON
          and write files using apply_patch.
        </System>
        <Context>
          {ctx.dependency(normalizedMappingJson, {
            as: "Normalized mapping JSON (11-normalize-mapping)",
            mode: "code",
          })}
          {ctx.dependency(mappingConventionsDoc, {
            as: "Mapping conventions (11-normalize-mapping)",
            mode: "quote",
          })}
          {ctx.actionResult("compute-parity-test-harness-plan", {
            as: "Computed parity test harness plan (canonical JSON)",
          })}
        </Context>
        <Instructions>{`Write JSON to \`{{assets.parity_test_harness_plan_json.path}}\` using apply_patch.
The JSON must be a single object and must exactly match the computed parity test harness plan.`}</Instructions>
      </Prompt>
    </Agent>
    <Agent
      id="write-parity-test-harness-plan-doc"
      needs={["write-parity-test-harness-plan-json"]}
      produces={["parity_test_harness_plan_doc"]}
      external_needs={[
        { alias: "normalizedMappingJson", agent: "write-normalized-mapping-json" },
        { alias: "mappingConventionsDoc", agent: "write-mapping-conventions-doc" },
      ]}
    >
      <Prompt>
        <System>
          You maintain a parity planning pipeline. You write deterministic
          operational docs and write files using apply_patch.
        </System>
        <Context>
          {ctx.dependency(normalizedMappingJson, {
            as: "Normalized mapping JSON (11-normalize-mapping)",
            mode: "code",
          })}
          {ctx.dependency(mappingConventionsDoc, {
            as: "Mapping conventions (11-normalize-mapping)",
            mode: "quote",
          })}
          {ctx.actionResult("compute-parity-test-harness-plan", {
            as: "Computed parity harness plan (JSON)",
          })}
        </Context>
        <Instructions>{`Write a parity test harness plan document to \`{{assets.parity_test_harness_plan_doc.path}}\` using apply_patch.

Requirements:
1) Deterministic: stable ordering and no timestamps.
2) Start with a short purpose statement: what the harness validates and how it should be used during Python->Rust porting.
3) Include a "Repo Scan Snapshot" section with:
   - Counts and short samples of Python tests/fixtures and Rust crates/tests from the computed JSON.
4) Include a "Harness Catalog" section that enumerates each harness:
   - id, kind, scope, ownerDomain, gating flags
   - fixture strategy (roots + examples)
   - comparison strategy (tolerances + normalization notes)
   - step-by-step procedure with command examples
   - top risks / flakiness mitigations
5) Include a "Gating Policy" section:
   - What runs by default (TF-free / Candle default)
   - What is best-effort (TF runtime lane)
   - How to record explicit skips and known gaps
6) Include a short "Next Actions" checklist: 6-10 concrete tasks to stand up the first harnesses, explicitly referencing the JSON fields and paths.
7) Keep this doc aligned with the conventions in the mapping conventions document and avoid inventing new naming requirements.`}</Instructions>
      </Prompt>
    </Agent>
  </Program>
);
