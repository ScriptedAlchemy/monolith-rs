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

import { mappingConventionsDoc, normalizedMappingJson } from "./11-normalize-mapping.ai.tsx";

export const nativeTrainingPlanDoc = assetRef("native_training_plan_doc");
export const nativeTrainingPlanJson = assetRef("native_training_plan_json");

type NativeTrainingFile = {
  path: string;
  lines: number;
};

type NormalizedRustTarget = {
  kind: string;
  crate?: string;
  path?: string;
  fullPath?: string;
  raw?: string;
};

type NormalizedRecord = {
  pythonPath: string;
  pythonLines: number;
  status: string;
  rustTargets: NormalizedRustTarget[];
  notes: string;
  source: string;
  issues: string[];
  naJustification?: string;
  suggestedRustTestLocation?: string;
};

type NormalizedMapping = {
  version: string;
  crates: string[];
  conventions?: unknown;
  counts?: unknown;
  warnings?: string[];
  records: NormalizedRecord[];
};

type NativeTrainingPlanInputs = {
  inputs: {
    pythonFilesGlob: string;
    protoFilesGlob: string;
    otherFilesGlob: string;
    normalizedMappingPath: string;
  };
  warnings: string[];
  pythonFiles: NativeTrainingFile[];
  protoFiles: string[];
  otherFiles: string[];
  normalized: {
    available: boolean;
    records: NormalizedRecord[];
    unmappedPythonFiles: string[];
    recordsWithIssues: Array<{ pythonPath: string; issues: string[] }>;
  };
};

function countLines(text: string): number {
  if (text.length === 0) return 0;
  return text.split(/\r?\n/).length;
}

export const computeNativeTrainingPlanInputs = action(
  async (actx): Promise<NativeTrainingPlanInputs> => {
    const pythonFilesGlob = "monolith/native_training/**/*.py";
    const protoFilesGlob = "monolith/native_training/**/*.proto";
    const otherFilesGlob = "monolith/native_training/**";
    const normalizedMappingPath =
      "generated/parity/11-normalize-mapping/normalized_mapping.json";

    const warnings: string[] = [];

    const pythonPaths = ((await actx.fs.glob(pythonFilesGlob)) as string[])
      .filter((p) => !p.endsWith("/"))
      .slice()
      .sort();

    const protoFiles = ((await actx.fs.glob(protoFilesGlob)) as string[])
      .filter((p) => !p.endsWith("/"))
      .slice()
      .sort();

    const otherFiles = ((await actx.fs.glob(otherFilesGlob)) as string[])
      .filter(
        (p) => !p.endsWith("/") && !p.endsWith(".py") && !p.endsWith(".proto"),
      )
      .slice()
      .sort();

    const pythonFiles: NativeTrainingFile[] = [];
    for (const p of pythonPaths) {
      try {
        const content = await actx.fs.readFile(p, "utf8");
        pythonFiles.push({ path: p, lines: countLines(content) });
      } catch {
        pythonFiles.push({ path: p, lines: 0 });
        warnings.push(`Failed to read ${p} for line counting.`);
      }
    }

    let normalizedRaw: string | undefined;
    try {
      normalizedRaw = await actx.fs.readFile(normalizedMappingPath, "utf8");
    } catch {
      warnings.push(
        `Missing ${normalizedMappingPath}; native training plan will be generated without normalized mapping context.`,
      );
    }

    let normalized: NormalizedMapping | undefined;
    if (normalizedRaw != null) {
      try {
        normalized = JSON.parse(normalizedRaw) as NormalizedMapping;
      } catch {
        warnings.push(
          `Failed to parse ${normalizedMappingPath} as JSON; native training plan will ignore it.`,
        );
      }
    }

    const normalizedRecords = (normalized?.records ?? []).filter((r) =>
      r.pythonPath.startsWith("monolith/native_training/"),
    );
    normalizedRecords.sort((a, b) => a.pythonPath.localeCompare(b.pythonPath));

    const normalizedSet = new Set(normalizedRecords.map((r) => r.pythonPath));
    const unmappedPythonFiles = pythonPaths.filter((p) => !normalizedSet.has(p));

    const recordsWithIssues = normalizedRecords
      .filter((r) => (r.issues ?? []).length > 0)
      .map((r) => ({
        pythonPath: r.pythonPath,
        issues: (r.issues ?? []).slice().sort(),
      }))
      .sort((a, b) => a.pythonPath.localeCompare(b.pythonPath));

    return {
      inputs: {
        pythonFilesGlob,
        protoFilesGlob,
        otherFilesGlob,
        normalizedMappingPath,
      },
      warnings,
      pythonFiles,
      protoFiles,
      otherFiles,
      normalized: {
        available: normalized != null,
        records: normalizedRecords,
        unmappedPythonFiles,
        recordsWithIssues,
      },
    };
  },
);

export default (
  <Program
    id="native-training-plan"
    target={{ language: "md" }}
    description="Segment monolith/native_training/** into implementable domains (data pipeline, distribution/runtime, export/checkpoint, hooks/metrics) and map each to Rust crates with parity risks and dependency ordering."
  >
    <Asset
      id="native_training_plan_doc"
      kind="doc"
      path="../generated/parity/40-native-training-plan/native_training_plan.md"
    />
    <Asset
      id="native_training_plan_json"
      kind="json"
      path="../generated/parity/40-native-training-plan/native_training_plan.json"
    />
    <Action
      id="compute-native-training-plan-inputs"
      export="computeNativeTrainingPlanInputs"
      cache
    />
    <Agent
      id="write-native-training-plan-json"
      produces={["native_training_plan_json"]}
      external_needs={[
        { alias: "normalizedMappingJson", agent: "write-normalized-mapping-json" },
        { alias: "mappingConventionsDoc", agent: "write-mapping-conventions-doc" },
      ]}
    >
      <Prompt>
        <System>
          You are generating an implementable, file-by-file parity sub-plan for
          porting monolith/native_training to Rust. You produce deterministic output,
          prefer stable ordering, and write files using apply_patch.
        </System>
        <Context>
          {ctx.dependency(normalizedMappingJson, {
            as: "Normalized mapping JSON (canonical)",
            mode: "code",
          })}
          {ctx.dependency(mappingConventionsDoc, {
            as: "Mapping conventions (canonical)",
            mode: "quote",
          })}
          {ctx.actionResult("compute-native-training-plan-inputs", {
            as: "Native training inventory + normalized records",
          })}
          {ctx.file("monolith/native_training/entry.py", { as: "entry.py", mode: "code" })}
          {ctx.file("monolith/native_training/cpu_training.py", {
            as: "cpu_training.py",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/estimator.py", {
            as: "estimator.py",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/save_utils.py", {
            as: "save_utils.py",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/dense_reload_utils.py", {
            as: "dense_reload_utils.py",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/distributed_ps.py", {
            as: "distributed_ps.py",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/data/datasets.py", {
            as: "data/datasets.py",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/data/parsers.py", {
            as: "data/parsers.py",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/model_export/saved_model_exporters.py", {
            as: "model_export/saved_model_exporters.py",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/model_export/export_utils.py", {
            as: "model_export/export_utils.py",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/hooks/ckpt_hooks.py", {
            as: "hooks/ckpt_hooks.py",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/sync_training_hooks.py", {
            as: "sync_training_hooks.py",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/alert/alert_manager.py", {
            as: "alert/alert_manager.py",
            mode: "code",
          })}
        </Context>
        <Instructions>{`Write a single JSON object to \`{{assets.native_training_plan_json.path}}\` using apply_patch.

Hard requirements:
1) Deterministic and planner-friendly: stable ordering; no timestamps; no prose outside JSON.
2) File accountability: include an entry for every Python file under monolith/native_training/** returned by the inventory, mapping each to a Rust target OR a justified N/A. Do not leave any untracked.
3) Respect mapping conventions: for Rust targets, prefer canonical \`monolith-rs/crates/<crate>/...\` full paths; use crates \`monolith-training\`, \`monolith-data\`, \`monolith-checkpoint\`, \`monolith-optimizer\`, \`monolith-hash-table\`, \`monolith-core\`, \`monolith-tensor\`, and \`monolith-proto\` as needed. If you introduce a new crate, mark it as a proposal and justify it.
4) Segment by subdomain: data pipeline (datasets/parsers/transform), distribution/runtime (distributed_ps, cluster/service discovery, sync/barrier ops), export/checkpoint (save_utils, dense_reload_utils, model_export/**), hooks/metrics/summary, and supporting utilities.
5) TensorFlow optionality: explicitly flag which workstreams require TensorFlow runtime and/or custom TF ops vs which can be Candle-only; do not assume vendoring TF runtime libraries.
6) Include a concrete execution order that is implementable (small steps, clear dependencies) and explicitly calls out TF-dependent milestones gated behind features.
7) Include parity verification gates for each workstream: what to test, what fixtures/goldens to add, and what "done" means.

Schema (must match exactly; fill all fields):
{
  "version": "1",
  "domain": "native_training",
  "pythonRoot": "monolith/native_training",
  "rust": {
    "defaultCrate": "monolith-training",
    "secondaryCrates": [ "monolith-data", "monolith-checkpoint", "monolith-optimizer", "monolith-hash-table", "monolith-core", "monolith-tensor", "monolith-proto" ],
    "crateRoots": {
      "monolith-training": "monolith-rs/crates/monolith-training",
      "monolith-data": "monolith-rs/crates/monolith-data",
      "monolith-checkpoint": "monolith-rs/crates/monolith-checkpoint",
      "monolith-optimizer": "monolith-rs/crates/monolith-optimizer",
      "monolith-hash-table": "monolith-rs/crates/monolith-hash-table"
    },
    "proposedModuleLayout": [
      { "path": "monolith-rs/crates/monolith-training/src/lib.rs", "purpose": "Training orchestration surface (entrypoints, runtime wiring, distributed control plane clients)." }
    ]
  },
  "inventory": {
    "pythonFiles": [ { "path": "monolith/native_training/...", "lines": 0, "status": "TODO|IN_PROGRESS|DONE|N/A|BLOCKED", "rustTargets": [ "monolith-rs/crates/..." ], "naJustification": "N/A: ... | null", "notes": "..." } ],
    "protoFiles": [ "monolith/native_training/..." ],
    "otherFiles": [ "monolith/native_training/..." ]
  },
  "interfaces": {
    "cliConfigEnv": {
      "entrypoints": [ "monolith/native_training/entry.py", "monolith/native_training/cpu_training.py", "monolith/native_training/demo.py" ],
      "behavior": "Describe flags/config/env precedence and any compatibility requirements that must be preserved for training jobs and tooling.",
      "compatibilityChecks": [ "..." ]
    },
    "formats": {
      "disk": [ "TFRecord", "SavedModel", "Checkpoint (TF v1/v2 as applicable)", "Warmup data formats", "Any custom protos or JSON configs" ],
      "wire": [ "gRPC/protobuf interfaces used by data services or distributed components, if any" ],
      "compatibilityChecks": [ "..." ]
    }
  },
  "harness": {
    "crossLangParity": {
      "strategy": "How to run Python and Rust on shared fixtures and compare outputs/errors deterministically (unit + integration).",
      "sharedFixturesRoot": "monolith-rs/fixtures/parity/native_training",
      "goldens": [ { "id": "...", "description": "...", "pythonInvocation": "...", "rustInvocation": "...", "artifacts": [ "..." ] } ]
    }
  },
  "workstreams": [
    {
      "id": "ws1_data_pipeline",
      "title": "Data pipeline parity (datasets/parsers/transform)",
      "dependsOn": [],
      "pythonFiles": [ "monolith/native_training/data/datasets.py", "monolith/native_training/data/parsers.py" ],
      "rustTargets": [ "monolith-rs/crates/monolith-data/src/lib.rs" ],
      "deliverables": [ "..." ],
      "parityChecks": [ "..." ],
      "tests": [ "..." ],
      "risks": [ "..." ],
      "tfDependency": { "required": false, "reasons": [ "..." ] }
    }
  ],
  "milestones": [
    { "id": "m1_data_pipeline_minimal", "definitionOfDone": [ "..." ], "blockedBy": [ "..." ] }
  ],
  "openGaps": [
    { "gap": "...", "severity": "high|medium|low", "files": [ "monolith/native_training/..." ], "mitigation": "..." }
  ]
}`}</Instructions>
      </Prompt>
    </Agent>
    <Agent
      id="write-native-training-plan-doc"
      needs={["write-native-training-plan-json"]}
      produces={["native_training_plan_doc"]}
      external_needs={[
        { alias: "normalizedMappingJson", agent: "write-normalized-mapping-json" },
        { alias: "mappingConventionsDoc", agent: "write-mapping-conventions-doc" },
      ]}
    >
      <Prompt>
        <System>
          You write deterministic engineering plans. You must reference output paths
          via assets bindings and write files using apply_patch.
        </System>
        <Context>
          {ctx.agent("write-native-training-plan-json", {
            artifacts: ["native_training_plan_json"],
            as: "Native training plan JSON (generated by this module)",
          })}
          {ctx.actionResult("compute-native-training-plan-inputs", {
            as: "Native training inventory + normalized records",
          })}
          {ctx.dependency(mappingConventionsDoc, {
            as: "Mapping conventions (canonical)",
            mode: "quote",
          })}
        </Context>
        <Instructions>{`Write a concrete plan document to \`{{assets.native_training_plan_doc.path}}\` using apply_patch.

Requirements:
1) Deterministic: stable ordering; no timestamps.
2) Keep it implementation-oriented: concise, but specific (file-level mapping, execution order, and test plan).
3) Structure:
   - Purpose + scope (native_training only; call out that runtime C++/ops are in native_training/runtime/** and may map to Rust crates or remain TF custom ops).
   - Domain segmentation: data pipeline; distribution/runtime; export/checkpoint; hooks/metrics/summary; discovery/consul/zk; utilities.
   - Target Rust crate/module layout (with key file paths per domain).
   - Execution order: list workstreams in order with dependency notes and feature gates for TF runtime optionality.
   - Parity strategy: which behaviors must be byte-for-byte identical (format parsing/serialization, checkpoint metadata) vs behaviorally equivalent (floating point numerics).
   - Test plan: Rust unit/integration tests mirroring Python tests where feasible; cross-language parity harness and goldens.
   - File accountability table: one row per Python file under monolith/native_training/** (path, status, Rust target(s), notes/N/A justification).
   - Top risks + mitigations (TF custom ops, tf.data semantics, distributed runtime behavior, checkpoint/SavedModel compatibility).
4) The "File accountability table" must align with the JSON plan's inventory.pythonFiles list.
5) Do not paste the entire JSON; summarize it and link to it by path via \`{{assets.native_training_plan_json.path}}\`.`}</Instructions>
      </Prompt>
    </Agent>
  </Program>
);
