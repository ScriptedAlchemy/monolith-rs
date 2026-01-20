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

export const corePlanDoc = assetRef("core_plan_doc");
export const corePlanJson = assetRef("core_plan_json");

type CoreFile = {
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

type CorePlanInputs = {
  inputs: {
    pythonFilesGlob: string;
    otherFilesGlob: string;
    normalizedMappingPath: string;
  };
  warnings: string[];
  pythonFiles: CoreFile[];
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

export const computeCorePlanInputs = action(async (actx): Promise<CorePlanInputs> => {
  const pythonFilesGlob = "monolith/core/**/*.py";
  const otherFilesGlob = "monolith/core/**";
  const normalizedMappingPath =
    "generated/parity/11-normalize-mapping/normalized_mapping.json";

  const warnings: string[] = [];

  const pythonPaths = ((await actx.fs.glob(pythonFilesGlob)) as string[])
    .filter((p) => !p.endsWith("/"))
    .slice()
    .sort();

  const otherFiles = ((await actx.fs.glob(otherFilesGlob)) as string[])
    .filter((p) => !p.endsWith("/") && !p.endsWith(".py"))
    .slice()
    .sort();

  const pythonFiles: CoreFile[] = [];
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
      `Missing ${normalizedMappingPath}; core plan will be generated without normalized mapping context.`,
    );
  }

  let normalized: NormalizedMapping | undefined;
  if (normalizedRaw != null) {
    try {
      normalized = JSON.parse(normalizedRaw) as NormalizedMapping;
    } catch {
      warnings.push(
        `Failed to parse ${normalizedMappingPath} as JSON; core plan will ignore it.`,
      );
    }
  }

  const normalizedRecords = (normalized?.records ?? []).filter((r) =>
    r.pythonPath.startsWith("monolith/core/"),
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
      otherFilesGlob,
      normalizedMappingPath,
    },
    warnings,
    pythonFiles,
    otherFiles,
    normalized: {
      available: normalized != null,
      records: normalizedRecords,
      unmappedPythonFiles,
      recordsWithIssues,
    },
  };
});

export default (
  <Program id="core-plan" target={{ language: "md" }} description="Produce a concrete sub-plan for monolith/core/** parity, prioritizing hyperparams, feature/env semantics, model_registry, and base_layer/task/model_params, including test parity requirements."><Asset id="core_plan_doc" kind="doc" path="../generated/parity/30-core-plan/core_plan.md" /><Asset id="core_plan_json" kind="json" path="../generated/parity/30-core-plan/core_plan.json" /><Action id="compute-core-plan-inputs" export="computeCorePlanInputs" cache /><Agent id="write-core-plan-json" produces={["core_plan_json"]} external_needs={[{ alias: "normalizedMappingJson", agent: "write-normalized-mapping-json" }, { alias: "mappingConventionsDoc", agent: "write-mapping-conventions-doc" }]}><Prompt><System>
          You are generating an implementable, file-by-file parity sub-plan for porting monolith/core to Rust. You produce deterministic output, prefer stable ordering, and write files using apply_patch.
        </System><Context>{ctx.dependency(normalizedMappingJson, { as: "Normalized mapping JSON (canonical)", mode: "code" })}{ctx.dependency(mappingConventionsDoc, { as: "Mapping conventions (canonical)", mode: "quote" })}{ctx.actionResult("compute-core-plan-inputs", { as: "Core inventory + normalized records" })}{ctx.file("monolith/core/hyperparams.py", { as: "hyperparams.py", mode: "code" })}{ctx.file("monolith/core/hyperparams_test.py", { as: "hyperparams_test.py", mode: "code" })}{ctx.file("monolith/core/feature.py", { as: "feature.py", mode: "code" })}{ctx.file("monolith/core/feature_test.py", { as: "feature_test.py", mode: "code" })}{ctx.file("monolith/core/model_registry.py", { as: "model_registry.py", mode: "code" })}{ctx.file("monolith/core/base_layer.py", { as: "base_layer.py", mode: "code" })}{ctx.file("monolith/core/base_layer_test.py", { as: "base_layer_test.py", mode: "code" })}{ctx.file("monolith/core/base_task.py", { as: "base_task.py", mode: "code" })}{ctx.file("monolith/core/base_model_params.py", { as: "base_model_params.py", mode: "code" })}{ctx.file("monolith/core/model.py", { as: "model.py", mode: "code" })}{ctx.file("monolith/core/dense.py", { as: "dense.py", mode: "code" })}{ctx.file("monolith/core/dense_test.py", { as: "dense_test.py", mode: "code" })}{ctx.file("monolith/core/util.py", { as: "util.py", mode: "code" })}{ctx.file("monolith/core/util_test.py", { as: "util_test.py", mode: "code" })}{ctx.file("monolith/core/py_utils.py", { as: "py_utils.py", mode: "code" })}{ctx.file("monolith/core/testing_utils.py", { as: "testing_utils.py", mode: "code" })}</Context><Instructions>{`Write a single JSON object to \`{{assets.core_plan_json.path}}\` using apply_patch.

Hard requirements:
1) Deterministic and planner-friendly: stable ordering; no timestamps; no prose outside JSON.
2) File accountability: include an entry for every Python file under monolith/core/** returned by the inventory, mapping each to a Rust target OR a justified N/A. Do not leave any untracked.
3) Respect mapping conventions: for Rust targets, prefer canonical \`monolith-rs/crates/<crate>/...\` full paths; default crate is \`monolith-core\` unless you justify otherwise (some layers may belong in \`monolith-layers\`).
4) Prioritize behavior parity over idiomatic APIs: match Python semantics, config/env precedence, error behavior, and serialization formats.
5) Include a concrete execution order that is implementable (small steps, clear dependencies) and explicitly calls out TensorFlow-dependent areas vs Candle-friendly areas.
6) Include parity verification gates for each workstream: what to test, what fixtures/goldens to add, and what "done" means.

Schema (must match exactly; fill all fields):
{
  "version": "1",
  "domain": "core",
  "pythonRoot": "monolith/core",
  "rust": {
    "defaultCrate": "monolith-core",
    "secondaryCrates": [ "monolith-layers", "monolith-config", "monolith-proto" ],
    "crateRoots": {
      "monolith-core": "monolith-rs/crates/monolith-core",
      "monolith-layers": "monolith-rs/crates/monolith-layers"
    },
    "proposedModuleLayout": [
      { "path": "monolith-rs/crates/monolith-core/src/hyperparams/mod.rs", "purpose": "Hyperparams parsing, defaults, merge semantics, and stable repr/serialization parity." }
    ]
  },
  "inventory": {
    "pythonFiles": [ { "path": "monolith/core/...", "lines": 0, "status": "TODO|IN_PROGRESS|DONE|N/A|BLOCKED", "rustTargets": [ "monolith-rs/crates/..." ], "naJustification": "N/A: ... | null", "notes": "..." } ],
    "otherFiles": [ "monolith/core/..." ]
  },
  "interfaces": {
    "configEnv": {
      "behavior": "Describe env var and flag/config precedence that must be preserved.",
      "compatibilityChecks": [ "..." ]
    },
    "serialization": {
      "formats": [ "Describe any JSON/text proto/flat config formats that must remain compatible." ],
      "compatibilityChecks": [ "..." ]
    }
  },
  "harness": {
    "crossLangParity": {
      "strategy": "How to run Python and Rust on shared fixtures and compare outputs/errors deterministically.",
      "sharedFixturesRoot": "monolith-rs/fixtures/parity/core",
      "goldens": [ { "id": "...", "description": "...", "pythonInvocation": "...", "rustInvocation": "...", "artifacts": [ "..." ] } ]
    }
  },
  "workstreams": [
    {
      "id": "ws1_hyperparams",
      "title": "Hyperparams + config semantics",
      "dependsOn": [],
      "pythonFiles": [ "monolith/core/hyperparams.py" ],
      "rustTargets": [ "monolith-rs/crates/monolith-core/src/hyperparams/mod.rs" ],
      "deliverables": [ "..." ],
      "parityChecks": [ "..." ],
      "tests": [ "..." ],
      "risks": [ "..." ]
    }
  ],
  "milestones": [
    { "id": "m1_hyperparams_parity", "definitionOfDone": [ "..." ], "blockedBy": [ "..." ] }
  ],
  "openGaps": [
    { "gap": "...", "severity": "high|medium|low", "files": [ "monolith/core/..." ], "mitigation": "..." }
  ]
}`}</Instructions>
        </Prompt></Agent><Agent id="write-core-plan-doc" needs={["write-core-plan-json"]} produces={["core_plan_doc"]} external_needs={[{ alias: "normalizedMappingJson", agent: "write-normalized-mapping-json" }, { alias: "mappingConventionsDoc", agent: "write-mapping-conventions-doc" }]}><Prompt><System>
          You write deterministic engineering plans. You must reference output paths via assets bindings and write files using apply_patch.
        </System><Context>{ctx.file("generated/parity/30-core-plan/core_plan.json", { as: "Core plan JSON (generated by this module)", mode: "code" })}{ctx.actionResult("compute-core-plan-inputs", { as: "Core inventory + normalized records" })}{ctx.dependency(mappingConventionsDoc, { as: "Mapping conventions (canonical)", mode: "quote" })}</Context><Instructions>{`Write a concrete plan document to \`{{assets.core_plan_doc.path}}\` using apply_patch.

Requirements:
1) Deterministic: stable ordering; no timestamps.
2) Keep it implementation-oriented: concise, but specific (file-level mapping, execution order, and test plan).
3) Structure:
   - Purpose + scope (core only).
   - Target Rust crate/module layout (with key file paths).
   - Execution order: list workstreams in order with dependency notes.
   - Parity strategy: which behaviors must be byte-for-byte identical (repr/serialization) vs behaviorally equivalent (numerics).
   - Test plan: Rust unit/integration tests mirroring Python tests; cross-language parity harness and goldens.
   - File accountability table: one row per Python file under monolith/core/** (path, status, Rust target(s), notes/N/A justification).
   - Top risks + mitigations.
4) The "File accountability table" must align with the JSON plan's inventory.pythonFiles list.
5) Do not paste the entire JSON; summarize it and link to it by path via \`{{assets.core_plan_json.path}}\`.`}</Instructions>
        </Prompt></Agent></Program>
);
