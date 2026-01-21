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

export const utilsEntrypointsPlanDoc = assetRef("utils_entrypoints_plan_doc");
export const utilsEntrypointsPlanJson = assetRef("utils_entrypoints_plan_json");

type FileWithLines = {
  path: string;
  lines: number;
};

type AbslFlag = {
  name: string;
  kind: string;
  definedIn: string;
};

type PythonImport = {
  module: string;
  definedIn: string;
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

type UtilsEntrypointsPlanInputs = {
  inputs: {
    scopedPythonFiles: string[];
    workspaceFiles: string[];
    normalizedMappingPath: string;
  };
  warnings: string[];
  pythonFiles: FileWithLines[];
  workspaceFiles: FileWithLines[];
  derived: {
    abslFlags: AbslFlag[];
    pythonImports: PythonImport[];
  };
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

function parseAbslFlags(params: {
  path: string;
  content: string;
}): AbslFlag[] {
  const { path, content } = params;
  const out: AbslFlag[] = [];
  const re = /flags\.DEFINE_([A-Za-z0-9_]+)\(\s*["']([^"']+)["']/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(content)) != null) {
    out.push({ kind: m[1] ?? "unknown", name: m[2] ?? "", definedIn: path });
  }
  out.sort(
    (a, b) =>
      a.definedIn.localeCompare(b.definedIn) ||
      a.name.localeCompare(b.name) ||
      a.kind.localeCompare(b.kind),
  );
  return out;
}

function parsePythonImports(params: {
  path: string;
  content: string;
}): PythonImport[] {
  const { path, content } = params;
  const out: PythonImport[] = [];
  const lines = content.split(/\r?\n/);
  const importRe = /^\s*import\s+([A-Za-z0-9_\.]+)\b/;
  const fromRe = /^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+/;
  for (const line of lines) {
    const im = importRe.exec(line);
    if (im != null) {
      out.push({ module: im[1] ?? "", definedIn: path });
      continue;
    }
    const fm = fromRe.exec(line);
    if (fm != null) {
      out.push({ module: fm[1] ?? "", definedIn: path });
    }
  }
  out.sort(
    (a, b) =>
      a.definedIn.localeCompare(b.definedIn) || a.module.localeCompare(b.module),
  );
  return out;
}

export const computeUtilsEntrypointsPlanInputs = action(
  async (actx): Promise<UtilsEntrypointsPlanInputs> => {
    const scopedPythonFiles = [
      "monolith/path_utils.py",
      "monolith/utils.py",
      "monolith/base_runner.py",
      "monolith/gpu_runner.py",
      "monolith/tpu_runner.py",
    ];
    const workspaceFiles = [
      "monolith/monolith_workspace.bzl",
      "monolith/tf_serving_workspace.bzl",
    ];
    const normalizedMappingPath =
      "generated/parity/11-normalize-mapping/normalized_mapping.json";

    const warnings: string[] = [];

    const pythonFiles: FileWithLines[] = [];
    const derivedFlags: AbslFlag[] = [];
    const derivedImports: PythonImport[] = [];

    for (const p of scopedPythonFiles) {
      try {
        const content = await actx.fs.readFile(p, "utf8");
        pythonFiles.push({ path: p, lines: countLines(content) });
        derivedFlags.push(...parseAbslFlags({ path: p, content }));
        derivedImports.push(...parsePythonImports({ path: p, content }));
      } catch {
        pythonFiles.push({ path: p, lines: 0 });
        warnings.push(`Failed to read ${p} for analysis.`);
      }
    }
    pythonFiles.sort((a, b) => a.path.localeCompare(b.path));

    const workspaceWithLines: FileWithLines[] = [];
    for (const p of workspaceFiles) {
      try {
        const content = await actx.fs.readFile(p, "utf8");
        workspaceWithLines.push({ path: p, lines: countLines(content) });
      } catch {
        workspaceWithLines.push({ path: p, lines: 0 });
        warnings.push(`Failed to read ${p} for analysis.`);
      }
    }
    workspaceWithLines.sort((a, b) => a.path.localeCompare(b.path));

    let normalizedRaw: string | undefined;
    try {
      normalizedRaw = await actx.fs.readFile(normalizedMappingPath, "utf8");
    } catch {
      warnings.push(
        `Missing ${normalizedMappingPath}; utils/entrypoints plan will be generated without normalized mapping context.`,
      );
    }

    let normalized: NormalizedMapping | undefined;
    if (normalizedRaw != null) {
      try {
        normalized = JSON.parse(normalizedRaw) as NormalizedMapping;
      } catch {
        warnings.push(
          `Failed to parse ${normalizedMappingPath} as JSON; utils/entrypoints plan will ignore it.`,
        );
      }
    }

    const normalizedRecords = (normalized?.records ?? []).filter((r) =>
      scopedPythonFiles.includes(r.pythonPath),
    );
    normalizedRecords.sort((a, b) => a.pythonPath.localeCompare(b.pythonPath));

    const normalizedSet = new Set(normalizedRecords.map((r) => r.pythonPath));
    const unmappedPythonFiles = scopedPythonFiles
      .slice()
      .sort()
      .filter((p) => !normalizedSet.has(p));

    const recordsWithIssues = normalizedRecords
      .filter((r) => (r.issues ?? []).length > 0)
      .map((r) => ({
        pythonPath: r.pythonPath,
        issues: (r.issues ?? []).slice().sort(),
      }))
      .sort((a, b) => a.pythonPath.localeCompare(b.pythonPath));

    const abslFlags = derivedFlags
      .filter((f) => f.name.length > 0)
      .slice()
      .sort(
        (a, b) =>
          a.definedIn.localeCompare(b.definedIn) ||
          a.name.localeCompare(b.name) ||
          a.kind.localeCompare(b.kind),
      );

    const pythonImports = derivedImports
      .filter((i) => i.module.length > 0)
      .slice()
      .sort(
        (a, b) =>
          a.definedIn.localeCompare(b.definedIn) ||
          a.module.localeCompare(b.module),
      );

    return {
      inputs: {
        scopedPythonFiles,
        workspaceFiles,
        normalizedMappingPath,
      },
      warnings,
      pythonFiles,
      workspaceFiles: workspaceWithLines,
      derived: {
        abslFlags,
        pythonImports,
      },
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
    id="utils-and-entrypoints-plan"
    target={{ language: "md" }}
    description="Plan parity for top-level monolith utilities + entrypoints (utils/path_utils/base_runner/gpu_runner/tpu_runner) plus Bazel workspace glue, mapping CLI/env/config semantics and Rust destinations."
  >
    <Asset
      id="utils_entrypoints_plan_doc"
      kind="doc"
      path="../generated/parity/35-utils-and-entrypoints-plan/utils_and_entrypoints_plan.md"
    />
    <Asset
      id="utils_entrypoints_plan_json"
      kind="json"
      path="../generated/parity/35-utils-and-entrypoints-plan/utils_and_entrypoints_plan.json"
    />
    <Action
      id="compute-utils-entrypoints-plan-inputs"
      export="computeUtilsEntrypointsPlanInputs"
      cache
    />
    <Agent
      id="write-utils-entrypoints-plan-json"
      produces={["utils_entrypoints_plan_json"]}
      external_needs={[
        {
          alias: "normalizedMappingJson",
          agent: "write-normalized-mapping-json",
        },
        {
          alias: "mappingConventionsDoc",
          agent: "write-mapping-conventions-doc",
        },
      ]}
    >
      <Prompt>
        <System>
          You are generating an implementable, file-by-file parity plan for
          porting monolith top-level utilities and entrypoints to Rust. You
          produce deterministic output, prefer stable ordering, and write files
          using apply_patch.
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
          {ctx.actionResult("compute-utils-entrypoints-plan-inputs", {
            as: "Scoped inventory + derived flags/imports + normalized records",
          })}
          {ctx.file("monolith/path_utils.py", {
            as: "path_utils.py",
            mode: "code",
          })}
          {ctx.file("monolith/utils.py", { as: "utils.py", mode: "code" })}
          {ctx.file("monolith/base_runner.py", {
            as: "base_runner.py",
            mode: "code",
          })}
          {ctx.file("monolith/gpu_runner.py", { as: "gpu_runner.py", mode: "code" })}
          {ctx.file("monolith/tpu_runner.py", { as: "tpu_runner.py", mode: "code" })}
          {ctx.file("monolith/monolith_workspace.bzl", {
            as: "monolith_workspace.bzl",
            mode: "code",
          })}
          {ctx.file("monolith/tf_serving_workspace.bzl", {
            as: "tf_serving_workspace.bzl",
            mode: "code",
          })}
        </Context>
        <Instructions>{`Write a single JSON object to \`{{assets.utils_entrypoints_plan_json.path}}\` using apply_patch.

Hard requirements:
1) Deterministic and planner-friendly: stable ordering; no timestamps; no prose outside JSON.
2) File accountability: include an entry for every scoped Python file and workspace file returned by the inventory, mapping each to Rust targets OR a justified N/A. Do not leave any untracked.
3) Respect mapping conventions: for Rust targets, prefer canonical \`monolith-rs/crates/<crate>/...\` full paths. If you suggest new crates/modules, justify.
4) Preserve runtime semantics: focus on CLI flags/env/config precedence, error behavior, path resolution, and filesystem behavior. For TF-specific code, clearly separate TF-runtime-required behavior vs TF-optional stubs.
5) Propose a realistic implementation order with small workstreams and explicit dependencies.
6) Include parity verification gates: what to test, what fixtures/goldens to add, and what "done" means.

Schema (must match exactly; fill all fields):
{
  "version": "1",
  "domain": "utils_and_entrypoints",
  "pythonRoot": "monolith",
  "scope": {
    "pythonFiles": [ "monolith/path_utils.py" ],
    "workspaceFiles": [ "monolith/monolith_workspace.bzl" ]
  },
  "rust": {
    "defaultCrate": "monolith-cli",
    "secondaryCrates": [ "monolith-core", "monolith-training", "monolith-config", "monolith-serving" ],
    "crateRoots": {
      "monolith-cli": "monolith-rs/crates/monolith-cli",
      "monolith-core": "monolith-rs/crates/monolith-core",
      "monolith-training": "monolith-rs/crates/monolith-training"
    },
    "proposedModuleLayout": [
      { "path": "monolith-rs/crates/monolith-cli/src/bin/monolith_gpu_runner.rs", "purpose": "CLI entrypoint parity for monolith/gpu_runner.py (flags, logging, exit codes, orchestration hooks)." }
    ]
  },
  "inventory": {
    "pythonFiles": [
      {
        "path": "monolith/path_utils.py",
        "lines": 0,
        "status": "TODO|IN_PROGRESS|DONE|N/A|BLOCKED",
        "rustTargets": [ "monolith-rs/crates/..." ],
        "naJustification": "N/A: ... | null",
        "notes": "..."
      }
    ],
    "workspaceFiles": [
      {
        "path": "monolith/monolith_workspace.bzl",
        "lines": 0,
        "status": "TODO|IN_PROGRESS|DONE|N/A|BLOCKED",
        "rustTargets": [ "monolith-rs/crates/..." ],
        "naJustification": "N/A: ... | null",
        "notes": "..."
      }
    ]
  },
  "interfaces": {
    "pathResolution": {
      "behavior": "Describe how Python code derives repo/base paths (including __file__-based logic and bazel/site-packages edge cases) and what Rust must replicate.",
      "compatibilityChecks": [ "..." ]
    },
    "filesystem": {
      "behavior": "Describe local vs remote filesystem semantics expected (tf.io.gfile copy/makedirs/rmtree/exists).",
      "compatibilityChecks": [ "..." ]
    },
    "entrypoints": {
      "base_runner": { "behavior": "Summarize BaseRunner contract and TF Summary writing behavior.", "compatibilityChecks": [ "..." ] },
      "gpu_runner": { "behavior": "Summarize GPURunner behavior: flags, Horovod/MPI handling, estimator loop semantics, summary writing.", "compatibilityChecks": [ "..." ] },
      "tpu_runner": { "behavior": "Summarize TPURunner behavior: TPU client/version config, embedding config, estimator config, eval loop behavior.", "compatibilityChecks": [ "..." ] }
    },
    "buildGlue": {
      "behavior": "Summarize Bazel WORKSPACE glue that affects runtime semantics (dependency pins, patches, custom ops).",
      "compatibilityChecks": [ "..." ]
    }
  },
  "harness": {
    "crossLangParity": {
      "strategy": "How to run Python and Rust on shared fixtures and compare outputs/errors deterministically (including CLI flag parsing and path resolution).",
      "sharedFixturesRoot": "monolith-rs/fixtures/parity/utils_entrypoints",
      "goldens": [
        { "id": "...", "description": "...", "pythonInvocation": "...", "rustInvocation": "...", "artifacts": [ "..." ] }
      ]
    }
  },
  "workstreams": [
    {
      "id": "ws1_path_utils",
      "title": "Path resolution + repo root discovery",
      "dependsOn": [],
      "pythonFiles": [ "monolith/path_utils.py" ],
      "workspaceFiles": [],
      "rustTargets": [ "monolith-rs/crates/monolith-core/src/path_utils.rs" ],
      "deliverables": [ "..." ],
      "parityChecks": [ "..." ],
      "tests": [ "..." ],
      "risks": [ "..." ]
    }
  ],
  "milestones": [
    { "id": "m1_path_utils_parity", "definitionOfDone": [ "..." ], "blockedBy": [ "..." ] }
  ],
  "openGaps": [
    { "gap": "...", "severity": "high|medium|low", "files": [ "monolith/..." ], "mitigation": "..." }
  ]
}`}</Instructions>
      </Prompt>
    </Agent>
    <Agent
      id="write-utils-entrypoints-plan-doc"
      needs={["write-utils-entrypoints-plan-json"]}
      produces={["utils_entrypoints_plan_doc"]}
      external_needs={[
        {
          alias: "normalizedMappingJson",
          agent: "write-normalized-mapping-json",
        },
        {
          alias: "mappingConventionsDoc",
          agent: "write-mapping-conventions-doc",
        },
      ]}
    >
      <Prompt>
        <System>
          You write deterministic engineering plans. You must reference output
          paths via assets bindings and write files using apply_patch.
        </System>
        <Context>
          {ctx.agent("write-utils-entrypoints-plan-json", {
            as: "Utils/Entrypoints plan JSON (generated by this module)",
            artifacts: ["utils_entrypoints_plan_json"],
          })}
          {ctx.actionResult("compute-utils-entrypoints-plan-inputs", {
            as: "Scoped inventory + derived flags/imports + normalized records",
          })}
          {ctx.dependency(mappingConventionsDoc, {
            as: "Mapping conventions (canonical)",
            mode: "quote",
          })}
        </Context>
        <Instructions>{`Write a concrete plan document to \`{{assets.utils_entrypoints_plan_doc.path}}\` using apply_patch.

Requirements:
1) Deterministic: stable ordering; no timestamps.
2) Implementation-oriented: concise but specific (file-level mapping, execution order, test plan).
3) Structure:
   - Purpose + scope (only monolith top-level utilities/entrypoints + workspace glue).
   - Target Rust crate/module layout (with key file paths).
   - CLI + config semantics: enumerate the flags and their expected effects for GPU/TPU runners; call out defaults and env/config precedence.
   - Filesystem + path semantics: gfile behavior expectations; repo root discovery parity requirements.
   - Execution order: list workstreams in order with dependency notes.
   - Parity strategy: what must be byte-for-byte compatible (paths, CLI flags behavior, logged messages/exit codes where relevant) vs best-effort.
   - Test plan: Rust unit/integration tests mirroring Python; cross-language parity harness and goldens.
   - File accountability table: one row per scoped file (python + workspace), including status, Rust target(s), notes/N/A justification.
   - Top risks + mitigations.
4) The file accountability table must align with the JSON plan's inventory lists.
5) Do not paste the entire JSON; summarize it and link to it by path via \`{{assets.utils_entrypoints_plan_json.path}}\`.`}</Instructions>
      </Prompt>
    </Agent>
  </Program>
);

