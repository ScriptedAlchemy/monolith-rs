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

export const agentServicePlanDoc = assetRef("agent_service_plan_doc");
export const agentServicePlanJson = assetRef("agent_service_plan_json");

type AgentServiceFile = {
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

type AgentServicePlanInputs = {
  inputs: {
    pythonFilesGlob: string;
    protoFilesGlob: string;
    otherFilesGlob: string;
    normalizedMappingPath: string;
  };
  warnings: string[];
  pythonFiles: AgentServiceFile[];
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

export const computeAgentServicePlanInputs = action(
  async (actx): Promise<AgentServicePlanInputs> => {
    const pythonFilesGlob = "monolith/agent_service/**/*.py";
    const protoFilesGlob = "monolith/agent_service/**/*.proto";
    const otherFilesGlob = "monolith/agent_service/**";
    const normalizedMappingPath =
      "generated/parity/11-normalize-mapping/normalized_mapping.json";

    const warnings: string[] = [];

    const pythonPaths = ((await actx.fs.glob(pythonFilesGlob)) as string[])
      .slice()
      .sort();
    const protoFiles = ((await actx.fs.glob(protoFilesGlob)) as string[])
      .slice()
      .sort();
    const otherFiles = ((await actx.fs.glob(otherFilesGlob)) as string[])
      .filter(
        (p) => !p.endsWith("/") && !p.endsWith(".py") && !p.endsWith(".proto"),
      )
      .slice()
      .sort();

    const pythonFiles: AgentServiceFile[] = [];
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
        `Missing ${normalizedMappingPath}; agent_service plan will be generated without normalized mapping context.`,
      );
    }

    let normalized: NormalizedMapping | undefined;
    if (normalizedRaw != null) {
      try {
        normalized = JSON.parse(normalizedRaw) as NormalizedMapping;
      } catch {
        warnings.push(
          `Failed to parse ${normalizedMappingPath} as JSON; agent_service plan will ignore it.`,
        );
      }
    }

    const normalizedRecords = (normalized?.records ?? []).filter((r) =>
      r.pythonPath.startsWith("monolith/agent_service/"),
    );
    normalizedRecords.sort((a, b) => a.pythonPath.localeCompare(b.pythonPath));

    const normalizedSet = new Set(normalizedRecords.map((r) => r.pythonPath));
    const unmappedPythonFiles = pythonPaths.filter(
      (p) => !normalizedSet.has(p),
    );

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
    id="agent-service-plan"
    target={{ language: "md" }}
    description="Produce a concrete, implementable sub-plan for monolith/agent_service/** parity (grouping, Rust module targets, fake ZK/TFServing test harness strategy, execution order)."
  >
    <Asset
      id="agent_service_plan_doc"
      kind="doc"
      path="../generated/parity/20-agent-service-plan/agent_service_plan.md"
    />
    <Asset
      id="agent_service_plan_json"
      kind="json"
      path="../generated/parity/20-agent-service-plan/agent_service_plan.json"
    />
    <Action
      id="compute-agent-service-plan-inputs"
      export="computeAgentServicePlanInputs"
      cache
    />
    <Agent
      id="write-agent-service-plan-json"
      produces={["agent_service_plan_json"]}
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
          You are generating an implementable, file-by-file parity sub-plan for
          porting monolith/agent_service to Rust. You produce deterministic
          output, prefer stable ordering, and write files using apply_patch.
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
          {ctx.actionResult("compute-agent-service-plan-inputs", {
            as: "Agent service inventory + normalized records",
          })}
          {ctx.file("monolith/agent_service/agent_service.proto", {
            as: "agent_service.proto",
            mode: "code",
          })}
          {ctx.file("monolith/agent_service/agent_service.py", {
            as: "agent_service.py",
            mode: "code",
          })}
          {ctx.file("monolith/agent_service/backends.py", {
            as: "backends.py",
            mode: "code",
          })}
          {ctx.file("monolith/agent_service/replica_manager.py", {
            as: "replica_manager.py",
            mode: "code",
          })}
          {ctx.file("monolith/agent_service/model_manager.py", {
            as: "model_manager.py",
            mode: "code",
          })}
          {ctx.file("monolith/agent_service/tfs_client.py", {
            as: "tfs_client.py",
            mode: "code",
          })}
          {ctx.file("monolith/agent_service/tfs_wrapper.py", {
            as: "tfs_wrapper.py",
            mode: "code",
          })}
          {ctx.file("monolith/agent_service/tfs_monitor.py", {
            as: "tfs_monitor.py",
            mode: "code",
          })}
          {ctx.file("monolith/agent_service/zk_mirror.py", {
            as: "zk_mirror.py",
            mode: "code",
          })}
          {ctx.file("monolith/agent_service/mocked_zkclient.py", {
            as: "mocked_zkclient.py",
            mode: "code",
          })}
          {ctx.file("monolith/agent_service/mocked_tfserving.py", {
            as: "mocked_tfserving.py",
            mode: "code",
          })}
          {ctx.file("monolith/agent_service/run.py", {
            as: "run.py",
            mode: "code",
          })}
        </Context>
        <Instructions>{`Write a single JSON object to \`{{assets.agent_service_plan_json.path}}\` using apply_patch.

Hard requirements:
1) Deterministic and planner-friendly: stable ordering; no timestamps; no prose outside JSON.
2) File accountability: include an entry for every Python file under monolith/agent_service/** returned by the inventory, mapping each to a Rust target OR a justified N/A. Do not leave any untracked.
3) Respect mapping conventions: for Rust targets, prefer canonical \`monolith-rs/crates/<crate>/...\` full paths; default crate is \`monolith-serving\` unless you justify otherwise.
4) Include a concrete execution order that is implementable (small steps, clear dependencies).
5) Include a fake/test harness strategy for:
   - ZK behaviors (watches, ephemeral nodes, mirroring)
   - TFServing behaviors (Predict, model status polling, error handling)
   - gRPC AgentService end-to-end contract compatibility (protobuf + server/client)
6) Include parity verification gates for each workstream: what to test, what fixtures/goldens to add, and what "done" means.

Schema (must match exactly; fill all fields):
{
  "version": "1",
  "domain": "agent_service",
  "pythonRoot": "monolith/agent_service",
  "rust": {
    "defaultCrate": "monolith-serving",
    "crateRoot": "monolith-rs/crates/monolith-serving",
    "proposedModuleLayout": [
      { "path": "monolith-rs/crates/monolith-serving/src/agent_service/mod.rs", "purpose": "..." }
    ]
  },
  "inventory": {
    "pythonFiles": [ { "path": "monolith/agent_service/...", "lines": 0, "status": "TODO|IN_PROGRESS|DONE|N/A|BLOCKED", "rustTargets": [ "monolith-rs/crates/..." ], "naJustification": "N/A: ... | null", "notes": "..." } ],
    "protoFiles": [ "monolith/agent_service/..." ],
    "otherFiles": [ "monolith/agent_service/..." ]
  },
  "interfaces": {
    "grpc": { "proto": "monolith/agent_service/agent_service.proto", "rustProtoCrate": "monolith-proto", "compatibilityChecks": [ "..." ] },
    "cli": { "pythonEntrypoints": [ "monolith/agent_service/run.py" ], "compatibilityChecks": [ "..." ] },
    "configEnv": { "files": [ "monolith/agent_service/agent.conf" ], "compatibilityChecks": [ "..." ] }
  },
  "harness": {
    "fakeZk": { "strategy": "...", "rustLocation": "monolith-rs/crates/monolith-serving/tests/support/fake_zk.rs", "pythonReference": "monolith/agent_service/mocked_zkclient.py" },
    "fakeTfServing": { "strategy": "...", "rustLocation": "monolith-rs/crates/monolith-serving/tests/support/fake_tfserving.rs", "pythonReference": "monolith/agent_service/mocked_tfserving.py" },
    "crossLangParity": { "strategy": "...", "sharedFixturesRoot": "monolith-rs/fixtures/parity/agent_service", "cases": [ { "id": "...", "description": "...", "inputs": [ "..." ], "expected": [ "..." ] } ] }
  },
  "workstreams": [
    {
      "id": "ws1_grpc_contract",
      "title": "...",
      "dependsOn": [],
      "pythonFiles": [ "monolith/agent_service/..." ],
      "rustTargets": [ "monolith-rs/crates/..." ],
      "deliverables": [ "..." ],
      "parityChecks": [ "..." ],
      "tests": [ "..." ],
      "risks": [ "..." ]
    }
  ],
  "milestones": [
    { "id": "m1_smoke", "definitionOfDone": [ "..." ], "blockedBy": [ "..." ] }
  ],
  "openGaps": [
    { "gap": "...", "severity": "high|medium|low", "files": [ "monolith/agent_service/..." ], "mitigation": "..." }
  ]
}`}</Instructions>
      </Prompt>
    </Agent>
    <Agent
      id="write-agent-service-plan-doc"
      needs={["write-agent-service-plan-json"]}
      produces={["agent_service_plan_doc"]}
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
          {ctx.file(
            "generated/parity/20-agent-service-plan/agent_service_plan.json",
            {
              as: "Agent service plan JSON (generated by this module)",
              mode: "code",
            },
          )}
          {ctx.actionResult("compute-agent-service-plan-inputs", {
            as: "Agent service inventory + normalized records",
          })}
          {ctx.dependency(mappingConventionsDoc, {
            as: "Mapping conventions (canonical)",
            mode: "quote",
          })}
        </Context>
        <Instructions>{`Write a concrete plan document to \`{{assets.agent_service_plan_doc.path}}\` using apply_patch.

Requirements:
1) Deterministic: stable ordering; no timestamps.
2) Keep it implementation-oriented: concise, but specific (file-level mapping, execution order, and test plan).
3) Structure:
   - Purpose + scope (agent_service only).
   - Target Rust crate/module layout (with key file paths).
   - Execution order: list workstreams in order with dependency notes.
   - Test/harness strategy: Fake ZK, Fake TFServing, cross-language parity harness, and gRPC contract checks.
   - File accountability table: one row per Python file under monolith/agent_service/** (path, status, Rust target(s), notes/N/A justification).
   - Top risks + mitigations.
4) The "File accountability table" must align with the JSON plan's inventory.pythonFiles list.
5) Do not paste the entire JSON; summarize it and link to it by path via \`{{assets.agent_service_plan_json.path}}\`.`}</Instructions>
      </Prompt>
    </Agent>
  </Program>
);
