import "./20-agent-service-plan.ai.tsx";

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
  agentServicePlanDoc,
  agentServicePlanJson,
} from "./20-agent-service-plan.ai.tsx";

export const agentServiceImplWorkItemsJson = assetRef(
  "agent_service_impl_work_items_json",
);
export const agentServiceImplWorkItemsDoc = assetRef(
  "agent_service_impl_work_items_doc",
);

type ExistingRustSurface = {
  crate: string;
  crateRoot: string;
  srcFiles: string[];
  testFiles: string[];
};

type AgentServiceImplInputs = {
  inputs: {
    agentServicePlanJsonPath: string;
    agentServicePlanDocPath: string;
    defaultCrate: string;
    crateRoot: string;
  };
  warnings: string[];
  plan: {
    available: boolean;
    workstreams: Array<{
      id: string;
      title: string;
      dependsOn: string[];
      pythonFiles: string[];
      rustTargets: string[];
      deliverables: string[];
      parityChecks: string[];
      tests: string[];
      risks: string[];
    }>;
    milestones: Array<{
      id: string;
      blockedBy: string[];
      definitionOfDone: string[];
    }>;
    openGaps: Array<{
      gap: string;
      severity: string;
      files: string[];
      mitigation: string;
    }>;
  };
  rust: {
    monolithServing: ExistingRustSurface;
  };
};

function safeParseJson(text: string): any | undefined {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function stableStringArray(xs: any): string[] {
  if (!Array.isArray(xs)) return [];
  return xs
    .filter((x) => typeof x === "string")
    .slice()
    .sort();
}

function stableObjectArray(xs: any): any[] {
  if (!Array.isArray(xs)) return [];
  return xs
    .filter((x) => x != null && typeof x === "object")
    .slice()
    .sort((a, b) => String(a.id ?? "").localeCompare(String(b.id ?? "")));
}

export const computeAgentServiceImplInputs = action(
  async (actx): Promise<AgentServiceImplInputs> => {
    const agentServicePlanJsonPath =
      "generated/parity/20-agent-service-plan/agent_service_plan.json";
    const agentServicePlanDocPath =
      "generated/parity/20-agent-service-plan/agent_service_plan.md";

    const defaultCrate = "monolith-serving";
    const crateRoot = `monolith-rs/crates/${defaultCrate}`;

    const warnings: string[] = [];

    let planRaw: string | undefined;
    try {
      planRaw = await actx.fs.readFile(agentServicePlanJsonPath, "utf8");
    } catch {
      warnings.push(
        `Missing ${agentServicePlanJsonPath}; will produce a stub implementation ticket set.`,
      );
    }

    const planParsed = planRaw != null ? safeParseJson(planRaw) : undefined;

    const planWorkstreams = stableObjectArray(planParsed?.workstreams).map(
      (ws) => {
        const id = String(ws.id ?? "").trim();
        const title = String(ws.title ?? "").trim();
        return {
          id,
          title,
          dependsOn: stableStringArray(ws.dependsOn),
          pythonFiles: stableStringArray(ws.pythonFiles),
          rustTargets: stableStringArray(ws.rustTargets),
          deliverables: stableStringArray(ws.deliverables),
          parityChecks: stableStringArray(ws.parityChecks),
          tests: stableStringArray(ws.tests),
          risks: stableStringArray(ws.risks),
        };
      },
    );

    const planMilestones = stableObjectArray(planParsed?.milestones).map(
      (m) => ({
        id: String(m.id ?? "").trim(),
        blockedBy: stableStringArray(m.blockedBy),
        definitionOfDone: stableStringArray(m.definitionOfDone),
      }),
    );

    const planOpenGaps = (
      Array.isArray(planParsed?.openGaps) ? (planParsed.openGaps as any[]) : []
    )
      .filter((g) => g != null && typeof g === "object")
      .map((g) => ({
        gap: String(g.gap ?? "").trim(),
        severity: String(g.severity ?? "").trim(),
        files: stableStringArray(g.files),
        mitigation: String(g.mitigation ?? "").trim(),
      }))
      .filter((g) => g.gap.length > 0)
      .sort((a, b) => a.gap.localeCompare(b.gap));

    const listIfExists = async (glob: string): Promise<string[]> => {
      try {
        return ((await actx.fs.glob(glob)) as string[]).slice().sort();
      } catch {
        warnings.push(`Glob failed: ${glob}`);
        return [];
      }
    };

    const srcFiles = await listIfExists(`${crateRoot}/src/**/*.rs`);
    const testFiles = await listIfExists(`${crateRoot}/tests/**/*.rs`);

    return {
      inputs: {
        agentServicePlanJsonPath,
        agentServicePlanDocPath,
        defaultCrate,
        crateRoot,
      },
      warnings,
      plan: {
        available: planParsed != null,
        workstreams: planWorkstreams.filter((ws) => ws.id.length > 0),
        milestones: planMilestones.filter((m) => m.id.length > 0),
        openGaps: planOpenGaps,
      },
      rust: {
        monolithServing: {
          crate: defaultCrate,
          crateRoot,
          srcFiles,
          testFiles,
        },
      },
    };
  },
);

export default (
  <Program
    id="agent-service-impl"
    target={{ language: "md" }}
    description="Produce an implementation-ready first slice (or ticketization) for monolith/agent_service/** parity in Rust (monolith-serving), based on the agent_service plan."
  >
    <Asset
      id="agent_service_impl_work_items_json"
      kind="json"
      path="../generated/parity/21-agent-service-impl/agent_service_impl_work_items.json"
    />
    <Asset
      id="agent_service_impl_work_items_doc"
      kind="doc"
      path="../generated/parity/21-agent-service-impl/agent_service_impl_work_items.md"
    />
    <Action
      id="compute-agent-service-impl-inputs"
      export="computeAgentServiceImplInputs"
      cache
    />
    <Agent
      id="write-agent-service-impl-work-items-json"
      produces={["agent_service_impl_work_items_json"]}
      external_needs={[
        {
          alias: "agentServicePlanJson",
          agent: "write-agent-service-plan-json",
        },
        { alias: "agentServicePlanDoc", agent: "write-agent-service-plan-doc" },
      ]}
    >
      <Prompt>
        <System>
          You produce implementation-ready engineering work items for porting
          monolith/agent_service to Rust. Output is deterministic, structured,
          and suitable for execution by a separate engineer. You write files
          using apply_patch.
        </System>
        <Context>
          {ctx.dependency(agentServicePlanJson, {
            as: "Agent service plan JSON (upstream)",
            mode: "code",
          })}
          {ctx.dependency(agentServicePlanDoc, {
            as: "Agent service plan doc (upstream)",
            mode: "quote",
          })}
          {ctx.actionResult("compute-agent-service-impl-inputs", {
            as: "Derived inputs (existing Rust surface + parsed plan)",
          })}
        </Context>
        <Instructions>{`Write a single JSON object to \`{{assets.agent_service_impl_work_items_json.path}}\` using apply_patch.

Hard requirements:
1) Deterministic: stable ordering; no timestamps; no prose outside JSON.
2) Focus: define an executable "first slice" for Rust implementation that unlocks follow-on work. Prefer stubbed+mocked infrastructure first (fake ZK + fake TFServing + minimal data defs) and minimal gRPC contract compilation.
3) Reference upstream plan: select workstreams/milestones from the agent_service plan JSON and break them into granular tickets with explicit deliverables and tests.
4) Keep scope realistic: each work item should be 0.5-2 days of work, with clear "done" checks.
5) Every work item must mention concrete repo file paths to create/modify, but do not assume those files already exist.

Schema (must match exactly; fill all fields):
{
  "version": "1",
  "domain": "agent_service",
  "inputs": {
    "agentServicePlanJsonPath": "generated/parity/20-agent-service-plan/agent_service_plan.json",
    "agentServicePlanDocPath": "generated/parity/20-agent-service-plan/agent_service_plan.md",
    "rustDefaultCrate": "monolith-serving",
    "rustCrateRoot": "monolith-rs/crates/monolith-serving"
  },
  "firstSlice": {
    "goal": "...",
    "nonGoals": [ "..." ],
    "rustTargets": [ "monolith-rs/crates/monolith-serving/..." ],
    "pythonReferences": [ "monolith/agent_service/..." ],
    "parityGates": [ "..." ]
  },
  "workItems": [
    {
      "id": "as_impl_001_...",
      "title": "...",
      "priority": "P0|P1|P2",
      "dependsOn": [ "as_impl_..." ],
      "derivedFromPlan": { "workstreams": [ "ws1_..." ], "milestones": [ "m1_..." ] },
      "scope": {
        "createOrModify": [ "monolith-rs/crates/monolith-serving/..." ],
        "touchesProto": true,
        "touchesGrpc": true,
        "touchesZk": true,
        "touchesTfServing": true
      },
      "deliverables": [ "..." ],
      "tests": [ "..." ],
      "parityChecks": [ "..." ],
      "acceptanceCriteria": [ "..." ],
      "risks": [ "..." ],
      "notes": "..."
    }
  ],
  "suggestedExecutionOrder": [ "as_impl_001_...", "as_impl_002_..." ],
  "openQuestions": [ "..." ]
}`}</Instructions>
      </Prompt>
    </Agent>
    <Agent
      id="write-agent-service-impl-work-items-doc"
      needs={["write-agent-service-impl-work-items-json"]}
      produces={["agent_service_impl_work_items_doc"]}
      external_needs={[
        {
          alias: "agentServicePlanJson",
          agent: "write-agent-service-plan-json",
        },
        { alias: "agentServicePlanDoc", agent: "write-agent-service-plan-doc" },
      ]}
    >
      <Prompt>
        <System>
          You write short, implementation-oriented engineering docs. You must
          reference output paths using assets bindings and write files using
          apply_patch.
        </System>
        <Context>
          {ctx.dependency(agentServicePlanJson, {
            as: "Agent service plan JSON (upstream)",
            mode: "code",
          })}
          {ctx.dependency(agentServicePlanDoc, {
            as: "Agent service plan doc (upstream)",
            mode: "quote",
          })}
          {ctx.file(
            "generated/parity/21-agent-service-impl/agent_service_impl_work_items.json",
            { as: "Work items JSON (generated by this module)", mode: "code" },
          )}
        </Context>
        <Instructions>{`Write an implementation handoff doc to \`{{assets.agent_service_impl_work_items_doc.path}}\` using apply_patch.

Requirements:
1) Deterministic: stable ordering; no timestamps.
2) Keep it short and execution-oriented.
3) Structure:
   - Purpose: what the first slice unlocks and why.
   - Inputs: reference the upstream agent_service plan artifacts by their paths (do not paste them).
   - First slice overview: key Rust targets, key Python references, and parity gates.
   - Work item list: one section per workItem (id, title, dependencies, key files, tests, acceptance criteria).
   - Open questions: list the items that may require clarification.
4) Do not embed the full JSON; link to it by path via \`{{assets.agent_service_impl_work_items_json.path}}\`.`}</Instructions>
      </Prompt>
    </Agent>
  </Program>
);
