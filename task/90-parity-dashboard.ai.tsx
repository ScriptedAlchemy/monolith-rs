import "./01-validate-inventory.ai.tsx";
import "./10-generate-mapping-table.ai.tsx";
import "./11-normalize-mapping.ai.tsx";
import "./20-agent-service-plan.ai.tsx";
import "./30-core-plan.ai.tsx";
import "./40-native-training-plan.ai.tsx";
import "./50-tf-runtime-plan.ai.tsx";

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
  inventoryValidationSummary,
  inventoryValidationReport,
} from "./01-validate-inventory.ai.tsx";
import {
  pythonRustMappingJson,
  pythonRustMappingGaps,
  pythonRustMappingTable,
} from "./10-generate-mapping-table.ai.tsx";
import {
  mappingConventionsDoc,
  normalizedMappingJson,
} from "./11-normalize-mapping.ai.tsx";
import {
  agentServicePlanDoc,
  agentServicePlanJson,
} from "./20-agent-service-plan.ai.tsx";
import { corePlanDoc, corePlanJson } from "./30-core-plan.ai.tsx";
import {
  nativeTrainingPlanDoc,
  nativeTrainingPlanJson,
} from "./40-native-training-plan.ai.tsx";
import { tfRuntimePlanDoc, tfRuntimePlanJson } from "./50-tf-runtime-plan.ai.tsx";

export const parityDashboardDoc = assetRef("parity_dashboard_doc");
export const parityDashboardJson = assetRef("parity_dashboard_json");

type Severity = "high" | "medium" | "low";

type Domain = "agent_service" | "core" | "native_training" | "tf_runtime";

type ParityWorkItem = {
  id: string;
  priority: "P0" | "P1" | "P2";
  domain: Domain | "global";
  title: string;
  rationale: string;
  dependsOn: string[];
  suggestedOwners: string[];
  artifacts: string[];
};

type ParityDashboard = {
  version: "1";
  generatedFrom: {
    inventorySummaryPath: string;
    mappingJsonPath: string;
    mappingGapsPath: string;
    normalizedMappingPath: string;
    mappingConventionsPath: string;
    domainPlanJsonPaths: Record<Domain, string>;
    domainPlanDocPaths: Record<Domain, string>;
  };
  snapshot: {
    pythonInventory: {
      expected: number;
      found: number;
      invariants: Record<string, boolean>;
      missingChecklists: number;
      extraChecklists: number;
      missingInIndex: number;
      extraInIndex: number;
    };
    mapping: {
      pythonFiles: number;
      crates: number;
      statusCounts: Record<string, number>;
      unknownMappings: number;
      naFiles: number;
      warnings: string[];
    };
    normalized: {
      records: number;
      withIssues: number;
      withCrateTargets: number;
      withNa: number;
      warnings: string[];
      topIssueKinds: Array<{ issue: string; count: number }>;
    };
    domains: Record<
      Domain,
      {
        pythonFiles: number;
        openGaps: Array<{ severity: Severity; gap: string; files: string[] }>;
        milestoneCount: number;
        workstreamCount: number;
      }
    >;
  };
  blockers: string[];
  nextWorkQueue: {
    p0: ParityWorkItem[];
    p1: ParityWorkItem[];
    p2: ParityWorkItem[];
  };
  notes: string[];
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
  return xs.filter((x) => typeof x === "string").slice().sort();
}

function countBy<T>(xs: T[], keyFn: (x: T) => string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const x of xs) {
    const k = keyFn(x);
    out[k] = (out[k] ?? 0) + 1;
  }
  return out;
}

function severityRank(s: string): number {
  if (s === "high") return 0;
  if (s === "medium") return 1;
  if (s === "low") return 2;
  return 3;
}

export const computeParityDashboardInputs = action(
  async (actx): Promise<ParityDashboard> => {
    const inventorySummaryPath =
      "generated/parity/01-validate-inventory/validation.summary.json";
    const mappingJsonPath =
      "generated/parity/10-generate-mapping-table/python_rust_mapping_table.json";
    const mappingGapsPath =
      "generated/parity/10-generate-mapping-table/python_rust_mapping_gaps.md";
    const normalizedMappingPath =
      "generated/parity/11-normalize-mapping/normalized_mapping.json";
    const mappingConventionsPath =
      "generated/parity/11-normalize-mapping/mapping_conventions.md";

    const domainPlanJsonPaths: Record<Domain, string> = {
      agent_service:
        "generated/parity/20-agent-service-plan/agent_service_plan.json",
      core: "generated/parity/30-core-plan/core_plan.json",
      native_training:
        "generated/parity/40-native-training-plan/native_training_plan.json",
      tf_runtime: "generated/parity/50-tf-runtime-plan/tf_runtime_plan.json",
    };

    const domainPlanDocPaths: Record<Domain, string> = {
      agent_service:
        "generated/parity/20-agent-service-plan/agent_service_plan.md",
      core: "generated/parity/30-core-plan/core_plan.md",
      native_training:
        "generated/parity/40-native-training-plan/native_training_plan.md",
      tf_runtime: "generated/parity/50-tf-runtime-plan/tf_runtime_plan.md",
    };

    const notes: string[] = [];
    const blockers: string[] = [];

    const readJsonIfExists = async (p: string) => {
      try {
        const raw = await actx.fs.readFile(p, "utf8");
        const parsed = safeParseJson(raw);
        if (parsed == null) notes.push(`Failed to parse JSON: ${p}`);
        return parsed;
      } catch {
        notes.push(`Missing file: ${p}`);
        return undefined;
      }
    };

    const inventory = await readJsonIfExists(inventorySummaryPath);
    const mapping = await readJsonIfExists(mappingJsonPath);
    const normalized = await readJsonIfExists(normalizedMappingPath);

    const invExpected = Number(inventory?.expected?.pythonFiles ?? 334);
    const invFound = Number(inventory?.counts?.pythonFiles ?? 0);
    const invInvariants = (() => {
      const raw = inventory?.invariants ?? {};
      const out: Record<string, boolean> = {};
      for (const k of Object.keys(raw)) out[k] = Boolean(raw[k]);
      return out;
    })();

    const invMissingChecklists = Number(
      inventory?.coverage?.missingChecklists?.length ?? 0,
    );
    const invExtraChecklists = Number(
      inventory?.coverage?.extraChecklists?.length ?? 0,
    );
    const invMissingInIndex = Number(
      inventory?.coverage?.missingInIndex?.length ?? 0,
    );
    const invExtraInIndex = Number(inventory?.coverage?.extraInIndex?.length ?? 0);

    if (invFound !== invExpected) {
      blockers.push(
        `Python inventory mismatch: expected ${invExpected} files but found ${invFound}.`,
      );
    }
    if (invMissingChecklists > 0) {
      blockers.push(`Missing parity checklists for ${invMissingChecklists} python files.`);
    }
    if (inventory?.parityIndex?.exists === false) {
      blockers.push("Parity index missing: monolith-rs/PYTHON_PARITY_INDEX.md.");
    }

    const mappingRecords = Array.isArray(mapping?.records) ? mapping.records : [];
    const mappingStatusCounts = countBy(mappingRecords, (r: any) =>
      String(r?.status ?? "TODO"),
    );
    const mappingCounts = mapping?.counts ?? {};
    const mappingWarnings = stableStringArray(mapping?.warnings);
    const mappingPythonFiles = Number(mappingCounts?.pythonFiles ?? mappingRecords.length);
    const mappingCrates = Array.isArray(mapping?.crates) ? mapping.crates.length : 0;
    const mappingUnknown = Number(mappingCounts?.unknownMappings ?? 0);
    const mappingNa = Number(mappingCounts?.naFiles ?? 0);

    const normalizedRecords = Array.isArray(normalized?.records)
      ? normalized.records
      : [];
    const normalizedCounts = normalized?.counts ?? {};
    const normalizedWarnings = stableStringArray(normalized?.warnings);
    const normalizedWithIssues = Number(normalizedCounts?.recordsWithIssues ?? 0);
    const normalizedWithCrateTargets = Number(
      normalizedCounts?.recordsWithCrateTargets ?? 0,
    );
    const normalizedWithNa = Number(normalizedCounts?.recordsWithNa ?? 0);

    const normalizedIssueKinds = (() => {
      const issues: string[] = [];
      for (const r of normalizedRecords) {
        const is = Array.isArray(r?.issues) ? r.issues : [];
        for (const i of is) {
          if (typeof i === "string" && i.length > 0) issues.push(i);
        }
      }
      const counted = countBy(issues, (x) => x);
      return Object.keys(counted)
        .sort((a, b) => {
          const da = counted[a] ?? 0;
          const db = counted[b] ?? 0;
          if (db !== da) return db - da;
          return a.localeCompare(b);
        })
        .slice(0, 30)
        .map((issue) => ({ issue, count: counted[issue] ?? 0 }));
    })();

    if (mappingUnknown > 0) {
      blockers.push(
        `Mapping table has ${mappingUnknown} python files with empty/unknown Rust targets; normalization and plans will be noisy until resolved.`,
      );
    }
    if (normalizedWithIssues > 0) {
      blockers.push(
        `Normalized mapping reports ${normalizedWithIssues} records with issues; resolve high-signal issue kinds (unknown_crate, missing_rust_target_or_na, na_missing_justification).`,
      );
    }

    const readDomainPlan = async (domain: Domain) => {
      const plan = await readJsonIfExists(domainPlanJsonPaths[domain]);
      const pythonFiles = Number(
        plan?.inventory?.pythonFiles?.length ?? plan?.inventory?.pythonFilesCount ?? 0,
      );

      const openGapsRaw = Array.isArray(plan?.openGaps) ? plan.openGaps : [];
      const openGaps = openGapsRaw
        .map((g: any) => ({
          severity: (String(g?.severity ?? "low") as Severity) || "low",
          gap: String(g?.gap ?? "").trim() || "unspecified_gap",
          files: stableStringArray(g?.files),
        }))
        .sort((a: any, b: any) => {
          const ra = severityRank(a.severity);
          const rb = severityRank(b.severity);
          if (ra !== rb) return ra - rb;
          return a.gap.localeCompare(b.gap);
        })
        .slice(0, 30);

      const milestoneCount = Array.isArray(plan?.milestones) ? plan.milestones.length : 0;
      const workstreamCount = Array.isArray(plan?.workstreams) ? plan.workstreams.length : 0;

      if (plan == null) blockers.push(`Missing ${domain} plan JSON.`);
      return { pythonFiles, openGaps, milestoneCount, workstreamCount };
    };

    const domains: ParityDashboard["snapshot"]["domains"] = {
      agent_service: await readDomainPlan("agent_service"),
      core: await readDomainPlan("core"),
      native_training: await readDomainPlan("native_training"),
      tf_runtime: await readDomainPlan("tf_runtime"),
    };

    const mkItem = (w: ParityWorkItem) => ({
      id: w.id,
      priority: w.priority,
      domain: w.domain,
      title: w.title,
      rationale: w.rationale,
      dependsOn: w.dependsOn.slice().sort(),
      suggestedOwners: w.suggestedOwners.slice().sort(),
      artifacts: w.artifacts.slice().sort(),
    });

    const workQueue: ParityDashboard["nextWorkQueue"] = {
      p0: [],
      p1: [],
      p2: [],
    };

    const artifactsCommon = [
      inventorySummaryPath,
      mappingJsonPath,
      normalizedMappingPath,
      mappingConventionsPath,
    ];

    const p0: ParityWorkItem[] = [];
    if (!invInvariants.checklistCoversAllPythonFiles && invMissingChecklists > 0) {
      p0.push(
        mkItem({
          id: "p0_checklists_cover_all_files",
          priority: "P0",
          domain: "global",
          title: "Make parity checklist coverage complete",
          rationale:
            "Downstream plans assume every Python file has a checklist entry; missing checklists hide unknown work and break rollups.",
          dependsOn: [],
          suggestedOwners: ["parity-infra"],
          artifacts: [inventorySummaryPath, "monolith-rs/parity/**"],
        }),
      );
    }
    if (mappingUnknown > 0) {
      p0.push(
        mkItem({
          id: "p0_close_unknown_mappings",
          priority: "P0",
          domain: "global",
          title: "Close unknown Rust target mappings (or mark N/A with justification)",
          rationale:
            "Canonical mapping is the dependency for normalized mapping and domain plans; unknowns prevent a correct work queue.",
          dependsOn: ["p0_checklists_cover_all_files"],
          suggestedOwners: ["parity-infra", "domain-owners"],
          artifacts: [mappingJsonPath, "generated/parity/10-generate-mapping-table/python_rust_mapping_table.md"],
        }),
      );
    }
    if (normalizedWithIssues > 0) {
      p0.push(
        mkItem({
          id: "p0_reduce_normalization_issues",
          priority: "P0",
          domain: "global",
          title: "Reduce normalized mapping issues to near-zero for targeted domains",
          rationale:
            "Normalization issues (unknown_crate, missing_rust_target_or_na, na_missing_justification) will cause downstream plan drift and inconsistent target paths.",
          dependsOn: ["p0_close_unknown_mappings"],
          suggestedOwners: ["parity-infra"],
          artifacts: [normalizedMappingPath, mappingConventionsPath],
        }),
      );
    }
    for (const it of p0) workQueue.p0.push(it);

    const p1: ParityWorkItem[] = [];
    p1.push(
      mkItem({
        id: "p1_agent_service_ws1",
        priority: "P1",
        domain: "agent_service",
        title: "Implement agent_service: gRPC contract + fake ZK + fake TFServing harness",
        rationale:
          "Serving parity is the highest-leverage early milestone; the harness unblocks incremental porting and cross-language verification.",
        dependsOn: ["p0_reduce_normalization_issues"],
        suggestedOwners: ["serving"],
        artifacts: [
          domainPlanJsonPaths.agent_service,
          domainPlanDocPaths.agent_service,
          "monolith/agent_service/agent_service.proto",
        ],
      }),
    );
    p1.push(
      mkItem({
        id: "p1_core_hyperparams_and_base_task",
        priority: "P1",
        domain: "core",
        title: "Implement core parity foundations (hyperparams/base_task/hooks registry)",
        rationale:
          "Core infra semantics and config are dependencies for training and serving. Early parity tests prevent hidden behavioral divergence.",
        dependsOn: ["p0_reduce_normalization_issues"],
        suggestedOwners: ["core-infra"],
        artifacts: [domainPlanJsonPaths.core, domainPlanDocPaths.core],
      }),
    );
    p1.push(
      mkItem({
        id: "p1_native_training_data_pipeline_minimal",
        priority: "P1",
        domain: "native_training",
        title: "Implement native_training minimal data pipeline parity (non-TF runtime where possible)",
        rationale:
          "Data pipeline is a large surface area; start with deterministic parsers/transforms and shared fixtures to reduce TF coupling.",
        dependsOn: ["p0_reduce_normalization_issues", "p1_core_hyperparams_and_base_task"],
        suggestedOwners: ["training"],
        artifacts: [domainPlanJsonPaths.native_training, domainPlanDocPaths.native_training],
      }),
    );
    p1.push(
      mkItem({
        id: "p1_tf_runtime_contract",
        priority: "P1",
        domain: "tf_runtime",
        title: "Finalize optional TF runtime contract + feature gates + minimal SavedModel IO",
        rationale:
          "SavedModel compatibility and custom op loading are required for full parity, but must remain optional to keep core crates usable without TF.",
        dependsOn: ["p0_reduce_normalization_issues"],
        suggestedOwners: ["runtime"],
        artifacts: [domainPlanJsonPaths.tf_runtime, domainPlanDocPaths.tf_runtime],
      }),
    );
    for (const it of p1) workQueue.p1.push(it);

    const p2: ParityWorkItem[] = [];
    p2.push(
      mkItem({
        id: "p2_cross_lang_harness",
        priority: "P2",
        domain: "global",
        title: "Stand up a cross-language parity harness runner (Python vs Rust) on shared fixtures",
        rationale:
          "Once harness exists, parity becomes a CI gate and reduces regressions across domains.",
        dependsOn: ["p1_agent_service_ws1", "p1_native_training_data_pipeline_minimal"],
        suggestedOwners: ["parity-infra"],
        artifacts: artifactsCommon,
      }),
    );
    for (const it of p2) workQueue.p2.push(it);

    return {
      version: "1",
      generatedFrom: {
        inventorySummaryPath,
        mappingJsonPath,
        mappingGapsPath,
        normalizedMappingPath,
        mappingConventionsPath,
        domainPlanJsonPaths,
        domainPlanDocPaths,
      },
      snapshot: {
        pythonInventory: {
          expected: invExpected,
          found: invFound,
          invariants: invInvariants,
          missingChecklists: invMissingChecklists,
          extraChecklists: invExtraChecklists,
          missingInIndex: invMissingInIndex,
          extraInIndex: invExtraInIndex,
        },
        mapping: {
          pythonFiles: mappingPythonFiles,
          crates: mappingCrates,
          statusCounts: mappingStatusCounts,
          unknownMappings: mappingUnknown,
          naFiles: mappingNa,
          warnings: mappingWarnings,
        },
        normalized: {
          records: normalizedRecords.length,
          withIssues: normalizedWithIssues,
          withCrateTargets: normalizedWithCrateTargets,
          withNa: normalizedWithNa,
          warnings: normalizedWarnings,
          topIssueKinds: normalizedIssueKinds,
        },
        domains,
      },
      blockers: blockers.slice().sort(),
      nextWorkQueue: {
        p0: workQueue.p0.slice().sort((a, b) => a.id.localeCompare(b.id)),
        p1: workQueue.p1.slice().sort((a, b) => a.id.localeCompare(b.id)),
        p2: workQueue.p2.slice().sort((a, b) => a.id.localeCompare(b.id)),
      },
      notes: notes.slice().sort(),
    };
  },
);

export default (
  <Program id="parity-dashboard" target={{ language: "md" }} description="Roll up inventory validation, mapping, conventions, and domain plans into a single parity dashboard: coverage status, open parity gaps, and prioritized next work queue."><Asset id="parity_dashboard_doc" kind="doc" path="../generated/parity/90-parity-dashboard/parity_dashboard.md" /><Asset id="parity_dashboard_json" kind="json" path="../generated/parity/90-parity-dashboard/parity_dashboard.json" /><Action id="compute-parity-dashboard-inputs" export="computeParityDashboardInputs" cache /><Agent id="write-parity-dashboard-json" produces={["parity_dashboard_json"]} external_needs={[{ alias: "inventoryValidationSummary", agent: "write-inventory-validation-summary" }, { alias: "inventoryValidationReport", agent: "write-inventory-validation-report" }, { alias: "pythonRustMappingJson", agent: "write-mapping-json" }, { alias: "pythonRustMappingTable", agent: "write-mapping-table" }, { alias: "pythonRustMappingGaps", agent: "write-mapping-gaps" }, { alias: "normalizedMappingJson", agent: "write-normalized-mapping-json" }, { alias: "mappingConventionsDoc", agent: "write-mapping-conventions-doc" }, { alias: "agentServicePlanJson", agent: "write-agent-service-plan-json" }, { alias: "agentServicePlanDoc", agent: "write-agent-service-plan-doc" }, { alias: "corePlanJson", agent: "write-core-plan-json" }, { alias: "corePlanDoc", agent: "write-core-plan-doc" }, { alias: "nativeTrainingPlanJson", agent: "write-native-training-plan-json" }, { alias: "nativeTrainingPlanDoc", agent: "write-native-training-plan-doc" }, { alias: "tfRuntimePlanJson", agent: "write-tf-runtime-plan-json" }, { alias: "tfRuntimePlanDoc", agent: "write-tf-runtime-plan-doc" }]}><Prompt><System>
          You maintain a parity planning pipeline. You produce strictly valid JSON and write files using apply_patch.
        </System><Context>{ctx.dependency(inventoryValidationSummary, { as: "Inventory validation summary (JSON)", mode: "code" })}{ctx.dependency(inventoryValidationReport, { as: "Inventory validation report (MD)", mode: "quote" })}{ctx.dependency(pythonRustMappingJson, { as: "Python->Rust mapping (JSON)", mode: "code" })}{ctx.dependency(pythonRustMappingTable, { as: "Python->Rust mapping (MD table)", mode: "quote" })}{ctx.dependency(pythonRustMappingGaps, { as: "Mapping gaps report (MD)", mode: "quote" })}{ctx.dependency(normalizedMappingJson, { as: "Normalized mapping (canonical JSON)", mode: "code" })}{ctx.dependency(mappingConventionsDoc, { as: "Mapping conventions (MD)", mode: "quote" })}{ctx.dependency(agentServicePlanJson, { as: "Agent service plan (JSON)", mode: "code" })}{ctx.dependency(agentServicePlanDoc, { as: "Agent service plan (MD)", mode: "quote" })}{ctx.dependency(corePlanJson, { as: "Core plan (JSON)", mode: "code" })}{ctx.dependency(corePlanDoc, { as: "Core plan (MD)", mode: "quote" })}{ctx.dependency(nativeTrainingPlanJson, { as: "Native training plan (JSON)", mode: "code" })}{ctx.dependency(nativeTrainingPlanDoc, { as: "Native training plan (MD)", mode: "quote" })}{ctx.dependency(tfRuntimePlanJson, { as: "TF runtime plan (JSON)", mode: "code" })}{ctx.dependency(tfRuntimePlanDoc, { as: "TF runtime plan (MD)", mode: "quote" })}{ctx.actionResult("compute-parity-dashboard-inputs", { as: "Computed dashboard payload (JSON)" })}</Context><Instructions>{`Write JSON to \`{{assets.parity_dashboard_json.path}}\` using apply_patch.
The JSON must be a single object and must exactly match the provided computed dashboard payload.`}</Instructions></Prompt></Agent><Agent id="write-parity-dashboard-doc" needs={["write-parity-dashboard-json"]} produces={["parity_dashboard_doc"]}><Prompt><System>
          You write deterministic engineering dashboards. You must reference output paths via assets bindings and write files using apply_patch.
        </System><Context>{ctx.agent("write-parity-dashboard-json", { artifacts: ["parity_dashboard_json"], as: "Dashboard JSON (generated by this module)" })}{ctx.dependency(mappingConventionsDoc, { as: "Mapping conventions (MD)", mode: "quote" })}</Context><Instructions>{`Write a parity dashboard to \`{{assets.parity_dashboard_doc.path}}\` using apply_patch.

Requirements:
1) Deterministic: stable ordering; no timestamps.
2) Keep it executable: focus on what is blocked, what is next, and where artifacts live.
3) Structure:
   - Title + purpose.
   - Snapshot table: inventory, mapping, normalized mapping, and per-domain plan counts.
   - Blockers (from the dashboard JSON) as a short bullet list.
   - "Next Work Queue" section with P0/P1/P2 lists (id, title, rationale, dependencies, and key artifacts).
   - "Domain Status" sections for: agent_service, core, native_training, tf_runtime.
   - "Key Artifacts" section linking (as code paths) to the upstream generated files and this module's JSON.
4) Do not paste raw JSON; summarize it and link to it by path via \`{{assets.parity_dashboard_json.path}}\`.`}</Instructions></Prompt></Agent></Program>
);
