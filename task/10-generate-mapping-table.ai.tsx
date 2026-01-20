import "./01-validate-inventory.ai.tsx";

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

import { inventoryValidationSummary } from "./01-validate-inventory.ai.tsx";

export const pythonRustMappingTable = assetRef("python_rust_mapping_table");
export const pythonRustMappingJson = assetRef("python_rust_mapping_json");
export const pythonRustMappingGaps = assetRef("python_rust_mapping_gaps");

type MappingSource = "seed_master_plan" | "seed_checklist" | "heuristic" | "unknown";

type RustTarget = {
  crate?: string;
  path?: string;
  raw?: string;
};

type MappingRecord = {
  pythonPath: string;
  pythonLines: number;
  status: string;
  rustTargets: RustTarget[];
  notes: string;
  source: MappingSource;
};

type MappingSummary = {
  inputs: {
    pythonFilesGlob: string;
    masterPlanPath: string;
    parityChecklistGlob: string;
    rustCratesGlob: string;
  };
  counts: {
    pythonFiles: number;
    rustTargetsSeeded: number;
    rustTargetsHeuristic: number;
    unknownMappings: number;
  };
  warnings: string[];
  records: MappingRecord[];
};

function countLines(text: string): number {
  if (text.length === 0) return 0;
  return text.split(/\r?\n/).length;
}

function normalizeStatus(s: string | undefined): string {
  const v = (s ?? "").trim();
  return v.length > 0 ? v : "TODO";
}

function parseRustTargetsFromCell(cell: string): RustTarget[] {
  const raw = cell.trim();
  if (raw.length === 0) return [];

  const parts = raw
    .split(",")
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  const targets: RustTarget[] = [];
  for (const part of parts) {
    const cleaned = part.replace(/`/g, "").trim();
    const m = cleaned.match(/monolith-rs\/crates\/([^/]+)\/(.*)$/);
    if (m) {
      targets.push({ crate: m[1], path: m[2].trim(), raw: cleaned });
      continue;
    }
    const m2 = cleaned.match(/monolith-rs\/crates\/([^/]+)\b/);
    if (m2) {
      targets.push({ crate: m2[1], raw: cleaned });
      continue;
    }
    targets.push({ raw: cleaned });
  }
  return targets;
}

function parseMasterPlanLineInventory(contents: string): Map<string, { status: string; rustCell: string }> {
  const map = new Map<string, { status: string; rustCell: string }>();
  const lines = contents.split(/\r?\n/);

  let inTable = false;
  for (const line of lines) {
    const trimmed = line.trim();

    if (!inTable) {
      if (
        trimmed === "| Python File | Lines | Status | Rust Mapping | Notes |" ||
        trimmed.startsWith("| Python File | Lines | Status | Rust Mapping |")
      ) {
        inTable = true;
      }
      continue;
    }

    if (!trimmed.startsWith("|")) break;
    if (trimmed.includes("---")) continue;

    const cols = trimmed
      .split("|")
      .map((c) => c.trim())
      .filter((c) => c.length > 0);

    if (cols.length < 4) continue;

    const pythonCol = cols[0];
    const statusCol = cols[2];
    const rustCol = cols[3];

    const m = pythonCol.match(/`?(monolith\/[^`]+?\.py)`?/);
    if (!m) continue;

    const pythonPath = m[1];
    map.set(pythonPath, { status: normalizeStatus(statusCol), rustCell: rustCol });
  }

  return map;
}

function chooseHeuristicTargets(params: {
  pythonPath: string;
  rustCoreModules: Set<string>;
  rustLayerModules: Set<string>;
}): { targets: RustTarget[]; status: string; notes: string; source: MappingSource } {
  const { pythonPath, rustCoreModules, rustLayerModules } = params;
  const file = pythonPath.split("/").pop() ?? pythonPath;
  const base = file.replace(/\.py$/i, "");
  const isTest = /_test\.py$/i.test(file) || /\/tests?\//i.test(pythonPath);

  const targets: RustTarget[] = [];
  let notes = "";

  if (pythonPath.startsWith("monolith/agent_service/")) {
    if (isTest) {
      targets.push({ crate: "monolith-serving", path: "tests", raw: "monolith-rs/crates/monolith-serving/tests" });
      return { targets, status: "TODO", notes: "Agent service test; port to Rust integration/unit tests.", source: "heuristic" };
    }

    const cliNames = new Set([
      "agent_client",
      "agent_controller",
      "client",
      "serving_client",
      "svr_client",
      "tfs_client",
      "run",
    ]);
    if (cliNames.has(base)) {
      targets.push({ crate: "monolith-cli", path: "src/bin", raw: "monolith-rs/crates/monolith-cli/src/bin" });
      return { targets, status: "TODO", notes: "Likely a CLI/entrypoint for agent serving.", source: "heuristic" };
    }

    targets.push({ crate: "monolith-serving", path: "src", raw: "monolith-rs/crates/monolith-serving/src" });
    return { targets, status: "TODO", notes: "Agent service runtime/logic; map to serving crate modules.", source: "heuristic" };
  }

  if (pythonPath.startsWith("monolith/core/")) {
    const coreCandidate = `${base}.rs`;
    if (rustLayerModules.has(coreCandidate)) {
      targets.push({ crate: "monolith-layers", path: `src/${coreCandidate}`, raw: `monolith-rs/crates/monolith-layers/src/${coreCandidate}` });
      return { targets, status: "TODO", notes: "Matches an existing Rust layers module by basename.", source: "heuristic" };
    }
    if (rustCoreModules.has(coreCandidate)) {
      targets.push({ crate: "monolith-core", path: `src/${coreCandidate}`, raw: `monolith-rs/crates/monolith-core/src/${coreCandidate}` });
      return { targets, status: "TODO", notes: "Matches an existing Rust core module by basename.", source: "heuristic" };
    }

    if (isTest) {
      targets.push({ crate: "monolith-core", path: "tests", raw: "monolith-rs/crates/monolith-core/tests" });
      return { targets, status: "TODO", notes: "Core test; port to Rust tests.", source: "heuristic" };
    }

    targets.push({ crate: "monolith-core", path: "src", raw: "monolith-rs/crates/monolith-core/src" });
    return { targets, status: "TODO", notes: "Core module; needs finer-grained Rust module mapping.", source: "heuristic" };
  }

  if (pythonPath.startsWith("monolith/native_training/")) {
    if (isTest) {
      targets.push({ crate: "monolith-training", path: "tests", raw: "monolith-rs/crates/monolith-training/tests" });
      return { targets, status: "TODO", notes: "Native training test; port to Rust tests/harnesses.", source: "heuristic" };
    }

    if (/checkpoint|export/i.test(pythonPath)) {
      targets.push({ crate: "monolith-checkpoint", path: "src", raw: "monolith-rs/crates/monolith-checkpoint/src" });
      notes = "Checkpoint/export oriented module.";
      return { targets, status: "TODO", notes, source: "heuristic" };
    }

    if (/data|dataset|tfrecord|reader|input/i.test(pythonPath)) {
      targets.push({ crate: "monolith-data", path: "src", raw: "monolith-rs/crates/monolith-data/src" });
      notes = "Data pipeline oriented module.";
      return { targets, status: "TODO", notes, source: "heuristic" };
    }

    targets.push({ crate: "monolith-training", path: "src", raw: "monolith-rs/crates/monolith-training/src" });
    return { targets, status: "TODO", notes: "Native training module; needs finer-grained domain mapping.", source: "heuristic" };
  }

  if (pythonPath.startsWith("monolith/common/")) {
    targets.push({ crate: "monolith-training", path: "src", raw: "monolith-rs/crates/monolith-training/src" });
    return { targets, status: "TODO", notes: "Common utilities; initial placement in training crate (may be adjusted).", source: "heuristic" };
  }

  if (pythonPath.startsWith("monolith/")) {
    targets.push({ crate: "monolith-core", path: "src", raw: "monolith-rs/crates/monolith-core/src" });
    return { targets, status: "TODO", notes: "Top-level monolith module; likely belongs to core/training; needs confirmation.", source: "heuristic" };
  }

  return { targets: [{ raw: "" }].filter((t) => t.raw && t.raw.length > 0), status: "TODO", notes: "Unknown module placement.", source: "unknown" };
}

export const computeMappingTable = action(async (actx): Promise<MappingSummary> => {
  const pythonFilesGlob = "monolith/**/*.py";
  const masterPlanPath = "monolith-rs/PYTHON_PARITY_MASTER_PLAN.md";
  const parityChecklistGlob = "monolith-rs/parity/**/*.md";
  const rustCratesGlob = "monolith-rs/crates/*/src/**/*.rs";

  const pythonFiles = (await actx.fs.glob(pythonFilesGlob)).slice().sort();
  const rustFiles = (await actx.fs.glob(rustCratesGlob)).slice().sort();

  const rustCoreModules = new Set(
    rustFiles
      .filter((p) => p.startsWith("monolith-rs/crates/monolith-core/src/"))
      .map((p) => p.split("/").pop() ?? p)
  );
  const rustLayerModules = new Set(
    rustFiles
      .filter((p) => p.startsWith("monolith-rs/crates/monolith-layers/src/"))
      .map((p) => p.split("/").pop() ?? p)
  );

  let masterPlanContents: string | undefined;
  const warnings: string[] = [];
  try {
    masterPlanContents = await actx.fs.readFile(masterPlanPath, "utf8");
  } catch {
    warnings.push(`Missing ${masterPlanPath}; mapping will rely on heuristics.`);
  }

  const masterPlanSeed =
    masterPlanContents != null ? parseMasterPlanLineInventory(masterPlanContents) : new Map<string, { status: string; rustCell: string }>();

  const checklistFiles = (await actx.fs.glob(parityChecklistGlob)).slice().sort();
  if (checklistFiles.length === 0) warnings.push(`No parity checklists found under ${parityChecklistGlob}.`);

  const records: MappingRecord[] = [];
  let rustTargetsSeeded = 0;
  let rustTargetsHeuristic = 0;
  let unknownMappings = 0;

  for (const py of pythonFiles) {
    let pythonLines = 0;
    try {
      const content = await actx.fs.readFile(py, "utf8");
      pythonLines = countLines(content);
    } catch {
      warnings.push(`Failed to read ${py} to compute line count.`);
    }

    const seed = masterPlanSeed.get(py);
    if (seed) {
      const rustTargets = parseRustTargetsFromCell(seed.rustCell);
      if (rustTargets.length > 0) rustTargetsSeeded += 1;
      else unknownMappings += 1;
      records.push({
        pythonPath: py,
        pythonLines,
        status: normalizeStatus(seed.status),
        rustTargets,
        notes: "",
        source: "seed_master_plan",
      });
      continue;
    }

    const heuristic = chooseHeuristicTargets({ pythonPath: py, rustCoreModules, rustLayerModules });
    if (heuristic.targets.length > 0) rustTargetsHeuristic += 1;
    else unknownMappings += 1;
    records.push({
      pythonPath: py,
      pythonLines,
      status: normalizeStatus(heuristic.status),
      rustTargets: heuristic.targets,
      notes: heuristic.notes,
      source: heuristic.source,
    });
  }

  records.sort((a, b) => a.pythonPath.localeCompare(b.pythonPath));

  return {
    inputs: { pythonFilesGlob, masterPlanPath, parityChecklistGlob, rustCratesGlob },
    counts: {
      pythonFiles: pythonFiles.length,
      rustTargetsSeeded,
      rustTargetsHeuristic,
      unknownMappings,
    },
    warnings,
    records,
  };
});

export default (
  <Program
    id="generate-mapping-table"
    target={{ language: "md" }}
    description="Generate/refresh a canonical Python->Rust mapping table for all monolith/**/*.py (target crates/modules, status, notes), seeding from existing draft mappings and per-file checklists."
  ><Asset id="python_rust_mapping_table" kind="doc" path="../generated/parity/10-generate-mapping-table/python_rust_mapping_table.md" /><Asset id="python_rust_mapping_json" kind="json" path="../generated/parity/10-generate-mapping-table/python_rust_mapping_table.json" /><Asset id="python_rust_mapping_gaps" kind="doc" path="../generated/parity/10-generate-mapping-table/python_rust_mapping_gaps.md" /><Action id="compute-mapping-table" export="computeMappingTable" cache /><Agent id="write-mapping-json" produces={["python_rust_mapping_json"]} external_needs={[{ alias: "inventoryValidationSummary", agent: "write-inventory-validation-summary" }]}><Prompt><System>
          You maintain a parity planning pipeline. You produce strictly valid JSON and write files using apply_patch.
        </System><Context>{ctx.dependency(inventoryValidationSummary, { as: "Inventory validation summary", mode: "code" })}{ctx.actionResult("compute-mapping-table", { as: "Computed mapping table (canonical JSON)" })}</Context><Instructions>{`Write JSON to \`{{assets.python_rust_mapping_json.path}}\` using apply_patch.
The JSON must be a single object and must exactly match the computed mapping table.`}</Instructions></Prompt></Agent><Agent id="write-mapping-table" needs={["write-mapping-json"]} produces={["python_rust_mapping_table"]} external_needs={[{ alias: "inventoryValidationSummary", agent: "write-inventory-validation-summary" }]}><Prompt><System>
          You maintain a parity planning pipeline. You write deterministic markdown tables and write files using apply_patch.
        </System><Context>{ctx.dependency(inventoryValidationSummary, { as: "Inventory validation summary", mode: "code" })}{ctx.actionResult("compute-mapping-table", { as: "Computed mapping table (JSON)" })}</Context><Instructions>{`Write a canonical mapping table to \`{{assets.python_rust_mapping_table.path}}\` using apply_patch.

Requirements:
1) Deterministic: no timestamps and stable ordering.
2) Cover every python file in the computed mapping table.
3) Include a top summary section with: python file count, seeded vs heuristic counts, unknown mapping count, and any warnings.
4) Include a markdown table with columns:
   - Python File
   - Lines
   - Status
   - Rust Targets (render as comma-separated backticked paths when available; otherwise show raw text)
   - Source
   - Notes

Render records in lexicographic order by python path.`}</Instructions></Prompt></Agent><Agent id="write-mapping-gaps" needs={["write-mapping-table"]} produces={["python_rust_mapping_gaps"]} external_needs={[{ alias: "inventoryValidationSummary", agent: "write-inventory-validation-summary" }]}><Prompt><System>
          You maintain a parity planning pipeline. You write concise, deterministic markdown and write files using apply_patch.
        </System><Context>{ctx.dependency(inventoryValidationSummary, { as: "Inventory validation summary", mode: "code" })}{ctx.actionResult("compute-mapping-table", { as: "Computed mapping table (JSON)" })}</Context><Instructions>{`Write a gaps/cleanup report to \`{{assets.python_rust_mapping_gaps.path}}\` using apply_patch.

Include:
1) A checklist of invariants to satisfy before normalization (no missing python files; every record has at least one crate OR an explicit N/A justification).
2) A list of python files whose rustTargets are empty or look non-pathlike (show up to 100).
3) A list of files where the Rust target uses a crate name that does not exist under monolith-rs/crates/ (show up to 100).
4) A short "next actions" list for normalize-mapping to apply (naming conventions, N/A format, crate/module normalization).

Keep it stable and deterministic (no timestamps).`}</Instructions></Prompt></Agent></Program>
);

