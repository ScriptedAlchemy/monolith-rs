import "./00-discover.ai.tsx";

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

import { discoverSummary } from "./00-discover.ai.tsx";

export const inventoryValidationSummary = assetRef("inventory_validation_summary");
export const inventoryValidationReport = assetRef("inventory_validation_report");

type ChecklistConvention = "replace_py_with_md" | "append_md_to_py" | "unknown";

type InventoryValidationSummary = {
  inputs: {
    pythonFilesGlob: string;
    parityIndexPath: string;
    parityChecklistGlob: string;
    parityChecklistRoot: string;
  };
  counts: {
    pythonFiles: number;
    parityChecklistFiles: number;
    parityIndexRows: number;
    parityIndexPythonRows: number;
  };
  parityIndex: {
    exists: boolean;
    parseNotes: string[];
  };
  checklist: {
    exists: boolean;
    convention: ChecklistConvention;
    expectedExample?: string;
  };
  coverage: {
    missingChecklists: string[];
    extraChecklists: string[];
    missingInIndex: string[];
    extraInIndex: string[];
  };
  invariants: {
    checklistCoversAllPythonFiles: boolean;
    parityIndexCoversAllPythonFiles: boolean;
    indexRowCountMatchesChecklistCount: boolean;
  };
  warnings: string[];
};

function detectChecklistConvention(params: {
  pythonFiles: string[];
  checklistFiles: string[];
  parityChecklistRoot: string;
}): { convention: ChecklistConvention; expectedForPython: (py: string) => string } {
  const { pythonFiles, checklistFiles, parityChecklistRoot } = params;
  const pythonSet = new Set(pythonFiles);

  const expectedAppendMdToPy = (py: string) => `${parityChecklistRoot}/${py}.md`;
  const expectedReplacePyWithMd = (py: string) =>
    `${parityChecklistRoot}/${py.replace(/\.py$/i, ".md")}`;

  if (checklistFiles.length === 0 || pythonFiles.length === 0) {
    return { convention: "unknown", expectedForPython: expectedReplacePyWithMd };
  }

  const countExisting = (expected: (py: string) => string) => {
    const existing = new Set(checklistFiles);
    let hits = 0;
    for (const py of pythonFiles) {
      if (existing.has(expected(py))) hits += 1;
    }
    return hits;
  };

  const appendHits = countExisting(expectedAppendMdToPy);
  const replaceHits = countExisting(expectedReplacePyWithMd);

  if (appendHits === 0 && replaceHits === 0) {
    return { convention: "unknown", expectedForPython: expectedReplacePyWithMd };
  }

  if (appendHits > replaceHits) {
    return { convention: "append_md_to_py", expectedForPython: expectedAppendMdToPy };
  }

  return { convention: "replace_py_with_md", expectedForPython: expectedReplacePyWithMd };
}

function checklistToPythonPath(params: {
  checklistPath: string;
  parityChecklistRoot: string;
  pythonSet: Set<string>;
}): string | undefined {
  const { checklistPath, parityChecklistRoot, pythonSet } = params;

  if (!checklistPath.startsWith(`${parityChecklistRoot}/`)) return undefined;
  const rel = checklistPath.slice(`${parityChecklistRoot}/`.length);

  if (rel.endsWith(".py.md")) {
    const py = rel.slice(0, -3);
    return pythonSet.has(py) ? py : undefined;
  }

  if (!rel.endsWith(".md")) return undefined;
  const base = rel.slice(0, -3);
  if (pythonSet.has(base)) return base;
  if (pythonSet.has(`${base}.py`)) return `${base}.py`;
  return undefined;
}

function parseParityIndexPythonRows(contents: string): {
  pythonPaths: string[];
  notes: string[];
  rowCount: number;
  pythonRowCount: number;
} {
  const lines = contents.split(/\r?\n/);
  const pythonPaths: string[] = [];
  const notes: string[] = [];

  let rowCount = 0;
  let pythonRowCount = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("|")) continue;
    if (trimmed.includes("---")) continue;
    rowCount += 1;

    const cols = trimmed.split("|").map((c) => c.trim());
    const py = cols.find((c) => c.startsWith("monolith/") && c.endsWith(".py"));
    if (py) {
      pythonRowCount += 1;
      pythonPaths.push(py);
    }
  }

  if (rowCount === 0) notes.push("No markdown table rows detected (lines starting with '|').");
  if (pythonRowCount === 0) notes.push("No python paths detected in table rows (monolith/*.py).");

  return {
    pythonPaths: pythonPaths.slice().sort(),
    notes,
    rowCount,
    pythonRowCount,
  };
}

export const validateInventory = action(async (actx): Promise<InventoryValidationSummary> => {
  const pythonFilesGlob = "monolith/**/*.py";
  const parityChecklistRoot = "monolith-rs/parity";
  const parityChecklistGlob = `${parityChecklistRoot}/**/*.md`;
  const parityIndexPath = "monolith-rs/PYTHON_PARITY_INDEX.md";

  const pythonFiles = (await actx.fs.glob(pythonFilesGlob)).slice().sort();
  const checklistFiles = (await actx.fs.glob(parityChecklistGlob)).slice().sort();

  let parityIndexExists = true;
  let parityIndexContents: string | undefined;
  try {
    parityIndexContents = await actx.fs.readFile(parityIndexPath, "utf8");
  } catch {
    parityIndexExists = false;
  }

  const parityIndexParseNotes: string[] = [];
  const parityIndexPythonPaths: string[] = [];
  let parityIndexRows = 0;
  let parityIndexPythonRows = 0;
  if (parityIndexExists && parityIndexContents != null) {
    const parsed = parseParityIndexPythonRows(parityIndexContents);
    parityIndexParseNotes.push(...parsed.notes);
    parityIndexPythonPaths.push(...parsed.pythonPaths);
    parityIndexRows = parsed.rowCount;
    parityIndexPythonRows = parsed.pythonRowCount;
  }

  const { convention, expectedForPython } = detectChecklistConvention({
    pythonFiles,
    checklistFiles,
    parityChecklistRoot,
  });

  const pythonSet = new Set(pythonFiles);
  const checklistPythonPaths = new Set<string>();
  const unmatchedChecklists: string[] = [];

  for (const c of checklistFiles) {
    const py = checklistToPythonPath({
      checklistPath: c,
      parityChecklistRoot,
      pythonSet,
    });
    if (py) checklistPythonPaths.add(py);
    else unmatchedChecklists.push(c);
  }

  const missingChecklists: string[] = [];
  const existingChecklists = new Set(checklistFiles);
  for (const py of pythonFiles) {
    const expected = expectedForPython(py);
    if (!existingChecklists.has(expected)) missingChecklists.push(py);
  }

  const parityIndexSet = new Set(parityIndexPythonPaths);
  const missingInIndex = pythonFiles.filter((py) => !parityIndexSet.has(py));
  const extraInIndex = parityIndexPythonPaths.filter((py) => !pythonSet.has(py));

  const extraChecklists = unmatchedChecklists.slice().sort();

  const warnings: string[] = [];
  if (pythonFiles.length === 0) warnings.push("No python files matched monolith/**/*.py.");
  if (!parityIndexExists) warnings.push(`Missing ${parityIndexPath}.`);
  if (checklistFiles.length === 0) warnings.push(`No parity checklists matched ${parityChecklistGlob}.`);
  if (convention === "unknown")
    warnings.push("Could not infer parity checklist naming convention from existing files; using default expectations.");
  if (missingChecklists.length > 0)
    warnings.push(`Missing parity checklists for ${missingChecklists.length} python files.`);
  if (extraChecklists.length > 0)
    warnings.push(`Found ${extraChecklists.length} parity checklist files that do not map to a python file.`);
  if (parityIndexExists && missingInIndex.length > 0)
    warnings.push(`Parity index is missing ${missingInIndex.length} python files.`);
  if (parityIndexExists && extraInIndex.length > 0)
    warnings.push(`Parity index contains ${extraInIndex.length} python paths not present on disk.`);

  const indexRowCountMatchesChecklistCount =
    parityIndexExists && parityIndexPythonRows > 0
      ? parityIndexPythonRows === checklistFiles.length
      : false;

  const checklistCoversAllPythonFiles = missingChecklists.length === 0 && pythonFiles.length > 0;
  const parityIndexCoversAllPythonFiles =
    parityIndexExists && pythonFiles.length > 0 ? missingInIndex.length === 0 : false;

  const expectedExample = pythonFiles.length > 0 ? expectedForPython(pythonFiles[0]) : undefined;

  return {
    inputs: {
      pythonFilesGlob,
      parityIndexPath,
      parityChecklistGlob,
      parityChecklistRoot,
    },
    counts: {
      pythonFiles: pythonFiles.length,
      parityChecklistFiles: checklistFiles.length,
      parityIndexRows,
      parityIndexPythonRows,
    },
    parityIndex: {
      exists: parityIndexExists,
      parseNotes: parityIndexParseNotes,
    },
    checklist: {
      exists: checklistFiles.length > 0,
      convention,
      expectedExample,
    },
    coverage: {
      missingChecklists,
      extraChecklists,
      missingInIndex,
      extraInIndex,
    },
    invariants: {
      checklistCoversAllPythonFiles,
      parityIndexCoversAllPythonFiles,
      indexRowCountMatchesChecklistCount,
    },
    warnings,
  };
});

export default (
  <Program
    id="validate-inventory"
    target={{ language: "md" }}
    description="Validate Python file inventory vs parity checklist coverage (missing/extra), and validate index row count matches checklist count; emit report + machine-readable JSON summary."
  ><Asset id="inventory_validation_summary" kind="json" path="../generated/parity/01-validate-inventory/validation.summary.json" /><Asset id="inventory_validation_report" kind="doc" path="../generated/parity/01-validate-inventory/validation.report.md" /><Action id="validate-inventory" export="validateInventory" cache /><Agent id="write-inventory-validation-summary" produces={["inventory_validation_summary"]} external_needs={[{ alias: "discoverSummary", agent: "write-discover-summary" }]}><Prompt><System>
          You are maintaining a parity planning pipeline. You produce strictly valid JSON and write files using apply_patch.
        </System><Context>{ctx.dependency(discoverSummary, { as: "Discovery summary", mode: "code" })}{ctx.actionResult("validate-inventory", { as: "Inventory validation (computed)" })}</Context><Instructions>{`Write JSON to \`{{assets.inventory_validation_summary.path}}\` using apply_patch.
The JSON must be a single object and must exactly match the provided inventory validation summary.`}</Instructions></Prompt></Agent><Agent id="write-inventory-validation-report" needs={["write-inventory-validation-summary"]} produces={["inventory_validation_report"]} external_needs={[{ alias: "discoverSummary", agent: "write-discover-summary" }]}><Prompt><System>
          You are maintaining a parity planning pipeline. You write concise, operational markdown and write files using apply_patch.
        </System><Context>{ctx.dependency(discoverSummary, { as: "Discovery summary", mode: "code" })}{ctx.actionResult("validate-inventory", { as: "Inventory validation (computed)" })}</Context><Instructions>{`Write a deterministic report to \`{{assets.inventory_validation_report.path}}\` using apply_patch.
Include:
1) The computed counts and invariant booleans.
2) Missing parity checklists (show first 50 paths and total).
3) Extra parity checklists (show first 50 paths and total).
4) Missing/extra python paths in the parity index (show first 50 paths and total) when the index exists.
5) The inferred checklist naming convention and an example expected checklist path.
Keep it stable and deterministic (no timestamps).`}</Instructions></Prompt></Agent></Program>
);
