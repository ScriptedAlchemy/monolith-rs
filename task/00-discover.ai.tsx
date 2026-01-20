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

export const discoverReport = assetRef("discover_report");
export const discoverSummary = assetRef("discover_summary");

type DiscoverySummary = {
  repo: {
    workingDir: string;
  };
  inputs: {
    pythonFilesGlob: string;
    parityChecklistGlob: string;
    parityIndexPath: string;
  };
  counts: {
    pythonFiles: number;
    parityChecklistFiles: number;
  };
  existence: {
    parityIndex: boolean;
    parityChecklistDir: boolean;
  };
  parityIndex: {
    head?: string;
    lineCount?: number;
  };
  samples: {
    pythonFile?: string;
    parityChecklistFile?: string;
  };
  notes: string[];
};

export const discoverRepo = action(async (actx): Promise<DiscoverySummary> => {
  const parityIndexPath = "monolith-rs/PYTHON_PARITY_INDEX.md";
  const pythonFilesGlob = "monolith/**/*.py";
  const parityChecklistGlob = "monolith-rs/parity/**/*.md";

  const pythonFiles = (await actx.fs.glob(pythonFilesGlob)).slice().sort();
  const parityChecklistFiles = (await actx.fs.glob(parityChecklistGlob)).slice().sort();

  let parityIndex = true;
  let parityIndexHead: string | undefined;
  let parityIndexLineCount: number | undefined;
  try {
    const content = await actx.fs.readFile(parityIndexPath, "utf8");
    const lines = content.split(/\r?\n/);
    parityIndexLineCount = lines.length;
    parityIndexHead = lines.slice(0, 60).join("\n");
  } catch {
    parityIndex = false;
  }

  const parityChecklistDir = parityChecklistFiles.length > 0;

  const notes: string[] = [];
  if (!parityIndex) notes.push(`Missing ${parityIndexPath}.`);
  if (!parityChecklistDir) notes.push("No parity checklist files found under monolith-rs/parity/.");

  if (pythonFiles.length === 0) {
    notes.push("No Python files matched monolith/**/*.py. Repo layout may differ from expectation.");
  }

  return {
    repo: { workingDir: actx.program.workingDir },
    inputs: { pythonFilesGlob, parityChecklistGlob, parityIndexPath },
    counts: { pythonFiles: pythonFiles.length, parityChecklistFiles: parityChecklistFiles.length },
    existence: { parityIndex, parityChecklistDir },
    parityIndex: { head: parityIndexHead, lineCount: parityIndexLineCount },
    samples: {
      pythonFile: pythonFiles[0],
      parityChecklistFile: parityChecklistFiles[0],
    },
    notes,
  };
});

export default (
  <Program id="discover-artifacts" target={{ language: "md" }} description="Sanity-check repo layout and load existing parity artifacts (PYTHON_PARITY_INDEX.md and monolith-rs/parity/**) as stable inputs for downstream phases."><Asset id="discover_summary" kind="json" path="generated/parity/00-discover/discovery.summary.json" /><Asset id="discover_report" kind="doc" path="generated/parity/00-discover/discovery.report.md" /><Action id="discover-repo" export="discoverRepo" cache /><Agent id="write-discover-summary" produces={["discover_summary"]}><Prompt><System>You are maintaining a parity planning pipeline. You produce strictly valid JSON and write files using apply_patch.</System><Context>{ctx.actionResult("discover-repo", { as: "Discovery summary (computed)" })}</Context><Instructions>
    Write JSON to `{{assets.discover_summary.path}}` using apply_patch.
    The JSON must be a single object and must exactly match the provided discovery summary.
  </Instructions></Prompt></Agent><Agent id="write-discover-report" needs={["write-discover-summary"]} produces={["discover_report"]}><Prompt><System>You are maintaining a parity planning pipeline. You write concise, operational markdown and write files using apply_patch.</System><Context>{ctx.actionResult("discover-repo", { as: "Discovery summary (computed)" })}</Context><Instructions>
    Write a brief report to `{{assets.discover_report.path}}` using apply_patch.
    Include:
    1) Whether the expected artifacts exist.
    2) Python file count vs parity checklist file count.
    3) Any notes/warnings from the discovery summary.
    Keep it stable and deterministic (do not include timestamps).
  </Instructions></Prompt></Agent></Program>
);
