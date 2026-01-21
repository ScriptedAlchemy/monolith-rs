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

export const discoverSummary = assetRef("discover_summary");

type DiscoverySummary = {
  repo: {
    workingDir: string;
    expectedRoots: {
      monolith: boolean;
      monolithRs: boolean;
    };
  };
  inputs: {
    pythonFilesGlob: string;
    parityChecklistGlob: string;
    parityIndexPath: string;
    monolithInitPath: string;
    monolithRsCargoTomlPath: string;
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
  const monolithInitPath = "monolith/__init__.py";
  const monolithRsCargoTomlPath = "monolith-rs/Cargo.toml";
  const parityIndexPath = "monolith-rs/PYTHON_PARITY_INDEX.md";
  const pythonFilesGlob = "monolith/**/*.py";
  const parityChecklistGlob = "monolith-rs/parity/**/*.md";

  const pythonFiles = (await actx.fs.glob(pythonFilesGlob)).slice().sort();
  const parityChecklistFiles = (await actx.fs.glob(parityChecklistGlob)).slice().sort();

  const monolith = (await actx.fs.glob(monolithInitPath)).length > 0;
  const monolithRs = (await actx.fs.glob(monolithRsCargoTomlPath)).length > 0;

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
  if (!monolith) notes.push(`Missing expected root/sentinel: ${monolithInitPath}.`);
  if (!monolithRs) notes.push(`Missing expected root/sentinel: ${monolithRsCargoTomlPath}.`);
  if (!parityIndex) notes.push(`Missing ${parityIndexPath}.`);
  if (!parityChecklistDir) notes.push("No parity checklist files found under monolith-rs/parity/.");

  if (pythonFiles.length === 0) {
    notes.push("No Python files matched monolith/**/*.py. Repo layout may differ from expectation.");
  }

  return {
    repo: { workingDir: actx.program.workingDir, expectedRoots: { monolith, monolithRs } },
    inputs: {
      pythonFilesGlob,
      parityChecklistGlob,
      parityIndexPath,
      monolithInitPath,
      monolithRsCargoTomlPath,
    },
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
  <Program
    id="discover-summary"
    target={{ language: "md" }}
    description="Sanity-check required repo roots (monolith/, monolith-rs/) and load existing parity artifacts (index + per-file checklist tree) into a bounded discovery summary JSON."
  >
    <Asset
      id="discover_summary"
      kind="json"
      path="../generated/parity/00-discover/discovery.summary.json"
    />
    <Action id="discover-repo" export="discoverRepo" cache />
    <Agent id="write-discover-summary" produces={["discover_summary"]}>
      <Prompt>
        <System>
          You maintain a parity planning pipeline. You produce strictly valid JSON and write files using apply_patch.
        </System>
        <Context>{ctx.actionResult("discover-repo", { as: "Discovery summary (computed)" })}</Context>
        <Instructions>{`Write JSON to \`{{assets.discover_summary.path}}\` using apply_patch.
The JSON must be a single object and must exactly match the provided discovery summary.`}</Instructions>
      </Prompt>
    </Agent>
  </Program>
);

