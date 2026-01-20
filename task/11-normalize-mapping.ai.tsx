import "./10-generate-mapping-table.ai.tsx";

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
  pythonRustMappingGaps,
  pythonRustMappingJson,
} from "./10-generate-mapping-table.ai.tsx";

export const normalizedMappingJson = assetRef("normalized_mapping_json");
export const mappingConventionsDoc = assetRef("mapping_conventions_doc");

type NormalizedRustTargetKind = "crate_path" | "crate_root" | "na" | "unknown";

type NormalizedRustTarget = {
  kind: NormalizedRustTargetKind;
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

type NormalizationConventions = {
  statusVocabulary: string[];
  rustTargetFormat: {
    canonical: string;
    examples: string[];
  };
  testMapping: {
    rule: string;
    examples: string[];
  };
  fixtureMapping: {
    rule: string;
    suggestedRoots: string[];
  };
  protoOpBoundary: {
    rule: string;
    examples: string[];
  };
  naPolicy: {
    rule: string;
    examples: string[];
  };
};

type NormalizedMapping = {
  version: "1";
  inputs: {
    upstreamMappingJsonPath: string;
    upstreamGapsReportPath: string;
    cratesGlob: string;
  };
  crates: string[];
  conventions: NormalizationConventions;
  counts: {
    pythonFiles: number;
    recordsWithCrateTargets: number;
    recordsWithEmptyTargets: number;
    recordsWithNa: number;
    recordsWithIssues: number;
  };
  warnings: string[];
  records: NormalizedRecord[];
};

type UpstreamMappingRecord = {
  pythonPath: string;
  pythonLines: number;
  status: string;
  rustTargets: Array<{ crate?: string; path?: string; raw?: string }>;
  notes: string;
  source: string;
};

type UpstreamMappingSummary = {
  records: UpstreamMappingRecord[];
};

function normalizePathLike(p: string): string {
  return p.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\.\//, "").trim();
}

function isPythonTestPath(pythonPath: string): boolean {
  const file = pythonPath.split("/").pop() ?? pythonPath;
  return /_test\.py$/i.test(file) || /\/tests?\//i.test(pythonPath);
}

function normalizeStatus(s: string | undefined): string {
  const v = (s ?? "").trim();
  if (v.length === 0) return "TODO";
  if (/^(n\/a|na)$/i.test(v)) return "N/A";
  const upper = v.toUpperCase().replace(/\s+/g, "_").replace(/-+/g, "_");
  if (upper === "INPROGRESS") return "IN_PROGRESS";
  if (upper === "IN_PROGRESS") return "IN_PROGRESS";
  return upper;
}

function normalizeRustTargets(params: {
  pythonPath: string;
  rustTargets: Array<{ crate?: string; path?: string; raw?: string }>;
  crates: Set<string>;
  notes: string;
  status: string;
}): {
  targets: NormalizedRustTarget[];
  issues: string[];
  naJustification?: string;
  suggestedRustTestLocation?: string;
} {
  const { pythonPath, rustTargets, crates, notes, status } = params;
  const issues: string[] = [];
  const targets: NormalizedRustTarget[] = [];

  const isTest = isPythonTestPath(pythonPath);

  const seen = new Set<string>();
  const pushUnique = (t: NormalizedRustTarget) => {
    const key =
      t.fullPath ?? `${t.kind}:${t.raw ?? ""}:${t.crate ?? ""}:${t.path ?? ""}`;
    if (!seen.has(key)) {
      seen.add(key);
      targets.push(t);
    }
  };

  let naJustification: string | undefined;
  const noteTrim = (notes ?? "").trim();
  const normalizedStatus = normalizeStatus(status);
  const allowedStatuses = new Set([
    "TODO",
    "IN_PROGRESS",
    "DONE",
    "N/A",
    "BLOCKED",
  ]);
  if (!allowedStatuses.has(normalizedStatus))
    issues.push(`unknown_status:${normalizedStatus}`);

  const noteHasNaPrefix = /^N\/A:/i.test(noteTrim);
  const statusIsNa = normalizedStatus === "N/A";
  if (statusIsNa || noteHasNaPrefix) {
    if (statusIsNa && !noteHasNaPrefix) issues.push("na_missing_justification");
    naJustification = noteHasNaPrefix ? noteTrim : `N/A: ${noteTrim}`;
    if (naJustification === "N/A:")
      naJustification = "N/A: Unspecified justification.";
    pushUnique({ kind: "na", raw: naJustification });
  }

  for (const rt of rustTargets ?? []) {
    const raw = (rt.raw ?? "").trim();
    const crate = (rt.crate ?? "").trim();
    const path = normalizePathLike(rt.path ?? "");

    if (raw.length === 0 && crate.length === 0 && path.length === 0) continue;

    if (/^N\/A:/i.test(raw)) {
      naJustification = raw;
      pushUnique({ kind: "na", raw });
      continue;
    }

    if (crate.length > 0) {
      if (!crates.has(crate)) issues.push(`unknown_crate:${crate}`);

      const trimmedPath = path.replace(/^\/+/, "");
      if (trimmedPath.length > 0) {
        if (!/^(src|tests|examples|benches)\b/.test(trimmedPath))
          issues.push("noncanonical_rust_target_prefix");
        const fullPath = `monolith-rs/crates/${crate}/${trimmedPath}`;
        pushUnique({
          kind: "crate_path",
          crate,
          path: trimmedPath,
          fullPath,
          raw: raw.length > 0 ? raw : undefined,
        });
      } else {
        const fullPath = `monolith-rs/crates/${crate}`;
        pushUnique({
          kind: "crate_root",
          crate,
          fullPath,
          raw: raw.length > 0 ? raw : undefined,
        });
      }
      continue;
    }

    if (raw.length > 0) {
      const cleaned = normalizePathLike(raw.replace(/`/g, ""));
      const m = cleaned.match(/^monolith-rs\/crates\/([^/]+)\/(.*)$/);
      if (m) {
        const c = m[1].trim();
        const p = normalizePathLike(m[2].trim());
        if (!crates.has(c)) issues.push(`unknown_crate:${c}`);
        if (p.length > 0 && !/^(src|tests|examples|benches)\b/.test(p))
          issues.push("noncanonical_rust_target_prefix");
        pushUnique({
          kind: p.length > 0 ? "crate_path" : "crate_root",
          crate: c,
          path: p.length > 0 ? p : undefined,
          fullPath: cleaned,
          raw,
        });
        continue;
      }
      pushUnique({ kind: "unknown", raw });
    }
  }

  const crateForTests = (() => {
    for (const t of targets) {
      if (t.crate && crates.has(t.crate)) return t.crate;
    }
    return undefined;
  })();

  let suggestedRustTestLocation: string | undefined;
  if (isTest) {
    const c = crateForTests ?? "monolith-core";
    suggestedRustTestLocation = `monolith-rs/crates/${c}/tests/parity/${pythonPath.replace(/^monolith\//, "").replace(/\.py$/i, "")}.rs`;
  }

  const hasCrateTargets = targets.some(
    (t) => t.kind === "crate_path" || t.kind === "crate_root",
  );
  if (statusIsNa && hasCrateTargets) issues.push("na_has_rust_targets");

  if (targets.length === 0 && naJustification == null) {
    issues.push("missing_rust_target_or_na");
  }

  if (isTest) {
    const anyTestsTarget = targets.some(
      (t) =>
        t.fullPath?.includes("/tests") ||
        t.path?.startsWith("tests/") ||
        t.path === "tests",
    );
    if (!anyTestsTarget && naJustification == null)
      issues.push("python_test_should_map_to_rust_tests");
  }

  return { targets, issues, naJustification, suggestedRustTestLocation };
}

export const computeNormalizedMapping = action(
  async (actx): Promise<NormalizedMapping> => {
    const upstreamMappingJsonPath =
      "generated/parity/10-generate-mapping-table/python_rust_mapping_table.json";
    const upstreamGapsReportPath =
      "generated/parity/10-generate-mapping-table/python_rust_mapping_gaps.md";
    const cratesGlob = "monolith-rs/crates/*/Cargo.toml";

    const warnings: string[] = [];

    let upstreamRaw: string;
    try {
      upstreamRaw = await actx.fs.readFile(upstreamMappingJsonPath, "utf8");
    } catch {
      return {
        version: "1",
        inputs: { upstreamMappingJsonPath, upstreamGapsReportPath, cratesGlob },
        crates: [],
        conventions: {
          statusVocabulary: ["TODO", "IN_PROGRESS", "DONE", "N/A", "BLOCKED"],
          rustTargetFormat: {
            canonical:
              "monolith-rs/crates/<crate>/(src|tests|examples|benches)/<path>",
            examples: [
              "monolith-rs/crates/monolith-core/src/lib.rs",
              "monolith-rs/crates/monolith-serving/tests/parity/agent_service/foo.rs",
            ],
          },
          testMapping: {
            rule: "Python tests map to Rust tests (prefer crates/<crate>/tests/parity/...).",
            examples: [
              "monolith/tests/foo_test.py -> monolith-rs/crates/monolith-core/tests/parity/tests/foo_test.rs",
            ],
          },
          fixtureMapping: {
            rule: "Golden fixtures used for parity live under a stable fixtures root and are shared by Python and Rust harnesses.",
            suggestedRoots: [
              "monolith-rs/fixtures/parity",
              "monolith-rs/testdata/parity",
            ],
          },
          protoOpBoundary: {
            rule: "Protos belong in monolith-proto; TensorFlow custom ops belong in dedicated op/runtime crates. Keep protobuf schema generation and TF op loading separated from core logic.",
            examples: [
              "monolith/**/*.proto -> monolith-rs/crates/monolith-proto",
              "monolith/native_training/**/ops -> monolith-rs/crates/monolith-tensor-ops (proposed)",
            ],
          },
          naPolicy: {
            rule: "When a Python file does not need a Rust port, set status to N/A and include a notes field starting with 'N/A: ...' with a concrete justification.",
            examples: ["N/A: Pure re-export __init__.py; no runtime behavior."],
          },
        },
        counts: {
          pythonFiles: 0,
          recordsWithCrateTargets: 0,
          recordsWithEmptyTargets: 0,
          recordsWithNa: 0,
          recordsWithIssues: 0,
        },
        warnings: [
          `Missing ${upstreamMappingJsonPath}; run 10-generate-mapping-table first.`,
        ],
        records: [],
      };
    }

    let upstream: UpstreamMappingSummary;
    try {
      upstream = JSON.parse(upstreamRaw) as UpstreamMappingSummary;
    } catch {
      warnings.push(
        `Failed to parse ${upstreamMappingJsonPath} as JSON; normalization will be empty.`,
      );
      upstream = { records: [] };
    }

    const crateTomls = ((await actx.fs.glob(cratesGlob)) as string[])
      .slice()
      .sort();
    const crates = crateTomls
      .map((p) => {
        const m = p.match(/^monolith-rs\/crates\/([^/]+)\/Cargo\.toml$/);
        return m ? m[1] : undefined;
      })
      .filter((c): c is string => typeof c === "string")
      .sort();
    const crateSet = new Set(crates);
    if (crates.length === 0)
      warnings.push(`No crates discovered via ${cratesGlob}.`);

    const conventions: NormalizationConventions = {
      statusVocabulary: ["TODO", "IN_PROGRESS", "DONE", "N/A", "BLOCKED"],
      rustTargetFormat: {
        canonical:
          "monolith-rs/crates/<crate>/(src|tests|examples|benches)/<path>",
        examples: [
          "monolith-rs/crates/monolith-core/src/lib.rs",
          "monolith-rs/crates/monolith-serving/src/grpc/mod.rs",
          "monolith-rs/crates/monolith-core/tests/parity/core/hyperparams.rs",
        ],
      },
      testMapping: {
        rule: "Python test modules (path contains /test(s)/ or filename *_test.py) map to Rust tests; prefer crates/<crate>/tests/parity/<python_rel>.rs.",
        examples: [
          "monolith/core/tests/foo_test.py -> monolith-rs/crates/monolith-core/tests/parity/core/tests/foo_test.rs",
        ],
      },
      fixtureMapping: {
        rule: "Parity fixtures should be shared inputs for both Python and Rust tests and live outside src/; prefer monolith-rs/fixtures/parity/<domain>/... with stable filenames.",
        suggestedRoots: [
          "monolith-rs/fixtures/parity",
          "monolith-rs/testdata/parity",
        ],
      },
      protoOpBoundary: {
        rule: "Keep protobuf schemas and TF op/runtime integration in dedicated crates. Map .proto schemas to monolith-proto; map TF custom op wrappers/loaders to op/runtime crates (not monolith-core).",
        examples: [
          "monolith/**/*.proto -> monolith-rs/crates/monolith-proto",
          "monolith/native_training/**/ops/* -> monolith-rs/crates/monolith-tensor-ops (proposed)",
        ],
      },
      naPolicy: {
        rule: "Use status N/A only when a Python file has no runtime behavior or is intentionally unported; notes must start with 'N/A: ...' and include a concrete justification.",
        examples: [
          "N/A: Documentation-only module; no runtime behavior used by production paths.",
        ],
      },
    };

    const records: NormalizedRecord[] = [];
    let recordsWithCrateTargets = 0;
    let recordsWithEmptyTargets = 0;
    let recordsWithNa = 0;
    let recordsWithIssues = 0;

    for (const r of upstream.records ?? []) {
      const status = normalizeStatus(r.status);
      const notes = (r.notes ?? "").trim();

      const { targets, issues, naJustification, suggestedRustTestLocation } =
        normalizeRustTargets({
          pythonPath: r.pythonPath,
          rustTargets: r.rustTargets ?? [],
          crates: crateSet,
          notes,
          status,
        });

      if (targets.length === 0) recordsWithEmptyTargets += 1;
      if (
        targets.some((t) => t.kind === "crate_path" || t.kind === "crate_root")
      )
        recordsWithCrateTargets += 1;
      if (
        naJustification != null ||
        status === "N/A" ||
        targets.some((t) => t.kind === "na")
      )
        recordsWithNa += 1;
      if (issues.length > 0) recordsWithIssues += 1;

      records.push({
        pythonPath: r.pythonPath,
        pythonLines: Number.isFinite(r.pythonLines) ? r.pythonLines : 0,
        status,
        rustTargets: targets,
        notes,
        source: (r.source ?? "unknown").trim(),
        issues,
        naJustification,
        suggestedRustTestLocation,
      });
    }

    records.sort((a, b) => a.pythonPath.localeCompare(b.pythonPath));

    return {
      version: "1",
      inputs: { upstreamMappingJsonPath, upstreamGapsReportPath, cratesGlob },
      crates,
      conventions,
      counts: {
        pythonFiles: records.length,
        recordsWithCrateTargets,
        recordsWithEmptyTargets,
        recordsWithNa,
        recordsWithIssues,
      },
      warnings,
      records,
    };
  },
);

export default (
  <Program
    id="normalize-mapping"
    target={{ language: "md" }}
    description="Normalize mapping conventions (crate/module naming, test/fixture placement, N/A justification format) and emit a conventions doc + JSON for downstream planners."
  >
    <Asset
      id="mapping_conventions_doc"
      kind="doc"
      path="../generated/parity/11-normalize-mapping/mapping_conventions.md"
    />
    <Asset
      id="normalized_mapping_json"
      kind="json"
      path="../generated/parity/11-normalize-mapping/normalized_mapping.json"
    />
    <Action
      id="compute-normalized-mapping"
      export="computeNormalizedMapping"
      cache
    />
    <Agent
      id="write-normalized-mapping-json"
      produces={["normalized_mapping_json"]}
      external_needs={[
        { alias: "pythonRustMappingJson", agent: "write-mapping-json" },
        { alias: "pythonRustMappingGaps", agent: "write-mapping-gaps" },
      ]}
    >
      <Prompt>
        <System>
          You maintain a parity planning pipeline. You produce strictly valid
          JSON and write files using apply_patch.
        </System>
        <Context>
          {ctx.dependency(pythonRustMappingJson, {
            as: "Upstream mapping JSON (10-generate-mapping-table)",
            mode: "code",
          })}
          {ctx.dependency(pythonRustMappingGaps, {
            as: "Upstream mapping gaps report (10-generate-mapping-table)",
            mode: "quote",
          })}
          {ctx.actionResult("compute-normalized-mapping", {
            as: "Computed normalized mapping (canonical JSON)",
          })}
        </Context>
        <Instructions>{`Write JSON to \`{{assets.normalized_mapping_json.path}}\` using apply_patch.
The JSON must be a single object and must exactly match the computed normalized mapping.`}</Instructions>
      </Prompt>
    </Agent>
    <Agent
      id="write-mapping-conventions-doc"
      needs={["write-normalized-mapping-json"]}
      produces={["mapping_conventions_doc"]}
      external_needs={[
        { alias: "pythonRustMappingJson", agent: "write-mapping-json" },
      ]}
    >
      <Prompt>
        <System>
          You maintain a parity planning pipeline. You write deterministic
          operational docs and write files using apply_patch.
        </System>
        <Context>
          {ctx.dependency(pythonRustMappingJson, {
            as: "Upstream mapping JSON (10-generate-mapping-table)",
            mode: "code",
          })}
          {ctx.actionResult("compute-normalized-mapping", {
            as: "Computed normalized mapping (JSON)",
          })}
        </Context>
        <Instructions>{`Write a conventions document to \`{{assets.mapping_conventions_doc.path}}\` using apply_patch.

Requirements:
1) Deterministic: stable ordering and no timestamps.
2) Start with a short purpose statement explaining how normalization is used downstream.
3) Document:
   - Status vocabulary and meaning (include N/A policy).
   - Canonical Rust target format and examples.
   - Test mapping conventions (Python test modules -> Rust tests).
   - Fixture placement conventions for golden/parity fixtures.
   - Proto/op boundary conventions (where protos live vs TF runtime/op integration).
4) Include a "Normalization Checks" section listing the specific issues the normalizer emits (e.g. unknown_crate, missing_rust_target_or_na, python_test_should_map_to_rust_tests).
5) Include a short "Next normalization improvements" list, prioritized, without adding new requirements beyond what's already computed.`}</Instructions>
      </Prompt>
    </Agent>
  </Program>
);
