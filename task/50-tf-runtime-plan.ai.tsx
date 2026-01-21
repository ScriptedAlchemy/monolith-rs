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

export const tfRuntimePlanDoc = assetRef("tf_runtime_plan_doc");
export const tfRuntimePlanJson = assetRef("tf_runtime_plan_json");

type TfRuntimeSearchResult = {
  path: string;
  signals: string[];
};

type TfRuntimePlanInputs = {
  inputs: {
    pythonGlob: string;
    tfRuntimeOpsGlob: string;
    normalizedMappingPath: string;
  };
  warnings: string[];
  python: {
    totalFiles: number;
    tensorflowImports: number;
    savedModelUsage: number;
    customOpLoads: number;
    candidates: TfRuntimeSearchResult[];
  };
  runtimeOps: {
    files: string[];
    sampleFiles: string[];
  };
  normalized: {
    available: boolean;
    tfDependentPythonFiles: Array<{
      pythonPath: string;
      status: string;
      rustTargets: string[];
      notes: string;
      issues: string[];
      naJustification?: string;
      suggestedRustTestLocation?: string;
    }>;
  };
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
  status: string;
  rustTargets: NormalizedRustTarget[];
  notes: string;
  issues: string[];
  naJustification?: string;
  suggestedRustTestLocation?: string;
};

type NormalizedMapping = {
  records: NormalizedRecord[];
};

type SignalPattern = { id: string; re: RegExp };

function extractSignals(text: string, patterns: SignalPattern[]): string[] {
  const out: string[] = [];
  for (const p of patterns) {
    if (p.re.test(text)) out.push(p.id);
  }
  return out.sort();
}

export const computeTfRuntimePlanInputs = action(
  async (actx): Promise<TfRuntimePlanInputs> => {
    const pythonGlob = "monolith/**/*.py";
    const tfRuntimeOpsGlob = "monolith/native_training/runtime/ops/**";
    const normalizedMappingPath =
      "generated/parity/11-normalize-mapping/normalized_mapping.json";

    const warnings: string[] = [];

    const pythonPaths = ((await actx.fs.glob(pythonGlob)) as string[])
      .filter((p) => !p.endsWith("/"))
      .slice()
      .sort();

    const tfPatterns: SignalPattern[] = [
      { id: "import_tensorflow_as_tf", re: /\bimport\s+tensorflow\s+as\s+tf\b/ },
      { id: "from_tensorflow_import", re: /\bfrom\s+tensorflow\b/ },
      { id: "tf_saved_model_api", re: /\btf\.saved_model\b/ },
      { id: "tf_compat_v1_saved_model", re: /\btf\.compat\.v1\.saved_model\b/ },
      { id: "saved_model_token", re: /\bsaved_model\b/ },
      { id: "parse_saved_model", re: /\bparse_saved_model\b/ },
      { id: "meta_graph_pb2", re: /\bmeta_graph_pb2\b/ },
      { id: "signature_def", re: /\bsignature_def\b/ },
      { id: "tf_session", re: /\btf\.(compat\.v1\.)?Session\b/ },
      { id: "session_run", re: /\bsess\.run\b/ },
      { id: "tf_load_library", re: /\btf\.load_library\b/ },
      { id: "load_library_token", re: /\bload_library\b/ },
      { id: "get_libops_path", re: /\bget_libops_path\b/ },
      { id: "tf_serving_token", re: /\bTFServing\b/i },
    ];

    const candidates: TfRuntimeSearchResult[] = [];
    let tensorflowImports = 0;
    let savedModelUsage = 0;
    let customOpLoads = 0;

    for (const p of pythonPaths) {
      let content = "";
      try {
        content = await actx.fs.readFile(p, "utf8");
      } catch {
        warnings.push(`Failed to read ${p}; skipping TF pattern scan.`);
        continue;
      }

      const signals = extractSignals(content, tfPatterns);
      if (signals.length > 0) {
        candidates.push({ path: p, signals });
      }

      if (
        /\bimport\s+tensorflow\s+as\s+tf\b/.test(content) ||
        /\bfrom\s+tensorflow\b/.test(content)
      )
        tensorflowImports += 1;
      if (
        /\btf\.saved_model\b/.test(content) ||
        /\bparse_saved_model\b/.test(content) ||
        /\bmeta_graph_pb2\b/.test(content) ||
        /\bsignature_def\b/.test(content)
      )
        savedModelUsage += 1;
      if (
        /\btf\.load_library\b/.test(content) ||
        /\bload_library\b/.test(content) ||
        /\bget_libops_path\b/.test(content)
      )
        customOpLoads += 1;
    }

    const runtimeOpsFiles = ((await actx.fs.glob(tfRuntimeOpsGlob)) as string[])
      .filter((p) => !p.endsWith("/"))
      .slice()
      .sort();

    const sampleFiles = runtimeOpsFiles
      .filter((p) => /\.(cc|h|proto|cu\.cc|py)$/i.test(p))
      .slice(0, 60);

    let normalizedRaw: string | undefined;
    try {
      normalizedRaw = await actx.fs.readFile(normalizedMappingPath, "utf8");
    } catch {
      warnings.push(
        `Missing ${normalizedMappingPath}; TF runtime plan will be generated without normalized mapping context.`,
      );
    }

    let normalized: NormalizedMapping | undefined;
    if (normalizedRaw != null) {
      try {
        normalized = JSON.parse(normalizedRaw) as NormalizedMapping;
      } catch {
        warnings.push(
          `Failed to parse ${normalizedMappingPath} as JSON; TF runtime plan will ignore it.`,
        );
      }
    }

    const candidateSignalsByPath = new Map(
      candidates.map((c) => [c.path, c.signals] as const),
    );

    const tfDependentPythonFiles = (normalized?.records ?? [])
      .filter((r) => typeof r?.pythonPath === "string")
      .filter((r) => {
        const p = r.pythonPath;
        const signals = candidateSignalsByPath.get(p) ?? [];
        if (signals.length > 0) return true;
        if (/\btensorflow\b/i.test(r.notes ?? "")) return true;
        return false;
      })
      .map((r) => ({
        pythonPath: String(r.pythonPath),
        status: String(r.status ?? "TODO"),
        rustTargets: (r.rustTargets ?? [])
          .map((t) => t?.fullPath ?? t?.raw ?? "")
          .filter((s) => typeof s === "string" && s.length > 0)
          .slice()
          .sort(),
        notes: String(r.notes ?? "").trim(),
        issues: (r.issues ?? []).filter((x) => typeof x === "string").slice().sort(),
        naJustification:
          typeof r.naJustification === "string" ? r.naJustification : undefined,
        suggestedRustTestLocation:
          typeof r.suggestedRustTestLocation === "string"
            ? r.suggestedRustTestLocation
            : undefined,
      }))
      .sort((a, b) => a.pythonPath.localeCompare(b.pythonPath));

    return {
      inputs: { pythonGlob, tfRuntimeOpsGlob, normalizedMappingPath },
      warnings,
      python: {
        totalFiles: pythonPaths.length,
        tensorflowImports,
        savedModelUsage,
        customOpLoads,
        candidates,
      },
      runtimeOps: {
        files: runtimeOpsFiles,
        sampleFiles,
      },
      normalized: {
        available: normalized != null,
        tfDependentPythonFiles,
      },
    };
  },
);

export default (
  <Program
    id="tf-runtime-plan"
    target={{ language: "md" }}
    description="Define an optional TensorFlow runtime integration contract (dynamic libtensorflow loading, custom op loading, signatures, SavedModel IO) and enumerate parity gaps that require TF runtime."
  >
    <Asset
      id="tf_runtime_plan_doc"
      kind="doc"
      path="../generated/parity/50-tf-runtime-plan/tf_runtime_plan.md"
    />
    <Asset
      id="tf_runtime_plan_json"
      kind="json"
      path="../generated/parity/50-tf-runtime-plan/tf_runtime_plan.json"
    />
    <Action
      id="compute-tf-runtime-plan-inputs"
      export="computeTfRuntimePlanInputs"
      cache
    />
    <Agent
      id="write-tf-runtime-plan-json"
      produces={["tf_runtime_plan_json"]}
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
          You are generating a deterministic, implementation-ready plan for an
          optional TensorFlow runtime integration in Rust. You must produce
          strictly valid JSON and write files using apply_patch.
        </System>
        <Context>
          {ctx.dependency(normalizedMappingJson, {
            as: "Normalized mapping JSON (canonical)",
            mode: "code",
          })}
          {ctx.dependency(mappingConventionsDoc, {
            as: "Mapping conventions (proto/op boundary, tests/fixtures)",
            mode: "quote",
          })}
          {ctx.actionResult("compute-tf-runtime-plan-inputs", {
            as: "TF runtime inventory (Python scan + native_training/runtime/ops tree)",
          })}
          {ctx.file("monolith/path_utils.py", {
            as: "path_utils.py",
            mode: "code",
          })}
          {ctx.file("monolith/utils.py", { as: "utils.py", mode: "code" })}
          {ctx.file(
            "monolith/native_training/runtime/ops/gen_monolith_ops.py",
            {
              as: "gen_monolith_ops.py",
              mode: "code",
            },
          )}
          {ctx.file(
            "monolith/native_training/model_export/saved_model_exporters.py",
            {
              as: "saved_model_exporters.py (excerpt)",
              mode: "code",
            },
          )}
          {ctx.file("monolith/native_training/estimator.py", {
            as: "estimator.py (SavedModel export/import behavior)",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/model_export/demo_predictor.py", {
            as: "demo_predictor.py (SavedModel load + signature usage)",
            mode: "code",
          })}
          {ctx.file("monolith/native_training/data/datasets.py", {
            as: "datasets.py (tf.data + custom op usage excerpt)",
            mode: "code",
          })}
          {ctx.file("monolith/base_runner.py", {
            as: "base_runner.py (TensorFlow runtime entrypoint surface)",
            mode: "code",
          })}
          {ctx.file("monolith/tpu_runner.py", {
            as: "tpu_runner.py (TensorFlow compat.v1 usage surface)",
            mode: "code",
          })}
        </Context>
        <Instructions>{`Write a single JSON object to \`{{assets.tf_runtime_plan_json.path}}\` using apply_patch.

Hard requirements:
1) Deterministic and planner-friendly: stable ordering; no timestamps; no prose outside JSON.
2) Treat TF runtime as optional: design a feature-gated integration contract that keeps core crates usable without TF.
3) Cover: dynamic libtensorflow loading, custom op loading, graph/session execution entrypoints, SavedModel IO, signature handling, and tensor type conversions.
4) Enumerate parity gaps requiring TF runtime vs those that can be Candle-only.
5) Provide an implementable milestone order with verification gates.

Schema (must match exactly; fill all fields):
{
  "version": "1",
  "domain": "tf_runtime",
  "goal": "Optional TensorFlow runtime integration for parity-critical flows (SavedModel IO + custom ops) without making the Rust stack depend on TF by default.",
  "scope": {
    "pythonHotspots": [ { "path": "monolith/...", "why": "...", "signals": [ "import tensorflow as tf", "tf.saved_model", "tf.load_library", "..." ] } ],
    "nativeOpsRoot": "monolith/native_training/runtime/ops",
    "notes": [ "..." ]
  },
  "rustDesign": {
    "featureGates": {
      "tf_runtime": { "default": false, "enables": [ "dynamic libtensorflow loading", "SavedModel IO via TF APIs" ] },
      "tf_custom_ops": { "default": false, "enables": [ "dlopen custom op .so and resolve op registrations" ] }
    },
    "proposedCrates": [
      { "crate": "monolith-tf-runtime", "status": "proposed|exists", "root": "monolith-rs/crates/monolith-tf-runtime", "responsibility": "..." },
      { "crate": "monolith-tensor-ops", "status": "proposed|exists", "root": "monolith-rs/crates/monolith-tensor-ops", "responsibility": "..." }
    ],
    "publicApi": {
      "dynamicLoading": [
        { "fn": "TfApi::load_from_paths", "signature": "paths: [string], symbols: ... -> Result<TfApi, TfError>", "notes": "Prefer dlopen + symbol resolution, no link-time TF dependency." }
      ],
      "customOpLoading": [
        { "fn": "TfCustomOps::load", "signature": "paths: [string] -> Result<TfCustomOps, TfError>", "notes": "Mirror tf.load_library behavior; define search paths and error semantics." }
      ],
      "savedModel": [
        { "fn": "SavedModel::load", "signature": "dir: string, tags: [string] -> Result<SavedModel, TfError>", "notes": "Expose signature discovery + session run." },
        { "fn": "SavedModel::signature", "signature": "name: string -> Result<Signature, TfError>", "notes": "Inputs/outputs TensorInfo-like metadata." }
      ],
      "execution": [
        { "fn": "Session::run", "signature": "feeds: Map<string, Tensor>, fetches: [string] -> Result<Map<string, Tensor>, TfError>", "notes": "Match TF error mapping; deterministic ordering." }
      ],
      "tensors": [
        { "type": "Tensor", "notes": "Represent dtype, shape, and backing buffer; include string tensor constraints and zero-copy options when possible." }
      ]
    },
    "errors": {
      "mapping": [ { "python": "tf.errors.NotFoundError", "rust": "TfError::NotFound", "parityExpectation": "..." } ],
      "notes": [ "Prefer explicit error kinds and preserve TF status message substrings where stable." ]
    }
  },
  "compatibility": {
    "savedModel": {
      "requirements": [
        "Load SavedModel directories produced by monolith/native_training/model_export/*",
        "Discover tags and signature_defs; select signature by name",
        "Feed tensors using tensor_info names and enforce shape/dtype checks"
      ],
      "nonGoals": [ "Training graph construction parity", "Full tf.data runtime parity" ]
    },
    "customOps": {
      "requirements": [
        "Support loading libtfkernel_monolith_ops_for_load.so (and follow-on dependencies) from a resolved path set",
        "Provide stable search path conventions (env var + relative to repo root for dev)",
        "Provide a manifest format listing required .so files and versions"
      ]
    }
  },
  "parityGaps": [
    {
      "gap": "SavedModel IO and signature execution requires TF runtime",
      "severity": "high|medium|low",
      "affectedPython": [ "monolith/native_training/model_export/saved_model_exporters.py" ],
      "rustTargets": [ "monolith-rs/crates/monolith-tf-runtime/src/saved_model.rs" ],
      "workaroundIfNoTfRuntime": "..."
    }
  ],
  "milestones": [
    {
      "id": "m1_dynamic_loading_skeleton",
      "dependsOn": [],
      "deliverables": [ "monolith-tf-runtime crate skeleton", "dlopen + symbol checks", "feature gate wiring" ],
      "verification": [ "Unit test: loads a fake dlopen library", "Integration test: validates missing lib reports expected error kind" ]
    }
  ],
  "verificationHarness": {
    "sharedFixturesRoot": "monolith-rs/fixtures/parity/tf_runtime",
    "cases": [
      {
        "id": "saved_model_signature_smoke",
        "pythonReference": "monolith/native_training/model_export/demo_predictor.py",
        "inputs": [ "a small SavedModel fixture directory", "signature name" ],
        "expected": [ "signature discovery", "feeds/fetches execute and produce deterministic shapes/dtypes" ]
      }
    ]
  }
}`}</Instructions>
      </Prompt>
    </Agent>
    <Agent
      id="write-tf-runtime-plan-doc"
      needs={["write-tf-runtime-plan-json"]}
      produces={["tf_runtime_plan_doc"]}
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
          {ctx.agent("write-tf-runtime-plan-json", {
            artifacts: ["tf_runtime_plan_json"],
            as: "TF runtime plan JSON (generated by this module)",
          })}
          {ctx.actionResult("compute-tf-runtime-plan-inputs", {
            as: "TF runtime inventory (Python scan + native_training/runtime/ops tree)",
          })}
          {ctx.dependency(mappingConventionsDoc, {
            as: "Mapping conventions (proto/op boundary, tests/fixtures)",
            mode: "quote",
          })}
        </Context>
        <Instructions>{`Write a concrete plan document to \`{{assets.tf_runtime_plan_doc.path}}\` using apply_patch.

Requirements:
1) Deterministic: stable ordering; no timestamps.
2) Explain the why: which parity goals require TF runtime (SavedModel + custom ops), and which do not.
3) Specify the contract:
   - Dynamic libtensorflow discovery and loading (search paths, env vars, error messages).
   - Custom op library loading behavior matching tf.load_library usage in Python.
   - SavedModel load + signature discovery + session execution: required subset only.
   - Tensor conversion rules (including string tensors) and shape/dtype checking behavior.
4) Provide an incremental milestone order with test gates and a feature-flag rollout plan.
5) Include a "Parity gaps requiring TF runtime" section that lists concrete Python hotspot files and the corresponding proposed Rust modules.
6) Do not paste the entire JSON; summarize it and link to it by path via \`{{assets.tf_runtime_plan_json.path}}\`.`}</Instructions>
      </Prompt>
    </Agent>
  </Program>
);
