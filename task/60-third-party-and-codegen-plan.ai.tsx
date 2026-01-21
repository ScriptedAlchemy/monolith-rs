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

export const thirdPartyAndCodegenPlanDoc = assetRef(
  "third_party_and_codegen_plan_doc",
);
export const thirdPartyAndCodegenPlanJson = assetRef(
  "third_party_and_codegen_plan_json",
);

type PatchFile = {
  path: string;
  hunks: number;
  addedLines: number;
  removedLines: number;
};

type RepoSignals = {
  envVars: string[];
  tensorflowPatchTouchedFiles: string[];
  protobufFiles: string[];
  customOpFiles: string[];
  bazelWorkspaceFiles: string[];
};

type ThirdPartyAndCodegenPlanInputs = {
  version: "1";
  inputs: {
    orgTensorflowReadmePath: string;
    orgTensorflowPatchPath: string;
    orgTensorflowGlob: string;
    protobufGlob: string;
    customOpGlob: string;
    bazelWorkspaceFiles: string[];
    normalizedMappingPath: string;
  };
  warnings: string[];
  orgTensorflow: {
    exists: boolean;
    readmeSummary?: string;
    patchStats?: {
      filesTouched: number;
      hunks: number;
      addedLines: number;
      removedLines: number;
      files: PatchFile[];
    };
    behaviorRelevantDiffs: Array<{
      id: string;
      description: string;
      signals: string[];
      touchedFiles: string[];
    }>;
  };
  proto: {
    totalFiles: number;
    files: string[];
    sampleFiles: string[];
  };
  customOps: {
    rootGlob: string;
    totalFiles: number;
    buildFiles: string[];
    sourceFiles: string[];
    sampleFiles: string[];
  };
  codegenAndBuild: {
    bazelFiles: string[];
    inferredCodegenSteps: Array<{
      id: string;
      purpose: string;
      owners: string[];
      inputs: string[];
      outputs: string[];
      parityNotes: string[];
    }>;
  };
  signals: RepoSignals;
};

function uniqSorted(xs: string[]): string[] {
  return Array.from(new Set(xs)).sort();
}

function shortenText(s: string, max: number): string {
  const t = (s ?? "").trim();
  if (t.length <= max) return t;
  return `${t.slice(0, Math.max(0, max - 3))}...`;
}

function parseTfPatchSummary(patchText: string): {
  files: PatchFile[];
  envVars: string[];
} {
  const lines = patchText.split(/\r?\n/);
  const files: PatchFile[] = [];
  const envVars: string[] = [];

  let current: PatchFile | undefined;
  for (const line of lines) {
    const mFile = line.match(/^diff --git a\/(.+?) b\/(.+)$/);
    if (mFile) {
      if (current) files.push(current);
      current = {
        path: mFile[2],
        hunks: 0,
        addedLines: 0,
        removedLines: 0,
      };
      continue;
    }

    if (current) {
      if (/^@@\s/.test(line) || /^@@$/.test(line)) current.hunks += 1;
      if (/^\+/.test(line) && !/^\+\+\+/.test(line)) current.addedLines += 1;
      if (/^-/.test(line) && !/^---/.test(line)) current.removedLines += 1;
    }

    const env1 = line.match(/\b(Read|ReadFloat)FromEnvVar\("([A-Z0-9_]+)"/);
    if (env1) envVars.push(env1[2]);
    const env2 = line.match(/\bgetenv\("([A-Z0-9_]+)"\)/);
    if (env2) envVars.push(env2[1]);
  }
  if (current) files.push(current);

  return { files, envVars: uniqSorted(envVars) };
}

export const computeThirdPartyAndCodegenPlanInputs = action(
  async (actx): Promise<ThirdPartyAndCodegenPlanInputs> => {
    const orgTensorflowReadmePath = "third_party/org_tensorflow/README.md";
    const orgTensorflowPatchPath = "third_party/org_tensorflow/tf.patch";
    const orgTensorflowGlob = "third_party/org_tensorflow/**";
    const protobufGlob = "monolith/**/*.proto";
    const customOpGlob = "monolith/native_training/runtime/ops/**";
    const bazelWorkspaceFiles = [
      "monolith/monolith_workspace.bzl",
      "monolith/tf_serving_workspace.bzl",
      "WORKSPACE",
      "WORKSPACE.bzlmod",
      "MODULE.bazel",
    ];
    const normalizedMappingPath =
      "generated/parity/11-normalize-mapping/normalized_mapping.json";

    const warnings: string[] = [];

    const orgTensorflowFiles = ((await actx.fs.glob(orgTensorflowGlob)) as string[])
      .filter((p) => !p.endsWith("/"))
      .slice()
      .sort();
    const orgTensorflowExists = orgTensorflowFiles.length > 0;

    let readmeSummary: string | undefined;
    try {
      const readme = await actx.fs.readFile(orgTensorflowReadmePath, "utf8");
      readmeSummary = shortenText(readme.replace(/\s+/g, " "), 900);
    } catch {
      if (orgTensorflowExists)
        warnings.push(`Failed to read ${orgTensorflowReadmePath}.`);
    }

    let patchStats:
      | ThirdPartyAndCodegenPlanInputs["orgTensorflow"]["patchStats"]
      | undefined;
    let patchTouchedFiles: string[] = [];
    let patchEnvVars: string[] = [];
    try {
      const patch = await actx.fs.readFile(orgTensorflowPatchPath, "utf8");
      const summary = parseTfPatchSummary(patch);
      patchTouchedFiles = uniqSorted(summary.files.map((f) => f.path));
      patchEnvVars = summary.envVars;
      patchStats = {
        filesTouched: summary.files.length,
        hunks: summary.files.reduce((acc, f) => acc + f.hunks, 0),
        addedLines: summary.files.reduce((acc, f) => acc + f.addedLines, 0),
        removedLines: summary.files.reduce((acc, f) => acc + f.removedLines, 0),
        files: summary.files.slice().sort((a, b) => a.path.localeCompare(b.path)),
      };
    } catch {
      if (orgTensorflowExists)
        warnings.push(`Failed to read ${orgTensorflowPatchPath}.`);
    }

    const protobufFiles = ((await actx.fs.glob(protobufGlob)) as string[])
      .filter((p) => !p.endsWith("/"))
      .slice()
      .sort();

    const customOpFiles = ((await actx.fs.glob(customOpGlob)) as string[])
      .filter((p) => !p.endsWith("/"))
      .slice()
      .sort();
    const customOpBuildFiles = customOpFiles.filter((p) =>
      /(^|\/)(BUILD(\.bazel)?|WORKSPACE(\.bazel)?|CMakeLists\.txt)$/.test(p),
    );
    const customOpSourceFiles = customOpFiles.filter((p) =>
      /\.(cc|cpp|c|cu|cuh|h|hpp|proto|py)$/.test(p),
    );

    const bazelFiles: string[] = [];
    for (const p of bazelWorkspaceFiles) {
      try {
        const st = await actx.fs.readFile(p, "utf8");
        if (st != null) bazelFiles.push(p);
      } catch {
        continue;
      }
    }
    bazelFiles.sort();

    const behaviorRelevantDiffs: Array<{
      id: string;
      description: string;
      signals: string[];
      touchedFiles: string[];
    }> = [
      {
        id: "tf_shared_lib_visibility",
        description:
          "TensorFlow BUILD visibility changes to support shared library usage (libtensorflow_framework.so).",
        signals: ["tf_shared_lib", "build_visibility"],
        touchedFiles: patchTouchedFiles.filter((p) => p === "tensorflow/BUILD"),
      },
      {
        id: "tf_master_rpc_timeout_and_device_failfast",
        description:
          "TF master/device discovery and RPC timeout behavior (prevents infinite waits; default RPC timeout).",
        signals: ["distributed_runtime", "rpc_timeout", "failfast"],
        touchedFiles: patchTouchedFiles.filter((p) =>
          /tensorflow\/core\/distributed_runtime\//.test(p),
        ),
      },
      {
        id: "tf_grpc_worker_concurrency_tuning",
        description:
          "Configurable GrpcWorkerService handler counts via env var multiplier.",
        signals: ["env:MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER", "grpc"],
        touchedFiles: patchTouchedFiles.filter((p) =>
          /grpc_worker_service\.cc$/.test(p),
        ),
      },
      {
        id: "hdfs_read_path_optimization",
        description:
          "HDFSRandomAccessFile can switch from Pread to Read when HDFS_OPTIMIZE_READ=1 for performance and caching behavior.",
        signals: ["env:HDFS_OPTIMIZE_READ", "hdfs_read_semantics"],
        touchedFiles: patchTouchedFiles.filter((p) =>
          /hadoop_file_system\.cc$/.test(p),
        ),
      },
      {
        id: "gpu_custom_op_support",
        description:
          "GPU kernel/build changes to support custom ops and GPU placements (gpu_device_array exposure; split/dynamic_partition int32 GPU paths).",
        signals: ["cuda", "custom_op", "gpu_kernels"],
        touchedFiles: patchTouchedFiles.filter((p) =>
          /tensorflow\/core\/kernels\//.test(p),
        ),
      },
    ];

    const inferredCodegenSteps: ThirdPartyAndCodegenPlanInputs["codegenAndBuild"]["inferredCodegenSteps"] =
      [
        {
          id: "protobuf_codegen_python_and_rust",
          purpose:
            "Generate protobuf bindings for Python (runtime + tests) and Rust (tonic/prost) while keeping wire compatibility stable.",
          owners: ["monolith-rs", "monolith-python"],
          inputs: ["monolith/**/*.proto"],
          outputs: [
            "python: generated .py modules (bazel rule outputs)",
            "rust: monolith-rs/crates/monolith-proto (proposed) via build.rs or pre-generated sources",
          ],
          parityNotes: [
            "Pin protoc + prost/tonic versions in Rust; track descriptor-set / file options differences.",
            "Golden gRPC compatibility tests should run without TensorFlow runtime.",
          ],
        },
        {
          id: "tensorflow_custom_ops_build_and_load",
          purpose:
            "Build TF custom ops (C++/CUDA) into shared libraries and load them from Python (tf.load_op_library / tf.load_library) and optionally from Rust TF runtime integration.",
          owners: ["monolith-native-training", "monolith-rs"],
          inputs: ["monolith/native_training/runtime/ops/**", "third_party/org_tensorflow/tf.patch"],
          outputs: [
            "shared objects: *.so for CPU/GPU ops (bazel outputs)",
            "runtime loader contract: paths, symbols, and registration names",
          ],
          parityNotes: [
            "Rust default path remains TF-optional; when TF runtime is present, ensure custom op loading mirrors Python behavior.",
            "Maintain feature flags: candle default; tf-runtime + custom-ops gated behind Linux x86_64 best-effort.",
          ],
        },
        {
          id: "tensorflow_behavior_patch_dependencies",
          purpose:
            "Track TF runtime semantics that depend on local TensorFlow patches (timeouts, HDFS read behavior, worker concurrency tuning).",
          owners: ["monolith-native-training", "monolith-rs"],
          inputs: ["third_party/org_tensorflow/**"],
          outputs: [
            "documented runtime invariants for parity harnesses",
            "explicit parity gaps in Rust (when TF runtime not used)",
          ],
          parityNotes: [
            "Some behaviors are not portable to Rust unless TF runtime is embedded; these must be tracked as explicit parity gaps.",
          ],
        },
      ];

    const envVars = uniqSorted([
      ...patchEnvVars,
      "MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER",
      "HDFS_OPTIMIZE_READ",
    ]);

    const signals: RepoSignals = {
      envVars,
      tensorflowPatchTouchedFiles: patchTouchedFiles,
      protobufFiles,
      customOpFiles,
      bazelWorkspaceFiles: bazelFiles,
    };

    return {
      version: "1",
      inputs: {
        orgTensorflowReadmePath,
        orgTensorflowPatchPath,
        orgTensorflowGlob,
        protobufGlob,
        customOpGlob,
        bazelWorkspaceFiles,
        normalizedMappingPath,
      },
      warnings,
      orgTensorflow: {
        exists: orgTensorflowExists,
        readmeSummary,
        patchStats,
        behaviorRelevantDiffs,
      },
      proto: {
        totalFiles: protobufFiles.length,
        files: protobufFiles,
        sampleFiles: protobufFiles.slice(0, 20),
      },
      customOps: {
        rootGlob: customOpGlob,
        totalFiles: customOpFiles.length,
        buildFiles: customOpBuildFiles,
        sourceFiles: customOpSourceFiles,
        sampleFiles: customOpFiles.slice(0, 50),
      },
      codegenAndBuild: {
        bazelFiles,
        inferredCodegenSteps,
      },
      signals,
    };
  },
);

export default (
  <Program
    id="third-party-and-codegen-plan"
    target={{ language: "md" }}
    description="Account for Python-adjacent behavior dependencies: third_party/org_tensorflow patches, custom op build/load surfaces, protobuf/codegen steps, and Python-dependent tooling that affects runtime semantics; map to Rust build/features and track explicit parity gaps."
  >
    <Asset
      id="third_party_and_codegen_plan_doc"
      kind="doc"
      path="../generated/parity/60-third-party-and-codegen-plan/third_party_and_codegen_plan.md"
    />
    <Asset
      id="third_party_and_codegen_plan_json"
      kind="json"
      path="../generated/parity/60-third-party-and-codegen-plan/third_party_and_codegen_plan.json"
    />
    <Action
      id="compute-third-party-and-codegen-plan-inputs"
      export="computeThirdPartyAndCodegenPlanInputs"
      cache
    />
    <Agent
      id="write-third-party-and-codegen-plan-json"
      produces={["third_party_and_codegen_plan_json"]}
      external_needs={[
        { alias: "normalizedMappingJson", agent: "write-normalized-mapping-json" },
        { alias: "mappingConventionsDoc", agent: "write-mapping-conventions-doc" },
      ]}
    >
      <Prompt>
        <System>
          You produce strictly valid JSON for a parity planning pipeline. You
          write files using apply_patch.
        </System>
        <Context>
          {ctx.dependency(normalizedMappingJson, {
            as: "Normalized Python->Rust mapping (11-normalize-mapping)",
            mode: "code",
          })}
          {ctx.dependency(mappingConventionsDoc, {
            as: "Mapping conventions (11-normalize-mapping)",
            mode: "quote",
          })}
          {ctx.actionResult("compute-third-party-and-codegen-plan-inputs", {
            as: "Computed third-party + codegen plan inputs (canonical JSON)",
          })}
        </Context>
        <Instructions>{`Write JSON to \`{{assets.third_party_and_codegen_plan_json.path}}\` using apply_patch.
The JSON must be a single object and must exactly match the computed plan inputs.`}</Instructions>
      </Prompt>
    </Agent>
    <Agent
      id="write-third-party-and-codegen-plan-doc"
      needs={["write-third-party-and-codegen-plan-json"]}
      produces={["third_party_and_codegen_plan_doc"]}
      external_needs={[
        { alias: "normalizedMappingJson", agent: "write-normalized-mapping-json" },
        { alias: "mappingConventionsDoc", agent: "write-mapping-conventions-doc" },
      ]}
    >
      <Prompt>
        <System>
          You write deterministic engineering plans for cross-language parity.
          You write files using apply_patch.
        </System>
        <Context>
          {ctx.dependency(mappingConventionsDoc, {
            as: "Mapping conventions (11-normalize-mapping)",
            mode: "quote",
          })}
          {ctx.dependency(normalizedMappingJson, {
            as: "Normalized mapping (11-normalize-mapping)",
            mode: "quote",
          })}
          {ctx.actionResult("compute-third-party-and-codegen-plan-inputs", {
            as: "Computed plan inputs (JSON)",
          })}
        </Context>
        <Instructions>{`Write a third-party + codegen plan document to \`{{assets.third_party_and_codegen_plan_doc.path}}\` using apply_patch.

Requirements:
1) Deterministic: stable ordering and no timestamps.
2) Start with a short purpose statement explaining why third_party and codegen matter for parity.
3) Include sections:
   - "TensorFlow Patch Inventory": summarize third_party/org_tensorflow/README.md and tf.patch, list touched TF paths, and call out behavior-impacting diffs (timeouts, HDFS read semantics, worker concurrency tuning, GPU kernel/build changes).
   - "Custom Ops Surface": describe how custom ops are built and loaded today; identify what Rust must support under an optional TF runtime (shared library loading, registration names, CPU/GPU variants) vs what can remain as explicit parity gaps under Candle-only.
   - "Protobuf & gRPC Codegen": propose Rust crate boundaries for protos (e.g., monolith-proto), generation strategy, and compatibility tests; highlight risks (file options, service naming, descriptor compatibility).
   - "Build & Tooling Parity": enumerate Python/Bazel entrypoints affecting runtime semantics (workspace bzl files, bazel rules), and propose how Rust builds/features model them.
   - "Explicit Parity Gaps": list gaps that exist when TF runtime/custom ops are unavailable (clearly labeled; include mitigation/testing strategy).
4) End with a short, prioritized "Next Actions" list with concrete artifacts (docs/tests/scripts) but no implementation code.`}</Instructions>
      </Prompt>
    </Agent>
  </Program>
);

