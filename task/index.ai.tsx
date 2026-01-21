import "./00-discover.ai.tsx";
import "./01-validate-inventory.ai.tsx";
import "./10-generate-mapping-table.ai.tsx";
import "./11-normalize-mapping.ai.tsx";
import "./20-agent-service-plan.ai.tsx";
import "./21-agent-service-impl.ai.tsx";
import "./30-core-plan.ai.tsx";
import "./35-utils-and-entrypoints-plan.ai.tsx";
import "./40-native-training-plan.ai.tsx";
import "./50-tf-runtime-plan.ai.tsx";
import "./60-third-party-and-codegen-plan.ai.tsx";
import "./80-parity-test-harness-plan.ai.tsx";
import "./90-parity-dashboard.ai.tsx";

import {
  Agent,
  Asset,
  Context,
  Instructions,
  Program,
  Prompt,
  System,
  assetRef,
  ctx,
} from "@unpack/ai";

import { parityDashboardDoc } from "./90-parity-dashboard.ai.tsx";

export const parityOrchestrationEntryDoc = assetRef(
  "parity_orchestration_entry_doc",
);

export default (
  <Program
    id="index"
    target={{ language: "md" }}
    description="Entrypoint orchestrator: imports and sequences phases (discover/validate -> mapping -> domain plans -> dashboard) and emits final human-facing deliverable(s)."
  >
    <Asset
      id="parity_orchestration_entry_doc"
      kind="doc"
      path="../generated/parity/index.md"
    />
    <Agent
      id="write-parity-orchestration-entry"
      produces={["parity_orchestration_entry_doc"]}
      external_needs={[
        { alias: "parityDashboardDoc", agent: "write-parity-dashboard-doc" },
      ]}
    >
      <Prompt>
        <System>
          You are the entrypoint writer for a multi-module parity pipeline. You
          write concise, deterministic markdown and write files using
          apply_patch.
        </System>
        <Context>
          {ctx.file("task/90-parity-dashboard.ai.tsx", {
            as: "Parity dashboard module (asset paths)",
            mode: "code",
          })}
          {ctx.dependency(parityDashboardDoc, {
            as: "Parity dashboard (MD)",
            mode: "quote",
          })}
        </Context>
        <Instructions>{`Write \`{{assets.parity_orchestration_entry_doc.path}}\` using apply_patch.

Requirements:
1) Deterministic: stable ordering; no timestamps.
2) Purpose: explain that this module is the pipeline entrypoint and the primary deliverable is the parity dashboard.
3) Include sections:
   - "How To Run" with the two commands:
     - \`unpack --plan task/index.ai.tsx\`
     - \`unpack --execute task/index.ai.tsx\`
   - "Primary Outputs" listing (as code paths) the parity dashboard doc + JSON paths and this entry doc path.
   - "What To Do Next" with 3-6 bullets that instruct readers to follow the dashboard's P0 queue and refer to the linked plans.
4) Do not paste raw JSON; link to it by path.`}</Instructions>
      </Prompt>
    </Agent>
  </Program>
);
