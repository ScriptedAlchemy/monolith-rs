import "./00-discover-summary.ai.tsx";

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

import { discoverSummary } from "./00-discover-summary.ai.tsx";

export const discoverReport = assetRef("discover_report");

export default (
  <Program
    id="discover-report"
    target={{ language: "md" }}
    description="Write a brief discovery report derived from the discovery summary."
  >
    <Asset
      id="discover_report"
      kind="doc"
      path="../generated/parity/00-discover/discovery.report.md"
    />
    <Agent
      id="write-discover-report"
      produces={["discover_report"]}
      external_needs={[{ alias: "discoverSummary", agent: "write-discover-summary" }]}
    >
      <Prompt>
        <System>
          You maintain a parity planning pipeline. You write concise, operational markdown and write files using apply_patch.
        </System>
        <Context>
          {ctx.dependency(discoverSummary, { as: "Discovery summary (JSON)", mode: "code" })}
        </Context>
        <Instructions>{`Write a brief report to \`{{assets.discover_report.path}}\` using apply_patch.
Include:
1) Whether the expected roots and artifacts exist.
2) Python file count vs parity checklist file count.
3) Any notes/warnings from the discovery summary.
Keep it stable and deterministic (do not include timestamps).`}</Instructions>
      </Prompt>
    </Agent>
  </Program>
);

