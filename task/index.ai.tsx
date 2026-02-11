import './00-discover-summary.ai.tsx';
import './00-discover-report.ai.tsx';
import './01-validate-inventory.ai.tsx';
import './10-generate-mapping-table.ai.tsx';
import './11-normalize-mapping.ai.tsx';
import './12-agent-service-zk-mirror.ai.tsx';
import './13-agent-service-tfs-client.ai.tsx';
import './14-agent-service-replica-manager.ai.tsx';
import './15-agent-service-cli.ai.tsx';
import './20-agent-service-plan.ai.tsx';
import './21-agent-service-impl.ai.tsx';
import './30-core-plan.ai.tsx';
import './35-utils-and-entrypoints-plan.ai.tsx';
import './40-native-training-plan.ai.tsx';
import './50-tf-runtime-plan.ai.tsx';
import './60-third-party-and-codegen-plan.ai.tsx';
import './80-parity-test-harness-plan.ai.tsx';
import './90-parity-dashboard.ai.tsx';

import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="index"
    target={{ language: 'rust' }}
    description="Entrypoint orchestrator: run direct porting modules (no mapping tables or plans)."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/index.md" />
    <Agent id="noop">
      <Prompt>
        <System>
          This module only sequences the imported porting tasks.
        </System>
        <Instructions>
          Run the imported modules. Do not produce additional reports or plans.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
