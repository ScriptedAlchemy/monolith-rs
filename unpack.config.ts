type SplitPart = {
  content: string;
  title?: string;
  startLine?: number;
  endLine?: number;
};

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

const MAX_CHECKLIST_TOKENS = 50_000;
const MIN_SPLIT_TOKENS = 120_000;

function trimHeading(line: string, prefix: string): string {
  const raw = line.slice(prefix.length).trim();
  return raw.replace(/^`/, '').replace(/`$/, '');
}

function findLineIndex(lines: string[], needle: string): number {
  for (let i = 0; i < lines.length; i += 1) {
    if (lines[i].trim() === needle) return i;
  }
  return -1;
}

function firstHeadingTitle(lines: string[], start: number, end: number): string {
  for (let i = start; i < end; i += 1) {
    const line = lines[i].trim();
    if (line.startsWith('### ')) return trimHeading(line, '### ');
  }
  for (let i = start; i < end; i += 1) {
    const line = lines[i].trim();
    if (line.startsWith('## ')) return trimHeading(line, '## ');
  }
  return 'request';
}

function countTokens(text: string): number {
  try {
    const mod = require('@dqbd/tiktoken') as {
      encoding_for_model?: (name: string) => { encode: (value: string) => number[]; free?: () => void };
      get_encoding?: (name: string) => { encode: (value: string) => number[]; free?: () => void };
    };
    let enc:
      | { encode: (value: string) => number[]; free?: () => void }
      | undefined;
    try {
      enc = mod.encoding_for_model?.('gpt-4o');
    } catch {
      enc = undefined;
    }
    if (!enc) {
      enc = mod.get_encoding?.('cl100k_base');
    }
    if (!enc) return Math.ceil(text.length / 4);
    const count = enc.encode(text).length;
    if (typeof enc.free === 'function') enc.free();
    return count;
  } catch {
    return Math.ceil(text.length / 4);
  }
}

function splitChecklistSection(lines: string[], start: number, end: number): SplitPart[] {
  const boundaries: number[] = [start];
  for (let i = start + 1; i < end; i += 1) {
    if (lines[i].startsWith('### ')) boundaries.push(i);
  }
  boundaries.push(end);

  const parts: SplitPart[] = [];
  let chunkStart = boundaries[0];
  let chunkEnd = boundaries[0];
  let chunkTokens = 0;

  for (let i = 0; i < boundaries.length - 1; i += 1) {
    const segmentStart = boundaries[i];
    const segmentEnd = boundaries[i + 1];
    const segmentText = lines.slice(segmentStart, segmentEnd).join('\n');
    const segmentTokens = countTokens(segmentText);
    const wouldExceed = chunkTokens > 0 && chunkTokens + segmentTokens > MAX_CHECKLIST_TOKENS;

    if (wouldExceed && chunkEnd > chunkStart) {
      parts.push({
        content: lines.slice(chunkStart, chunkEnd).join('\n'),
        title: firstHeadingTitle(lines, chunkStart, chunkEnd),
        startLine: chunkStart + 1,
        endLine: chunkEnd,
      });
      chunkStart = segmentStart;
      chunkEnd = segmentStart;
      chunkTokens = 0;
    }

    chunkEnd = segmentEnd;
    chunkTokens += segmentTokens;
  }

  if (chunkEnd > chunkStart) {
    parts.push({
      content: lines.slice(chunkStart, chunkEnd).join('\n'),
      title: firstHeadingTitle(lines, chunkStart, chunkEnd),
      startLine: chunkStart + 1,
      endLine: chunkEnd,
    });
  }

  return parts;
}

export default {
  generate: {
    splitInput: (input: string): SplitPart[] | null => {
      if (!input || countTokens(input) < MIN_SPLIT_TOKENS) return null;
      const normalized = input.replace(/\r\n/g, '\n');
      const lines = normalized.split('\n');

      const lineLevelIdx = findLineIndex(lines, '## Line-Level Inventory (All Python Files)');
      const checklistIdx = findLineIndex(lines, '## Per-File Parity Checklists (All Python Files)');

      if (lineLevelIdx === -1 || checklistIdx === -1 || checklistIdx <= lineLevelIdx) {
        return null;
      }

      const parts: SplitPart[] = [];
      if (lineLevelIdx > 0) {
        parts.push({
          content: lines.slice(0, lineLevelIdx).join('\n'),
          title: 'overview',
          startLine: 1,
          endLine: lineLevelIdx,
        });
      }

      if (checklistIdx > lineLevelIdx) {
        parts.push({
          content: lines.slice(lineLevelIdx, checklistIdx).join('\n'),
          title: 'line-level-inventory',
          startLine: lineLevelIdx + 1,
          endLine: checklistIdx,
        });
      }

      parts.push(...splitChecklistSection(lines, checklistIdx, lines.length));
      return parts;
    },
  },
};
