---
name: deepresearch-code-audit
description: Unified audit report template for deep code review. Use when synthesizing reviewer verdicts across multiple phases into a single comprehensive report.
---

## What I do

Provide a structured template for the unified audit report produced at the end of a deep code review. The orchestrator collects verdicts from per-phase reviewers and docs-consistency findings, then fills out this template.

## When to use me

Load this skill after all per-phase reviewers have reported back. Fill out the template using:
- All reviewer verdicts (from Phase 2 of the deep-review workflow)
- All docs-consistency findings (from Phase 1 of the deep-review workflow)

## Template

Copy and fill out every section. If a section has no findings, write "None" — do not leave placeholders.

---

### Docs consistency

For each finding: which docs disagree, what the contradiction is, severity, reference.

| Doc pair | Finding | Severity | Reference |
| --- | --- | --- | --- |
| e.g., Architecture.md vs DesignBrief.md | e.g., "X says Y, Z says W" | blocker / warning / note | `file:section` |
| ... | ... | ... | ... |

Check for:
- **Contradictions** between canonical docs (Architecture, DesignBrief) and the implementation plan
- **Stale references** — docs pointing to modules, sections, or contracts that don't exist in code
- **Missing definitions** — terms used in docs but not defined in the glossary
- **Implementation log accuracy** — does the log match what's actually in the tree?
- **Phase status accuracy** — does the phase status table match reality?

### Per-phase summary

One row per shipped phase.

| Phase | Verdict | Key findings | Test result |
| --- | --- | --- | --- |
| <name/number> | CLEAN / ISSUES / BLOCKERS | one-line summary | N passed, M failed |
| ... | ... | ... | ... |

### Cross-cutting issues

Patterns that span multiple phases. Not single-phase findings — systemic concerns.

- e.g., "Inconsistent naming pattern across phases 2-5: snake_case vs camelCase"
- e.g., "Hermetic test rule violated in 3 of 8 phases"
- e.g., "Architecture §5 module layout not followed in 2 phases"

### Overall assessment

**Health**: one of: Clean / Minor issues / Major issues / Critical gaps

**Priority actions** — ordered by severity, most critical first:

1. **[blocker]** <action> — affects: <scope>
2. **[blocker]** <action> — affects: <scope>
3. **[warning]** <action> — affects: <scope>
4. **[warning]** <action> — affects: <scope>
5. **[note]** <action> — affects: <scope>

---

## Rules

1. **Every section must be filled.** If a section has no findings, write "None" with a one-line reason.
2. **Severity is mandatory.** Every finding must be classified as blocker, warning, or note.
3. **References must be specific.** Use `file:line` or `file:section` format. Vague references are not useful.
4. **Cross-cutting != duplicated.** If the same finding appears in 3 phases, list it once in cross-cutting and reference the per-phase rows.
5. **The report is read-only output.** It is not a task list for the orchestrator — it is a report for the user. The user decides what to act on.
