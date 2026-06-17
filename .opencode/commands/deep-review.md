---
description: Comprehensive audit of the codebase against its design documentation. Checks docs consistency and reviews every shipped phase in parallel.
agent: orchestrator
---

# Deep Code Review

Perform a comprehensive audit of the codebase against its design documentation. The workflow is project-agnostic: phases are discovered dynamically from the implementation plan, not hardcoded.

## 1. Docs consistency audit

Read all documentation files in `docs/` and `CONTEXT.md`. For this project that includes:
- `docs/Architecture.md` — canonical architecture
- `docs/DesignBrief.md` — design rationale
- `docs/ImplementationPlan.md` — phased build plan + exit criteria
- `docs/Implementation.md` — implementation log
- `CONTEXT.md` — domain glossary

For other projects, read whatever docs are present. The check is: do the docs agree with each other, and do they match reality?

Cross-reference for:
- **Contradictions** — does the canonical architecture contradict the design brief or the implementation plan?
- **Stale references** — does any doc point to a module, section, or contract that doesn't exist in code?
- **Missing definitions** — does any doc use a term not defined in the glossary?
- **Implementation log accuracy** — does the log match what's actually in the tree?
- **Phase status accuracy** — does the phase status table match reality (modules exist, tests pass)?

Record findings with severity (blocker / warning / note) and a file:line reference. Keep a running list for step 3.

## 2. Per-phase code audit

Read the phase status table in the implementation plan. For each phase marked **shipped**, spawn one `reviewer` subagent **in parallel**.

For each shipped phase, pass the reviewer:
- Phase number and name
- Exit criteria (verbatim from the implementation plan)
- Module/file list for this phase (from the architecture's module layout and the plan's deliverables)
- Instruction: "Review the entire code for this phase, not a diff. Run `uv run pytest` (or the project's test command) on the phase's test files."
- Instruction: "Verify the implementation log entry for this phase is accurate against the code."

Each reviewer reports back using its structured verdict format: APPROVED or BLOCKED, with categorized findings and file:line references.

**Do not edit code during this audit.** This is a read-only review. You are producing a report, not fixing issues.

## 3. Synthesis

Load the `deepresearch-code-audit` skill for the unified report template.

Collect all reviewer verdicts + all docs-consistency findings. Fill out the template:

- **Docs consistency table** — from step 1 findings
- **Per-phase summary table** — one row per shipped phase, with verdict and test result
- **Cross-cutting issues** — patterns spanning multiple phases (not single-phase findings)
- **Overall assessment** — health summary + severity-prioritized action list

### What you do NOT do

- Do not write or propose code fixes. The audit is read-only.
- Do not update any docs, including the implementation plan's phase status. Status updates are a developer action, not an audit action.
- Do not delegate to `developer`. Only `reviewer` is spawned here.
- Do not mark findings as "resolved" by editing code. List them; the user decides what to fix.

## Hard rules

- The audit is project-agnostic. Do not assume a fixed number of phases — discover them from the implementation plan.
- Canonical docs (`Architecture.md`, `DesignBrief.md`, `CONTEXT.md`) are not updated as part of this audit. If you find they need updating, flag the finding; the user decides.
- The implementation plan's phase status table is not updated as part of this audit. Report on its current state.
- Every finding must have a severity (blocker / warning / note) and a specific reference (`file:line` or `file:section`).
