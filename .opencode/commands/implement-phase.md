---
description: implement a phase of the implementation plan.
agent: orchestrator
---

# Structured implementation

Implement phase "$1" of the Deep Research framework. Follow this workflow:

## 1. Read the phase spec

Read the definition of "$1" from `docs/ImplementationPlan.md` — its goal, deliverables, bringup steps, exit criteria, and tests.

## 2. Check for missing design decisions

Read `docs/Architecture.md` and verify all design decisions needed for implementation are present.
- IF NOT: Ask the user with the `question` tool. Provide a concrete recommendation for each missing piece.
- IF YES: Proceed.

## 3. Build the structured spec

Load the `deepresearch-phase-spec` skill and fill out the template. Every section must be filled (or marked N/A with a reason).

If the spec flags **Architecture divergence** from `docs/Architecture.md` or `docs/DesignBrief.md`:
- Ask the user for approval before proceeding. Canonical docs are not updated silently.
- If the user approves the divergence, note it for the finalize step (step 7).

## 4. Self-check the spec

Re-read the completed spec. Every file, function, and test must be concrete and traceable to a section in `docs/Architecture.md` or `docs/ImplementationPlan.md`. If anything is vague:
- Re-read the relevant doc section and tighten the spec.
- If still ambiguous, ask the user.

Do not spawn the developer with a vague spec.

## 5. Spawn the developer

Spawn the `developer` subagent. Pass it:
- The phase number
- The exit criteria
- The completed spec from step 3

The developer writes code, tests, runs `uv run pytest` and `uv run ruff check .`, and reports back using its **completion report** format.

## 6. Pre-handoff verification

Before spawning the reviewer, run `uv run pytest` and `uv run ruff check .` yourself to sanity-check the developer's work.
- If either fails, spawn the developer again with the specific failure to fix.
- If both pass, proceed to the review.

## 7. Spawn the reviewer

Spawn the `reviewer` subagent. Pass it:
- The phase number
- The exit criteria
- The list of files created/modified
- Any architecture divergences flagged in the spec or completion report

The reviewer reports back using its **verdict format**: APPROVED or BLOCKED, with categorized findings and file:line references.

## 8. Iterate on review findings

If the reviewer reports blockers:
- Spawn the developer again with the specific blockers to fix
- Re-verify and re-review
- Repeat until the reviewer reports APPROVED

## 9. Finalize

Once the reviewer reports APPROVED:
- Confirm `docs/ImplementationPlan.md` phase status row is updated to "shipped" (the developer should have done this; verify it).
- If architecture divergences were approved by the user earlier, update `docs/Architecture.md` (and `docs/DesignBrief.md` if needed) now.
- Report completion: which phase was implemented, what modules were created/modified, and a summary of the test results.
