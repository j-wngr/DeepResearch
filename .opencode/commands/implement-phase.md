---
description: implement a phase of the implementation plan.
agent: orchestrator
---

# Structured implementation

Implement phase "$1" of the Deep Research framework. Follow this workflow:

## 1. Read the phase spec
Read the definition of "$1" from `docs/ImplementationPlan.md` — its goal, deliverables, bringup steps, and exit criteria.

## 2. Check for missing design decisions
Read `docs/Architecture.md` and verify all design decisions needed for implementation are present.
- IF NOT: Ask the user with the question tool. Provide a concrete recommendation for each missing piece.
- IF YES: Proceed.

## 3. Create a detailed implementation plan
Create a unambiguous, step-by-step implementation plan that covers:
- Which modules to create or modify (from `docs/Architecture.md §5` module layout)
- What tests to write, including which invariants to test and which fakes to use
- Any new models, state fields, or config knobs needed
- Whether `docs/Architecture.md` or `docs/DesignBrief.md` needs updating

The plan must be concrete enough that the developer can execute it without guessing.

## 4. Spawn the developer
Spawn the `developer` subagent with the implementation plan. The developer will write code and tests, run `pytest` and `ruff`, and verify the work.

## 5. Spawn the reviewer
Once the developer completes, spawn the `reviewer` subagent. The reviewer will:
- Check architecture compliance (§5 dependency direction, contracts)
- Verify test coverage and hermetic test rules
- Run `ruff` and `pytest`
- Check docs consistency
- Report blockers vs suggestions

## 6. Iterate on review findings
If the reviewer reports blockers:
- Spawn the developer again with the specific findings to fix
- Re-spawn the reviewer to verify the fixes
- Repeat until the reviewer approves

## 7. Finalize
Once the reviewer approves:
- Update `docs/Architecture.md` and `docs/DesignBrief.md` if the user approves.
- Report completion: which phase was implemented, what modules were created/modified, and a summary of the test results
