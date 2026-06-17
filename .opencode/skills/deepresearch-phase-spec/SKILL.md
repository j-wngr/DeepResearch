---
name: deepresearch-phase-spec
description: Structured specification template for implementing a DeepResearch phase. Use when planning a phase of the implementation plan to produce an unambiguous spec the developer can execute without guessing.
---

## What I do

Provide a concrete, fill-in-the-blanks template for specifying a phase of the DeepResearch implementation. Every file, function, test scenario, and config change is enumerated — the developer should never need to guess.

## When to use me

Load this skill when you are the orchestrator and about to spawn the `developer` subagent for a phase. Fill out the template, then pass the completed spec as the developer's task prompt.

## Template

Copy and fill out every section. Delete placeholder lines that don't apply.

---

### Phase identity

- **Phase number**: [N]
- **Phase name**: [from `docs/ImplementationPlan.md` phase table]
- **Arch milestone**: [from `docs/Architecture.md §13`]

### Exit criteria

Copy verbatim from `docs/ImplementationPlan.md`'s phase spec, plus any "Tests (unit/integration)" items.

- [ ] Criterion 1
- [ ] Criterion 2
- ...

### Files to create

For each: absolute or repo-relative path, purpose in one line, key public exports.

- `src/deepresearch/<path>` — purpose; exports: `foo`, `Bar`
- ...

### Files to modify

For each: path, what changes, why (reference architecture section or invariant).

- `src/deepresearch/<path>` — what: ...; why: Architecture §X
- ...

### Functions / classes to implement

For each: name, signature, contract (from Architecture §6 if applicable), edge cases.

- `function_name(arg: type) -> return_type` — contract: ...; edge cases: ...
- `class ClassName` — fields, methods, invariants
- ...

### Test scenarios

For each: test file path, test name, what invariant or behavior it asserts, which fakes from `tests/fakes/` it uses.

- `tests/test_xxx.py::test_name` — asserts: ...; uses: FakeChat, FakeEmbeddings, ...
- ...

### Model / state / config changes

- **New Pydantic models**: list with brief purpose
- **New state fields**: list with type, default, reducer
- **New config knobs**: list with env var name, default, purpose

### Architecture divergence

If anything in this spec diverges from `docs/Architecture.md` or `docs/DesignBrief.md`, list it here. The orchestrator will ask the user for approval before proceeding. The developer must **never** update Architecture.md or DesignBrief.md directly.

- [ ] None — spec aligns with Architecture
- [ ] Divergence: describe what diverges and why

### Bringup steps

Manual verification the orchestrator will run after the developer reports completion.

- `uv run pytest` — expect all green
- `uv run ruff check .` — expect clean
- `uv run deepresearch --help` — expect verbs listed
- Other concrete steps from ImplementationPlan.md's "Bringup" section

---

## Rules

1. **Every section must be filled.** If a section truly doesn't apply, write "N/A" with a one-line reason — do not leave placeholders.
2. **Be specific.** "Add tests" is not specific. "Add `tests/test_rag.py::test_concurrent_ingest` asserting N threads ingesting simultaneously produce a consistent count" is specific.
3. **Reference the docs.** Every file, function, and test should be traceable to a section in `docs/Architecture.md` or `docs/ImplementationPlan.md`. If it can't be, flag it in Architecture divergence.
4. **The developer does not guess.** If a design choice is genuinely ambiguous, the developer reports it back in the completion report — they do not invent an answer.
