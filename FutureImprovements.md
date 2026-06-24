# Future Improvements

Findings from a code review on 2026-06-24, grouped by severity.

---

## Critical

### 1. Gate deletes pool files that parallel sub-agents may have whitelisted
**File:** `src/deepresearch/gate.py:352`, `src/deepresearch/nodes/subagent.py:352-353`

When sub-agent B rejects source X and calls `pool.remove()` + `store.delete()`, sub-agent A may have already whitelisted X. The writer then calls `pool_module.get_ref(source_id, bibliography_dir)` for sources cited by A's subreport — that file no longer exists → `FileNotFoundError`, crashing the writer node.

**Fix:** Never delete from the pool during a gate run. Mark rejected sources in a per-run reject log (or skip them on retrieval) instead. Pool cleanup should be a separate, post-run reconciliation step.

---

### 2. `_CACHE_LOCK` held across the LLM call, serializing all parallel gate judgments
**File:** `src/deepresearch/gate.py:135-162`

`_CACHE_LOCK` is a process-wide `threading.Lock`. The full body of `judge()` — including the `chat_fn` LLM call — runs inside the lock. In a 6-subtopic run with 10 candidates each, all 60 gate LLM calls are fully serialized regardless of how many threads LangGraph dispatches.

**Fix:** Split into three phases: (1) lock, read cache, check hit, unlock; (2) if miss, call `chat_fn` without the lock; (3) lock, write result, unlock. Track in-flight keys with a `dict[str, threading.Event]` to avoid duplicate LLM calls for concurrent misses on the same key.

---

## Major

### 3. Unobtainable sources are re-prompted on every subagent iteration
**File:** `src/deepresearch/nodes/subagent.py:110-166`

When a user marks a URL unobtainable (`kind="unobtainable"`), it is recorded in `acquisition_gaps`. But on the next quality-gate fail → loop-back-to-acquire cycle, `_acquire_node` starts fresh: it hits the same paywall again, creates a new `AcquisitionRequest` for the same URL, and interrupts the user again.

**Fix:** Build a `skip_ids` set at the start of `_acquire_node` by parsing `acquisition_gaps` entries, and skip those source IDs when processing search hits.

---

### 4. `elif gaps and not state.get("candidates")` is always True at the quality gate
**File:** `src/deepresearch/nodes/subagent.py:499`

`_gate_node` always returns `{"candidates": [], ...}`. By the time `_quality_gate_node` runs, `state.get("candidates")` is always `[]`, so `not state.get("candidates")` is always `True`. The condition reduces to `elif gaps:`, removing the guard that was meant to only short-circuit when there are no more candidates to try.

**Fix:** Either remove the dead guard and document the condition as `elif gaps:`, or restructure so the gate passes remaining candidates through.

---

### 5. LLM prompted for fields that are immediately discarded
**File:** `src/deepresearch/nodes/evaluate.py:130-134` (prompt) vs `140-173` (parse)

The `_score_coverage` prompt asks the LLM for `followups`, `queued_additions`, `coverage_score`, and `fully_covered`. None of these are consumed: `followups` and `queued_additions` are re-derived mechanically from gaps, and `coverage_score` / `fully_covered` are recomputed locally. Wastes tokens and can confuse the model by setting inconsistent expectations.

**Fix:** Either use the LLM's computed values or remove those fields from the prompt.

---

### 6. `reconcile` never re-indexes a source whose content changed
**File:** `src/deepresearch/rag/index.py:209`

```python
if source_id in indexed:
    continue
```

If `save_web` re-fetches and overwrites a source with newer content (the `content_hash` changes), the Chroma store still holds the old chunks. RAG results continue to reflect stale content.

**Fix:** Compare `content_hash` in chunk metadata against the current frontmatter value; re-index on mismatch.

---

## Minor

### 7. Dead parameters in `_fetch_from_search_hit`
**File:** `src/deepresearch/nodes/subagent.py:239-248`

`store` and `embeddings` appear in the signature and are passed from `_acquire_node`, but the function body never references them. Indexing was moved to `_gate_node`; these are dead parameters.

**Fix:** Remove `store` and `embeddings` from the signature and all call sites.

---

### 8. `evaluate_node` can crash if `brief` is None
**File:** `src/deepresearch/nodes/evaluate.py:80,89`

`_mark_all_dirty(state["brief"])` and `_mark_dirty_for_gaps(state["brief"], coverage)` are called without a None guard. `_score_coverage` handles `brief is None` but these call sites do not.

**Fix:** Add a None check for `brief` at the start of `evaluate_node` and return early (auto-approve) if it is missing.

---

### 9. `run_state.py` defaults to hardcoded `Path("research")`
**File:** `src/deepresearch/run_state.py:41`

```python
summary = load_run(state_dir, output_dir or Path("research"), thread_id)
```

When `output_dir` is None the fallback is a hardcoded relative path, not `get_config().output_dir`. The `has_brief` / `has_report` flags in `RunSummary` will be wrong for non-default output directories.

**Fix:** Default to `get_config().output_dir` instead of `Path("research")`.

---

### 10. Test helper omits required `SubAgentState` fields
**File:** `tests/test_subagent.py:120-131`

`_run_subagent_with_subtopic` omits `pending_acquisitions`, `acquisition_gaps`, and `isolated` from the initial state dict. Tests work today because nodes initialize the first two and `isolated` is never read before being set, but a refactor that reads them before initialization would fail silently.

**Fix:** Add the missing fields to the initial state dict in the test helper.

---

### 11. Dead parameters in `acquisition.apply_response`
**File:** `src/deepresearch/acquisition.py:37`

`chat_fn` and `tavily_client` are accepted and immediately deleted (`del chat_fn, tavily_client`). These are dead weight left over from an older version and mislead callers.

**Fix:** Remove them from the signature and all call sites.

---

### 12. `verify._revise` silently abandons revision on line-count mismatch
**File:** `src/deepresearch/verify.py:158-159`

When the LLM returns fewer lines than unsupported verdicts, the entire revision is silently dropped and the unsupported claims remain in the report. The caller receives the original body unchanged with no indication that revision was attempted.

**Fix:** At minimum emit a warning on mismatch.

---

### 13. `link_density` denominator counts raw markdown tokens, not words
**File:** `src/deepresearch/sources/quality.py:42-44`

`total` is computed on raw markdown (which includes `[`, `](`, `https://...`, `)` as tokens) while `linked` counts only anchor-text words. High-density link pages have URLs that inflate `total`, under-reporting link density.

**Fix:** Use the same `word_count(markdown)` helper for the denominator as for the linked-text count.

---

## Improvements

- **Gate lock granularity** (`gate.py`): release the lock before `chat_fn`, re-acquire before writing. Track in-flight keys with a `dict[str, threading.Event]` to avoid duplicate LLM calls for concurrent misses on the same key.
- **`pool.save_web` idempotency** (`sources/pool.py`): skip the write when `content_hash` matches the existing file, to avoid unnecessary disk I/O and spurious `retrieved_at` timestamp updates for unchanged sources.
- **`embeddings.py`** creates a new `OllamaEmbeddings` instance on every `embed`/`embed_query` call. The instance is cheap but inconsistent with how the CLI reuses a single instance via `_build_configurable`.
- **`_build_configurable` missing `retrieve_fn`** (`cli.py`): `retrieve_fn` is a test seam injected via `configurable` in tests but intentionally absent from the real CLI config. A comment at the injection site would clarify the seam contract for future readers.
