# Day 08 Lab Report — LangGraph Agentic Orchestration

## 1. Team / student

- **Team Name**: TeamFlash
- **Repository / Track**: DAY23-Track3-TeamFlash
- **Date**: 2026-08-25

## 2. Architecture

The support-ticket agent workflow is implemented as a cyclic, stateful graph using **LangGraph**:

- **11 Registered Nodes**:
  1. `intake`: Normalizes raw ticket query and initiates audit message.
  2. `classify`: Intent classification using LLM with structured output (`ClassificationResult`).
  3. `tool`: Mock execution of lookups/approved actions with simulated transient errors.
  4. `evaluate`: Evaluates tool output quality / error flags to gate the retry loop.
  5. `answer`: Grounded response generation using LLM based strictly on gathered context.
  6. `clarify`: Requests missing details for vague queries or asks for alternatives on rejection.
  7. `risky_action`: Prepares and formats sensitive side-effect actions for human review.
  8. `approval`: Human-in-the-loop gate (mock approved by default, supports `interrupt()`).
  9. `retry`: Increments attempt counter and logs transient errors.
  10. `dead_letter`: Handles retry exhaustion and prepares technical support escalation.
  11. `finalize`: Emits final audit event ensuring all branches terminate cleanly.

- **8 Fixed Edges**:
  - `START -> intake -> classify`
  - `tool -> evaluate`
  - `risky_action -> approval`
  - `answer -> finalize -> END`
  - `clarify -> finalize -> END`
  - `dead_letter -> finalize -> END`

- **4 Conditional Edges**:
  - `classify`: Routes to `answer`, `tool`, `clarify`, `risky_action`, or `retry`.
  - `evaluate`: Routes to `retry` (needs_retry) or `answer` (success).
  - `retry`: Routes to `tool` (attempt < max_attempts) or `dead_letter` (attempt >= max_attempts).
  - `approval`: Routes to `tool` (approved) or `clarify` (rejected).

- **Termination Guarantee**: All branches reach `finalize` before routing to `END`.

## 3. State schema

The state is defined as `AgentState(TypedDict, total=False)`. All node functions adhere to the
immutable contract: they compute updates in local variables and return partial update dicts.

| Field | Reducer | Type | Why / Usage |
|---|---|---|---|
| `messages` | append (`add`) | `list[str]` | Conversational history and tracking log |
| `tool_results` | append (`add`) | `list[str]` | Chronological history of tool execution outputs |
| `errors` | append (`add`) | `list[str]` | Accumulated log of transient errors and failures |
| `events` | append (`add`) | `list[dict]` | Standardized audit events (`LabEvent`) for each node |
| `thread_id` | overwrite | `str` | Unique thread identifier isolating execution runs |
| `scenario_id` | overwrite | `str` | Evaluation scenario ID (never used for branching) |
| `query` | overwrite | `str` | Normalized user ticket query |
| `route` | overwrite | `str` | Classified intent category (kept intact for metrics) |
| `risk_level` | overwrite | `str` | Risk classification (`low`, `medium`, `high`) |
| `attempt` | overwrite | `int` | Current retry counter (incremented by `retry_node`) |
| `max_attempts` | overwrite | `int` | Bounded retry ceiling per run |
| `final_answer` | overwrite | `Optional[str]` | Grounded final response or escalation message |
| `evaluation_result` | overwrite | `Optional[str]` | Tool verdict (`success`/`needs_retry`) |
| `pending_question` | overwrite | `Optional[str]` | Question for missing info or rejection |
| `proposed_action` | overwrite | `Optional[str]` | Description of pending side-effect action |
| `approval` | overwrite | `Optional[dict]` | Serializable decision (`ApprovalDecision`) |

## 4. Scenario results

Results collected from `outputs/metrics.json` over 7 evaluation scenarios:

**Summary Metrics:**
- **Total Scenarios:** 7
- **Success Rate:** 100.00%
- **Avg Nodes Visited:** 6.43
- **Total Retries:** 3
- **Total Interrupts:** 2

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | True | 0 | 0 |
| S02_tool | tool | tool | True | 0 | 0 |
| S03_missing | missing_info | missing_info | True | 0 | 0 |
| S04_risky | risky | risky | True | 0 | 1 |
| S05_error | error | error | True | 2 | 0 |
| S06_delete | risky | risky | True | 0 | 1 |
| S07_dead_letter | error | error | True | 1 | 0 |

## 5. Failure analysis

### Failure Mode 1: Transient Tool Failures & Unbounded Retry Loops
- **Root Cause & Detection**: External APIs can suffer from timeouts. In our graph, `tool_node`
  simulates transient errors, and `evaluate_node` checks `tool_results[-1]` for error signatures,
  setting `evaluation_result = "needs_retry"`.
- **Containment & Routing**: `retry_or_fallback_node` is the sole owner of `attempt` (+1).
  `route_after_retry` checks `attempt < max_attempts`:
  - If attempts remain, it loops back to `tool`.
  - If attempts reach `max_attempts` (e.g., `S07_dead_letter` with `max_attempts=1`), it routes to
    `dead_letter_node`.
- **Termination Guarantee**: `dead_letter_node` formats an escalation message and connects
  directly to `finalize -> END`.
- **Residual Risk**: Outages resolving on attempt N > max_attempts will still escalate to humans.

### Failure Mode 2: Unauthorized Execution of Risky Actions
- **Root Cause & Detection**: Sensitive operations (refunds, deletions) carry severe side effects.
  `classify_node` assigns top priority to `risky` intents (`risk_level="high"`).
- **Containment & Routing**: The graph routes to `risky_action_node` to format the proposal, then
  enters `approval_node`.
  - If approved (`approved=True`), execution proceeds to `tool_node`.
  - If rejected (`approved=False`), `route_after_approval` redirects immediately to `clarify_node`,
    completely bypassing `tool_node`.
- **Termination Guarantee**: `clarify_node` generates a polite notification explaining rejection
  and directs to `finalize -> END`.
- **Residual Risk**: Subtle adversarial phrasing is mitigated by structured prompt instructions
  and fallback to clarification.

## 6. Persistence / recovery evidence

- **Thread ID Isolation**: Every run is assigned a unique `thread_id` (`f"thread-{scenario_id}"`),
  ensuring independent checkpoint streams.
- **State History**: Using `graph.get_state_history(config)`, full checkpoint timelines can be
  retrieved (verified with 6 to 10 distinct checkpoint snapshots per scenario run).
- **SQLite Persistence & Crash-Resume Evidence**:
  - Implemented `SqliteSaver` in `persistence.py` with `PRAGMA journal_mode=WAL;`.
  - Verified by running a scenario against `checkpoints.db`, terminating the graph instance, and
    launching a brand-new graph instance connecting to the same SQLite database.
  - Successfully retrieved the exact prior state (`route = "tool"`, `final_answer`, and 8 full
    checkpoint snapshots) without data loss.

## 7. Extension work

1. **SQLite Persistence & Crash Recovery**: Full `SqliteSaver` integration with WAL journal mode.
2. **LLM Structured Output Classification**: Robust Pydantic model (`ClassificationResult`)
   ensuring type-safe route extraction and reasoning.
3. **Grounded LLM Generation**: `answer_node` dynamically synthesizes grounded context from tool
   results, approval status, and queries.
4. **Human-in-the-Loop Interrupt Support**: Integrated `langgraph.types.interrupt()` when
   `LANGGRAPH_INTERRUPT=true` is enabled.

## 8. Improvement plan

1. **Parallel Tool Fan-Out (`Send()`)**:
   Implement dynamic map-reduce tool execution using LangGraph's `Send()` API to allow parallel
   lookup of multiple orders/records concurrently.
2. **Interactive Streamlit Review UI**:
   Build an interactive operations dashboard to visualize live graph execution, inspect checkpoint
   state history (Time-Travel debugging), and review pending HITL approval requests.
3. **LLM-as-Judge Tool Evaluation**:
   Upgrade `evaluate_node` from pattern matching to a full LLM-as-judge evaluator that scores tool
   output relevance and completeness before routing to `answer`.
