from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict[str, Any]:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── CLASSIFICATION SCHEMA & NODE ────────────────────────────────────
class ClassificationResult(BaseModel):
    """Structured output for intent classification."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description=(
            "The intent category: "
            "'risky' for actions with irreversible side-effects (refunds, deletions); "
            "'tool' for information lookups (order status, tracking); "
            "'missing_info' for vague/incomplete queries lacking specifics; "
            "'error' for system failures, timeouts, service crashes; "
            "'simple' for general questions/FAQs answerable without tools."
        )
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Risk level: 'high' for risky routes, 'low' for others.",
    )
    reasoning: str = Field(
        default="",
        description="Short justification of why this route was selected based on priority.",
    )


CLASSIFY_SYSTEM_PROMPT = (
    "You are an intent classification system for an AI support ticket agent.\n"
    "Classify the user's support ticket query into exactly one route based on strict priority:\n\n"
    "1. 'risky' (HIGHEST PRIORITY): Actions with side-effects or sensitive operations "
    "(refunds, cancellations, deleting accounts, sending emails).\n"
    "2. 'tool': Read-only information lookups requiring tools/database access "
    "(order status, package tracking, account lookups).\n"
    "3. 'missing_info': Vague, ambiguous, or incomplete queries lacking essential details "
    "(e.g., 'Can you fix it?', 'Why is this broken?', 'Help me').\n"
    "4. 'error': Reports of system failures, timeouts, service crashes, or server errors.\n"
    "5. 'simple' (LOWEST PRIORITY): General questions, FAQs, how-to guides that can be answered "
    "directly without tool calls (e.g., 'How do I reset my password?').\n\n"
    "Priority rule: risky > tool > missing_info > error > simple."
)


def classify_node(state: AgentState) -> dict[str, Any]:
    """Classify the query into a route using an LLM with structured output."""
    query = state.get("query", "").strip()
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(ClassificationResult)
        response = structured_llm.invoke(
            f"{CLASSIFY_SYSTEM_PROMPT}\n\nUser Query: {query}"
        )
        if isinstance(response, ClassificationResult):
            route = response.route
            risk_level = "high" if route == "risky" else response.risk_level
        elif isinstance(response, dict):
            route = response.get("route", "simple")
            risk_level = "high" if route == "risky" else response.get("risk_level", "low")
        else:
            route = "simple"
            risk_level = "low"
    except Exception as exc:
        route = "simple"
        risk_level = "low"
        return {
            "route": route,
            "risk_level": risk_level,
            "errors": [f"classify_node LLM error: {exc}"],
            "events": [
                make_event(
                    "classify",
                    "failed",
                    f"LLM classification failed, fallback to {route}",
                    error=str(exc),
                )
            ],
        }

    return {
        "route": route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified route={route} risk={risk_level}",
                route=route,
                risk_level=risk_level,
            )
        ],
    }


def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute a mock tool call with simulated transient errors for retry testing."""
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")
    proposed_action = state.get("proposed_action")

    if route == "error" and attempt < 2:
        result = f"ERROR: Tool execution failed due to timeout on attempt {attempt}"
        event = make_event("tool", "failed", "simulated tool failure", attempt=attempt)
    elif route == "risky":
        action_desc = proposed_action or f"Action for: {query}"
        result = f"Successfully executed approved risky action: {action_desc}"
        event = make_event("tool", "completed", "executed approved action", attempt=attempt)
    else:
        result = f"Mock tool result: '{query}' -> Status: Verified (attempt {attempt})"
        event = make_event("tool", "completed", "mock tool succeeded", attempt=attempt)

    return {
        "tool_results": [result],
        "events": [event],
    }


def evaluate_node(state: AgentState) -> dict[str, Any]:
    """Evaluate tool results — the retry-loop gate."""
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else ""

    if "ERROR" in latest_result.upper():
        eval_result = "needs_retry"
        message = "evaluation failed: error detected in tool result"
    else:
        eval_result = "success"
        message = "evaluation passed: tool result verified"

    return {
        "evaluation_result": eval_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                message,
                verdict=eval_result,
            )
        ],
    }


def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate a final grounded response using an LLM."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    proposed_action = state.get("proposed_action")

    context_lines = [f"User Query: {query}"]
    if tool_results:
        context_lines.append(f"Tool Results: {tool_results[-1]}")
    if approval:
        context_lines.append(f"Approval Status: {approval.get('approved')}")
    if proposed_action:
        context_lines.append(f"Proposed Action: {proposed_action}")

    prompt = (
        "You are a helpful customer support agent.\n"
        "Generate a clear, grounded response based ONLY on the following context:\n"
        + "\n".join(context_lines)
        + "\n\nAnswer:"
    )

    try:
        llm = get_llm()
        res = llm.invoke(prompt)
        content = res.content if hasattr(res, "content") else str(res)
        final_answer = content if isinstance(content, str) else str(content)
    except Exception as exc:
        final_answer = f"I have processed your request: '{query}'."
        return {
            "final_answer": final_answer,
            "errors": [f"answer_node LLM error: {exc}"],
            "events": [
                make_event(
                    "answer",
                    "fallback",
                    f"LLM answer generation failed: {exc}",
                )
            ],
        }

    return {
        "final_answer": final_answer,
        "events": [
            make_event(
                "answer",
                "completed",
                "grounded response generated",
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict[str, Any]:
    """Ask for missing information or clarify rejected actions."""
    query = state.get("query", "")
    approval = state.get("approval")
    proposed_action = state.get("proposed_action")

    if approval and not approval.get("approved", True):
        comment = approval.get("comment", "Request rejected by reviewer")
        question = (
            f"Your request '{proposed_action or query}' could not be completed because "
            f"it was not approved ({comment}). Please provide additional details."
        )
        event = make_event(
            "clarify",
            "completed",
            "clarification requested after rejection",
            reason="rejection",
        )
    else:
        question = (
            f"Could you please provide more details for your request: '{query}'? "
            "(e.g., account ID, order number, or specific error message)"
        )
        event = make_event(
            "clarify",
            "completed",
            "clarification requested for missing info",
            reason="missing_info",
        )

    return {
        "pending_question": question,
        "final_answer": question,
        "events": [event],
    }


def risky_action_node(state: AgentState) -> dict[str, Any]:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    risk_level = state.get("risk_level", "high")
    proposed_action = f"Perform sensitive action for query: '{query}'"
    return {
        "proposed_action": proposed_action,
        "events": [
            make_event(
                "risky_action",
                "completed",
                f"proposed action: {proposed_action}",
                risk_level=risk_level,
            )
        ],
    }


def approval_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval step (default mock, optional interrupt)."""
    raw_approval = state.get("approval")
    if raw_approval is not None:
        approval = dict(raw_approval)
    elif os.getenv("LANGGRAPH_INTERRUPT", "").lower() in ("true", "1"):
        proposed_action = state.get("proposed_action", "")
        try:
            from langgraph.types import interrupt

            decision = interrupt({"proposed_action": proposed_action})
            if isinstance(decision, dict):
                approval = {
                    "approved": bool(decision.get("approved", True)),
                    "reviewer": str(decision.get("reviewer", "human-reviewer")),
                    "comment": str(decision.get("comment", "Approved via interrupt")),
                }
            else:
                approval = {
                    "approved": bool(decision),
                    "reviewer": "human-reviewer",
                    "comment": "Approved via interrupt",
                }
        except ImportError:
            approval = {
                "approved": True,
                "reviewer": "mock-reviewer",
                "comment": "Approved automatically (mock)",
            }
    else:
        approval = {
            "approved": True,
            "reviewer": "mock-reviewer",
            "comment": "Approved automatically (mock)",
        }

    return {
        "approval": approval,
        "events": [
            make_event(
                "approval",
                "completed",
                f"approval decision: approved={approval['approved']}",
                **approval,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict[str, Any]:
    """Record a retry attempt and update counter."""
    current_attempt = state.get("attempt", 0)
    new_attempt = current_attempt + 1
    max_attempts = state.get("max_attempts", 3)
    error_msg = f"Attempt {new_attempt}/{max_attempts} failed. Scheduling retry."

    return {
        "attempt": new_attempt,
        "errors": [error_msg],
        "events": [
            make_event(
                "retry",
                "completed",
                f"retry attempt {new_attempt} recorded",
                attempt=new_attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict[str, Any]:
    """Handle unresolvable failures after max retries exceeded."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    msg = (
        f"Request failed after reaching maximum retry limit ({attempt}/{max_attempts} attempts). "
        "The ticket has been routed to engineering support for manual intervention."
    )
    return {
        "final_answer": msg,
        "events": [
            make_event(
                "dead_letter",
                "completed",
                "max retries exhausted, escalated to dead letter",
                attempt=attempt,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Emit final audit event before termination."""
    return {
        "events": [make_event("finalize", "completed", "workflow finished")]
    }
