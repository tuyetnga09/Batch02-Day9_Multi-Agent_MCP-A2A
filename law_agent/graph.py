"""Law Agent LangGraph StateGraph definition.

Graph topology:
    analyze_law → check_routing → (parallel) call_tax + call_compliance → aggregate → END

The parallel branches (call_tax / call_compliance) are dispatched via LangGraph's
Send API so that both sub-agent calls happen concurrently.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from common.llm import get_llm

logger = logging.getLogger(__name__)

MAX_DELEGATION_DEPTH = 3


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

def _last_wins(a: str, b: str) -> str:
    """Reducer: keep the most recently written value."""
    return b if b else a


class LawState(TypedDict):
    question: str
    context_id: str
    trace_id: str
    delegation_depth: int
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    # Annotated so parallel branches can both write without conflict
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    final_answer: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def analyze_law(state: LawState) -> dict:
    """LLM analysis from a contract / general law perspective."""
    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                "You are a senior corporate litigation attorney specialising in contract law, "
                "tort law, and general business law. Analyse the legal aspects of the question "
                "thoroughly, covering relevant statutes, case law principles, and liability exposure."
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    return {"law_analysis": result.content}


# Từ khóa định tuyến — tối ưu latency: thay LLM routing bằng keyword matching,
# bỏ hẳn 1 LLM call nối tiếp khỏi critical path.
_TAX_KEYWORDS = (
    "tax", "irs", "evasion", "penalt", "fbar", "fatca", "deduction",
    "audit", "revenue", "thuế",
)
_COMPLIANCE_KEYWORDS = (
    "complian", "regulat", "sec", "sox", "sarbanes", "fcpa", "aml",
    "gdpr", "securities", "sanction", "antitrust", "tuân thủ",
)


async def check_routing(state: LawState) -> dict:
    """Determine whether tax and/or compliance sub-agents are needed.

    Tối ưu: dùng keyword matching (không gọi LLM) → tức thời, không thêm
    độ trễ vào critical path. Vẫn tôn trọng MAX_DELEGATION_DEPTH.
    """
    depth = state.get("delegation_depth", 0)
    if depth >= MAX_DELEGATION_DEPTH:
        logger.info("Max delegation depth reached (%d); skipping sub-agents", depth)
        return {"needs_tax": False, "needs_compliance": False}

    q = state["question"].lower()
    needs_tax = any(kw in q for kw in _TAX_KEYWORDS)
    needs_compliance = any(kw in q for kw in _COMPLIANCE_KEYWORDS)
    logger.info(
        "Routing decision (keyword): needs_tax=%s needs_compliance=%s",
        needs_tax, needs_compliance,
    )
    return {"needs_tax": needs_tax, "needs_compliance": needs_compliance}


def route_to_subagents(state: LawState) -> list[Send]:
    """Routing function: dispatch parallel Send objects based on routing flags.

    This function is used with add_conditional_edges; it returns a list of
    Send objects which LangGraph executes as parallel branches.
    """
    # Tối ưu: analyze_law luôn chạy SONG SONG với tax/compliance (output của
    # nó chỉ dùng ở aggregate), thay vì nối tiếp trước như baseline.
    sends: list[Send] = [Send("analyze_law", state)]
    if state.get("needs_tax"):
        sends.append(Send("call_tax", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance", state))
    return sends


async def call_tax(state: LawState) -> dict:
    """Delegate to the Tax Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    try:
        endpoint = await discover("tax_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Tax Agent returned %d chars", len(result))
        return {"tax_result": result}
    except Exception as exc:
        logger.exception("call_tax failed: %s", exc)
        return {"tax_result": f"[Tax analysis unavailable: {exc}]"}


async def call_compliance(state: LawState) -> dict:
    """Delegate to the Compliance Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    try:
        endpoint = await discover("compliance_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Compliance Agent returned %d chars", len(result))
        return {"compliance_result": result}
    except Exception as exc:
        logger.exception("call_compliance failed: %s", exc)
        return {"compliance_result": f"[Compliance analysis unavailable: {exc}]"}


async def aggregate(state: LawState) -> dict:
    """Combine law_analysis, tax_result, and compliance_result into a final answer."""
    llm = get_llm()

    sections: list[str] = []
    if state.get("law_analysis"):
        sections.append(f"## Legal Analysis\n{state['law_analysis']}")
    if state.get("tax_result"):
        sections.append(f"## Tax Analysis\n{state['tax_result']}")
    if state.get("compliance_result"):
        sections.append(f"## Regulatory Compliance Analysis\n{state['compliance_result']}")

    combined = "\n\n---\n\n".join(sections)

    messages = [
        SystemMessage(
            content=(
                "You are a senior legal counsel synthesising specialist analyses into a "
                "comprehensive, well-structured response for the client. Combine the following "
                "analyses into a cohesive answer with clear sections. Avoid redundancy. "
                "End with a brief disclaimer that the analysis is educational and the client "
                "should consult licensed attorneys for their specific situation."
            )
        ),
        HumanMessage(content=combined),
    ]
    result = await llm.ainvoke(messages)
    return {"final_answer": result.content}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_graph():
    """Build and compile the Law Agent StateGraph."""
    graph = StateGraph(LawState)

    graph.add_node("analyze_law", analyze_law)
    graph.add_node("check_routing", check_routing)
    graph.add_node("call_tax", call_tax)
    graph.add_node("call_compliance", call_compliance)
    graph.add_node("aggregate", aggregate)

    # Tối ưu topology: check_routing (keyword, tức thời) là entry point, rồi
    # fan-out SONG SONG tới analyze_law + call_tax + call_compliance, cùng hội
    # tụ ở aggregate. Critical path bên law agent giảm từ 4 tầng nối tiếp
    # (analyze→route→specialists→aggregate) xuống 2 tầng (route→{song song}→aggregate).
    graph.set_entry_point("check_routing")
    graph.add_conditional_edges(
        "check_routing",
        route_to_subagents,
        ["analyze_law", "call_tax", "call_compliance"],
    )

    graph.add_edge("analyze_law", "aggregate")
    graph.add_edge("call_tax", "aggregate")
    graph.add_edge("call_compliance", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()