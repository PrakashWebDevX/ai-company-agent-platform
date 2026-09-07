import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.agents.company import (
    architect_task,
    code_task,
    database_task,
    extract_memories,
    plan_task,
    qa_task,
    rag_answer,
    research_task,
    review_task,
)
from backend.config.settings import settings
from backend.memory.store import MemoryStore
from backend.rag.pipeline import RAGPipeline, get_rag_pipeline


class CompanyState(TypedDict, total=False):
    task: str
    user_id: int
    plan: str
    research: str
    architecture: str
    schema: str
    code: str
    rag_context: str
    rag_answer: str
    qa: str
    review: str
    memories_saved: str
    # `plan` and `research` now run in the same superstep (see build_graph),
    # and both append to `logs`. A plain `list[str]` field only accepts one
    # write per superstep in LangGraph -- two concurrent writes raise
    # InvalidUpdateError at runtime. `Annotated[..., operator.add]` tells
    # LangGraph to concatenate concurrent/successive writes instead, so
    # each node can return just its own contribution (see `_log`).
    logs: Annotated[list[str], operator.add]
    final: str


# Module-level singleton: each run_company() call traverses 9 graph nodes,
# 2 of which previously did `MemoryStore()` themselves. MemoryStore()
# re-executes the full schema script + a seed INSERT on every
# construction, so a single request was paying that setup cost twice.
# RAGPipeline is *not* constructed here anymore -- it's shared with
# backend/api/main.py via get_rag_pipeline() (see backend/rag/pipeline.py
# for why that matters for the retrieval cache's correctness).
_memory_store: MemoryStore | None = None


def _memory() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


def _rag() -> RAGPipeline:
    return get_rag_pipeline()


def _log(name: str) -> list[str]:
    """Return this node's sole contribution to `logs` (see the Annotated
    reducer above); LangGraph concatenates it onto the accumulated list
    itself, including when two nodes contribute in the same superstep.
    """
    return [name]


def node_rag(state: CompanyState) -> CompanyState:
    context = _rag().context_block(state["task"])
    answer = rag_answer(state["task"], context) if context else ""
    return {"rag_context": context, "rag_answer": answer, "logs": _log("RAG")}


def node_plan(state: CompanyState) -> CompanyState:
    mems = _memory().list_memories(state.get("user_id") or settings.default_user_id)
    mem_text = "\n".join(m["memory"] for m in mems[:20])
    plan = plan_task(state["task"], state.get("rag_context") or "", mem_text)
    return {"plan": plan, "logs": _log("CEO Planner")}


def node_research(state: CompanyState) -> CompanyState:
    return {"research": research_task(state["task"]), "logs": _log("Research")}


def node_architect(state: CompanyState) -> CompanyState:
    architecture = architect_task(state["task"], state.get("plan") or "", state.get("research") or "")
    return {"architecture": architecture, "logs": _log("Architect")}


def node_database(state: CompanyState) -> CompanyState:
    schema = database_task(state["task"], state.get("architecture") or "")
    return {"schema": schema, "logs": _log("Database")}


def node_coder(state: CompanyState) -> CompanyState:
    code = code_task(state["task"], state.get("architecture") or "", state.get("schema") or "")
    return {"code": code, "logs": _log("Coder")}


def node_qa(state: CompanyState) -> CompanyState:
    qa = qa_task(state.get("code") or "", state.get("schema") or "")
    return {"qa": qa, "logs": _log("QA")}


def node_review(state: CompanyState) -> CompanyState:
    review = review_task(
        {
            "plan": state.get("plan") or "",
            "research": state.get("research") or "",
            "architecture": state.get("architecture") or "",
            "schema": state.get("schema") or "",
            "code": state.get("code") or "",
            "rag": state.get("rag_answer") or "",
            "qa": state.get("qa") or "",
        }
    )
    return {"review": review, "final": review, "logs": _log("Reviewer")}


def node_memory(state: CompanyState) -> CompanyState:
    user_id = state.get("user_id") or settings.default_user_id
    notes = extract_memories(state["task"], state.get("final") or "")
    store = _memory()
    if notes and "API keys are not configured" not in notes:
        store.add_memory(user_id, notes)
    store.add_conversation(user_id, state["task"], state.get("final") or "")
    return {"memories_saved": notes, "logs": _log("Memory")}


def build_graph():
    graph = StateGraph(CompanyState)
    graph.add_node("rag", node_rag)
    graph.add_node("plan", node_plan)
    graph.add_node("research", node_research)
    graph.add_node("architect", node_architect)
    graph.add_node("database", node_database)
    graph.add_node("coder", node_coder)
    graph.add_node("qa", node_qa)
    graph.add_node("review", node_review)
    graph.add_node("memory", node_memory)

    graph.add_edge(START, "rag")
    # `plan` only reads task/rag_context/memories, and `research` only
    # reads task -- neither depends on the other's output (architect is
    # the first node that needs both). Fanning out from "rag" to both,
    # then fanning back in at "architect", runs them concurrently instead
    # of paying their latency twice in sequence.
    graph.add_edge("rag", "plan")
    graph.add_edge("rag", "research")
    graph.add_edge("plan", "architect")
    graph.add_edge("research", "architect")
    graph.add_edge("architect", "database")
    graph.add_edge("database", "coder")
    graph.add_edge("coder", "qa")
    graph.add_edge("qa", "review")
    graph.add_edge("review", "memory")
    graph.add_edge("memory", END)
    return graph.compile()


_APP = None


def get_app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def run_company(task: str, user_id: int | None = None) -> dict:
    result = get_app().invoke(
        {
            "task": task,
            "user_id": user_id or settings.default_user_id,
            "logs": [],
        }
    )
    return dict(result)


def run_company_stream(task: str, user_id: int | None = None):
    """Yield (node_name, partial_state) as each graph node/superstep completes.

    Wraps LangGraph's own `.stream(..., stream_mode="updates")`, which
    yields one dict per completed superstep -- `{"plan": {...}}` for a
    single node, or `{"plan": {...}, "research": {...}}` when nodes ran in
    parallel in the same superstep (see the fan-out/fan-in edges below).
    Used by the `/chat/stream` SSE route so a client sees per-agent
    progress on a run that otherwise takes 30-90+ seconds end to end.
    """
    initial = {
        "task": task,
        "user_id": user_id or settings.default_user_id,
        "logs": [],
    }
    for update in get_app().stream(initial, stream_mode="updates"):
        for node_name, partial_state in update.items():
            yield node_name, partial_state
