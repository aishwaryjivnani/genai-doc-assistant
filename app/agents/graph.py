"""
Task 8: Agent-based reasoning.

Four agents, matching the "Agent Roles" from the class notes, each a node
in a LangGraph state graph:

  Planner   -> decides the steps/rewrites the question into a search query
  Retriever -> fetches content from the knowledge base (vector DB)
  Reasoning -> analyses the content and generates a grounded answer
  Response  -> here implemented as a Validator that checks the answer
               against the context before it's returned; on failure it
               loops back to Reasoning, up to MAX_REASONING_RETRIES times
               (this doubles as the "Result Verification agent" control
               from Task 9)

LLM: Google Gemini free tier (no OpenAI, no Ollama), via langchain-google-genai.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from app.agents.state import AgentState
from app.core.config import settings
from app.utils.guardrails import (
    NO_CONTEXT_RESPONSE,
    has_sufficient_context,
    log_event,
    strip_prompt_injection_markers,
)
from app.services.vectorstore import similarity_search

_llm = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0,
        )
    return _llm


# ---------------------------------------------------------------------------
# Agent 1: Planner
# ---------------------------------------------------------------------------
def planner_node(state: AgentState) -> AgentState:
    question = state["question"]
    llm = get_llm()
    prompt = (
        "You rewrite user questions into a focused search query for a vector "
        "database of enterprise documents. Return ONLY the rewritten query, "
        "no explanation.\n\nUser question: " + question
    )
    result = llm.invoke([SystemMessage(content="You are a precise query planning agent."),
                          HumanMessage(content=prompt)])
    search_query = result.content.strip() or question

    trace = state.get("trace", [])
    trace.append(f"Planner: rewrote question into search query -> '{search_query}'")
    log_event("planner_done", search_query=search_query)
    return {**state, "search_query": search_query, "plan_notes": search_query, "trace": trace}


# ---------------------------------------------------------------------------
# Agent 2: Retriever
# ---------------------------------------------------------------------------
def retriever_node(state: AgentState) -> AgentState:
    query = state.get("search_query") or state["question"]
    results = similarity_search(query)

    chunks = [
        {
            "text": strip_prompt_injection_markers(doc.page_content),
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
        }
        for doc, score in results
    ]

    trace = state.get("trace", [])
    trace.append(f"Retriever: found {len(chunks)} candidate chunks for query")
    log_event("retriever_done", chunk_count=len(chunks))
    return {
        **state,
        "retrieved_chunks": chunks,
        "context_sufficient": has_sufficient_context(results),
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Agent 3: Reasoning (RAG generation)
# ---------------------------------------------------------------------------
def reasoning_node(state: AgentState) -> AgentState:
    trace = state.get("trace", [])

    if not state.get("context_sufficient"):
        trace.append("Reasoning: insufficient context, returning fallback response")
        return {**state, "answer": NO_CONTEXT_RESPONSE, "trace": trace}

    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in state["retrieved_chunks"]
    )
    feedback_note = ""
    if state.get("validation_feedback"):
        feedback_note = (
            "\n\nA previous answer attempt was rejected for this reason: "
            f"{state['validation_feedback']}. Fix that issue this time."
        )

    llm = get_llm()
    system = (
        "You are an enterprise document Q&A assistant. Answer ONLY using the "
        "provided context. If the context does not contain the answer, say "
        "you don't know. Cite the source file name(s) you used."
    )
    user = (
        f"Context:\n{context_block}\n\n"
        f"Question: {state['question']}"
        f"{feedback_note}"
    )
    result = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    trace.append("Reasoning: generated grounded answer from retrieved context")
    log_event("reasoning_done")
    return {**state, "answer": result.content.strip(), "trace": trace}


# ---------------------------------------------------------------------------
# Agent 4: Response / Validator (result verification agent)
# ---------------------------------------------------------------------------
def validator_node(state: AgentState) -> AgentState:
    trace = state.get("trace", [])

    if not state.get("context_sufficient"):
        # Nothing to validate against; the fallback answer is already safe.
        trace.append("Response/Validator: skipped (no context to check against)")
        return {**state, "validation_passed": True, "trace": trace}

    context_block = "\n\n".join(c["text"] for c in state["retrieved_chunks"])
    llm = get_llm()
    system = (
        "You are a strict fact-checking agent. Given a context and an answer, "
        "reply with exactly one line: 'PASS' if every claim in the answer is "
        "supported by the context, or 'FAIL: <short reason>' if the answer "
        "contains unsupported claims or hallucinations."
    )
    user = f"Context:\n{context_block}\n\nAnswer to check:\n{state['answer']}"
    result = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    verdict = result.content.strip()

    passed = verdict.upper().startswith("PASS")
    trace.append(f"Response/Validator: {verdict}")
    log_event("validation_done", passed=passed)
    return {
        **state,
        "validation_passed": passed,
        "validation_feedback": "" if passed else verdict,
        "retries": state.get("retries", 0) + (0 if passed else 1),
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------
def _should_retry(state: AgentState) -> str:
    if state.get("validation_passed"):
        return "end"
    if state.get("retries", 0) >= settings.MAX_REASONING_RETRIES:
        return "end"
    return "retry"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("validator", validator_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "reasoning")
    graph.add_edge("reasoning", "validator")
    graph.add_conditional_edges(
        "validator", _should_retry, {"retry": "reasoning", "end": END}
    )
    return graph.compile()


_compiled_graph = None


def run_agentic_rag(question: str) -> AgentState:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    initial_state: AgentState = {"question": question, "retries": 0, "trace": []}
    return _compiled_graph.invoke(initial_state)
