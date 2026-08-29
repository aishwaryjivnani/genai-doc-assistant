"""
The shared state object that flows through every node in the agent graph.
Each agent reads what it needs and writes its own output back into this dict.
"""
from typing import List, TypedDict


class AgentState(TypedDict, total=False):
    question: str                 # original user question
    search_query: str             # possibly rewritten query from the planner
    plan_notes: str                # planner's reasoning, shown in the trace
    retrieved_chunks: List[dict]   # [{text, source, score}, ...]
    context_sufficient: bool
    answer: str
    validation_passed: bool
    validation_feedback: str
    retries: int
    trace: List[str]              # human-readable log of what each agent did
