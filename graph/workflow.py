"""
Workflow assembly using a Pydantic state model.

Linear POC flow:
    business_profiler -> hazard_identifier -> loss_predictor -> coverage_designer

Key details:
- The graph's state type is a Pydantic model (BaseModel) to satisfy LangGraph.
- Node wrappers convert the Pydantic state <-> dict so node implementations
  can stay simple and return dictionaries.
- A small adapter lets callers pass either a dict or a WorkflowState into
  `.invoke(...)` and always get a plain dict back.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field
from langgraph.graph import END, StateGraph


class WorkflowState(BaseModel):
    """
    Pydantic state carried through the graph. Extra keys are allowed so nodes can
    attach scratch data (e.g., rationales) without strict schema churn.
    """

    model_config = ConfigDict(extra="allow")

    request: Dict[str, Any] = Field(default_factory=dict)
    profile: Optional[Dict[str, Any]] = None
    hazard_scores: Optional[Dict[str, Any]] = None
    loss_estimates: Optional[Dict[str, Any]] = None
    recommendation: Optional[Dict[str, Any]] = None


def build_workflow(*, llm: Any):
    """
    Build and compile the underwriting workflow graph.
    """
    # Local imports to avoid import errors before nodes exist.
    from .nodes.business_profiler import run as bp_run
    from .nodes.hazard_identifier import run as hz_run
    from .nodes.loss_predictor import run as ls_run
    from .nodes.coverage_designer import run as cd_run

    # ---- Node wrappers: WorkflowState -> dict -> WorkflowState ----

    def _bp_node(state: WorkflowState) -> WorkflowState:
        new_state = bp_run(state=state.model_dump(), llm=llm)
        return WorkflowState.model_validate(new_state)

    def _hz_node(state: WorkflowState) -> WorkflowState:
        new_state = hz_run(state=state.model_dump(), llm=llm)
        return WorkflowState.model_validate(new_state)

    def _ls_node(state: WorkflowState) -> WorkflowState:
        new_state = ls_run(state=state.model_dump())
        return WorkflowState.model_validate(new_state)

    def _cd_node(state: WorkflowState) -> WorkflowState:
        new_state = cd_run(state=state.model_dump(), llm=llm)
        return WorkflowState.model_validate(new_state)

    graph = StateGraph(WorkflowState)
    graph.add_node("business_profiler", _bp_node)
    graph.add_node("hazard_identifier", _hz_node)
    graph.add_node("loss_predictor", _ls_node)
    graph.add_node("coverage_designer", _cd_node)

    graph.set_entry_point("business_profiler")
    graph.add_edge("business_profiler", "hazard_identifier")
    graph.add_edge("hazard_identifier", "loss_predictor")
    graph.add_edge("loss_predictor", "coverage_designer")
    graph.add_edge("coverage_designer", END)

    compiled = graph.compile()

    class _CompiledWorkflowAdapter:
        """
        Small adapter to make `.invoke(...)` ergonomic:
        - Accepts dict OR WorkflowState
        - Always returns a plain dict
        """

        def __init__(self, inner):
            self._inner = inner

        def invoke(self, input_state: Any) -> Dict[str, Any]:
            if isinstance(input_state, WorkflowState):
                state = input_state
            else:
                state = WorkflowState.model_validate(input_state)
            out = self._inner.invoke(state)
            # `out` is a WorkflowState; normalize to dict for API layer.
            if isinstance(out, WorkflowState):
                return out.model_dump()
            return out

    return _CompiledWorkflowAdapter(compiled)
