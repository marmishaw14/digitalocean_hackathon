from gradient.types.chat.completion_create_response import CompletionCreateResponse
import os
from typing import Any, TypedDict, cast
from gradient import AsyncGradient
from gradient_adk import entrypoint, RequestContext
from langgraph.graph.state import CompiledStateGraph, StateGraph


class State(TypedDict):
    input: str
    output: str | None


async def llm_call(state: State) -> State:
    inference_client: AsyncGradient = AsyncGradient(
        model_access_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY")
    )

    resp: CompletionCreateResponse = await inference_client.chat.completions.create(
        messages=[{"role": "user", "content": state["input"]}],
        model="openai-gpt-oss-120b",
    )

    content = cast(Any, resp).choices[0].message.content
    state["output"] = cast(str, content)
    return state


@entrypoint
async def main(input: dict[str, Any], context: RequestContext) -> str:
    initial_state: State = {"input": str(input.get("prompt", "")), "output": None}

    graph: StateGraph[State, None, State, State] = StateGraph[
        State, None, State, State
    ](state_schema=State)
    graph.add_node("llm_call", llm_call)
    graph.set_entry_point(key="llm_call")

    app: CompiledStateGraph[State, None, State, State] = graph.compile()

    result_any: dict[str, Any] | Any = await app.ainvoke(input=initial_state)
    result: State = cast(State, result_any)

    return result["output"] or ""
