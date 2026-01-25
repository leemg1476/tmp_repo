from typing import TypedDict, List, Any

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="Basic Agent Server")


class AgentRequest(BaseModel):
    message: str


class AgentResponse(BaseModel):
    reply: str


class AgentState(TypedDict):
    messages: List[BaseMessage]


@tool
async def get_samsung_close_prices_tool() -> Any:
    """Mock MCP tool: return Samsung Electronics close prices for the last 7 days."""
    async with streamablehttp_client("http://localhost:8010/mcp") as (
        read,
        write,
        _get_session_id,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_samsung_close_prices", {})
            return getattr(result, "content", result)


def build_graph() -> StateGraph:
    model = ChatOpenAI(model="gpt-5-mini")
    model_with_tools = model.bind_tools([get_samsung_close_prices_tool])

    async def call_model(state: AgentState) -> AgentState:
        response = await model_with_tools.ainvoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    tool_node = ToolNode([get_samsung_close_prices_tool])

    graph = StateGraph(AgentState)
    graph.add_node("chat", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    return graph.compile()


graph = build_graph()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/agent/respond", response_model=AgentResponse)
async def respond(payload: AgentRequest) -> AgentResponse:
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=payload.message)]}
    )
    last_message = result["messages"][-1]
    if isinstance(last_message, AIMessage):
        reply = last_message.content
    else:
        reply = str(last_message.content)
    return AgentResponse(reply=reply)
