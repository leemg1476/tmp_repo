import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


@pytest.fixture
def mcp_app():
    from mcp_server import mcp

    # MCP 서버를 ASGI 앱으로 구성해 실제 서버 없이도 툴 호출을 테스트한다.
    return mcp.http_app(transport="streamable-http", path="/mcp")


@pytest.fixture
def mcp_client_factory(mcp_app):
    # MCP 클라이언트를 in-process ASGI 앱에 연결하는 팩토리를 제공한다.
    def _factory(headers=None, timeout=None, auth=None):
        # Provide an httpx client wired to the in-process MCP ASGI app.
        # This keeps tests fast and avoids binding to real sockets.
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mcp_app, lifespan="on"),
            base_url="http://localhost:8010",
            headers=headers,
            timeout=timeout,
            auth=auth,
            follow_redirects=True,
        )

    return _factory


@pytest.fixture
def agent_app(monkeypatch, mcp_client_factory):
    # 에이전트 앱에서 MCP 호출을 in-process로 라우팅하고 LLM을 대체한다.
    import agent
    from mcp.client.streamable_http import streamablehttp_client as real_streamablehttp_client

    def _streamablehttp_client(
        url,
        headers=None,
        timeout=30,
        sse_read_timeout=60 * 5,
        terminate_on_close=True,
        auth=None,
    ):
        return real_streamablehttp_client(
            url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            terminate_on_close=terminate_on_close,
            httpx_client_factory=mcp_client_factory,
            auth=auth,
        )

    class DummyGraph:
        async def ainvoke(self, state):
            # Bypass the LLM so the test focuses on MCP tool execution and wiring.
            tool_result = await agent.get_samsung_close_prices_tool()
            content = json.dumps(tool_result, ensure_ascii=True)
            return {"messages": state["messages"] + [AIMessage(content=content)]}

    monkeypatch.setattr(agent, "streamablehttp_client", _streamablehttp_client)
    monkeypatch.setattr(agent, "graph", DummyGraph())
    return agent.app


# MCP 서버 툴이 1주일치 종가 배열을 올바른 키로 반환하는지 확인한다.
@pytest.mark.asyncio
async def test_mcp_server_tool_returns_prices(mcp_client_factory):
    # Validate the MCP tool returns 7 daily close prices with expected keys.
    async with streamablehttp_client(
        "http://localhost:8010/mcp",
        httpx_client_factory=mcp_client_factory,
    ) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_samsung_close_prices", {})
            data = getattr(result, "content", result)

    assert isinstance(data, list)
    assert len(data) == 7
    assert all("date" in item and "close" in item for item in data)


# 에이전트 서버의 헬스 체크가 정상 응답(status=ok)을 주는지 확인한다.
@pytest.mark.asyncio
async def test_agent_health_endpoint(agent_app):
    # Ensure the agent FastAPI health endpoint responds with status=ok.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent_app, lifespan="on"),
        base_url="http://localhost:8000",
    ) as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json().get("status") == "ok"


# 에이전트 응답이 MCP 툴 결과를 JSON 문자열로 반환하는지 확인한다.
@pytest.mark.asyncio
async def test_agent_responds_with_mcp_tool(agent_app):
    # Ensure /agent/respond returns MCP tool output serialized in reply.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent_app, lifespan="on"),
        base_url="http://localhost:8000",
    ) as client:
        response = await client.post(
            "/agent/respond",
            json={"message": "Samsung Electronics last 7 days close prices"},
        )
        assert response.status_code == 200
        reply = response.json().get("reply")
        assert reply
        parsed = json.loads(reply)
        assert isinstance(parsed, list)
        assert len(parsed) == 7
