from datetime import date, timedelta

from fastmcp import FastMCP


mcp = FastMCP("mock-mcp")


@mcp.tool()
def get_samsung_close_prices() -> list[dict]:
    today = date.today()
    base_close = 70000
    step = 150
    items: list[dict] = []
    for i in range(7):
        current_day = today - timedelta(days=i)
        close = base_close - (i * step)
        items.append({"date": current_day.isoformat(), "close": close})
    return list(reversed(items))


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8010,
        path="/mcp",
    )
