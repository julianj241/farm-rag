"""MCP server exposing the farm weather functions as tools over stdio."""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from weather import format_weather
from weather import get_forecast as fetch_forecast
from weather import get_historical as fetch_historical

app = Server("farm-weather")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_forecast",
            description=(
                "Get the daily weather forecast for El Cajon, CA for the next N days "
                "(1-14). Use this for upcoming weather conditions when making "
                "forward-looking recommendations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 14,
                        "default": 3,
                        "description": "Number of forecast days (1-14).",
                    }
                },
            },
        ),
        Tool(
            name="get_historical",
            description=(
                "Get historical daily weather for El Cajon, CA between two YYYY-MM-DD "
                "dates. Use this to look up past weather conditions during specific "
                "log periods."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)."},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)."},
                },
                "required": ["start_date", "end_date"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_forecast":
        days = int(arguments.get("days", 3))
        if not 1 <= days <= 14:
            raise ValueError("days must be between 1 and 14")
        data = fetch_forecast(days=days)
        return [TextContent(type="text", text=format_weather(data))]

    if name == "get_historical":
        start_date = arguments.get("start_date")
        end_date = arguments.get("end_date")
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required (YYYY-MM-DD)")
        data = fetch_historical(start_date, end_date)
        return [TextContent(type="text", text=format_weather(data))]

    raise ValueError(f"Unknown tool: {name}")


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
