import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from carlinpedia.mcp_server import build_server


@pytest.fixture
def server(ingested, embedder, project):
    return build_server(ingested, embedder, project)


def call(server, name, args):
    result = asyncio.run(server.call_tool(name, args))
    return json.loads(result.content[0].text)


def test_server_exposes_tools_and_prompt(server):
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert tools == {"search_passages", "get_passage", "get_bit", "list_works", "list_themes", "set_favorite"}
    prompts = asyncio.run(server.list_prompts())
    assert [p.name for p in prompts] == ["carlin_on"]


def test_search_passages_tool(server):
    payload = call(server, "search_passages", {"query": "storage unit stuff", "limit": 3})
    assert payload["count"] == len(payload["results"]) <= 3
    assert payload["results"][0]["bit"]["title"] == "Stuff"


def test_search_passages_with_filters(server):
    payload = call(server, "search_passages", {"query": "vote", "themes": ["Voting and democracy"], "kinds": ["album"]})
    assert {r["bit"]["title"] for r in payload["results"]} == {"Voting"}


def test_tool_errors_reach_the_model_as_messages(server):
    with pytest.raises(ToolError, match="Unknown theme"):
        asyncio.run(server.call_tool("search_passages", {"query": "x", "themes": ["nope"]}))
    with pytest.raises(ToolError, match="No passage with id 424242"):
        asyncio.run(server.call_tool("get_passage", {"passage_id": 424242}))
    with pytest.raises(ToolError, match="empty"):
        asyncio.run(server.call_tool("search_passages", {"query": " "}))


def test_set_favorite_then_filter(server):
    pid = call(server, "search_passages", {"query": "storage unit stuff", "limit": 1})["results"][0]["passage_id"]
    assert call(server, "set_favorite", {"passage_id": pid}) == {"passage_id": pid, "favorite": True}
    favorites = call(server, "search_passages", {"query": "stuff", "favorites_only": True})
    assert [r["passage_id"] for r in favorites["results"]] == [pid]


def test_list_tools_and_get_bit(server):
    works = call(server, "list_works", {})["works"]
    assert [w["slug"] for w in works] == ["test-special-1990", "test-album-1992"]
    themes = call(server, "list_themes", {})
    assert any(t["slug"] == "language.euphemism" for t in themes["themes"])
    passage = call(server, "search_passages", {"query": "vote", "limit": 1})["results"][0]
    bit = call(server, "get_bit", {"bit_id": passage["bit"]["id"]})
    assert bit["title"] == passage["bit"]["title"]


def test_carlin_on_prompt_embeds_the_story(server):
    result = asyncio.run(server.get_prompt("carlin_on", {"news": "Airline adds a legroom fee"}))
    text = result.messages[0].content.text
    assert "Airline adds a legroom fee" in text and "verbatim" in text
