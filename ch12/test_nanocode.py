"""Tests for Nanocode v1.0 (Chapter 12 - The Capstone)."""

import os
import tempfile

from nanocode import (
    Agent, AgentStop, Thought, ToolCall, Memory, ToolContext,
    ReadFile, WriteFile, WritePlan, ListFiles, SearchCodebase, SaveMemory, RunCommand, SearchWeb,
    BRAINS, tools, get_tool, tool_definitions
)


# --- Fake Brain for Testing ---

class FakeBrain:
    """Fake brain for testing - returns predictable responses."""
    context_limit = 200_000
    last_input_tokens = 0

    def __init__(self, responses=None, memory=None, tools=None):
        self.memory = memory
        self.tools = tools or []
        self.responses = responses or [Thought(text="Fake response", raw_content=[{"type": "text", "text": "Fake response"}])]
        self.call_count = 0
        self.last_conversation = None

    def think(self, conversation):
        self.last_conversation = list(conversation)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return Thought(text="No more responses", raw_content=[{"type": "text", "text": "No more responses"}])


# --- Chapter 11: SearchWeb Tool Tests ---

def test_search_web_tool_exists():
    """Verify SearchWeb class exists with required attributes."""
    tool = SearchWeb()
    assert tool.name == "search_web"
    assert tool.description is not None
    assert tool.input_schema is not None
    assert "query" in tool.input_schema["properties"]


def test_search_web_in_tools_list():
    """Verify SearchWeb is registered in the tools list."""
    tool_names = [t.name for t in tools]
    assert "search_web" in tool_names


def test_search_web_can_be_found():
    """Verify get_tool can find search_web."""
    tool = get_tool(tools, "search_web")
    assert tool is not None
    assert tool.name == "search_web"


def test_search_web_in_tool_definitions():
    """Verify search_web appears in tool definitions for API."""
    definitions = tool_definitions(tools)
    names = [d["name"] for d in definitions]
    assert "search_web" in names


def test_search_web_execute_success(monkeypatch):
    """Verify SearchWeb.execute() returns formatted results."""
    fake_results = [
        {"title": "Python 3.13", "href": "https://python.org", "body": "Latest release"},
    ]
    monkeypatch.setattr("nanocode.DDGS", lambda: type("FakeDDGS", (), {"text": lambda self, q, max_results=3: fake_results})())
    tool = SearchWeb()
    context = ToolContext()
    result = tool.execute(context, "latest python version")
    assert "Python 3.13" in result
    assert "https://python.org" in result


def test_search_web_execute_no_results(monkeypatch):
    """Verify SearchWeb.execute() handles empty results."""
    monkeypatch.setattr("nanocode.DDGS", lambda: type("FakeDDGS", (), {"text": lambda self, q, max_results=3: []})())
    tool = SearchWeb()
    context = ToolContext()
    result = tool.execute(context, "impossible query xyz")
    assert "No results found" in result


def test_search_web_execute_error(monkeypatch):
    """Verify SearchWeb.execute() handles errors gracefully."""
    def raise_error():
        raise RuntimeError("Network down")
    monkeypatch.setattr("nanocode.DDGS", lambda: type("FakeDDGS", (), {"text": lambda self, q, max_results=3: raise_error()})())
    tool = SearchWeb()
    context = ToolContext()
    result = tool.execute(context, "test query")
    assert "Error" in result


# --- Previous Chapter Tests (Cumulative) ---

def test_handle_input_returns_string():
    """Basic test: handle_input returns a string."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    result = agent.handle_input("Hello")
    assert isinstance(result, str)


def test_quit_command_raises_agent_stop():
    """Verify /q raises AgentStop."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    try:
        agent.handle_input("/q")
        assert False, "Should have raised AgentStop"
    except AgentStop:
        pass


def test_messages_accumulate():
    """Verify conversation history accumulates."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    agent.handle_input("First message")
    agent.handle_input("Second message")
    # Should have 4 messages: user, assistant, user, assistant
    assert len(agent.conversation) == 4


def test_brains_registry_has_expected_providers():
    """Verify BRAINS registry contains all providers."""
    assert "claude" in BRAINS
    assert "deepseek" in BRAINS
    assert "ollama" in BRAINS


def test_read_file_adds_line_numbers():
    """Verify ReadFile prefixes each line with line numbers."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("line one\nline two\nline three\n")
        temp_path = f.name

    try:
        tool = ReadFile()
        context = ToolContext()
        result = tool.execute(context, temp_path)
        assert "1 | line one" in result
        assert "2 | line two" in result
        assert "3 | line three" in result
    finally:
        os.unlink(temp_path)


def test_write_file_creates_file():
    """Verify WriteFile creates a file with content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.txt")
        tool = WriteFile()
        context = ToolContext()
        result = tool.execute(context, path, "hello world")

        assert os.path.exists(path)
        assert "Successfully wrote" in result
        with open(path) as f:
            assert f.read() == "hello world"


def test_plan_mode_hides_write_tools():
    """Verify plan mode removes write tools from brain's menu."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")
    tool_names = [t["name"] for t in agent.brain.tools]
    assert "write_file" not in tool_names
    assert "edit_file" not in tool_names
    assert "run_command" not in tool_names
    assert "write_plan" in tool_names
    assert "read_file" in tool_names
    assert "search_web" in tool_names


def test_act_mode_shows_all_tools():
    """Verify act mode includes all tools in brain's menu."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="act")
    tool_names = [t["name"] for t in agent.brain.tools]
    assert "write_file" in tool_names
    assert "edit_file" in tool_names
    assert "run_command" in tool_names


def test_mode_switch_updates_brain_tools():
    """Verify switching mode updates the brain's tool menu."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")
    plan_names = [t["name"] for t in agent.brain.tools]
    assert "write_file" not in plan_names
    agent.handle_input("/mode act")
    act_names = [t["name"] for t in agent.brain.tools]
    assert "write_file" in act_names


def test_write_plan_saves_file():
    """Verify WritePlan creates PLAN.md with content."""
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            tool = WritePlan()
            context = ToolContext()
            result = tool.execute(context, content="# My Plan")
            assert "Plan saved" in result
        finally:
            os.chdir(original_dir)


def test_write_file_writes_file():
    """Verify WriteFile creates a file with content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.py")
        tool = WriteFile()
        context = ToolContext()
        result = tool.execute(context, file_path, "print('hello')")
        assert "Successfully wrote" in result


def test_run_command_executes():
    """Verify RunCommand works."""
    tool = RunCommand()
    context = ToolContext()
    result = tool.execute(context, "echo hello")
    assert "hello" in result


def test_ollama_in_brains_registry():
    """Verify Ollama is in the BRAINS registry."""
    assert "ollama" in BRAINS


def test_version_is_1_0():
    """Verify version number is 1.0 in main()."""
    import nanocode
    import inspect
    source = inspect.getsource(nanocode.main)
    assert "v1.0" in source


# --- Context Compaction Tests ---

def test_compact_conversation_summarizes():
    """Verify compaction replaces conversation with summary."""
    summary = Thought(
        text="Summary of conversation",
        raw_content=[{"type": "text", "text": "Summary of conversation"}]
    )
    responses = [
        Thought(text="Response 1", raw_content=[{"type": "text", "text": "Response 1"}]),
        summary,
    ]
    brain = FakeBrain(responses=responses)
    brain.last_input_tokens = 200_000  # Over 75% threshold
    agent = Agent(brain=brain, tools=tools, mode="act")

    agent.handle_input("Do something")

    # After compaction, conversation should be short (summary + current response)
    assert len(agent.conversation) <= 4
    assert "Summary" in str(agent.conversation[0]["content"])


# --- Runtime regression tests ---

import pytest
from nanocode import Ollama

def test_deepseek_sends_supported_model_with_thinking_disabled(monkeypatch):
    import nanocode

    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")
    payloads = []

    class Response:
        status_code = 200

        def json(self):
            return {"content": [{"type": "text", "text": "hello"}],
                    "usage": {"input_tokens": 12}}

    def post(url, headers, json, timeout):
        assert url == "https://api.deepseek.com/anthropic/v1/messages"
        payloads.append(json)
        return Response()

    monkeypatch.setattr(nanocode.requests, "post", post)
    thought = nanocode.DeepSeek().think([{"role": "user", "content": "hello"}])

    assert thought.text == "hello"
    assert payloads[0]["model"] == "deepseek-v4-pro"
    assert payloads[0]["thinking"] == {"type": "disabled"}


def test_plan_dispatch_refuses_write_requested_by_brain(tmp_path):
    path = tmp_path / "forbidden.py"
    call = ToolCall("write-1", "write_file", {"path": str(path), "content": "unsafe"})
    brain = FakeBrain(responses=[
        Thought(tool_calls=[call], raw_content=[{
            "type": "tool_use", "id": call.id, "name": call.name, "input": call.args,
        }]),
        Thought(text="done", raw_content=[{"type": "text", "text": "done"}]),
    ])
    agent = Agent(brain=brain, tools=tools, mode="plan")

    agent.handle_input("write this file")

    assert not path.exists()
    result = agent.conversation[2]["content"][0]["content"]
    assert "Error" in result
    assert "plan mode" in result.lower()


def test_act_dispatch_allows_write(tmp_path):
    path = tmp_path / "allowed.py"
    agent = Agent(brain=FakeBrain(), tools=tools, mode="act")

    agent._execute_tool("write_file", {"path": str(path), "content": "allowed"})

    assert path.read_text() == "allowed"


def test_saved_memory_refreshes_next_model_request(tmp_path):
    memory = Memory(path=str(tmp_path / ".nanocode" / "memory.md"))
    call = ToolCall("memory-1", "save_memory", {"content": "Remember the blue project"})
    seen_systems = []

    class RecordingBrain(FakeBrain):
        def think(self, conversation):
            seen_systems.append(self.system)
            return super().think(conversation)

    brain = RecordingBrain(responses=[
        Thought(tool_calls=[call], raw_content=[{
            "type": "tool_use", "id": call.id, "name": call.name, "input": call.args,
        }]),
        Thought(text="saved", raw_content=[{"type": "text", "text": "saved"}]),
        Thought(text="next turn", raw_content=[{"type": "text", "text": "next turn"}]),
    ])
    agent = Agent(brain=brain, tools=tools, memory=memory, mode="plan")

    agent.handle_input("remember this")
    agent.handle_input("what do you remember?")

    assert "Remember the blue project" not in seen_systems[0]
    assert all("Remember the blue project" in system for system in seen_systems[1:])
    assert all("PLAN mode" in system for system in seen_systems)
    assert __import__("pathlib").Path(memory.path).read_text() == "Remember the blue project"


@pytest.mark.parametrize("name,args", [
    ("edit_file", {"path": "unused", "old_text": "x", "new_text": "y"}),
    ("run_command", {"command": "exit 0"}),
])
def test_plan_dispatch_refuses_act_only_tools(name, args, monkeypatch):
    agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")
    tool = get_tool(agent.tools, name)

    def forbidden(*args, **kwargs):
        pytest.fail("act-only tool executed in plan mode")

    monkeypatch.setattr(tool, "execute", forbidden)
    result = agent._execute_tool(name, args)
    assert "Error" in result
    assert "plan mode" in result.lower()


@pytest.mark.parametrize("status", [0, 7])
def test_run_command_reports_silent_exit_status(status):
    result = RunCommand().execute(ToolContext(), command=f"exit {status}")
    assert f"Exit code: {status}" in result


def test_run_command_keeps_output_with_failure_status():
    result = RunCommand().execute(
        ToolContext(), command="printf success; printf failure >&2; exit 3"
    )
    assert "Exit code: 3" in result
    assert "STDOUT:\nsuccess" in result
    assert "STDERR:\nfailure" in result


def test_run_command_timeout_still_reports_error(monkeypatch):
    import nanocode

    def timeout(*args, **kwargs):
        raise nanocode.subprocess.TimeoutExpired("slow", 1)

    monkeypatch.setattr(nanocode.subprocess, "run", timeout)
    assert "timed out" in RunCommand().execute(ToolContext(), "slow")


def test_compaction_refreshes_saved_memory(tmp_path):
    memory = Memory(path=str(tmp_path / ".nanocode" / "memory.md"))
    seen_systems = []

    class RecordingBrain(FakeBrain):
        def think(self, conversation):
            seen_systems.append(self.system)
            return Thought(text="summary")

    brain = RecordingBrain()
    agent = Agent(brain=brain, tools=tools, memory=memory)
    agent._execute_tool("save_memory", {"content": "new memory for summary"})
    agent.conversation = [{"role": "user", "content": "prior work"}]

    agent._compact_conversation()

    assert "new memory for summary" in seen_systems[0]
    assert "PLAN mode" in seen_systems[0]


def test_ollama_constructor_needs_no_server(monkeypatch):
    import nanocode

    def unexpected(*args, **kwargs):
        pytest.fail("Ollama constructor contacted the server")

    monkeypatch.setattr(nanocode.requests, "get", unexpected)
    monkeypatch.setattr(nanocode.requests, "post", unexpected)
    Ollama()


@pytest.mark.parametrize("configured,loaded", [
    ("qwen3-coder:30b", "qwen3-coder:30b"),
    ("my-model", "my-model:latest"),
    ("registry.local:5000/team/model", "registry.local:5000/team/model:latest"),
])
def test_ollama_uses_loaded_context_before_inference(monkeypatch, configured, loaded):
    import nanocode

    monkeypatch.setenv("OLLAMA_MODEL", configured)
    requests_seen = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [
                {"name": "unrelated:latest", "model": "unrelated:latest",
                 "context_length": 131072},
                {"name": loaded, "model": loaded, "context_length": 8192},
            ]}

    def get(url, timeout):
        assert url == "http://localhost:11434/api/ps"
        requests_seen.append("ps")
        return Response()

    brain = Ollama()

    def infer(url, headers, payload):
        assert requests_seen[-1] == "ps"
        assert brain.context_limit == 8192
        assert "options" not in payload
        requests_seen.append("messages")
        return type("MessageResponse", (), {"json": lambda self: {
            "content": [{"type": "text", "text": "answer"}],
            "usage": {"input_tokens": 7000},
        }})()

    monkeypatch.setattr(nanocode.requests, "get", get)
    monkeypatch.setattr(nanocode, "request_with_retry", infer)
    agent = Agent(brain=brain, tools=tools, brain_name="ollama")

    agent.handle_input("hello")

    # 7000 tokens exceeds 75% of the served 8192 window, so summarize.
    assert requests_seen == ["ps", "messages", "ps", "messages"]
    assert "Previous conversation summary: answer" in agent.conversation[0]["content"]


@pytest.mark.parametrize("models", [
    [],
    [{"name": "qwen3-coder:7b", "context_length": 8192}],
    [{"name": "qwen3-coder:30b-extra", "context_length": 8192}],
    [{"name": "qwen3-coder:30b"}],
    [{"name": "qwen3-coder:30b", "context_length": None}],
    [{"name": "qwen3-coder:30b", "context_length": 0}],
    [{"name": "qwen3-coder:30b", "context_length": -1}],
    [{"name": "qwen3-coder:30b", "context_length": "8192"}],
    [{"name": "qwen3-coder:30b", "context_length": True}],
    None,
])
def test_ollama_refuses_unknown_effective_context(monkeypatch, models):
    import nanocode

    monkeypatch.setenv("OLLAMA_MODEL", "qwen3-coder:30b")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": models}

    monkeypatch.setattr(nanocode.requests, "get", lambda *args, **kwargs: Response())
    brain = Ollama()

    def unexpected(*args, **kwargs):
        pytest.fail("inference sent without validated loaded context")

    monkeypatch.setattr(nanocode, "request_with_retry", unexpected)
    with pytest.raises(ValueError, match="(?i)(load|context)"):
        brain.think([{"role": "user", "content": "hello"}])


@pytest.mark.parametrize("failure", ["http", "connection", "json"])
def test_ollama_metadata_failure_blocks_inference(monkeypatch, failure):
    import nanocode

    class Response:
        def raise_for_status(self):
            if failure == "http":
                raise nanocode.requests.HTTPError("server failure")

        def json(self):
            raise ValueError("invalid JSON")

    def get(*args, **kwargs):
        if failure == "connection":
            raise nanocode.requests.ConnectionError("offline")
        return Response()

    monkeypatch.setattr(nanocode.requests, "get", get)
    brain = Ollama()

    def unexpected(*args, **kwargs):
        pytest.fail("inference sent after failed metadata lookup")

    monkeypatch.setattr(nanocode, "request_with_retry", unexpected)
    with pytest.raises((ValueError, RuntimeError), match="(?i)(ollama|context|load)"):
        brain.think([{"role": "user", "content": "hello"}])
