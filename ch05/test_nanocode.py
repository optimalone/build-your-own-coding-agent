import os
import tempfile
import pytest
from nanocode import (
    Agent, AgentStop, Thought, ToolCall,
    ReadFile, WriteFile, get_tool, tool_definitions, tools,
)


# --- Fake Brain for Testing ---

class FakeBrain:
    """Fake brain for testing - returns predictable responses."""

    def __init__(self, responses=None, tools=None):
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


# --- Tests from previous chapters ---

def test_quit_command_raises_agent_stop():
    """Verify /q raises AgentStop exception."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    with pytest.raises(AgentStop):
        agent.handle_input("/q")


def test_empty_input_returns_empty_string():
    """Verify empty/whitespace input returns empty string."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    assert agent.handle_input("") == ""
    assert agent.handle_input("   ") == ""


def test_handle_input_returns_brain_response():
    """Verify handle_input returns the brain's response text."""
    brain = FakeBrain(responses=[
        Thought(text="Hello!", raw_content=[{"type": "text", "text": "Hello!"}])
    ])
    agent = Agent(brain=brain, tools=tools)
    result = agent.handle_input("hi")
    assert result == "Hello!"


# --- Tool class tests ---

def test_read_file_adds_line_numbers():
    """Verify ReadFile prefixes each line with line numbers."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("line one\nline two\nline three\n")
        temp_path = f.name

    try:
        tool = ReadFile()
        result = tool.execute(temp_path)
        assert "1 | line one" in result
        assert "2 | line two" in result
        assert "3 | line three" in result
    finally:
        os.unlink(temp_path)


def test_read_file_handles_missing_file():
    """Verify ReadFile returns error for missing file."""
    tool = ReadFile()
    result = tool.execute("/nonexistent/path/file.txt")
    assert "Error" in result


def test_write_file_creates_file():
    """Verify WriteFile creates a file with content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.txt")
        tool = WriteFile()
        result = tool.execute(path, "hello world")

        assert os.path.exists(path)
        assert "Successfully wrote" in result
        assert "11 characters" in result
        with open(path) as f:
            assert f.read() == "hello world"


def test_write_file_overwrites_existing():
    """Verify WriteFile overwrites existing content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.txt")
        tool = WriteFile()

        tool.execute(path, "original content")
        tool.execute(path, "new content")

        with open(path) as f:
            assert f.read() == "new content"


def test_write_file_handles_bad_path():
    """Verify WriteFile returns error for invalid path."""
    tool = WriteFile()
    result = tool.execute("/nonexistent/path/file.txt", "content")
    assert "Error" in result


# --- Tool definition tests ---

def test_tool_has_required_attributes():
    """Verify tool classes have name, description, input_schema."""
    tool = ReadFile()
    assert tool.name == "read_file"
    assert tool.description is not None
    assert tool.input_schema is not None


def test_get_tool_finds_by_name():
    """Verify get_tool finds a tool by name."""
    tool = get_tool(tools, "read_file")
    assert tool is not None
    assert tool.name == "read_file"


def test_get_tool_returns_none_for_unknown():
    """Verify get_tool returns None for unknown tool name."""
    tool = get_tool(tools, "unknown_tool")
    assert tool is None


def test_tool_definitions_for_api():
    """Verify tool_definitions returns correct format for API."""
    defs = tool_definitions(tools)
    assert len(defs) == 2
    for d in defs:
        assert "name" in d
        assert "description" in d
        assert "input_schema" in d
        # Should not include execute method
        assert "execute" not in d


# --- Agent tool execution tests ---

def test_agent_execute_tool_finds_tool():
    """Verify agent can execute a registered tool."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    result = agent._execute_tool("read_file", {"path": __file__})
    assert "import" in result  # This file contains 'import'


def test_agent_execute_tool_unknown_tool():
    """Verify agent returns error for unknown tool."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    result = agent._execute_tool("unknown_tool", {})
    assert "not found" in result


def test_agent_tools_definitions():
    """Verify tool definitions are correctly formatted for API."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    definitions = tool_definitions(agent.tools)

    assert len(definitions) == 2
    assert definitions[0]["name"] == "read_file"
    assert "description" in definitions[0]
    assert "input_schema" in definitions[0]


# --- Agentic loop tests ---

def test_agentic_loop_executes_tool_calls():
    """Verify agentic loop executes tool calls and continues."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content\n")
        temp_path = f.name

    try:
        # Brain returns a tool call, then a final response
        brain = FakeBrain(responses=[
            Thought(
                text="Let me read that file.",
                tool_calls=[ToolCall(id="1", name="read_file", args={"path": temp_path})],
                raw_content=[
                    {"type": "text", "text": "Let me read that file."},
                    {"type": "tool_use", "id": "1", "name": "read_file", "input": {"path": temp_path}}
                ]
            ),
            Thought(
                text="The file contains test content.",
                raw_content=[{"type": "text", "text": "The file contains test content."}]
            )
        ])
        agent = Agent(brain=brain, tools=tools)
        result = agent.handle_input("Read the file")

        assert "Let me read that file." in result
        assert "The file contains test content." in result
        assert brain.call_count == 2  # Called twice (tool call + final)
    finally:
        os.unlink(temp_path)


def test_thought_stores_raw_content():
    """Verify Thought stores raw_content for message history."""
    raw = [{"type": "text", "text": "Hello"}]
    thought = Thought(text="Hello", raw_content=raw)
    assert thought.raw_content == raw


# --- Runtime regression tests ---

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
        assert headers["x-api-key"] == "offline-test-key"
        assert timeout == 120
        payloads.append(json)
        return Response()

    monkeypatch.setattr(nanocode.requests, "post", post)
    thought = nanocode.DeepSeek().think([{"role": "user", "content": "hello"}])

    assert thought.text == "hello"
    assert payloads[0]["model"] == "deepseek-flash"
    assert payloads[0]["thinking"] == {"type": "disabled"}
    assert payloads[0]["max_tokens"] == 4096
    assert payloads[0]["messages"] == [{"role": "user", "content": "hello"}]


# --- Recovery and search privacy regressions ---

class RecoveryBrain:
    last_input_tokens = 0
    context_limit = 200_000

    def __init__(self, responses):
        self.responses = iter(responses)
        self.tools = []

    def think(self, conversation):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def assert_tool_results_follow_requests(conversation):
    for index, message in enumerate(conversation):
        if message["role"] != "assistant":
            continue
        requests = [block["id"] for block in message["content"]
                    if block["type"] == "tool_use"]
        if requests:
            following = conversation[index + 1]
            assert following["role"] == "user"
            assert [block["tool_use_id"] for block in following["content"]
                    if block["type"] == "tool_result"] == requests


def test_inference_failure_preserves_completed_tool_results(tmp_path):
    import nanocode

    source = tmp_path / "source.py"
    source.write_text("preserved source\n")
    call = nanocode.ToolCall(id="read-1", name="read_file", args={"path": str(source)})
    brain = RecoveryBrain([
        nanocode.Thought(tool_calls=[call], raw_content=[
            {"type": "tool_use", "id": "read-1", "name": "read_file", "input": call.args}
        ]),
        RuntimeError("inference unavailable"),
        nanocode.Thought(text="Recovered", raw_content=[{"type": "text", "text": "Recovered"}]),
    ])
    agent = nanocode.Agent(brain=brain, tools=[nanocode.ReadFile()])

    assert "inference unavailable" in agent.handle_input("Read the source")
    assert len(agent.conversation) == 3
    assert "preserved source" in agent.conversation[2]["content"][0]["content"]
    assert_tool_results_follow_requests(agent.conversation)

    agent.handle_input("Continue")
    assert agent.conversation[-1] == {
        "role": "assistant", "content": [{"type": "text", "text": "Recovered"}]
    }
    assert "preserved source" in agent.conversation[2]["content"][0]["content"]
    assert_tool_results_follow_requests(agent.conversation)


def test_initial_inference_failure_removes_only_new_user_message():
    import nanocode

    brain = RecoveryBrain([
        nanocode.Thought(text="Hello", raw_content=[{"type": "text", "text": "Hello"}]),
        RuntimeError("inference unavailable"),
        nanocode.Thought(text="Recovered", raw_content=[{"type": "text", "text": "Recovered"}]),
    ])
    agent = nanocode.Agent(brain=brain, tools=[])
    agent.handle_input("Hello")
    previous = list(agent.conversation)

    assert "inference unavailable" in agent.handle_input("Failed turn")
    assert agent.conversation == previous
    agent.handle_input("Try again")
    assert agent.conversation[-1]["content"] == [{"type": "text", "text": "Recovered"}]


def test_tool_exception_returns_results_for_all_calls_and_session_continues(tmp_path):
    import nanocode

    class BrokenReadFile(nanocode.ReadFile):
        name = "broken_read"

        def execute(self, *args, **kwargs):
            raise RuntimeError("unexpected tool failure")

    target = tmp_path / "created.py"
    calls = [
        nanocode.ToolCall(id="write-1", name="write_file",
                          args={"path": str(target), "content": "completed write\n"}),
        nanocode.ToolCall(id="broken-1", name="broken_read", args={"path": str(target)}),
        nanocode.ToolCall(id="read-1", name="read_file", args={"path": str(target)}),
    ]
    brain = RecoveryBrain([
        nanocode.Thought(tool_calls=calls, raw_content=[
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.args}
            for call in calls
        ]),
        nanocode.Thought(text="Finished", raw_content=[{"type": "text", "text": "Finished"}]),
        nanocode.Thought(text="Continued", raw_content=[{"type": "text", "text": "Continued"}]),
    ])
    agent = nanocode.Agent(brain=brain, tools=[
        nanocode.WriteFile(), BrokenReadFile(), nanocode.ReadFile()
    ])
    agent.mode = "act"

    agent.handle_input("Write then read")
    assert target.read_text() == "completed write\n"
    assert len(agent.conversation) == 4
    assert_tool_results_follow_requests(agent.conversation)
    results = agent.conversation[2]["content"]
    assert results[1]["content"].startswith("Error:")
    assert "unexpected tool failure" in results[1]["content"]
    assert "completed write" in results[2]["content"]

    agent.handle_input("Continue")
    assert agent.conversation[-1]["content"] == [{"type": "text", "text": "Continued"}]
    assert_tool_results_follow_requests(agent.conversation)


# --- Current provider HTTP contracts ---

def test_claude_sends_adaptive_thinking_contract(monkeypatch):
    import nanocode

    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test-key")
    requests = []
    conversation = [{"role": "user", "content": "hello"}]

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"content": [{"type": "text", "text": "hello"}],
                    "usage": {"input_tokens": 12}}

    def post(url, headers, json, timeout):
        requests.append((url, headers, json, timeout))
        return Response()

    monkeypatch.setattr(nanocode.requests, "post", post)
    thought = nanocode.Claude().think(conversation)

    assert thought.text == "hello"
    assert len(requests) == 1
    url, headers, payload, timeout = requests[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers == {"x-api-key": "offline-test-key",
                       "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
    assert timeout == 120
    assert payload == {
        "model": "claude-sonnet-5-5",
        "max_tokens": 16000,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "medium"},
        "messages": conversation,
    }
    assert "budget_tokens" not in payload["thinking"]


@pytest.mark.parametrize("thinking", ["", "Read the source before answering."])
def test_tool_continuation_preserves_signed_and_redacted_thinking(monkeypatch, tmp_path, thinking):
    import copy
    import nanocode

    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test-key")
    source = tmp_path / "source.py"
    source.write_text("preserved source\n")
    content = [
        {"type": "thinking", "thinking": thinking, "signature": "signed-thinking"},
        {"type": "text", "text": "Checking the source."},
        {"type": "redacted_thinking", "data": "opaque-thinking"},
        {"type": "tool_use", "id": "read-1", "name": "read_file",
         "input": {"path": str(source)}},
    ]
    responses = iter([content, [{"type": "text", "text": "done"}]])
    requests = []

    class Response:
        status_code = 200

        def __init__(self, blocks):
            self.blocks = blocks

        def json(self):
            return {"content": self.blocks, "usage": {"input_tokens": 12}}

    def post(url, headers, json, timeout):
        requests.append(copy.deepcopy(json))
        return Response(next(responses))

    monkeypatch.setattr(nanocode.requests, "post", post)
    tools = [nanocode.ReadFile()]
    brain = nanocode.Claude(tools=nanocode.tool_definitions(tools))
    agent = nanocode.Agent(brain=brain, tools=tools)

    assert agent.handle_input("Read the source") == "Checking the source.\ndone"
    assert len(requests) == 2
    assert requests[1]["messages"][1] == {"role": "assistant", "content": content}
    assert requests[1]["messages"][2] == {
        "role": "user", "content": [{"type": "tool_result", "tool_use_id": "read-1",
                                      "content": "1 | preserved source\n"}]
    }
