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
