import os
import tempfile
import pytest
from nanocode import (
    Agent, AgentStop, Thought, ToolCall, ToolContext, Memory,
    ReadFile, WriteFile, SaveMemory, get_tool, tool_definitions, tools,
)


# --- Fake Brain for Testing ---

class FakeBrain:
    """Fake brain for testing - returns predictable responses."""

    def __init__(self, responses=None, memory=None, tools=None):
        self.memory = memory
        self.tools = tools or []
        self.responses = responses or [Thought(text="Fake response", raw_content=[{"type": "text", "text": "Fake response"}])]
        self.call_count = 0

    def think(self, conversation):
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


# --- Memory class tests ---

def test_memory_creates_default_file():
    """Verify Memory creates file with default content if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "memory.md")
        memory = Memory(path=path)

        assert os.path.exists(path)
        assert "Nanocode" in memory.content


def test_memory_loads_existing_content():
    """Verify Memory loads content from existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "memory.md")

        # Pre-create file with custom content
        with open(path, 'w') as f:
            f.write("Custom memory content")

        memory = Memory(path=path)
        assert memory.content == "Custom memory content"


def test_memory_save_updates_content_and_file():
    """Verify Memory.save() updates both content and file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "memory.md")
        memory = Memory(path=path)

        memory.save("New content")

        assert memory.content == "New content"
        with open(path) as f:
            assert f.read() == "New content"


# --- SaveMemory tool tests ---

def test_save_memory_updates_memory():
    """Verify SaveMemory updates the Memory object."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = Memory(path=os.path.join(tmpdir, "memory.md"))
        tool = SaveMemory()
        context = ToolContext(memory=memory)

        result = tool.execute(context, "Updated preferences")

        assert "successfully" in result.lower()
        assert memory.content == "Updated preferences"


def test_save_memory_fails_without_memory():
    """Verify SaveMemory returns error when memory is None."""
    tool = SaveMemory()
    context = ToolContext(memory=None)
    result = tool.execute(context, "test")
    assert "Error" in result


# --- Agent with Memory tests ---

def test_agent_has_save_memory_tool():
    """Verify agent has save_memory tool."""
    agent = Agent(brain=FakeBrain(), tools=tools)

    tool_names = [t.name for t in agent.tools]
    assert "save_memory" in tool_names


def test_agent_execute_save_memory_tool():
    """Verify save_memory tool updates the Memory object through agent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = Memory(path=os.path.join(tmpdir, "memory.md"))
        agent = Agent(brain=FakeBrain(), tools=tools, memory=memory)

        result = agent._execute_tool("save_memory", {"content": "Updated preferences"})

        assert "successfully" in result.lower()
        assert memory.content == "Updated preferences"


def test_brain_receives_memory_content():
    """Verify brain has access to memory content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = Memory(path=os.path.join(tmpdir, "memory.md"))
        memory.save("Custom system prompt")

        brain = FakeBrain(memory=memory)
        assert brain.memory.content == "Custom system prompt"


# --- Tool execution tests ---

def test_agent_execute_tool_finds_tool():
    """Verify agent can execute a registered tool."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    result = agent._execute_tool("read_file", {"path": __file__})
    assert "import" in result


def test_agent_execute_tool_unknown_tool():
    """Verify agent returns error for unknown tool."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    result = agent._execute_tool("unknown_tool", {})
    assert "not found" in result


def test_agent_tools_definitions():
    """Verify tool definitions include all tools."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    definitions = tool_definitions(agent.tools)

    tool_names = [d["name"] for d in definitions]
    assert "save_memory" in tool_names
    assert "read_file" in tool_names
    assert "write_file" in tool_names
