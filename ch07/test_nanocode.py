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
        f.write("line one\nline two\n")
        temp_path = f.name

    try:
        tool = ReadFile()
        context = ToolContext()
        result = tool.execute(context, temp_path)
        assert "1 | line one" in result
        assert "2 | line two" in result
    finally:
        os.unlink(temp_path)


# --- Mode tests ---

def test_agent_defaults_to_plan_mode():
    """Verify agent starts in plan mode by default."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    assert agent.mode == "plan"


def test_agent_can_start_in_act_mode():
    """Verify agent can be initialized in act mode."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="act")
    assert agent.mode == "act"


def test_mode_command_switches_to_act():
    """Verify /mode act switches to act mode."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")
    result = agent.handle_input("/mode act")

    assert agent.mode == "act"
    assert "ACT" in result


def test_mode_command_switches_to_plan():
    """Verify /mode plan switches to plan mode."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="act")
    result = agent.handle_input("/mode plan")

    assert agent.mode == "plan"
    assert "PLAN" in result


def test_mode_command_defaults_to_plan():
    """Verify /mode without argument defaults to plan mode."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="act")
    result = agent.handle_input("/mode")

    assert agent.mode == "plan"


# --- Write file with mode tests (using tool directly) ---

def test_write_file_blocked_in_plan_mode():
    """Verify WriteFile is blocked for non-PLAN.md files in plan mode."""
    tool = WriteFile()
    context = ToolContext(mode="plan")
    result = tool.execute(context, path="test.py", content="hello")

    assert "BLOCKED" in result
    assert "plan mode" in result


def test_write_file_allowed_to_plan_md_in_plan_mode():
    """Verify WriteFile allows PLAN.md even in plan mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = os.path.join(tmpdir, "PLAN.md")
        tool = WriteFile()
        context = ToolContext(mode="plan")

        result = tool.execute(context, plan_path, "# My Plan")

        assert "Plan saved" in result
        assert os.path.exists(plan_path)
        with open(plan_path) as f:
            assert f.read() == "# My Plan"


def test_write_file_allowed_in_act_mode():
    """Verify WriteFile works for any file in act mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.py")
        tool = WriteFile()
        context = ToolContext(mode="act")

        result = tool.execute(context, file_path, "print('hello')")

        assert "Successfully wrote" in result
        assert os.path.exists(file_path)


# --- Write file with mode tests (through agent) ---

def test_agent_write_file_blocked_in_plan_mode():
    """Verify write_file is blocked for non-PLAN.md files in plan mode."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")
    result = agent._execute_tool("write_file", {"path": "test.py", "content": "hello"})

    assert "BLOCKED" in result
    assert "plan mode" in result


def test_agent_write_file_allowed_to_plan_md_in_plan_mode():
    """Verify write_file allows PLAN.md even in plan mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = os.path.join(tmpdir, "PLAN.md")
        agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")

        result = agent._execute_tool("write_file", {"path": plan_path, "content": "# My Plan"})

        assert "Plan saved" in result
        assert os.path.exists(plan_path)
        with open(plan_path) as f:
            assert f.read() == "# My Plan"


def test_agent_write_file_allowed_in_act_mode():
    """Verify write_file works for any file in act mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.py")
        agent = Agent(brain=FakeBrain(), tools=tools, mode="act")

        result = agent._execute_tool("write_file", {"path": file_path, "content": "print('hello')"})

        assert "Successfully wrote" in result
        assert os.path.exists(file_path)


def test_mode_switch_affects_write_file():
    """Verify changing mode affects write_file behavior."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.py")
        agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")

        # Should be blocked in plan mode
        result1 = agent._execute_tool("write_file", {"path": file_path, "content": "v1"})
        assert "BLOCKED" in result1

        # Switch to act mode
        agent.handle_input("/mode act")

        # Should work in act mode
        result2 = agent._execute_tool("write_file", {"path": file_path, "content": "v2"})
        assert "Successfully wrote" in result2


# --- Agent has all tools ---

def test_agent_has_all_tools():
    """Verify agent has all tools including write_file and save_memory."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    tool_names = [t.name for t in agent.tools]
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "save_memory" in tool_names
