import os
import tempfile
import pytest
from nanocode import (
    Agent, AgentStop, Thought, ToolCall, ToolContext, Memory,
    ReadFile, WriteFile, WritePlan, EditFile, ListFiles, SearchCodebase, SaveMemory, RunCommand,
    get_tool, tool_definitions, tools,
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


def test_read_file_adds_line_numbers():
    """Verify ReadFile prefixes each line with line numbers."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("line one\nline two\n")
        temp_path = f.name

    try:
        tool = ReadFile()
        context = ToolContext()
        result = tool.execute(context, path=temp_path)
        assert "1 | line one" in result
        assert "2 | line two" in result
    finally:
        os.unlink(temp_path)


def test_mode_command_switches_to_act():
    """Verify /mode act switches to act mode."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")
    agent.handle_input("/mode act")
    assert agent.mode == "act"


def test_write_file_writes_file():
    """Verify WriteFile writes content to the given path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.py")
        tool = WriteFile()
        context = ToolContext()
        result = tool.execute(context, file_path, "print('hello')")
        assert "Successfully wrote" in result
        assert os.path.exists(file_path)


def test_write_plan_saves_file():
    """Verify WritePlan writes to PLAN.md."""
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            tool = WritePlan()
            context = ToolContext()
            result = tool.execute(context, content="# My Plan")
            assert "Plan saved" in result
            assert os.path.exists("PLAN.md")
        finally:
            os.chdir(original_dir)


# --- Tool filtering by mode ---

def test_plan_mode_hides_write_tools():
    """Verify plan mode does not expose write tools to the brain."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")
    tool_names = [t["name"] for t in agent.brain.tools]

    assert "write_file" not in tool_names
    assert "edit_file" not in tool_names
    assert "run_command" not in tool_names
    assert "write_plan" in tool_names
    assert "read_file" in tool_names


def test_act_mode_shows_all_tools():
    """Verify act mode exposes all tools to the brain."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="act")
    tool_names = [t["name"] for t in agent.brain.tools]

    assert "write_file" in tool_names
    assert "edit_file" in tool_names
    assert "run_command" in tool_names
    assert "write_plan" in tool_names


def test_mode_switch_updates_brain_tools():
    """Verify switching mode changes the brain's tool menu."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")

    plan_names = [t["name"] for t in agent.brain.tools]
    assert "write_file" not in plan_names
    assert "edit_file" not in plan_names
    assert "run_command" not in plan_names

    agent.handle_input("/mode act")
    act_names = [t["name"] for t in agent.brain.tools]
    assert "write_file" in act_names
    assert "edit_file" in act_names
    assert "run_command" in act_names


# --- New tests for Chapter 8: Awareness tools ---

def test_list_files_returns_file_tree():
    """Verify ListFiles returns a tree structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        os.makedirs(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "README.md"), 'w') as f:
            f.write("# Test")
        with open(os.path.join(tmpdir, "src", "main.py"), 'w') as f:
            f.write("print('hello')")

        tool = ListFiles()
        context = ToolContext()
        result = tool.execute(context, path=tmpdir)

        assert "README.md" in result
        assert "src/" in result
        assert "main.py" in result


def test_list_files_skips_git_and_pycache():
    """Verify ListFiles skips .git and __pycache__ directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directories that should be skipped
        os.makedirs(os.path.join(tmpdir, ".git"))
        os.makedirs(os.path.join(tmpdir, "__pycache__"))
        os.makedirs(os.path.join(tmpdir, "src"))

        with open(os.path.join(tmpdir, ".git", "config"), 'w') as f:
            f.write("git config")
        with open(os.path.join(tmpdir, "__pycache__", "cache.pyc"), 'w') as f:
            f.write("cache")
        with open(os.path.join(tmpdir, "src", "main.py"), 'w') as f:
            f.write("print('hello')")

        tool = ListFiles()
        context = ToolContext()
        result = tool.execute(context, path=tmpdir)

        assert "config" not in result
        assert "cache.pyc" not in result
        assert "main.py" in result


def test_search_codebase_finds_matches():
    """Verify SearchCodebase finds text in files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test.py"), 'w') as f:
            f.write("def hello_world():\n    print('hello')\n")

        tool = SearchCodebase()
        context = ToolContext()
        result = tool.execute(context, query="hello_world", path=tmpdir)

        assert "test.py" in result
        assert "hello_world" in result
        assert ":1:" in result  # Line number


def test_search_codebase_case_insensitive():
    """Verify SearchCodebase is case-insensitive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test.py"), 'w') as f:
            f.write("def HelloWorld():\n    pass\n")

        tool = SearchCodebase()
        context = ToolContext()
        result = tool.execute(context, query="helloworld", path=tmpdir)
        assert "HelloWorld" in result


def test_search_codebase_no_matches():
    """Verify SearchCodebase returns message when no matches found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test.py"), 'w') as f:
            f.write("def foo():\n    pass\n")

        tool = SearchCodebase()
        context = ToolContext()
        result = tool.execute(context, query="nonexistent_function", path=tmpdir)
        assert "No matches found" in result


def test_search_codebase_skips_git():
    """Verify SearchCodebase skips .git directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, ".git"))
        with open(os.path.join(tmpdir, ".git", "config"), 'w') as f:
            f.write("unique_search_term")
        with open(os.path.join(tmpdir, "main.py"), 'w') as f:
            f.write("print('hello')")

        tool = SearchCodebase()
        context = ToolContext()
        result = tool.execute(context, query="unique_search_term", path=tmpdir)
        assert "No matches found" in result


# --- Agent has new tools ---

def test_agent_has_list_files_tool():
    """Verify agent has list_files tool."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    tool_names = [t.name for t in agent.tools]
    assert "list_files" in tool_names


def test_agent_has_search_codebase_tool():
    """Verify agent has search_codebase tool."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    tool_names = [t.name for t in agent.tools]
    assert "search_codebase" in tool_names


def test_agent_execute_list_files():
    """Verify agent can execute list_files tool."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test.txt"), 'w') as f:
            f.write("hello")

        agent = Agent(brain=FakeBrain(), tools=tools)
        result = agent._execute_tool("list_files", {"path": tmpdir})

        assert "test.txt" in result


def test_agent_execute_search_codebase():
    """Verify agent can execute search_codebase tool."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "code.py"), 'w') as f:
            f.write("def my_function():\n    pass\n")

        agent = Agent(brain=FakeBrain(), tools=tools)
        result = agent._execute_tool("search_codebase", {"query": "my_function", "path": tmpdir})

        assert "my_function" in result
        assert "code.py" in result


# --- New tests for Chapter 9: RunCommand tool ---

def test_run_command_executes():
    """Verify run_command executes a command."""
    tool = RunCommand()
    context = ToolContext()
    result = tool.execute(context, command="echo hello")

    assert "STDOUT" in result
    assert "hello" in result


def test_run_command_captures_stderr():
    """Verify run_command captures error output."""
    tool = RunCommand()
    context = ToolContext()
    result = tool.execute(context, command="python -c \"import sys; sys.stderr.write('error!')\"")

    assert "STDERR" in result
    assert "error!" in result


def test_run_command_handles_nonexistent_command():
    """Verify run_command handles commands that don't exist."""
    tool = RunCommand()
    context = ToolContext()
    result = tool.execute(context, command="nonexistent_command_xyz_12345")

    # Should have some error output (either STDERR or Error message)
    assert "STDERR" in result or "Error" in result or "not found" in result.lower()


def test_run_command_runs_python():
    """Verify run_command can run Python scripts."""
    tool = RunCommand()
    context = ToolContext()
    result = tool.execute(context, command="python -c \"print('hello from python')\"")

    assert "hello from python" in result


def test_agent_has_run_command_tool():
    """Verify agent has run_command tool."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    tool_names = [t.name for t in agent.tools]
    assert "run_command" in tool_names


def test_agent_execute_run_command():
    """Verify agent can execute run_command."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="act")
    result = agent._execute_tool("run_command", {"command": "echo test"})

    assert "test" in result


def test_tool_definitions_includes_run_command():
    """Verify tool_definitions includes run_command."""
    defs = tool_definitions(tools)
    tool_names = [d["name"] for d in defs]
    assert "run_command" in tool_names


# --- EditFile Tests ---

def test_edit_file_replaces_text():
    """Verify EditFile replaces text in a file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("x = 1\ny = 2\nz = 3\n")
        temp_path = f.name

    try:
        tool = EditFile()
        context = ToolContext()
        result = tool.execute(context, temp_path, "y = 2", "y = 42")

        assert "Successfully" in result
        with open(temp_path) as f:
            content = f.read()
        assert "y = 42" in content
        assert "y = 2" not in content
    finally:
        os.unlink(temp_path)


def test_edit_file_not_found():
    """Verify EditFile returns error when text not found."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("x = 1\n")
        temp_path = f.name

    try:
        tool = EditFile()
        context = ToolContext()
        result = tool.execute(context, temp_path, "not in file", "replacement")

        assert "Error" in result
        assert "Could not find" in result
    finally:
        os.unlink(temp_path)


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


def test_compact_conversation_restores_tools_on_error():
    """Verify compaction restores tools even if summarization fails."""
    class FailBrain(FakeBrain):
        def think(self, conversation):
            if not self.tools:
                raise RuntimeError("API error during compaction")
            return super().think(conversation)

    responses = [
        Thought(text="Response", raw_content=[{"type": "text", "text": "Response"}]),
    ]
    brain = FailBrain(responses=responses)
    brain.tools = tool_definitions(tools)
    brain.last_input_tokens = 200_000  # Trigger compaction
    agent = Agent(brain=brain, tools=tools, mode="act")

    agent.handle_input("Do something")

    # Tools should be restored despite the compaction error
    assert len(brain.tools) > 0


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
