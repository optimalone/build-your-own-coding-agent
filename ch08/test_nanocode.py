import os
import tempfile
import pytest
from nanocode import (
    Agent, AgentStop, Thought, ToolCall, ToolContext, Memory,
    ReadFile, WriteFile, WritePlan, ListFiles, SearchCodebase, SaveMemory,
    get_tool, tool_definitions, tools,
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


# --- Tool filtering by mode ---

def test_plan_mode_hides_write_file():
    """Verify plan mode does not expose write_file to the brain."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")
    tool_names = [t["name"] for t in agent.brain.tools]

    assert "write_file" not in tool_names
    assert "write_plan" in tool_names
    assert "read_file" in tool_names


def test_act_mode_shows_all_tools():
    """Verify act mode exposes all tools to the brain."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="act")
    tool_names = [t["name"] for t in agent.brain.tools]

    assert "write_file" in tool_names
    assert "write_plan" in tool_names
    assert "read_file" in tool_names


def test_mode_switch_updates_brain_tools():
    """Verify switching mode changes the brain's tool menu."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")

    plan_names = [t["name"] for t in agent.brain.tools]
    assert "write_file" not in plan_names

    agent.handle_input("/mode act")
    act_names = [t["name"] for t in agent.brain.tools]
    assert "write_file" in act_names

    agent.handle_input("/mode plan")
    plan_names = [t["name"] for t in agent.brain.tools]
    assert "write_file" not in plan_names


# --- WritePlan tool ---

def test_write_plan_saves_file():
    """Verify WritePlan writes to PLAN.md."""
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            tool = WritePlan()
            context = ToolContext()
            result = tool.execute(context, content="# My Plan\n\nStep 1: Read code")

            assert "Plan saved" in result
            assert os.path.exists("PLAN.md")
            with open("PLAN.md") as f:
                assert f.read() == "# My Plan\n\nStep 1: Read code"
        finally:
            os.chdir(original_dir)


# --- WriteFile tool (no mode checks, always works) ---

def test_write_file_writes_file():
    """Verify WriteFile writes content to the given path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.py")
        tool = WriteFile()
        context = ToolContext()

        result = tool.execute(context, file_path, "print('hello')")

        assert "Successfully wrote" in result
        assert os.path.exists(file_path)


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


def test_search_codebase_skips_env_files_and_symlink_aliases(tmp_path):
    import nanocode

    (tmp_path / ".env").write_text("marker=private-api-key\n")
    (tmp_path / ".env.local").write_text("marker=private-local-key\n")
    nested = tmp_path / "config"
    nested.mkdir()
    (nested / ".env.production").write_text("marker=private-production-key\n")
    (tmp_path / "settings.py").symlink_to(tmp_path / ".env")
    (tmp_path / "app.py").write_text("marker = 'public-source'\n")

    result = nanocode.SearchCodebase().execute(None, "marker", str(tmp_path))

    assert "public-source" in result
    assert "private-api-key" not in result
    assert "private-local-key" not in result
    assert "private-production-key" not in result
    assert "settings.py" not in result


def test_search_codebase_skips_dot_venv(tmp_path):
    import nanocode

    environment = tmp_path / ".venv"
    environment.mkdir()
    (environment / "dependency.py").write_text("marker = 'dependency-noise'\n")
    (tmp_path / "app.py").write_text("marker = 'project-source'\n")

    result = nanocode.SearchCodebase().execute(None, "marker", str(tmp_path))

    assert "project-source" in result
    assert "dependency-noise" not in result


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
                       "anthropic-beta": "thinking-binding-controls-2026-08-01",
                       "content-type": "application/json"}
    assert timeout == 120
    assert payload == {
        "model": "claude-sonnet-5-5",
        "max_tokens": 16000,
        "thinking": {"type": "adaptive", "display": "summarized",
                     "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
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


# --- Signed-prefix transition HTTP regressions ---

def prefix_transition_http(monkeypatch, responses):
    """Keep Agent/provider parsing real; replace only the external HTTP boundary."""
    import copy
    import json as json_module
    import nanocode

    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test-key")
    replies = iter(responses)
    sent = []

    class Response:
        status_code = 200
        headers = {}

        def __init__(self, content, input_tokens):
            self.content = content
            self.input_tokens = input_tokens

        def json(self):
            return {"content": self.content, "usage": {"input_tokens": self.input_tokens}}

    def post(url, headers, json, timeout, **kwargs):
        assert url == "https://api.anthropic.com/v1/messages"
        assert timeout == 120
        sent.append((copy.deepcopy(headers), copy.deepcopy(json)))
        return Response(*next(replies))

    monkeypatch.setattr(nanocode.requests, "post", post)
    return sent


def signed_prefix_blocks():
    return [
        {"type": "thinking", "thinking": "", "signature": "empty+/signature=="},
        {"type": "redacted_thinking", "data": "opaque-redacted"},
        {"type": "thinking", "thinking": "Keep the context.", "signature": "signed+/block=="},
    ]


def assert_prefix_drop_contract(sent):
    for headers, payload in sent:
        assert headers == {
            "x-api-key": "offline-test-key",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "thinking-binding-controls-2026-08-01",
            "content-type": "application/json",
        }
        assert payload["thinking"] == {
            "type": "adaptive", "display": "summarized",
            "block_binding": {"prefix_mismatch_behavior": "drop_block"},
        }
        assert payload["model"] == "claude-sonnet-5-5"
        assert payload["max_tokens"] == 16000
        assert payload["output_config"] == {"effort": "medium"}


def test_signed_prefix_memory_refresh_http(monkeypatch, tmp_path):
    import nanocode

    memory = nanocode.Memory(path=str(tmp_path / "memory.md"))
    memory.save("Original preferences")
    content = signed_prefix_blocks() + [{
        "type": "tool_use", "id": "save-1", "name": "save_memory",
        "input": {"content": "Updated preferences"},
    }]
    sent = prefix_transition_http(monkeypatch, [
        (content, 12), ([{"type": "text", "text": "done"}], 15),
    ])
    agent = nanocode.Agent(brain=nanocode.Claude(memory=memory),
                           tools=[nanocode.SaveMemory()], memory=memory)

    agent.handle_input("Remember my preferences")

    assert len(sent) == 2
    assert "Original preferences" in sent[0][1]["system"]
    assert "Updated preferences" in sent[1][1]["system"]
    continued = sent[1][1]["messages"]
    assert continued[1] == {"role": "assistant", "content": content}
    assert agent.conversation[1]["content"] == content
    assert continued[2]["content"] == [{"type": "tool_result", "tool_use_id": "save-1",
                                         "content": "Memory updated successfully."}]
    assert memory.content == "Updated preferences"
    assert_prefix_drop_contract(sent)


def test_signed_prefix_mode_switch_http(monkeypatch, tmp_path):
    import nanocode

    forbidden = tmp_path / "forbidden.py"
    signed = signed_prefix_blocks() + [{"type": "text", "text": "ready"}]
    blocked = [{"type": "tool_use", "id": "write-1", "name": "write_file",
                "input": {"path": str(forbidden), "content": "unsafe"}}]
    sent = prefix_transition_http(monkeypatch, [
        (signed, 12), (blocked, 15), ([{"type": "text", "text": "done"}], 18),
    ])
    agent = nanocode.Agent(brain=nanocode.Claude(),
                           tools=[nanocode.ReadFile(), nanocode.WriteFile()], mode="act")

    agent.handle_input("Get ready")
    agent.handle_input("/mode plan")
    agent.handle_input("Try the write")

    assert len(sent) == 3
    assert "system" not in sent[0][1]
    assert "PLAN mode" in sent[1][1]["system"]
    assert [t["name"] for t in sent[0][1]["tools"]] == ["read_file", "write_file"]
    assert [t["name"] for t in sent[1][1]["tools"]] == ["read_file"]
    assert sent[1][1]["messages"][1] == {"role": "assistant", "content": signed}
    assert agent.conversation[1]["content"] == signed
    assert sent[2][1]["messages"][3] == {"role": "assistant", "content": blocked}
    assert sent[2][1]["messages"][4]["content"] == [{
        "type": "tool_result", "tool_use_id": "write-1",
        "content": "Error: Tool 'write_file' is not allowed in plan mode",
    }]
    assert not forbidden.exists()
    assert_prefix_drop_contract(sent)
