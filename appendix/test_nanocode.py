"""Tests for Nanocode Streaming (Appendix A)."""

import json
from nanocode import Thought, ToolCall


# --- Test Helpers ---

def parse_sse_events(raw_lines):
    """Parse raw SSE lines into a list of event dicts."""
    events = []
    for line in raw_lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        try:
            events.append(json.loads(line[6:]))
        except (json.JSONDecodeError, ValueError):
            continue
    return events


def build_thought_from_events(events, print_fn=None, thinking_print_fn=None):
    """Build a Thought from parsed SSE events. Returns (Thought, input_tokens)."""
    text_parts = []
    thinking_parts = []
    raw_content = []
    tool_calls = []
    current_tool = None
    input_tokens = 0

    for data in events:
        event_type = data.get("type")

        if event_type == "message_start":
            input_tokens = data.get("message", {}).get("usage", {}).get("input_tokens", 0)

        elif event_type == "error":
            error = data.get("error", {})
            raise Exception(f"Stream error: {error.get('message', data)}")

        elif event_type == "content_block_start":
            block = data.get("content_block", {})
            if block.get("type") == "tool_use":
                current_tool = {"id": block["id"], "name": block["name"], "input": ""}

        elif event_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "thinking_delta":
                text = delta["thinking"]
                if thinking_print_fn:
                    thinking_print_fn(text)
                thinking_parts.append(text)
            elif delta.get("type") == "text_delta":
                text = delta["text"]
                if print_fn:
                    print_fn(text)
                text_parts.append(text)
            elif delta.get("type") == "input_json_delta":
                if current_tool:
                    current_tool["input"] += delta.get("partial_json", "")

        elif event_type == "content_block_stop":
            if current_tool:
                tool_input = json.loads(current_tool["input"]) if current_tool["input"] else {}
                tool_calls.append(ToolCall(
                    id=current_tool["id"],
                    name=current_tool["name"],
                    args=tool_input
                ))
                current_tool = None

    full_text = "".join(text_parts)
    full_thinking = "".join(thinking_parts)
    if full_thinking:
        raw_content.append({"type": "thinking", "thinking": full_thinking})
    if full_text:
        raw_content.append({"type": "text", "text": full_text})
    for tc in tool_calls:
        raw_content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args})

    return Thought(
        text=full_text or None,
        tool_calls=tool_calls,
        raw_content=raw_content,
        thinking=full_thinking or None
    ), input_tokens


# --- SSE Parsing Tests ---

def test_parse_sse_events_extracts_data_lines():
    """Verify parse_sse_events extracts JSON from data: lines."""
    raw_lines = [
        b'event: message_start',
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":42}}}',
        b'',
        b'event: content_block_delta',
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}',
    ]
    events = parse_sse_events(raw_lines)
    assert len(events) == 2
    assert events[0]["type"] == "message_start"
    assert events[1]["type"] == "content_block_delta"


def test_parse_sse_events_skips_non_data_lines():
    """Verify non-data lines are ignored."""
    raw_lines = [
        b'event: ping',
        b': comment',
        b'',
        b'data: {"type":"message_stop"}',
    ]
    events = parse_sse_events(raw_lines)
    assert len(events) == 1


def test_parse_sse_events_handles_string_input():
    """Verify parse_sse_events works with string lines (not just bytes)."""
    raw_lines = [
        'data: {"type":"message_stop"}',
    ]
    events = parse_sse_events(raw_lines)
    assert len(events) == 1


def test_parse_sse_events_skips_malformed_json():
    """Verify malformed JSON data lines are skipped."""
    raw_lines = [
        b'data: not json',
        b'data: {"type":"message_stop"}',
    ]
    events = parse_sse_events(raw_lines)
    assert len(events) == 1
    assert events[0]["type"] == "message_stop"


# --- Thought Building Tests ---

def test_build_thought_text_only():
    """Verify text-only response is assembled correctly."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 100}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    thought, tokens = build_thought_from_events(events)
    assert thought.text == "Hello world"
    assert tokens == 100
    assert not thought.tool_calls
    assert len(thought.raw_content) == 1
    assert thought.raw_content[0]["type"] == "text"


def test_build_thought_with_tool_call():
    """Verify tool call response is assembled correctly."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 200}}},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "tool_123", "name": "read_file"
        }},
        {"type": "content_block_delta", "index": 0, "delta": {
            "type": "input_json_delta", "partial_json": '{"path":'
        }},
        {"type": "content_block_delta", "index": 0, "delta": {
            "type": "input_json_delta", "partial_json": ' "test.py"}'
        }},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    thought, tokens = build_thought_from_events(events)
    assert thought.text is None
    assert len(thought.tool_calls) == 1
    assert thought.tool_calls[0].name == "read_file"
    assert thought.tool_calls[0].args == {"path": "test.py"}
    assert tokens == 200


def test_build_thought_mixed_text_and_tool():
    """Verify response with both text and tool calls."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 50}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me read that."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {
            "type": "tool_use", "id": "tool_456", "name": "read_file"
        }},
        {"type": "content_block_delta", "index": 1, "delta": {
            "type": "input_json_delta", "partial_json": '{"path": "main.py"}'
        }},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]
    thought, tokens = build_thought_from_events(events)
    assert thought.text == "Let me read that."
    assert len(thought.tool_calls) == 1
    assert thought.tool_calls[0].name == "read_file"
    assert len(thought.raw_content) == 2


def test_build_thought_tracks_streaming_output():
    """Verify print_fn is called for each text delta."""
    chunks = []
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "one"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " two"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " three"}},
        {"type": "message_stop"},
    ]
    thought, _ = build_thought_from_events(events, print_fn=chunks.append)
    assert chunks == ["one", " two", " three"]
    assert thought.text == "one two three"


def test_build_thought_empty_response():
    """Verify empty event stream produces empty Thought."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {"type": "message_stop"},
    ]
    thought, tokens = build_thought_from_events(events)
    assert thought.text is None
    assert thought.tool_calls == []
    assert tokens == 5


def test_build_thought_tool_with_empty_input():
    """Verify tool call with no input args."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 30}}},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "tool_789", "name": "list_files"
        }},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    thought, _ = build_thought_from_events(events)
    assert len(thought.tool_calls) == 1
    assert thought.tool_calls[0].name == "list_files"
    assert thought.tool_calls[0].args == {}


def test_build_thought_with_thinking():
    """Verify thinking deltas are accumulated and stored."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 50}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": " reason"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Here's the answer."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]
    thought, tokens = build_thought_from_events(events)
    assert thought.thinking == "Let me reason"
    assert thought.text == "Here's the answer."
    assert tokens == 50
    assert len(thought.raw_content) == 2  # thinking + text
    assert thought.raw_content[0]["type"] == "thinking"


def test_build_thought_thinking_print_fn():
    """Verify thinking_print_fn is called for each thinking delta."""
    chunks = []
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "step one"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": " step two"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    thought, _ = build_thought_from_events(events, thinking_print_fn=chunks.append)
    assert chunks == ["step one", " step two"]
    assert thought.thinking == "step one step two"


def test_build_thought_error_event_raises():
    """Verify SSE error events raise an exception."""
    import pytest
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    ]
    with pytest.raises(Exception, match="Stream error: Overloaded"):
        build_thought_from_events(events)


# --- Real HTTP Stream Parser Tests ---

import pytest
from nanocode import Brain


@pytest.fixture
def stream_brain(monkeypatch):
    """Replace only HTTP transport; exercise the production SSE parser."""
    def run(events):
        class Response:
            status_code = 200
            headers = {}

            def iter_lines(self):
                for event in events:
                    yield ("event: " + event["type"]).encode()
                    yield ("data: " + json.dumps(event)).encode()
                    yield b""

        monkeypatch.setattr("nanocode.requests.post", lambda *a, **kw: Response())
        brain = Brain()
        thought = brain._stream_response("https://example.test/messages", {}, {})
        return brain, thought
    return run


def stream_start():
    return {"type": "message_start", "message": {
        "id": "msg_test", "type": "message", "role": "assistant",
        "content": [], "model": "test-model", "stop_reason": None,
        "stop_sequence": None, "usage": {"input_tokens": 42, "output_tokens": 1},
    }}


def text_events(index=0):
    return [
        {"type": "content_block_start", "index": index,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": index,
         "delta": {"type": "text_delta", "text": "Reading now."}},
        {"type": "content_block_stop", "index": index},
    ]


def tool_events(index=1):
    return [
        {"type": "content_block_start", "index": index,
         "content_block": {"type": "tool_use", "id": "tool_1", "name": "read_file", "input": {}}},
        {"type": "content_block_delta", "index": index,
         "delta": {"type": "input_json_delta", "partial_json": '{"path":'}},
        {"type": "content_block_delta", "index": index,
         "delta": {"type": "input_json_delta", "partial_json": '"main.py"}'}},
        {"type": "content_block_stop", "index": index},
    ]


def stream_end():
    return [
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
         "usage": {"output_tokens": 20}},
        {"type": "message_stop"},
    ]


@pytest.mark.parametrize("events", [
    [stream_start()],
    [stream_start(), *text_events()[:-1]],
    [stream_start(), *text_events(), *tool_events()],
], ids=["empty", "mid_text", "closed_tool"])
def test_stream_response_rejects_missing_message_stop(stream_brain, events):
    with pytest.raises(Exception, match="[Ss]tream.*message_stop"):
        stream_brain(events)


@pytest.mark.parametrize("block_events", [text_events()[:-1], tool_events(0)[:-1]],
                         ids=["text", "tool"])
def test_stream_response_rejects_message_stop_with_open_block(stream_brain, block_events):
    with pytest.raises(Exception, match="[Ss]tream.*[Uu]nclosed"):
        stream_brain([stream_start(), *block_events, *stream_end()])


def test_stream_response_complete_text_and_tool(stream_brain, capsys):
    brain, thought = stream_brain([stream_start(), *text_events(), *tool_events(), *stream_end()])
    assert thought.text == "Reading now."
    assert [(tc.id, tc.name, tc.args) for tc in thought.tool_calls] == [
        ("tool_1", "read_file", {"path": "main.py"})]
    assert thought.raw_content == [
        {"type": "text", "text": "Reading now."},
        {"type": "tool_use", "id": "tool_1", "name": "read_file", "input": {"path": "main.py"}},
    ]
    assert brain.last_input_tokens == 42
    assert capsys.readouterr().out == "Reading now.\n"


@pytest.mark.parametrize("thinking", ["Check the file.", ""], ids=["visible", "omitted"])
def test_stream_response_preserves_signed_and_redacted_thinking_order(stream_brain, thinking):
    events = [
        stream_start(),
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": thinking}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta", "signature": "opaque+"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta", "signature": "signature=="}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "redacted_thinking", "data": "encrypted+/opaque=="}},
        {"type": "content_block_stop", "index": 1},
        *text_events(2), *tool_events(3), *stream_end(),
    ]
    _, thought = stream_brain(events)
    assert thought.raw_content == [
        {"type": "thinking", "thinking": thinking, "signature": "opaque+signature=="},
        {"type": "redacted_thinking", "data": "encrypted+/opaque=="},
        {"type": "text", "text": "Reading now."},
        {"type": "tool_use", "id": "tool_1", "name": "read_file", "input": {"path": "main.py"}},
    ]
    assert thought.thinking == (thinking or None)


def test_stream_response_error_after_closed_tool_raises(stream_brain):
    with pytest.raises(Exception, match="Stream error: Overloaded"):
        stream_brain([stream_start(), *tool_events(0),
                      {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}])


# --- Shared Runtime Integration Regressions ---

from nanocode import Agent, Memory, ToolContext, RunCommand, tools, get_tool


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


# --- Runtime regression tests ---

import pytest
from nanocode import Ollama

def test_deepseek_sends_supported_model_with_thinking_disabled(monkeypatch):
    import nanocode

    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")
    payloads = []

    class Response:
        status_code = 200
        headers = {}

        def iter_lines(self):
            events = [stream_start(), *text_events(), *stream_end()]
            events[0]["message"]["usage"]["input_tokens"] = 12
            events[2]["delta"]["text"] = "hello"
            for event in events:
                yield ("data: " + __import__("json").dumps(event)).encode()

    def post(url, headers, json, stream, timeout):
        assert stream is True
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

    class StreamResponse:
        status_code = 200
        headers = {}

        def iter_lines(self):
            events = [stream_start(), *text_events(), *stream_end()]
            events[0]["message"]["usage"]["input_tokens"] = 7000
            events[2]["delta"]["text"] = "answer"
            for event in events:
                yield ("data: " + json.dumps(event)).encode()

    def infer(url, headers, json, stream, timeout):
        assert url == "http://localhost:11434/v1/messages"
        assert stream is True
        assert requests_seen[-1] == "ps"
        assert brain.context_limit == 8192
        assert "options" not in json
        requests_seen.append("messages")
        return StreamResponse()

    monkeypatch.setattr(nanocode.requests, "get", get)
    monkeypatch.setattr(nanocode.requests, "post", infer)
    brain = Ollama()
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
    def unexpected(*args, **kwargs):
        pytest.fail("inference sent without validated loaded context")

    monkeypatch.setattr(nanocode.requests, "post", unexpected)
    brain = Ollama()
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
    def unexpected(*args, **kwargs):
        pytest.fail("inference sent after failed metadata lookup")

    monkeypatch.setattr(nanocode.requests, "post", unexpected)
    brain = Ollama()
    with pytest.raises((ValueError, RuntimeError), match="(?i)(ollama|context|load)"):
        brain.think([{"role": "user", "content": "hello"}])


@pytest.mark.parametrize("configured,loaded,equivalent", [
    ("library/qwen3-coder:30b", "qwen3-coder:30b", True),
    ("registry.ollama.ai/library/qwen3-coder:30b", "qwen3-coder:30b", True),
    ("qwen3-coder:30b", "library/qwen3-coder:30b", True),
    ("qwen3-coder:30b", "registry.ollama.ai/library/qwen3-coder:30b", True),
    ("QWEN3-CODER:30B", "qwen3-coder:30b", True),
    ("REGISTRY.OLLAMA.AI/LIBRARY/Qwen3-Coder:30B", "qwen3-coder:30b", True),
    ("qwen3-coder:30b", "REGISTRY.OLLAMA.AI/Library/QWEN3-CODER:30B", True),
    ("library/my-model", "registry.ollama.ai/library/my-model:latest", True),
    ("registry.local:5000/team/model", "REGISTRY.LOCAL:5000/TEAM/MODEL:latest", True),
    ("https://registry.ollama.ai/library/qwen3-coder:30b", "qwen3-coder:30b", True),
    ("qwen3-coder:30b", "team/qwen3-coder:30b", False),
    ("team/qwen3-coder:30b", "qwen3-coder:30b", False),
    ("team/qwen3-coder:30b", "other/qwen3-coder:30b", False),
    ("qwen3-coder:30b", "registry.local/library/qwen3-coder:30b", False),
    ("registry.local/library/qwen3-coder:30b", "qwen3-coder:30b", False),
    ("registry.local:5000/team/model", "registry.local:5001/team/model:latest", False),
    ("registry.local:5000/team/model", "registry.local/team/model:latest", False),
    ("registry.ollama.ai:443/library/model", "model:latest", False),
    ("registry.one/team/model", "registry.two/team/model:latest", False),
    ("registry.local:5000/team/model", "registry.local:5000/other/model:latest", False),
])
def test_ollama_matches_loaded_context_by_model_identity(monkeypatch, configured, loaded, equivalent):
    import nanocode

    monkeypatch.setenv("OLLAMA_MODEL", configured)
    requests_seen = []

    class LoadedResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{
                "name": loaded, "model": loaded, "size": 1000, "size_vram": 1000,
                "digest": "sha256:offline", "expires_at": "2026-11-04T23:00:00Z",
                "details": {"parent_model": "", "format": "gguf", "family": "qwen3",
                            "families": ["qwen3"], "parameter_size": "30B",
                            "quantization_level": "Q4_K_M"},
                "context_length": 8192, "runner": "llama_cpp",
            }]}

    def get(url, timeout):
        assert url == "http://localhost:11434/api/ps"
        requests_seen.append("ps")
        return LoadedResponse()

    class MessageResponse:
        status_code = 200
        headers = {}

        def iter_lines(self):
            events = [stream_start(), *text_events(), *stream_end()]
            events[0]["message"]["usage"]["input_tokens"] = 64
            events[2]["delta"]["text"] = "answer"
            for event in events:
                yield ("data: " + json.dumps(event)).encode()

    def post(url, headers, json, stream, timeout):
        assert url == "http://localhost:11434/v1/messages"
        assert stream is True
        assert json["model"] == configured
        assert brain.context_limit == 8192
        requests_seen.append("messages")
        return MessageResponse()

    monkeypatch.setattr(nanocode.requests, "get", get)
    monkeypatch.setattr(nanocode.requests, "post", post)
    brain = Ollama()
    conversation = [{"role": "user", "content": "hello"}]

    if equivalent:
        thought = brain.think(conversation)
        assert thought.text == "answer"
        assert brain.context_limit == 8192
        assert brain.last_input_tokens == 64
        assert requests_seen == ["ps", "messages"]
    else:
        with pytest.raises(ValueError, match="is not loaded"):
            brain.think(conversation)
        assert requests_seen == ["ps"]
