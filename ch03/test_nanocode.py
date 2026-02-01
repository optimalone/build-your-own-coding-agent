import pytest
from nanocode import Agent, AgentStop, Thought, ToolCall


# --- Fake Brain for Testing ---

class FakeBrain:
    """Fake brain for testing - returns predictable responses."""

    def __init__(self, responses=None):
        self.responses = responses or [Thought(text="Fake response")]
        self.call_count = 0
        self.last_conversation = None

    def think(self, conversation):
        self.last_conversation = list(conversation)  # Store a copy
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return Thought(text="No more responses")


# --- Tests from Chapter 1 ---

def test_quit_command_raises_agent_stop():
    """Verify /q raises AgentStop exception."""
    agent = Agent(brain=FakeBrain())
    with pytest.raises(AgentStop):
        agent.handle_input("/q")


def test_quit_command_with_whitespace():
    """Verify /q works with surrounding whitespace."""
    agent = Agent(brain=FakeBrain())
    with pytest.raises(AgentStop):
        agent.handle_input("  /q  ")


def test_empty_input_returns_empty_string():
    """Verify empty/whitespace input returns empty string."""
    agent = Agent(brain=FakeBrain())
    assert agent.handle_input("") == ""
    assert agent.handle_input("   ") == ""
    assert agent.handle_input("\n") == ""


# --- New tests for Chapter 3 ---

def test_handle_input_returns_brain_response():
    """Verify handle_input returns the brain's response text."""
    brain = FakeBrain(responses=[Thought(text="Hello from brain!")])
    agent = Agent(brain=brain)
    result = agent.handle_input("hi")
    assert result == "Hello from brain!"


def test_conversation_accumulates():
    """Verify conversation list grows with each interaction."""
    brain = FakeBrain(responses=[
        Thought(text="Response 1"),
        Thought(text="Response 2")
    ])
    agent = Agent(brain=brain)

    agent.handle_input("First message")
    assert len(agent.conversation) == 2  # user + assistant

    agent.handle_input("Second message")
    assert len(agent.conversation) == 4  # 2 users + 2 assistants


def test_conversation_contains_correct_roles():
    """Verify conversation has correct role alternation."""
    brain = FakeBrain(responses=[Thought(text="AI response")])
    agent = Agent(brain=brain)

    agent.handle_input("User message")

    assert agent.conversation[0]["role"] == "user"
    assert agent.conversation[0]["content"] == "User message"
    assert agent.conversation[1]["role"] == "assistant"
    assert agent.conversation[1]["content"] == "AI response"


def test_brain_receives_conversation():
    """Verify brain.think is called with the conversation list."""
    brain = FakeBrain()
    agent = Agent(brain=brain)

    agent.handle_input("Test message")

    assert brain.last_conversation is not None
    assert len(brain.last_conversation) == 1
    assert brain.last_conversation[0]["content"] == "Test message"


def test_failed_brain_call_removes_user_message():
    """Verify failed brain call removes the user message from history."""
    class FailingBrain:
        def think(self, conversation):
            raise Exception("API Error")

    agent = Agent(brain=FailingBrain())
    result = agent.handle_input("Test message")

    assert "Error" in result
    assert len(agent.conversation) == 0  # Message should be removed


# --- Thought and ToolCall tests ---

def test_thought_with_text():
    """Verify Thought stores text."""
    thought = Thought(text="Hello")
    assert thought.text == "Hello"
    assert thought.tool_calls == []


def test_thought_with_tool_calls():
    """Verify Thought stores tool calls."""
    calls = [ToolCall(id="1", name="read_file", args={"path": "test.txt"})]
    thought = Thought(text="Let me read that", tool_calls=calls)
    assert thought.text == "Let me read that"
    assert len(thought.tool_calls) == 1
    assert thought.tool_calls[0].name == "read_file"


def test_tool_call_stores_attributes():
    """Verify ToolCall stores id, name, and args."""
    call = ToolCall(id="abc123", name="write_file", args={"path": "x.txt", "content": "hi"})
    assert call.id == "abc123"
    assert call.name == "write_file"
    assert call.args["path"] == "x.txt"
    assert call.args["content"] == "hi"
