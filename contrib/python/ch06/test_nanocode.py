#!/bin/env python3
import pytest
import tempfile 
import os
from Agent     import Agent, BRAINS
from AgentStop import AgentStop 
from Thought   import Thought 
from ToolCall  import ToolCall
from ToolContext import ToolContext
from Brain     import Brain
from Memory    import Memory
from Tools import ReadFile, WriteFile, SaveMemory, tools, get_tool, tool_definitions


# --- Fake Brain for Testing ---

class FakeBrain(Brain):
    """Fake brain for testing - returns predictable responses."""
    def __init__(self, responses=None, memory=None, tools=None):
        self.tools = tools or []
        self.responses = responses or [Thought(text="Fake response", raw_content=[{"type": "text", "text": "Fake response"}])]
        self.call_count = 0 # number of fake reponses

        self.memory = memory
        
        self.last_conversation = None
        for tht in self.responses :
            print (f"in FakeBrain.__init__ text:{tht.text} raw_content:{tht.raw_content}")

    def think(self, conversation):
        self.last_conversation = list(conversation)

        # generate a fake reponse - after which just keep generating "no more responses" 
        if self.call_count < len(self.responses): 
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return Thought(text="No more responses", raw_content=[{"type": "text", "text": "No more responses"}])



# --- Tests from Chapter 1 ---

def test_quit_command_raises_agent_stop():
    """Verify /q raises AgentStop exception."""
    agent = Agent(brain=FakeBrain(), tools=tools )
    with pytest.raises(AgentStop):
        agent.handle_input("/q")


def test_quit_command_with_whitespace():
    """Verify /q works with surrounding whitespace."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    with pytest.raises(AgentStop):
        agent.handle_input("  /q  ")


def test_empty_input_returns_empty_string():
    """Verify empty/whitespace input returns empty string."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    assert agent.handle_input("") == ""
    assert agent.handle_input("   ") == ""
    assert agent.handle_input("\n") == ""


# --- New tests for Chapter 3 ---

def test_handle_input_returns_brain_response():
    """Verify handle_input returns the brain's response text."""
    brt = "Hello from brain!"
    brain = FakeBrain(responses=[
        Thought( text=f"{brt}", raw_content=[{"type": "text", "text": f"{brt}"}] )
    ])
    agent = Agent(brain=brain, tools=tools)
    result = agent.handle_input("hi")
    print (f"{result}")
    assert result == f"{brt}"


def test_conversation_accumulates():
    """Verify conversation list grows with each interaction."""

    # using brts and fb_responses to help with fake brain initialization.  
    brts = ["Response 1", "Response 2"] # variable brts stand for: brain-response-text-strings
    fb_responses = []
    for brt in brts:
        fb_responses.append(Thought(text=f"{brt}", raw_content= [{"type": "text", "text": f"{brt}"}] ))
        
    brain = FakeBrain(responses=fb_responses)
    
    agent = Agent(brain=brain, tools=tools)

    agent.handle_input("First message")
    assert len(agent.conversation) == 2  # user + assistant

    agent.handle_input("Second message")
    assert len(agent.conversation) == 4  # 2 users + 2 assistants

    print (f"conversation 1 :   {agent.conversation}")



def test_conversation_contains_correct_roles():
    """Verify conversation has correct role alternation."""
    #brain = FakeBrain(responses=[Thought(text="AI response")])
    brts = ["AI response"]
    fb_responses = []
    for brt in brts:
        fb_responses.append(Thought(text=f"{brt}", raw_content= [{"type": "text", "text": f"{brt}"}] ))
        
    brain = FakeBrain(responses=fb_responses)

    agent = Agent(brain=brain, tools=tools)

    agent.handle_input("User message")
    print (f"conversation 2:   {agent.conversation}")


    assert agent.conversation[0]["role"] == "user"
    assert agent.conversation[0]["content"] == "User message"
    assert agent.conversation[1]["role"] == "assistant"
    assert agent.conversation[1]["content"][0]["text"] == "AI response"


def test_brain_receives_conversation():
    """Verify brain.think is called with the conversation list."""
    brain = FakeBrain()
    agent = Agent(brain=brain, tools=tools)

    agent.handle_input("Test message")

    assert brain.last_conversation is not None
    assert len(brain.last_conversation) == 1
    assert brain.last_conversation[0]["content"] == "Test message"


def test_failed_brain_call_removes_user_message():
    """Verify failed brain call removes the user message from history."""
    class FailingBrain:
        def think(self, conversation):
            raise Exception("API Error")

    agent = Agent(brain=FailingBrain(), tools=tools)
    result = agent.handle_input("Test message")

    assert "Error" in result
    assert len(agent.conversation) == 0  # Message should be removed


# --- Thought and ToolCall tests ---

def test_thought_with_text():
    """Verify Thought stores text."""
    thought = Thought(text="Hello")
    assert thought.text == "Hello"
    assert thought.tool_calls == []
    assert thought.thinking is None


def test_thought_with_thinking():
    """Verify Thought stores thinking."""
    thought = Thought(text="Hello", thinking="Let me consider this...")
    assert thought.thinking == "Let me consider this..."
    assert thought.text == "Hello"


def test_thought_with_tool_calls():
    """Verify Thought stores tool calls."""
    calls = [ToolCall(id="1", name="read_file", args={"path": "test.txt"})]
    thought = Thought(text="Let me read that", tool_calls=calls)
    assert thought.text == "Let me read that"
    assert len(thought.tool_calls) == 1
    assert thought.tool_calls[0].name == "read_file"


def test_tool_call_stores_attributes():
    """Verify ToolCall stores id, name, and args."""
    call = ToolCall(id="abc123", name="write_file", args={"path": "x.txt", "content": "hi", "mode" : "w"})
    assert call.id == "abc123"
    assert call.name == "write_file"
    assert call.args["path"] == "x.txt"
    assert call.args["content"] == "hi"
    assert call.args["mode"] == "w"


# --- New tests for Chapter 4: Multiple Brains ---

def test_agent_stores_brain_name():
    """Verify agent stores the brain name."""
    agent = Agent(brain=FakeBrain(), brain_name="claude", tools=tools)
    assert agent.brain_name == "claude"

    agent = Agent(brain=FakeBrain(), brain_name="deepseek", tools=tools)
    assert agent.brain_name == "deepseek"


def test_brains_registry_has_expected_providers():
    """Verify BRAINS registry contains expected providers."""
    assert "claude" in BRAINS
    assert "deepseek" in BRAINS


def test_switch_command_toggles_brain_name():
    """Verify /switch updates brain_name (using FakeBrain for both)."""
    agent = Agent(brain=FakeBrain(), brain_name="claude", tools= tools)

    # Mock BRAINS to use FakeBrain for switching
    original_brains = BRAINS.copy()
    BRAINS["claude"] = FakeBrain
    BRAINS["deepseek"] = FakeBrain

    try:
        result = agent.handle_input("/switch")
        assert "deepseek" in result
        assert agent.brain_name == "deepseek"

        result = agent.handle_input("/switch")
        assert "claude" in result
        assert agent.brain_name == "claude"
    finally:
        BRAINS.clear()
        BRAINS.update(original_brains)

# ----------------- Tool tests ------------------------
def test_ReadFile_adds_line_numbers():
    tf_path = ""
    context = ToolContext(memory=None)
    with tempfile.NamedTemporaryFile(mode= "w" , suffix=".tempfile", delete=False) as tf:
        tf.write(f"first line\nsecond line\nthird line\n")
        tf_path = tf.name

    try:
        tool = ReadFile()
        contents = tool.execute(context, tf_path)
        assert "1 | first line" in contents
        assert "2 | second line" in contents
        assert "3 | third line" in contents
    finally:
        os.unlink(tf_path)



def test_ReadFile_handles_missing_file():
    tool = ReadFile()
    context = ToolContext(memory=None) 
    result = tool.execute(context, "/path/to/missing/file.txt")
    assert "Error" in result 

def test_write_file_creates_file():
    """Verify WriteFile creates a file with content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.txt")
        tool = WriteFile()
        context = ToolContext(memory=None) 

        result = tool.execute(context, path, "hello world", "w")

        assert os.path.exists(path)
        assert "successfully" in result
        assert "11 characters" in result
        with open(path) as f:
            assert f.read() == "hello world"



def test_write_file_overwrites_existing():
    """Verify WriteFile overwrites existing content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.txt")
        tool = WriteFile()
        context = ToolContext(memory=None) 

        tool.execute(context, path, "original content", "w")
        tool.execute(context, path, "new content", "w")

        with open(path) as f:
            assert f.read() == "new content"

def test_write_file_appends_existing():
    """Verify WriteFile overwrites existing content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.txt")
        tool = WriteFile()
        context = ToolContext(memory=None) 

        tool.execute(context, path, "original content\n", "w")
        tool.execute(context, path, "new content\n", "a")

        with open(path) as f:
            assert f.read() == "original content\nnew content\n"


def test_write_file_handles_bad_path():
    """Verify WriteFile returns error for invalid path."""
    tool = WriteFile()
    context = ToolContext(memory=None) 
    result = tool.execute(context, "/nonexistent/path/file.txt", "content" , "w")
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
    assert len(defs) == 3
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

    assert len(definitions) == 3
    assert definitions[0]["name"] == "read_file"
    assert "description" in definitions[0]
    assert "input_schema" in definitions[0]



def test_thought_stores_raw_content():
    """Verify Thought stores raw_content for message history."""
    raw = [{"type": "text", "text": "Hello"}]
    thought = Thought(text="Hello", raw_content=raw)
    assert thought.raw_content == raw


def test_agentic_loop_executes_tool_calls():
    """Verify agentic loop executes tool calls and continues."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content\n")
        temp_path = f.name

    try:
        # Brain reyturns toolcalls followed by responses 
        brain = FakeBrain(responses=[

          Thought(
                text="Let me read that file.",
                tool_calls=[
                    ToolCall(id="1", 
                             name="read_file", 
                             args={"path": temp_path})
                    ],
                raw_content=[
                    {"type": "text", "text": "Let me read that file."},
                    {"type": "tool_use", "id": "1", "name": "read_file", "input": {"path": temp_path}}]),

          Thought(
                text="The file contains test content.",
                raw_content=[{"type": "text", "text": "The file contains test content."}]),

          Thought(
                text="Let me write to that file.",
                tool_calls=[
                    ToolCall(id="2", 
                             name="write_file", 
                             args={"path": temp_path, 
                                   "content" : "new content", 
                                   "mode" : "w" })],
                raw_content=[
                    {"type": "text", 
                     "text": "Let me write to that file."
                    },

                    {"type": "tool_use", "id": "1", 
                     "name": "write_file", 
                     "input": {
                         "path": temp_path, 
                         "content" : "new content", 
                         "mode" : "w"      
                         }}]),

           Thought(
               text="The file contains new content.",
               raw_content=[{"type": "text", "text": "The file contains new content."}]),

           Thought(
                text="Let me append to that file.",
                tool_calls=[
                    ToolCall(id="3", 
                             name="write_file", 
                             args={"path": temp_path, 
                                   "content" : "new content 2", 
                                   "mode" : "a" })],
                raw_content=[
                    {"type": "text", 
                     "text": "Let me append to that file."
                    },

                    {"type": "tool_use", "id": "1", 
                     "name": "write_file", 
                     "input": {
                         "path": temp_path, 
                         "content" : "new content 2", 
                         "mode" : "a"      
                         }}]),

           Thought(
               text="The file contains new content and new content 2.",
               raw_content=[{"type": "text", "text": "The file contains new content and new content 2."}]),

        ])
        agent = Agent(brain=brain, tools=tools)
        result = agent.handle_input("Read the file")

        assert "Let me read that file." in result
        assert "The file contains test content." in result
        assert brain.call_count == 2  # Called twice (tool call + final)
 
        result = agent.handle_input("Write to the file")
        assert "Let me write to that file." in result
        assert "The file contains new content." in result

        assert brain.call_count == 4  # Called 4 times (tool call + response + tool call + final  )

        result = agent.handle_input("Append to the file")
        assert "Let me append to that file." in result
        assert "The file contains new content and new content 2." in result

        assert brain.call_count == 6  # Called 6 times 3 tool calls + 3 responses


    finally:
        os.unlink(temp_path)



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
