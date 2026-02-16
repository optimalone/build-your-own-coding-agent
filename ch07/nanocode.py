import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()


# --- HTTP Helpers ---

def request_with_retry(url, headers, payload, max_retries=5):
    """Make HTTP POST with retry on rate limit (429), server errors (5xx), and network failures."""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.exceptions.RequestException as e:
            wait_time = 2 ** attempt
            print(f"Network error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            wait_time = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
            print(f"Error {response.status_code}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue

        if response.status_code >= 400:
            try:
                error_msg = response.json()["error"]["message"]
            except (KeyError, ValueError):
                error_msg = response.text
            raise Exception(f"API error ({response.status_code}): {error_msg}")

        return response

    raise Exception(f"Request failed after {max_retries} retries")


# --- Exceptions ---

class AgentStop(Exception):
    """Raised when the agent should stop processing."""
    pass


# --- Brain Response Types ---

class ToolCall:
    """A tool invocation request from the brain."""

    def __init__(self, id, name, args):
        self.id = id
        self.name = name
        self.args = args  # dict


class Thought:
    """Standardized response from any Brain."""

    def __init__(self, text=None, tool_calls=None, raw_content=None):
        self.text = text  # str or None
        self.tool_calls = tool_calls or []  # list of ToolCall
        self.raw_content = raw_content  # original API response for message history


# --- Memory Class ---

class Memory:
    """Persistent scratchpad for the agent."""

    def __init__(self, path=".nanocode/memory.md"):
        self.path = path
        self._ensure_exists()
        self.content = self._load()

    def _ensure_exists(self):
        """Create memory file with default content if needed."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            default = "I am Nanocode, a helpful coding assistant.\n"
            with open(self.path, "w") as f:
                f.write(default)

    def _load(self):
        """Load content from disk."""
        with open(self.path, 'r') as f:
            return f.read()

    def save(self, content):
        """Update memory content and persist to disk."""
        self.content = content
        with open(self.path, 'w') as f:
            f.write(content)


# --- Tool Context ---

class ToolContext:
    """What tools need to know about the agent's state."""

    def __init__(self, mode=None, memory=None):
        self.mode = mode      # "plan" or "act"
        self.memory = memory  # Memory object or None


# --- Brain Interface ---

class Brain:
    """Base class for LLM providers."""

    def think(self, conversation):
        """Process conversation, return Thought."""
        raise NotImplementedError


class Claude(Brain):
    """Claude API - the brain of our agent."""

    def __init__(self, memory=None, tools=None):
        self.memory = memory
        self.tools = tools or []
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")
        self.model = "claude-sonnet-4-5-20250929"
        self.url = "https://api.anthropic.com/v1/messages"

    def think(self, conversation):
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": conversation
        }
        if self.memory:
            payload["system"] = self.memory.content
        if self.tools:
            payload["tools"] = self.tools

        print("(Claude is thinking...)")
        response = request_with_retry(self.url, headers, payload)
        return self._parse_response(response.json()["content"])

    def _parse_response(self, content):
        """Convert Claude's response format to Thought."""
        text_parts = []
        tool_calls = []

        for block in content:
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    args=block["input"]
                ))

        return Thought(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            raw_content=content
        )


class DeepSeek(Brain):
    """DeepSeek API (Anthropic-compatible, with tool support)."""

    def __init__(self, memory=None, tools=None):
        self.memory = memory
        self.tools = tools or []
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in .env")
        self.model = "deepseek-chat"
        self.url = "https://api.deepseek.com/anthropic/v1/messages"

    def think(self, conversation):
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": conversation
        }
        if self.memory:
            payload["system"] = self.memory.content
        if self.tools:
            payload["tools"] = self.tools

        print("(DeepSeek is thinking...)")
        response = request_with_retry(self.url, headers, payload)
        return self._parse_response(response.json()["content"])

    def _parse_response(self, content):
        """Convert Anthropic response format to Thought."""
        text_parts = []
        tool_calls = []

        for block in content:
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    args=block["input"]
                ))

        return Thought(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            raw_content=content
        )


# Available brains
BRAINS = {
    "claude": Claude,
    "deepseek": DeepSeek,
}


# --- Tool Classes ---

class ReadFile:
    """Reads a file from the filesystem."""
    name = "read_file"
    description = "Reads a file from the filesystem. Use this to examine code."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The relative path to the file"}
        },
        "required": ["path"]
    }

    def execute(self, context, path):
        print(f"  → Reading {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            numbered_lines = [f"{i+1} | {line}" for i, line in enumerate(lines)]
            return "".join(numbered_lines)
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFile:
    """Writes content to a file (mode-aware)."""
    name = "write_file"
    description = "Writes content to a file. In plan mode, only PLAN.md is allowed."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The relative path to the file"},
            "content": {"type": "string", "description": "The full content to write"}
        },
        "required": ["path", "content"]
    }

    def execute(self, context, path, content):
        # Always allow writing to PLAN.md
        if os.path.basename(path) == "PLAN.md":
            print(f"  → Writing {path}")
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return "Plan saved successfully to PLAN.md"
            except Exception as e:
                return f"Error saving plan: {e}"

        # Block in plan mode
        if context.mode == "plan":
            return f"BLOCKED: Cannot write to '{path}' in plan mode. Write to 'PLAN.md' instead."

        # Allow in act mode
        print(f"  → Writing {path}")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file: {e}"


class SaveMemory:
    """Updates the agent's internal memory/scratchpad."""
    name = "save_memory"
    description = "Updates your internal memory/scratchpad. Use this to remember user preferences."
    input_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The full text to save."}
        },
        "required": ["content"]
    }

    def execute(self, context, content):
        print(f"  → Saving memory")
        if context.memory is None:
            return "Error: Memory not available"
        context.memory.save(content)
        return "Memory updated successfully."


# --- Tool Helpers ---

def get_tool(tools, name):
    """Find a tool by name, or None if not found."""
    return next((t for t in tools if t.name == name), None)


def tool_definitions(tools):
    """Return tool definitions for the API."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


# --- Tools List ---

tools = [ReadFile(), WriteFile(), SaveMemory()]


# --- Agent Class ---

class Agent:
    """A coding agent with tools, memory, and safety mode."""

    def __init__(self, brain, tools, memory=None, mode="plan", brain_name="claude"):
        self.brain = brain
        self.tools = list(tools)  # Copy the tools list
        self.memory = memory
        self.mode = mode  # "plan" or "act"
        self.brain_name = brain_name
        self.conversation = []

    def handle_input(self, user_input):
        """Handle user input. Returns output string, raises AgentStop to quit."""
        if user_input.strip() == "/q":
            raise AgentStop()

        if user_input.strip() == "/switch":
            return self._switch_brain()

        if not user_input.strip():
            return ""

        # Handle mode switching
        if user_input.strip().startswith("/mode"):
            return self._handle_mode_command(user_input)

        self.conversation.append({"role": "user", "content": user_input})

        try:
            return self._agentic_loop()
        except Exception as e:
            self.conversation.pop()  # Remove failed user message
            return f"Error: {e}"

    def _handle_mode_command(self, user_input):
        """Handle /mode command to switch between plan and act."""
        parts = user_input.strip().split()
        if len(parts) > 1 and parts[1] == "act":
            self.mode = "act"
            return "⚠️  Switched to ACT MODE (Writing Enabled)"
        else:
            self.mode = "plan"
            return "🛡️  Switched to PLAN MODE (Read-Only)"

    def _agentic_loop(self):
        """Process brain responses, executing tools until done."""
        output_parts = []

        while True:
            thought = self.brain.think(self.conversation)

            # Store raw content for message history
            self.conversation.append({"role": "assistant", "content": thought.raw_content})

            # Collect text output
            if thought.text:
                output_parts.append(thought.text)

            # Check for tool calls
            if not thought.tool_calls:
                break

            # Execute tools and collect results
            tool_results = []
            for tool_call in thought.tool_calls:
                result = self._execute_tool(tool_call.name, tool_call.args)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result
                })

            self.conversation.append({"role": "user", "content": tool_results})

        return "\n".join(output_parts)

    def _execute_tool(self, name, args):
        """Execute a tool by name with given arguments."""
        tool = get_tool(self.tools, name)
        if tool is None:
            return f"Error: Tool '{name}' not found"
        try:
            context = ToolContext(mode=self.mode, memory=self.memory)
            return tool.execute(context, **args)
        except TypeError as e:
            return f"Error: Invalid arguments - {e}"

    def _switch_brain(self):
        """Toggle to the next brain."""
        names = list(BRAINS.keys())
        idx = names.index(self.brain_name)
        new_name = names[(idx + 1) % len(names)]

        try:
            self.brain = BRAINS[new_name](memory=self.memory, tools=tool_definitions(self.tools))
            self.brain_name = new_name
            return f"Switched to: {new_name}"
        except ValueError as e:
            return f"Cannot switch to {new_name}: {e}"


# --- Main Loop ---

def main():
    # Parse mode from CLI
    mode = "act" if len(sys.argv) > 1 and sys.argv[1] == "--act" else "plan"
    brain_name = os.getenv("NANOCODE_BRAIN", "claude")

    memory = Memory()
    brain = BRAINS[brain_name](memory=memory, tools=tool_definitions(tools))
    agent = Agent(brain=brain, tools=tools, memory=memory, mode=mode, brain_name=brain_name)

    print(f"⚡ Nanocode v0.6")
    print(f"Commands: /q quit, /switch toggle brain, /mode [plan|act]")
    print(f"Brain: {brain_name}")
    if mode == "act":
        print("Mode: ACT (Writing Enabled)")
    else:
        print("Mode: PLAN (Read-Only)")

    while True:
        try:
            user_input = input(f"[{agent.brain_name}:{agent.mode}] ❯ ")
            output = agent.handle_input(user_input)
            if output:
                print(f"\n{output}\n")

        except (AgentStop, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
