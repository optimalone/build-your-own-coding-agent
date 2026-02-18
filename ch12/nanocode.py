import os
import sys
import subprocess
import time
import requests
from dotenv import load_dotenv

load_dotenv()


# --- HTTP Helpers ---

def request_with_retry(url, headers, payload, max_retries=10):
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
            retry_after = response.headers.get("retry-after")
            wait_time = int(retry_after) if retry_after else 2 ** attempt
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
    context_limit = 200_000
    last_input_tokens = 0

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
        self.model = "claude-sonnet-4-6"
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
        data = response.json()
        self.last_input_tokens = data.get("usage", {}).get("input_tokens", 0)
        return self._parse_response(data["content"])

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
    context_limit = 128_000

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
        data = response.json()
        self.last_input_tokens = data.get("usage", {}).get("input_tokens", 0)
        return self._parse_response(data["content"])

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


class Ollama(Brain):
    """Ollama local model (Anthropic-compatible API, with tool support)."""
    context_limit = 32_000

    def __init__(self, memory=None, tools=None):
        self.memory = memory
        self.tools = tools or []
        self.model = os.getenv("OLLAMA_MODEL", "qwen3-coder:30b")
        self.url = "http://localhost:11434/v1/messages"
        self._detect_context_limit()

    def _detect_context_limit(self):
        """Query Ollama for the model's context window size."""
        try:
            response = requests.post(
                "http://localhost:11434/api/show",
                json={"model": self.model},
                timeout=2
            )
            model_info = response.json().get("model_info", {})
            for key, value in model_info.items():
                if key.endswith(".context_length"):
                    self.context_limit = value
                    return
        except Exception:
            pass  # Keep default

    def think(self, conversation):
        headers = {
            "x-api-key": "ollama",  # Required but not validated
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

        print(f"({self.model} is thinking...)")
        response = request_with_retry(self.url, headers, payload)
        data = response.json()
        self.last_input_tokens = data.get("usage", {}).get("input_tokens", 0)
        return self._parse_response(data["content"])

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
    "ollama": Ollama,
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


class EditFile:
    """Replaces text in a file (surgical edit)."""
    name = "edit_file"
    description = "Replaces specific text in a file. Use for surgical edits instead of rewriting entire files."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "old_text": {"type": "string", "description": "Exact text to find and replace"},
            "new_text": {"type": "string", "description": "Text to replace it with"}
        },
        "required": ["path", "old_text", "new_text"]
    }

    def execute(self, context, path, old_text, new_text):
        print(f"  → Editing {path}")
        if context.mode == "plan":
            return "BLOCKED: Cannot edit files in plan mode."

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            if old_text not in content:
                return f"Error: Could not find the specified text in {path}"
            new_content = content.replace(old_text, new_text, 1)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing file: {e}"


class ListFiles:
    """Lists files in the project structure."""
    name = "list_files"
    description = "Lists all files in the project structure. Useful to understand the project layout."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The root path (default '.')"}
        }
    }

    def execute(self, context, path="."):
        print(f"  → Listing {path}")
        try:
            file_list = []
            for root, dirs, files in os.walk(path):
                if any(p in root for p in [".git", "__pycache__", "venv", ".nanocode"]):
                    continue

                level = root.replace(path, '').count(os.sep)
                indent = ' ' * 4 * (level)
                file_list.append(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    file_list.append(f"{subindent}{f}")

            return "\n".join(file_list)
        except Exception as e:
            return f"Error listing files: {e}"


class SearchCodebase:
    """Searches for a string in all files."""
    name = "search_codebase"
    description = "Searches the entire codebase for a text string. Useful to find where functions or variables are defined."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The string to search for"},
            "path": {"type": "string", "description": "The root path (default '.')"}
        },
        "required": ["query"]
    }

    def execute(self, context, query, path="."):
        print(f"  → Searching for '{query}'")
        results = []
        try:
            for root, dirs, files in os.walk(path):
                if any(p in root for p in [".git", "__pycache__", "venv", ".nanocode"]):
                    continue

                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for i, line in enumerate(f):
                                if query.lower() in line.lower():
                                    results.append(f"{file_path}:{i+1}: {line.strip()}")
                    except Exception:
                        continue

            return "\n".join(results) if results else "No matches found."
        except Exception as e:
            return f"Error searching: {e}"


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


class RunCommand:
    """Executes shell commands (mode-aware)."""
    name = "run_command"
    description = "Executes a terminal command. Use this to run scripts, tests, or install packages. Blocked in plan mode."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run (e.g., 'python test.py')"}
        },
        "required": ["command"]
    }

    def execute(self, context, command):
        print(f"  → Running: {command[:50]}...")
        # Block in plan mode - running commands can modify state
        if context.mode == "plan":
            return "BLOCKED: Cannot run commands in plan mode. Switch to act mode first."

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd()
            )

            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            if not output:
                output = "(No output)"

            return output.strip()

        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds."
        except Exception as e:
            return f"Error executing command: {e}"


class SearchWeb:
    """Searches the internet using DuckDuckGo."""
    name = "search_web"
    description = "Searches the internet for current information. Use when you need knowledge beyond your training data."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }

    def execute(self, context, query):
        print(f"  → Searching web for '{query}'")
        try:
            from ddgs import DDGS
            results = DDGS().text(query, max_results=3)
            if not results:
                return "No results found."

            formatted = []
            for r in results:
                formatted.append(f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}\n")

            return "\n".join(formatted)
        except Exception as e:
            return f"Error searching web: {e}"


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

tools = [
    ReadFile(),
    WriteFile(),
    EditFile(),
    ListFiles(),
    SearchCodebase(),
    SaveMemory(),
    RunCommand(),
    SearchWeb(),
]


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
        max_iterations = 50

        for _iteration in range(max_iterations):
            thought = self.brain.think(self.conversation)

            # Compact if approaching context limit
            if self.brain.last_input_tokens > self.brain.context_limit * 0.75:
                self._compact_conversation()

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
        else:
            output_parts.append("(Stopped: too many iterations)")

        return "\n".join(output_parts)

    def _compact_conversation(self):
        """Summarize old messages to stay within context limits."""
        print("(Compacting conversation...)")
        history = "\n".join(
            f"{m['role']}: {str(m['content'])[:500]}"
            for m in self.conversation
        )
        prompt = [{
            "role": "user",
            "content": f"Summarize this conversation for continuity. "
                       f"Focus on what was accomplished, what's in progress, "
                       f"and key decisions:\n\n{history}"
        }]
        thought = self.brain.think(prompt)
        self.conversation = [
            {"role": "user", "content": f"Previous conversation summary: {thought.text}"},
        ]

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

    print(f"⚡ Nanocode v1.0")
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
