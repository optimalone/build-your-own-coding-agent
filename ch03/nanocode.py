import os
import requests
from dotenv import load_dotenv

load_dotenv()


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

    def __init__(self, text=None, tool_calls=None):
        self.text = text  # str or None
        self.tool_calls = tool_calls or []  # list of ToolCall


# --- Claude (The Brain) ---

class Claude:
    """Claude API - the brain of our agent."""

    def __init__(self):
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

        print("(Claude is thinking...)")
        response = requests.post(self.url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
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
            tool_calls=tool_calls
        )


# --- Agent Class ---

class Agent:
    """A coding agent with conversation memory."""

    def __init__(self, brain):
        self.brain = brain
        self.conversation = []

    def handle_input(self, user_input):
        """Handle user input. Returns output string, raises AgentStop to quit."""
        if user_input.strip() == "/q":
            raise AgentStop()

        if not user_input.strip():
            return ""

        self.conversation.append({"role": "user", "content": user_input})

        try:
            thought = self.brain.think(self.conversation)
            text = thought.text or ""
            self.conversation.append({"role": "assistant", "content": text})
            return text
        except Exception as e:
            self.conversation.pop()  # Remove failed user message
            return f"Error: {e}"


# --- Main Loop ---

def main():
    brain = Claude()
    agent = Agent(brain)
    print("⚡ Nanocode v0.2 (Conversation Memory)")
    print("Type '/q' to quit.\n")

    while True:
        try:
            user_input = input("❯ ")
            output = agent.handle_input(user_input)
            if output:
                print(f"\n{output}\n")

        except (AgentStop, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
