from Thought   import Thought 
from ToolCall  import ToolCall

""" --- Brain Interface --- """
""" this is an abstract class - deriving classes must implement the think method """

class Brain:
    """Base class for LLM providers."""

    def think(self, conversation):
        """Process conversation, return Thought."""
        raise NotImplementedError

    def _parse_response(self, content):
        """Convert Claude's response format to Thought."""
        text_parts = []
        tool_calls = []
        thinking = None

        for block in content:
            if block["type"] == "thinking":
                thinking = block["thinking"]
            elif block["type"] == "text":
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
            raw_content = content,
            thinking=thinking
        )

