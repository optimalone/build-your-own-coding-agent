import os 

default_memory_str = """
I am Nanocode, a helpful coding assistant.

## Rule
- Always read a file before editing it.
- Run tests after every code change.
- Never modify files outside project directory.

## Style
- Use clear variaLE names. No single letter variables.
- Prefer small focussed functions

"""
class Memory:
    """Persistent scratchpad for the agent."""

    def __init__(self, path=".nanocode/memory.md"):
        self.path = path
        self._ensure_exists()
        self.content = self._load()

    def _ensure_exists(self):
        global default_memory_str 
        """Create memory file with default content if needed."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write(default_memory_str)

    def _load(self):
        """Load content from disk."""
        with open(self.path, 'r') as f:
            return f.read()

    def save(self, content):
        """Update memory content and persist to disk."""
        self.content = content
        with open(self.path, 'w') as f:
            f.write(content)

