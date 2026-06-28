class ReadFile:
    """Reads a file from the filesystem."""
    name = "read_file"
    description = "Reads a file from the filesystem. Use this to examine code."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The path to the file"}
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
    """Write given contents into a file"""

    name = "write_file"
    description = "Write content to file, Overwrites/Appends content."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The path to the file."},
            "content" : {"type": "string", "description" : "The full content to write to the file." }, 
            "mode" : {"type": "string", "description" : "How to modify the file?: w or a " }, 
        },
        "required": ["path", "content", "mode"]
    }

    def execute(self,context, path, content, mode):

        w_mode = {"w" : "Overwriting",
                  "a" : "Appending" }
         
        print(f"  - {w_mode[mode]} file {path} with given content" )
        try:
            with open(path, mode , encoding="utf-8") as f:
                f.write(content)
            return f"{w_mode[mode]} {len(content)} characters to {path} successfully"
        except Exception as e:
            return f"Error writing file {path}: {e}"


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





tools = [ReadFile(), WriteFile(), SaveMemory()]

def get_tool(tools, name):
    """Find a tool by name, or None if not found."""
    return next((t for t in tools if t.name == name), None)


def tool_definitions(tools):
    """Return tool definitions for the API."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


