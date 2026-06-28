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

    def execute(self, path):
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
    description = "Write content to file, Overwrites existing content."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The path to the file."},
            "conbtent" : {"type": "string", "description" : "The full content to write to the file." }, 
        },
        "required": ["path", "content"]
    }

    def execute(self,path, content):
        print(f"  - Overwriting file {path} with given content" )
        try:
            with open(path, "w" , encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file {path}: {e}"


tools = [ReadFile(), WriteFile()]

def get_tool(tools, name):
    """Find a tool by name, or None if not found."""
    return next((t for t in tools if t.name == name), None)


def tool_definitions(tools):
    """Return tool definitions for the API."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


