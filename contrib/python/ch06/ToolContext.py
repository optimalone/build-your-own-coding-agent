class ToolContext:
    """What tools need to know about the agent's state."""

    def __init__(self, memory=None):
        self.memory = memory  # Memory object or None
