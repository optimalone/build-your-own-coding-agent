class ToolCall:
    """tool invocation request from the brain """

    """ 
    id uniqui identifier to track responses from LLM 
    name : name of tool to invoke : str
    args : arguments to the tool : dict 
    """
    def __init__ (self, id, name, args ):
        self.id = id
        self.name = name
        self.args = args 

