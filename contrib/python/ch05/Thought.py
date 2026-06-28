""" response from LLM - after its done thinking and eturns response to user """
import json

class Thought:
    def __init__ ( self, text=None, tool_calls=None , raw_content=None,  thinking= None):
        """ response an be ..."""

        """ text response """
        self.text = text # string response 

        """ OR """
        """ list of tool_calls """
        self.tool_calls = tool_calls or [] # list of ToolCall objects
       
        self.raw_content = raw_content  # original API response for message history

        """ model's reasoning summary  ( TODO define this ) """
        self.thinking = thinking # str or None

    def __str__(self):
        thestr = f"Thought:\n"
        if (self.text):
           thestr += f"text: {self.text}\n"
        if (self.tool_calls):   
           thestr += f"tool_calls: {self.tool_calls}\n"
        if (self.raw_content):   
           thestr += f"raw_content: {json.dumps(self.raw_content, sort_keys=True, indent=4)}\n"
        
        if (self.thinking):
           thestr += f"thinking:"
           lines = self.thinking.strip().split("\n")[:5]
           for i, line in enumerate(lines):
              prefix = "  💭 " if i == 0 else "     "
              thestr += (f"\033[2m{prefix}{line}\033[0m")
           thestr += "\n"
           
        return thestr    
     






