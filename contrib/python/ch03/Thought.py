""" response from LLM - after its done thinking and eturns response to user """
class Thought:
    def __init__ ( self, text=None, tool_calls=None , thinking= None):
        """ response an be ..."""

        """ text response """
        self.text = text # string response 

        """ OR """
        """ list of tool_calls """
        self.tool_calls = tool_calls or [] # list of ToolCall objects
        
        """ model's reasoning summary  ( TODO define this ) """
        self.thinking = thinking # str or None 


