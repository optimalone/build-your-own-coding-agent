from AgentStop import AgentStop
from Brain     import Brain 
from Thought   import Thought
from Claude    import Claude
from DeepSeek  import DeepSeek
from Tools     import ReadFile, WriteFile, SaveMemory, tools, get_tool, tool_definitions
from ToolContext import ToolContext
import json 
# import ToolCall

# Available brains
BRAINS = {
    "claude": Claude,
    "deepseek": DeepSeek,
}

class Agent:
   
   def __init__ (self, brain, tools, memory=None,  brain_name="claude" ):
       self.brain = brain # the LLM wrapper
       self.brain_name = brain_name # name of brain we want to use 
       self.conversation = []  # the context - accumaltes the user-brain interaction
       self.tools = list(tools)  
       self.memory = memory
   
   # switch the LLM to a differnt provider 
   def _switch_brain(self):
       """Toggle to the next brain."""
       names = list(BRAINS.keys())
       idx = names.index(self.brain_name)
       new_name = names[(idx + 1) % len(names)]

       try:
           self.brain = BRAINS[new_name](tools=tool_definitions(self.tools), memory=self.memory )
           self.brain_name = new_name
           return f"Switched to: {new_name}"
       except ValueError as e:
           return f"Cannot switch to {new_name}: {e}"
 

   def _agentic_loop(self):
       """Process brain responses, executing tools until done."""
       output_parts = []

       while True:
           print (f"=================Start Context=========================")   
           for part in self.conversation:
              print (json.dumps(part,sort_keys=True, indent=4))  
           print (f"==================End Context==========================")

           thought = self.brain.think(self.conversation)
                  
           print (f"==================Start {self.brain_name} output=====================")       
           print (f"{thought}" )
           print (f"==================End {self.brain_name} output========================")   

           # Display thinking
           if thought.thinking:
               lines = thought.thinking.strip().split("\n")[:5]
               for i, line in enumerate(lines):
                   prefix = "  💭 " if i == 0 else "     "
                   print(f"\033[2m{prefix}{line}\033[0m")

           # Store raw content for message history (Claude expects this format)
           self.conversation.append({"role": "assistant", "content": thought.raw_content})

           # Collect text output
           if thought.text:
               output_parts.append(thought.text)

           # Check for tool calls
           if not thought.tool_calls:   
               break  # No more tools to execute

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

       return "\n".join(output_parts)




   def _execute_tool(self, name, args):
        """Execute a tool by name with given arguments."""
        tool = get_tool(self.tools, name)
        if tool is None:
            return f"Error: Tool '{name}' not found"
        try:
            context = ToolContext(memory = self.memory)
            return tool.execute(context=context,  **args)
        except TypeError as e:
            return f"Error: Invalid arguments - {e}"

   


   def handle_input( self, user_input):
      # /q quits the session 
      if (user_input.strip() == '/q'):
          raise AgentStop()

      # handle empty lines
      if (not user_input.strip()) :
          return ""

      # switch the LLMs
      if user_input.strip() == "/switch":
            return self._switch_brain()

      # encapsulate user specified input before sending it to the LLM for processing
      user_input_dict = {
         "role"    : "user",
         "content" : user_input
      }
      self.conversation.append(user_input_dict)

      try:
          return self._agentic_loop()
      except  Exception as e:
          self.conversation.pop() # remove offending message from the context
          return f"Error: could not process {user_input}\nGot error {e}"



