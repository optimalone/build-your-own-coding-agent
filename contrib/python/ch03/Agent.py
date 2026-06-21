from AgentStop import AgentStop
from Brain     import Brain 
from Thought   import Thought
from Claude    import Claude
from DeepSeek  import DeepSeek
# import ToolCall

# Available brains
BRAINS = {
    "claude": Claude,
    "deepseek": DeepSeek,
}

class Agent:
   
   def __init__ (self, brain, brain_name="claude" ):
       self.brain = brain # the LLM wrapper
       self.brain_name = brain_name # name of brain we want to use 
       self.conversation = []  # the context - accumaltes the user-brain interaction
   
   # switch the LLM to a differnt provider 
   def _switch_brain(self):
       """Toggle to the next brain."""
       names = list(BRAINS.keys())
       idx = names.index(self.brain_name)
       new_name = names[(idx + 1) % len(names)]

       try:
           self.brain = BRAINS[new_name]()
           self.brain_name = new_name
           return f"Switched to: {new_name}"
       except ValueError as e:
           return f"Cannot switch to {new_name}: {e}"
 

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
          # send context to LLM - this is where it all starts 
          thought = self.brain.think(self.conversation)
          
          # if response has a "thinking" component i.e. information showing how LLM is thinking about the problem
          # extract it from the Thought 
          if thought.thinking:
             lines = thought.thinking.strip().split("\n")[:5]
             for i, line in enumerate(lines):
                 prefix = "  💭 " if i == 0 else "     "
                 print(f"\033[2m{prefix}{line}\033[0m")

          # extract the text part of the response 
          text = thought.text or ""

          # incrementally build the contect - append llm response to the conversation 
          brain_response = {
            "role" : "assistant",
            "content" : text
          }
          self.conversation.append(brain_response)

          return text

      except  Exception as e:
          self.conversation.pop() # remove offending message from the context
          return f"Error: could not process {user_input}\nGot error {e}"
