import AgentStop
import Thought
# import ToolCall

class Agent:
   
   def __init__ (self, brain):
       self.brain = brain # the LLM wrapper 
       self.conversation = []  # the context - accumaltes the user-brain interaction
       

   def handle_input( self, user_input):
      if (user_input.strip() == '/q'):
          raise AgentStop.AgentStop()

      if (not user_input.strip()) :
          return ""

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
          # process it
          if thought.thinking:
             lines = thought.thinking.strip().split("\n")[:5]
             for i, line in enumerate(lines):
                 prefix = "  💭 " if i == 0 else "     "
                 print(f"\033[2m{prefix}{line}\033[0m")
          text = thought.text or ""
          brain_response = {
            "role" : "assistant",
            "content" : text
          }
          self.conversation.append(brain_response)
          return text

      except  Exception as e:
          self.conversation.pop() # remove offending message from the context
          return f"Error: could not process {user_input}\nGot error {e}"
