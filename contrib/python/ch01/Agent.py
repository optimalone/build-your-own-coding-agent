import AgentStop

class Agent:
   # ctor
   def __init__ (self):
       pass

   def handle_input( self, user_input):
      if (user_input.strip() == '/q'):
          raise AgentStop.AgentStop()

      if (not user_input.strip()) :
          return ""

      return f" You said {user_input}\n Agent not connected"

