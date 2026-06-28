#!/bin/env python3

from Agent     import Agent, BRAINS
from AgentStop import AgentStop 
from Thought   import Thought 
from ToolCall  import ToolCall
from Claude    import Claude
from DeepSeek  import DeepSeek
from helpers import request_with_retry
from Tools import ReadFile, WriteFile,  tools, get_tool, tool_definitions

import os
 

def main():
    
    brain_name = os.getenv("NANOCODE_BRAIN", "claude")
   

    brain = BRAINS[brain_name](tools=tool_definitions(tools))
    agent = Agent(brain=brain, tools=tools, brain_name=brain_name)

    print("⚡ Nanocode v0.4 (Tools Enabled)")
    print(f"Commands: /q quit, /switch toggle brain")
    print(f"Brain: {brain_name}\n")

    while True:
        try:
            user_input = input(f"[{agent.brain_name}]❯ ")
            output = agent.handle_input(user_input)
            if output:
                print(f"\n{output}\n")

        except (AgentStop, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
