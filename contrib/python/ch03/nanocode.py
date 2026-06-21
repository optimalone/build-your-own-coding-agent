#!/bin/env python3

from Agent     import Agent, BRAINS
from AgentStop import AgentStop 
from Thought   import Thought 
from ToolCall  import ToolCall
from Claude    import Claude
from DeepSeek  import DeepSeek
from helpers import request_with_retry

import os
 

def main():
    
    brain_name = os.getenv("NANOCODE_BRAIN", "claude")
    brain = BRAINS[brain_name]()
    agent = Agent(brain, brain_name)

    print("⚡ Nanocode v0.2 (Conversation Memory)")
    print(f"Commands: /q quit, /switch toggle brain")
    print(f"Brain: {brain_name}\n")

    while True:
        try:
            user_input = input("❯ ")
            output = agent.handle_input(user_input)
            if output:
                print(f"\n{output}\n")

        except (AgentStop, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
