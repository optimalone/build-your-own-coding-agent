#!/bin/env python3

from Agent     import Agent
from AgentStop import AgentStop 
from Thought   import Thought 
from ToolCall  import ToolCall
from Claude    import Claude

def main():
    brain = Claude()
    agent = Agent(brain)
    print("⚡ Nanocode v0.2 (Conversation Memory)")
    print("Type '/q' to quit.\n")

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
