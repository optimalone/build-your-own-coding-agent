
"""Claude LLM erapper 
encapsulates the REST interface to Anthropic's Claude series of LLMs 
"""

import os
from Brain     import Brain
from dotenv import load_dotenv
import helpers

class Claude(Brain):
    """Claude API - the brain of our agent."""

    def __init__(self, tools=None):
        load_dotenv()
        self.tools = tools or None
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
           raise ValueError("ANTHROPIC_API_KEY not found in .env")
        self.model = "claude-sonnet-4-6"
        self.url = "https://api.anthropic.com/v1/messages"

    # this acrually dispatches user requet to the LLM
    def think(self, conversation):
        print ("In Claude  think") 
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": 16000,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 10000
            },
            "messages": conversation
        }
        if (self.tools):
           payload["tools"] = self.tools 

        # call the resilient Http response handler 
        # response = requests.post(self.url, headers=headers, json=payload, timeout=120)
        response = helpers.request_with_retry(self.url, headers,  payload )
        return self._parse_response(response.json()["content"])

   

