
""" DeepSeek LLM wrapper 
encapsulates the REST interface to Anthropic's Claude series of LLMs 
""" 

import os
from Brain     import Brain
from dotenv import load_dotenv
from helpers import request_with_retry

class DeepSeek(Brain):
    """DeepSeek API (Anthropic-compatible)."""

    def __init__(self, tools=None):
        load_dotenv()
        self.tools = tools or []
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in .env")
        self.model = "deepseek-v4-flash"
        self.url = "https://api.deepseek.com/anthropic/v1/messages"

    def think(self, conversation):
        print ("In Deepseek  think") 
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": conversation
        }

        if self.tools:
           payload["tools"] = self.tools
            
        response = request_with_retry(self.url, headers, payload)
        return self._parse_response(response.json()["content"])
