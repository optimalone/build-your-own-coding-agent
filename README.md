# Nanocode: Build Your Own AI Coding Agent

This is the companion code repository for [**Build a Coding Agent**](https://leanpub.com/build-your-own-coding-agent) — a book that teaches you how to build an autonomous AI coding agent from scratch using pure Python.

No LangChain. No vector databases. No "orchestration frameworks." Just `requests`, `subprocess`, and code you can debug with `print()`.

## What You'll Build

**Nanocode** is a terminal-based AI coding agent that can:

- Read, write, and edit files in your codebase
- Execute shell commands and iterate on errors
- Search code using `git grep`
- Remember context across sessions
- Search the web for current information
- Run on Claude, DeepSeek, or local models via Ollama

By the end of the book, you'll have a ~700-line Python script that rivals commercial coding assistants — and you'll understand every line because you wrote it.

## Repository Structure

Each `chXX/` folder contains a complete, runnable snapshot of the code at the end of that chapter:

| Chapter | What You Build |
|---------|----------------|
| `ch01/` | Event loop — the skeleton |
| `ch02/` | API test — first contact with Claude |
| `ch03/` | Stateful chatbot — memory within a session |
| `ch04/` | Multi-provider — Claude + DeepSeek |
| `ch05/` | Tools — read and write files |
| `ch06/` | Persistent memory — cross-session context |
| `ch07/` | Plan mode — safety harness |
| `ch08/` | Codebase context — list and search files |
| `ch09/` | Feedback loop — run commands, fix errors |
| `ch10/` | Local models — Ollama integration |
| `ch11/` | Web search — DuckDuckGo tool |
| `ch12/` | Capstone — build a Snake game with AI |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/owenthereal/build-a-coding-agent.git
cd build-a-coding-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Running Examples

```bash
# Chapter 1: Event loop (no AI yet)
python ch01/nanocode.py

# Chapter 3+: Needs API key in .env
python ch03/nanocode.py

# Chapter 10+: Can use local models
NANOCODE_BRAIN=ollama python ch10/nanocode.py

# Run tests (no API key needed)
cd ch09
pytest
```

## Requirements

- Python 3.10+
- An Anthropic API key (get one at [console.anthropic.com](https://console.anthropic.com))
- Optional: DeepSeek API key, Ollama for local models

## The Book

This code accompanies the book **Build a Coding Agent**, available on:

- [Leanpub](https://leanpub.com/build-your-own-coding-agent)
- Amazon (coming soon)

The book follows a "Zero Magic" philosophy — every abstraction is explained with physical metaphors, and every line of code serves a purpose you'll understand.

## License

MIT License. See [LICENSE](LICENSE) for details.
