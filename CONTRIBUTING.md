# Contributing a Language Implementation

The reference implementation of Nanocode is Python, but the architecture is simple enough to port to any language. If you want to build it in JavaScript, Go, C#, Java, Rust, or anything else — we'd love to include it.

## Structure

Community implementations live in `contrib/<language>/`. Inside, mirror the chapter structure — one folder per chapter, each a complete runnable snapshot.

```
contrib/
└── javascript/
    ├── ch01/          # Event loop (no AI)
    ├── ch03/          # Stateful chatbot
    ├── ch05/          # Tools (read/write files)
    ├── ...
    ├── ch12/          # Capstone
    ├── README.md      # Setup instructions
    └── package.json   # Language-specific config
```

## Guidelines

1. **Follow the book's progression.** Each chapter folder should match what the book builds in that chapter. Readers should be able to read the book and follow along in your language.

2. **Name the main file `nanocode.<ext>`.** For example, `nanocode.js`, `nanocode.go`, `nanocode.cs`. This makes it easy to find the agent across languages.

3. **Keep it minimal.** The book's philosophy is "Zero Magic" — no frameworks, no heavy abstractions. Use your language's standard library and HTTP client. Avoid pulling in agent frameworks or LLM SDKs.

4. **Include tests.** The Python implementation has a `test_nanocode.py` per chapter using a `FakeBrain` test double. Do the same in your language's test framework.

5. **Add a README.** Your `contrib/<language>/README.md` should cover setup (dependencies, environment variables, how to run).

6. **Skip chapters that don't apply.** Chapter 2 is a throwaway test script. Chapter 12 is a capstone demo. You can skip these if they don't add value for your language.

## Submitting

1. Fork this repo.
2. Create your directory under `contrib/` (e.g., `contrib/go/`).
3. Implement as many chapters as you like (starting from ch01).
4. Open a PR. We'll review and merge it, then add your language to the table in the main README.

Partial implementations are welcome — you don't need all 12 chapters to open a PR. Start with ch01–ch05 and build from there.

## Questions?

Open an issue if you're unsure about how to map a concept to your language. The core ideas are:

- **Brain:** A class/interface that wraps an HTTP call to an LLM API and returns structured responses.
- **Tools:** Functions/classes with a name, JSON schema, and an execute method.
- **Agent:** Holds conversation state, calls the brain, dispatches tool calls in a loop.
- **Main loop:** Read input, call agent, print output, repeat.

That's it. The rest is details.
