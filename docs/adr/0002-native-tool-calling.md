# Native tool calling over text-embedded JSON

The original protocol had the model emit one JSON object as free text, which the host parsed, retrying once on malformed JSON. That parse is where small local models fail (fences, prose before the object, wrong nesting), and it makes "prose or tool call?" ambiguous. The agent uses OpenAI-style `tools`/`tool_calls` with a per-tool JSON Schema; a reply with no `tool_calls` is the final answer, and invalid arguments become observations.

Consequences: a tool-capable model is a hard requirement (Ollama returns HTTP 400 `does not support tools`); Ollama ignores `tool_choice`, so deciding to call a tool remains model judgment; schemas must stay flat because Ollama does not resolve `$ref`/`$defs`.
