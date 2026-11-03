# Spec: Sandboxed Local Coding Agent (v1)

## Problem Statement

I want a coding agent that runs entirely on my machine, driven by my local model, so that neither my prompts nor my source code leave the host. Existing agents either require a cloud API or run model-generated code directly on my machine with my full filesystem and network privileges. I need the generated code to execute somewhere that cannot read my home directory, cannot reach the network, cannot spawn processes, and cannot run forever — and I need to know, after the fact, exactly what it tried to do.

## Solution

A single Rust CLI, `dont-get-rusty "task"`, that drives an OpenAI-compatible local model and executes its code only inside an embedded Wasmtime sandbox running a CPython 3.15.2 WASI Guest. The Guest gets two read-only Capabilities: the Runtime Image at `/` and the Workspace at `/workspace`. All durable writes go through host tools, which are policy-checked, audited, and quota-limited. The Agent Loop uses native tool calling, pauses after repeated failed Runs, compacts context deterministically, and appends one Audit Event per Tool Call to a JSONL log outside the Workspace.

## User Stories

1. As a local developer, I want to run a coding task with one command, so that I don't need a hosted service.
2. As a local developer, I want the Agent Loop to use my local model endpoint, so that my prompts and code never leave my machine.
3. As a local developer, I want model-generated code to run only inside a WebAssembly Sandbox, so that a bad generation cannot damage my host.
4. As a local developer, I want the Guest to have no network access, so that generated code cannot exfiltrate data or reach internal services.
5. As a local developer, I want the Guest to see only the Runtime Image and the Workspace, so that it cannot read my home directory or system files.
6. As a local developer, I want the Workspace mounted read-only inside the Sandbox, so that generated code cannot silently overwrite my files.
7. As a local developer, I want every durable write to go through an audited host tool, so that each change is attributable to a Tool Call.
8. As a local developer, I want to confirm overwrites of files that existed before the session, so that the agent cannot destroy my work.
9. As a local developer, I want files the agent created in this session to be overwritten without a prompt, so that iteration is not noisy.
10. As a local developer, I want a `--yes` flag, so that unattended runs do not block on prompts.
11. As a local developer, I want a required confirmation in a non-TTY session to become a denial Observation, so that pipelines never hang.
12. As a local developer, I want a wall-clock timeout per Run, so that an infinite loop ends.
13. As a local developer, I want a memory cap per Run, so that a memory bomb traps without taking down the host.
14. As a local developer, I want stdout and stderr truncated at a fixed size with a discarded-byte count, so that an output flood cannot exhaust host memory and I know what was lost.
15. As a local developer, I want every Run to start from a fresh Sandbox state, so that state cannot leak between Runs.
16. As a local developer, I want timeout, OOM, guest exit, and arbitrary traps mapped to distinct exit codes and host-written stderr lines, so that I can tell them apart.
17. As a local developer, I want a Python Guest with its standard library, so that ordinary scripts run without installing anything.
18. As a local developer, I want the Python Guest to start in well under a second from a cached module, so that iteration stays fast.
19. As a local developer, I want the agent to read Workspace files, so that it works from my actual code and data.
20. As a local developer, I want the agent to list directories, so that it can discover the Workspace layout.
21. As a local developer, I want the agent to write Workspace files, so that it can produce changes.
22. As a local developer, I want a Workspace size quota enforced before each write, so that the agent cannot fill my disk.
23. As a local developer, I want the agent to run its code before claiming success, so that final answers are grounded in observed output.
24. As a local developer, I want the agent to retry a failing script a bounded number of times, so that it can fix its own bugs.
25. As a local developer, I want the Agent Loop to pause after three consecutive failed Runs and ask whether to continue, so that a stuck loop returns control to me.
26. As a local developer, I want all Tool Calls in one model turn to execute in order, so that a batched read of two files costs one round trip.
27. As a local developer, I want a Step limit, so that time and cost are bounded.
28. As a local developer, I want older Observations compacted deterministically to a character budget, so that the model's context window does not overflow.
29. As a local developer, I want a compacted Observation to keep the tool name, exit code, byte counts, and first stderr line, so that the useful signal survives.
30. As a local developer, I want a Policy denial to become an error Observation, so that the agent can route around it instead of ending the session.
31. As a local developer, I want every Tool Call and Policy decision appended to a JSONL audit log, so that I can review exactly what happened.
32. As a local developer, I want the audit log outside the Workspace, so that the Guest cannot read it.
33. As a local developer, I want the audit log to store a hash of full arguments rather than code or file bodies, so that I can prove what crossed the boundary without duplicating sensitive data.
34. As a local developer, I want asset hashes verified at startup, so that a tampered or corrupted Guest is refused before it runs.
35. As a local developer, I want a pinned fetch script for the Guest assets, so that setup is reproducible and the repository stays small.
36. As a local developer, I want the compiled Guest module cached, so that startup does not pay the full compile cost each time.
37. As a local developer, I want the model name and endpoint configurable in policy, so that I can switch models without rebuilding.
38. As a local developer, I want a clear error when the model does not support tool calling, so that I know what to fix.
39. As a local developer, I want the tool to work offline after setup, so that the only network dependency is my local model.
40. As a local developer, I want the final answer to state what was done, which files changed, and how to run the result, so that I can verify the work.

## Implementation Decisions

- **One-shot CLI.** `dont-get-rusty "task"` with `--workspace <dir>` (overrides `[fs].root`), `--policy <file>`, and `--yes`. No subcommands in v1.
- **Modules.** CLI/wiring, Agent Loop, LLM client, tool router, Sandbox, Workspace FS wrapper, Policy, Audit, asset loader. The Agent Loop is the only place that owns step accounting, failure accounting, and context compaction.
- **Guest runtime.** CPython 3.15.2 WASI build, `wasm32-wasip1`, preview 1 core module. Python only in v1; the internal language enum stays extensible. JavaScript (quickjs-ng) is post-v1.
- **Capabilities.** The Guest receives exactly two read-only preopens: the Runtime Image at `/` (the CPython dist, which is not embedded in the wasm) and the Workspace at `/workspace`. No environment variables, no network, no processes, no threads.
- **Writes.** The Guest cannot write. `write_file` is the only durable write path; it uses `cap-std` ≥ 4.0.3 and enforces the quota by summing Workspace sizes before writing. `read_file` and `list_files` also use `cap-std`.
- **Tool protocol.** Native OpenAI-style `tools`/`tool_calls` with a flat JSON Schema per tool, generated by `schemars`. A reply with no `tool_calls` is the final answer. No text-embedded JSON, no re-parse retry. Schemas must avoid `$ref`/`$defs`/`anyOf` because Ollama does not resolve them.
- **Tool surface.** `run_code { code, stdin? }` (language fixed to Python in v1), `read_file { path }`, `write_file { path, content }`, `list_files { path? }`.
- **Run result.** `RunResult { exit_code: i32, stdout: String, stderr: String, timed_out: bool, duration_ms: u64 }`. Partial stdout is retained on every path; `duration_ms` spans instantiation through trap or exit.
- **Termination mapping.** Guest `proc_exit(n)` → `exit_code=n`; epoch deadline → `timed_out=true, exit_code=124`; memory-limit trap → `exit_code=137`; any other trap → `exit_code=1`. Every trap appends a host-written `sandbox: …` line to stderr.
- **Limits.** 10 s wall-clock epoch interruption with a ticker thread; 256 MB per-linear-memory `StoreLimits`; no fuel metering (combined fuel + epoch accounting is buggy in Wasmtime 50.0.1). Output is captured through a custom `OutputStream` that retains 64 KB per stream, drains and counts the rest, and appends `[sandbox: output truncated; N bytes discarded]`; retained bytes decode with `from_utf8_lossy`.
- **Network.** p1 sockets must be linked because CPython imports them; `add_to_linker_sync` registers all p1 functions and in 50.0.1 they are nonfunctional stubs. `WasiCtxBuilder` defaults deny TCP/UDP/DNS. Do not omit the imports — CPython then fails to instantiate.
- **Step semantics.** One Step is one LLM turn. Every Tool Call returned in that turn executes in order; all resulting Observations go back in one message. `max_steps = 25`.
- **Failure semantics.** A failed Run is a `run_code` with a non-zero exit code (including traps). Three consecutive failed Runs pause the loop; the CLI asks `continue? [y/N]`; yes resets the counter, no or a non-interactive session ends with the last stderr.
- **Policy.** Host-side only, loaded from `policy.toml`. The model cannot read or modify it. A denial becomes an error Observation; it never aborts the loop.
- **Overwrite confirmation.** `write_file` prompts only when the target existed before the session. Agent-created paths are tracked in memory and overwrite silently. `--yes` disables prompts. A required confirmation without a TTY becomes a denial Observation.
- **Context budgets.** Capture keeps 64 KB per stream. Before each LLM call, the message list is compacted to `context_budget_chars` (default 16000): newest Observations verbatim until the budget is spent, older ones stubbed to tool, exit code, byte counts, and first stderr line; an oversized newest Observation is sent head+tail with a marker. No LLM summarization.
- **LLM client.** Non-streaming; one `/v1/chat/completions` request per Step; 120 s timeout; one retry on connection error or 5xx; 4xx surfaced verbatim. `base_url` and `model` come from policy; no capability preflight. Default model `qwen3.5:4b`, with `qwen2.5-coder:7b` documented as the upgrade.
- **Audit.** Append-only JSONL at `<cwd>/.agent/audit.jsonl`, outside the Workspace mount. One event per Tool Call: timestamp, session id, step index, tool, args truncated to 2 KiB plus `sha256` of the full payload, decision, exit code, stdout/stderr byte counts, truncated flag, duration.
- **Assets.** `scripts/fetch-assets.sh` downloads pinned release URLs into a gitignored `assets/` and verifies SHA-256. Startup re-verifies; a missing or mismatched asset refuses to run with the fix command. The compiled `.cwasm` is cached under `.agent/cache/`, keyed by wasm hash and Wasmtime version.
- **Measured constraints.** Cold compile of `python.wasm` is ~1.3 s; a cached `.cwasm` runs `print(1)` in ~0.02 s. CPython runs correctly with a read-only Runtime Image (verified: `import json`, exit 0, no `__pycache__`).
- **ADRs.** Guest filesystem is read-only with host-mediated writes; native tool calling replaces text-embedded JSON.

## Testing Decisions

A good test asserts externally observable behavior: process exit status, captured output, files on disk, audit events, and the requests the host sends to the model. It must not reach into private functions or assert on internal call sequences.

Seams, highest first:

1. **CLI process.** Spawn `dont-get-rusty "task"` against a stub OpenAI-compatible server and the real CPython Guest, in a temporary Workspace. Asserts the end-to-end contract: request shape, tool execution, final answer, files written, audit file.
2. **Agent Loop.** In-process with a fake LLM implementation, the real Sandbox, and a temporary Workspace. Asserts Step accounting, batched call order, denial-as-Observation, failed-Run pause, and context compaction.
3. **Sandbox.** In-process `Sandbox::run` with the real CPython Guest. Asserts timeout, OOM trap, output truncation, read-only preopens, and escape attempts.
4. **Workspace FS.** Pure Rust with `tempfile` and `proptest`. Asserts traversal, absolute paths, symlink escapes, and quota enforcement without executing wasm.

Prior art: none; the repository is greenfield. The test matrix follows the design spec: `limits` (infinite loop → 124, 300 MB allocation → 137, print flood → truncation marker), `escape` (`/etc/passwd`, `/workspace/../x`, host paths with `..`/absolute/symlink), `agent_loop` (fake LLM scripted replies), `llm_wire` (stub server validates `tools` array and parses `tool_calls`), and one `#[ignore]`-free CPython smoke test that imports the stdlib and prints. Sandbox tests share one Engine and one compiled Module per test binary; the `.cwasm` cache keeps the 32 MB module cost to one compile per binary.

## Out of Scope

- JavaScript Guest (QuickJS) — post-v1.
- TUI (`ratatui`) — post-v1; the CLI is one-shot only.
- Network access from the Guest, including opt-in proxied networking.
- Runtime package installs inside the Guest.
- Multi-user or remote hosting, GPU access, browser UI.
- LLM summarization of old context; streaming responses.
- Audit log rotation.
- Writing files from Guest code; artifacts leave via stdout or the model's `write_file`.

## Further Notes

- The issue tracker and triage labels are not configured in this environment; the spec is stored at `docs/specs/v1-sandboxed-local-agent.md`. Run `/setup-matt-pocock-skills`, then publish it with the `ready-for-agent` label.
- Open risk: `qwen3.5:4b` is installed and advertises tool calling but is untested against these exact flat schemas. If it produces malformed `run_code` arguments, the fallback is `qwen2.5-coder:7b`.
- Ollama ignores `tool_choice`, so "no `tool_calls` means final answer" depends on the model's judgment. Ollama returns HTTP 400 when the model lacks tool support.
- Wasmtime must be pinned at 50.0.1; the p1 API and socket stubs are version-specific.
- The Runtime Image is ~13 MB of `.py` sources with no `.pyc`; cold imports are slower than a normal CPython install.
