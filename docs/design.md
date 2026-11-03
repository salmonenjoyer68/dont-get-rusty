# Design: Sandboxed Local Coding Agent

Working implementation spec. Vocabulary is defined in [CONTEXT.md](../CONTEXT.md); this file holds architecture, stack, limits, and milestones, and is expected to change as facts and decisions land.

## Goal

A coding agent written in Rust that runs fully on the user's machine. The LLM writes code, and that code executes only inside a WebAssembly sandbox (embedded Wasmtime). Untrusted, model-generated code never touches the host filesystem, network, or processes unless explicitly granted.

## Core Principles

- Local first: no required cloud calls. Local model via Ollama or llama.cpp's OpenAI-compatible server.
- Deny by default: the guest gets no files, no network, no env vars, and no sockets unless granted.
- Capabilities are explicit, logged, and revocable.
- Generated code is never run on the host. It runs only in WASM.
- Small, auditable core. Few dependencies. No `unsafe` in agent code (`#![forbid(unsafe_code)]`).

## Architecture

```
User → CLI/TUI → Agent Loop → LLM client (local HTTP)
                     │
                     ├─ Tool Router (serde-validated)
                     │    ├─ run_code    → Sandbox (Wasmtime + WASI, RO workspace)
                     │    ├─ read_file   → Workspace FS (cap-std)
                     │    ├─ write_file  → Workspace FS
                     │    └─ list_files  → Workspace FS
                     │
                     └─ Policy + Audit Log (host-side only)
```

## Stack

| Concern | Crate |
|---|---|
| WASM runtime | `wasmtime` / `wasmtime-wasi` 50.0.1 (WASI p1 core modules) |
| Async | `tokio` |
| HTTP to LLM | `reqwest` (rustls, json, stream) |
| Serialization | `serde`, `serde_json`, `toml` |
| Tool schemas | `schemars` |
| Sandboxed FS | `cap-std` ≥ 4.0.3 (host tools only; blocks `..` and symlink escapes) |
| CLI | `clap` |
| TUI (later) | `ratatui`, `crossterm` |
| Errors | `thiserror` (libs), `anyhow` (bin) |
| Logging/audit | `tracing`, `tracing-subscriber`, JSONL writer |
| Tests | built-in `#[test]`, `tempfile`, `proptest` for path fuzzing |

## Guest Runtimes (compiled to wasm32-wasip1, run inside Wasmtime)

| Language | Guest | Notes |
|---|---|---|
| Python | CPython 3.15.2 WASI build (brettcannon), p1 | Only guest in v1. Stdlib dist mounted read-only at `/`; 32.4 MB wasm. |
| JavaScript | quickjs-ng 0.18.0, p1 | Post-v1. 1.5 MB wasm, no preopens needed for `-e`; `Lang` stays extensible. |
| Rust | Not in v1 | Would need `rustc` to wasm32-wasip1; run compile step on host in a locked-down way, or skip. |

Assets are fetched by `scripts/fetch-assets.sh` from pinned release URLs into a gitignored `assets/` (SHA-256 verified). Startup re-verifies; a missing or mismatched asset refuses to run with the fix command. Pre-compile with `Module::serialize` and cache under `.agent/cache/`, keyed by wasm hash + Wasmtime version.

## Sandbox Limits (defaults)

- Memory: 256 MB via `StoreLimitsBuilder::memory_size` (CPython's linear memory starts at 40 MiB; the cap is per linear memory)
- CPU: epoch interruption (`Config::epoch_interruption(true)`) with a ticker thread; 10 s wall-clock timeout. Fuel is not used: an open Wasmtime bug (PR #14647) loses accounting when fuel and epoch are combined, and v1 needs no deterministic cap.
- Output: stdout/stderr captured through a custom `OutputStream` that retains 64 KB per stream, drains and counts the remainder, and appends `[sandbox: output truncated; N bytes discarded]`; retained bytes decoded with `from_utf8_lossy`
- Filesystem: Guest gets a read-only preopen of the runtime dist at `/` (stdlib) and of `./workspace` at `/workspace`; all durable writes go through host tools (`cap-std`); 50 MB total quota enforced host-side
- Network: p1 sockets must be linked (CPython imports `sock_*`); `add_to_linker_sync` registers all of them and in 50.0.1 they are nonfunctional stubs. `WasiCtxBuilder` defaults deny TCP/UDP/DNS; omitting the imports makes CPython fail at instantiation.
- Env/args: no env; argv is the script path
- Processes/threads: none

Run termination:

| Guest outcome | `exit_code` | `timed_out` | `stderr` |
|---|---|---|---|
| `proc_exit(n)` | `n` | false | guest output |
| epoch deadline | 124 | true | guest output + `sandbox: timed out after 10s` |
| memory-limit trap | 137 | false | guest output + `sandbox: memory limit exceeded (256 MB)` |
| any other trap | 1 | false | guest output + `sandbox: trap: <message>` |

Partial stdout is retained on every path. `duration_ms` spans instantiation through trap or exit.

## Core Types

```rust
// tools.rs
#[derive(JsonSchema, Deserialize)]
pub struct RunCodeArgs { pub code: String, pub stdin: Option<String> }
// language is fixed to Python in v1; the tool schema does not expose it.

#[derive(JsonSchema, Deserialize)]
pub struct ReadFileArgs { pub path: String }

#[derive(JsonSchema, Deserialize)]
pub struct WriteFileArgs { pub path: String, pub content: String }

#[derive(JsonSchema, Deserialize)]
pub struct ListFilesArgs { pub path: Option<String> }

pub enum ToolCall {
    RunCode(RunCodeArgs),
    ReadFile(ReadFileArgs),
    WriteFile(WriteFileArgs),
    ListFiles(ListFilesArgs),
}

#[derive(Serialize)]
pub struct RunResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub duration_ms: u64,
}
```

Tool schemas are generated with `schemars` and must stay flat: Ollama renders them into the prompt without resolving `$ref`/`$defs` and without validating arguments. Inline subschemas and avoid `anyOf`; invalid arguments from the model become error observations.

```rust
// sandbox.rs (shape)
pub struct Sandbox { engine: Engine, python: Module, cfg: SandboxCfg }

impl Sandbox {
    pub fn new(cfg: SandboxCfg) -> Result<Self>;
    pub async fn run(&self, lang: Lang, code: &str, stdin: Option<&str>, ws: &Dir) -> Result<RunResult>;
}
```

Run flow inside `run`:

1. Fresh `Store` per call (no state leaks between runs).
2. Build `WasiCtx` with read-only preopens (runtime dist at `/`, workspace at `/workspace`), piped stdio, no env, no network.
3. Apply `StoreLimits`, set epoch deadline.
4. Instantiate guest, call `_start`, catch trap (timeout / OOM / exit code).
5. Return `RunResult`; never panic on guest failure.

## Agent Loop

```rust
loop {
    let reply = llm.chat(&messages, &tools).await?;   // native tool calling
    if reply.tool_calls.is_empty() {
        break reply.text;                              // final answer
    }
    for call in reply.tool_calls {
        let out = match policy.check(&call) {
            Ok(()) => router.execute(call).await,      // errors become observations
            Err(denied) => denied.into_observation(),  // audited, loop continues
        };
        audit.log(&call, &out);
        messages.push(tool_result(out));
    }
    steps += 1;
    if steps >= cfg.max_steps { break }
}
```

Rules:

- Max 25 steps, where a step is one LLM turn; every tool call in that turn executes in order.
- A failed run is `run_code` returning a non-zero `exit_code` (including traps). Three consecutive failed runs pause the loop; the CLI asks `continue? [y/N]`. Yes resets the counter; no, or a non-interactive session, ends the session with the last stderr.
- A policy denial becomes an error observation; it never aborts the loop.
- The agent must run its code before claiming it works.
- `write_file` prompts only when the target existed before the session; agent-created files overwrite silently. `--yes` disables prompts; without a TTY a required confirmation becomes a denial observation.
- Two budgets: capture keeps 64 KB per stream; before each LLM call, compact the message list to `context_budget_chars` from policy — newest results verbatim until the budget is spent, older ones stubbed to `{tool, exit_code, stdout_bytes, first stderr line}`; a newest result that alone exceeds the budget is sent as head+tail with a marker. No LLM summarization.
- Invalid tool arguments become an error observation; the loop continues.

## LLM Client

Non-streaming: one `/v1/chat/completions` request per Step, `tools` built from the `schemars` schemas, 120 s timeout, one retry on connection error or 5xx, 4xx surfaced verbatim. `base_url` and `model` come from policy; no capability preflight.

## CLI

`dont-get-rusty "task"` — one-shot. Flags: `--workspace <dir>` (overrides `[fs].root`), `--policy <file>`, `--yes`. No subcommands in v1.

## Policy (`policy.toml`)

```toml
[sandbox]
memory_mb = 256
timeout_s = 10
max_output_kb = 64
network = false

[fs]
root = "./workspace"
max_total_mb = 50
confirm_overwrite = true

[agent]
max_steps = 25
max_failed_runs = 3
context_budget_chars = 16000

[llm]
base_url = "http://localhost:11434/v1"
model = "qwen3.5:4b"  # upgrade: qwen2.5-coder:7b after `ollama pull`
request_timeout_s = 120
```

`max_total_mb` is enforced before each `write_file` by summing workspace file sizes; a write that would exceed it is denied.

## Audit

Append-only JSONL at `<cwd>/.agent/audit.jsonl`, outside `[fs].root` so the Guest cannot read it. One event per Tool Call:

```json
{"ts":"2026-11-03T12:00:00Z","session":"…","step":3,"tool":"write_file","args_sha256":"…","args_prefix":"…","decision":"allow","exit_code":null,"stdout_bytes":0,"stderr_bytes":0,"truncated":false,"duration_ms":4}
```

Args are truncated to 2 KiB plus a SHA-256 of the full payload; code and file contents are never stored.

## Tests

Sandbox tests run the real CPython guest, so `scripts/fetch-assets.sh` must run first. One `Engine` and one compiled `Module` are shared per test binary; the `.cwasm` cache makes subsequent loads milliseconds, so the 32 MB module is paid once per binary, not per test. Host FS tests are pure Rust (`tempfile`, `proptest`); the Agent Loop uses a fake `Llm` implementation.

| Test | Guest script | Asserts |
|---|---|---|
| limits | `while True: pass` | epoch trap, `timed_out=true`, `exit_code=124` |
| limits | `bytearray(300 * 1024 * 1024)` | OOM trap, `exit_code=137`, host unaffected |
| limits | `while True: print("x" * 1024)` | 64 KB retained, discarded count in marker |
| escape | `open("/etc/passwd")` | denied by the preopen set |
| escape | `open("/workspace/../x", "w")` | read-only preopen |
| escape | host `write_file` with `..`, absolute path, symlink | `cap-std` error |
| agent_loop | scripted fake LLM | batch order, denial observation, failed-run pause |

## System Prompt (for the LLM)

```
You are a coding agent running locally. You can write and run code only inside
a WebAssembly sandbox with no network and a restricted workspace folder.

- Use tools to inspect files and run code. Do not guess outputs.
- Python is available. Standard library only.
- Keep code small and testable. Run it, read the result, fix errors.
- If a task needs network or files outside the workspace, say so and ask the user.
- Use the provided tools. Reply without a tool call only when you have the final answer.
- Final answer: what you did, which files changed, how to run it.
```

## Project Layout

```
agent/
├── Cargo.toml
├── scripts/
│   └── fetch-assets.sh
├── assets/
│   ├── python.wasm
│   ├── python-stdlib/
│   └── checksums.txt
├── policy.toml
├── workspace/
├── src/
│   ├── main.rs        # clap CLI, wiring
│   ├── agent.rs       # loop, context mgmt
│   ├── llm.rs         # OpenAI-compatible client
│   ├── tools.rs       # ToolCall enum, router
│   ├── sandbox.rs     # Wasmtime embedding, limits
│   ├── fs.rs          # cap-std workspace wrapper
│   ├── policy.rs      # load + enforce policy
│   └── audit.rs       # JSONL audit log
└── tests/
    ├── escape.rs      # traversal, symlink, absolute path, fd tricks
    ├── limits.rs      # timeout, memory bomb, output flood
    └── agent_loop.rs  # fake Llm end-to-end
```

## Security Checklist

- [ ] `#![forbid(unsafe_code)]` in the agent crate
- [ ] No `std::process::Command` on model output anywhere
- [ ] All host file access goes through `cap-std` ≥ 4.0.3 `Dir`; the Guest preopen is read-only
- [ ] Traversal/symlink tests pass (proptest on paths)
- [ ] `while True:` guest killed by epoch timeout
- [ ] Memory bomb trapped cleanly, host unaffected
- [ ] Output flood truncated
- [ ] Prompt injection in file contents cannot alter policy
- [ ] Guest `.wasm` hashes verified at startup
- [ ] Audit log records every call and policy decision

## Known Limitations

- No native C extensions in the Python guest.
- No threads or subprocesses in the guest.
- Cold compile of `python.wasm` is ~1.3 s (17.4 MB `.cwasm`); a cached module loads and runs `print(1)` in ~0.02 s. The Wasmtime CLI's default cache hides the compile, so the embedder must use its own `.cwasm` cache.
- Verified: CPython 3.15.2 runs with a read-only stdlib mount (`import json` works, exit 0, no `__pycache__` written).
- CPython requires its stdlib dist (~13 MB, `.py` sources only, no `.pyc`) mounted read-only at `/`.
- Guest code cannot write files; artifacts leave via stdout or are written by the model with `write_file`.
- Small local models produce flaky tool calls; schemas stay flat and invalid arguments become observations.
- Wasmtime WASI API changes between versions; pin the version.

## Milestones

1. `cargo run`: Wasmtime runs hello-world Python, stdout captured.
2. Epoch timeout, memory limit, output cap enforced.
3. Read-only workspace preopen + host FS wrapper, escape tests green.
4. Agent loop with `run_code` against a local model.
5. File tools, policy.toml, audit log.
6. Overwrite confirmation, ratatui TUI.
7. Hardening pass, fuzzing, hash-pinned assets.

## Out of Scope (v1)

- Multi-user / remote hosting
- GPU access from the guest
- Runtime package installs
- Network access from the guest
- JavaScript guest (QuickJS) — post-v1; `Lang` stays extensible
- Browser UI

## Open Questions

- Opt-in host-side proxied network mode later, or stay fully offline?
