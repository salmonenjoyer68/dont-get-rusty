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
User → CLI ────────┐
Browser → Web Demo ┘→ Agent Loop → LLM client (local HTTP)
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
| Sandboxed FS | `cap-std` ≥ 4.0.3 (model-reachable paths only; blocks `..` and symlink escapes) |
| CLI | `clap` |
| Web Demo | `axum` 0.8, `async-stream`, `futures-core` — optional `web` feature |
| TUI (later) | `ratatui`, `crossterm` |
| Errors | `thiserror` (libs), `anyhow` (bin) |
| Audit | hand-written JSONL (`serde_json`), RFC 3339 timestamps from `time` |
| Tests | built-in `#[test]`, `tempfile`, `proptest` for path fuzzing |

## Guest Runtimes (compiled to wasm32-wasip1, run inside Wasmtime)

| Language | Guest | Notes |
|---|---|---|
| Python | CPython 3.15.2 WASI build (brettcannon), p1 | Only guest in v1. Stdlib dist mounted read-only at `/`; 32.4 MB wasm. |
| JavaScript | quickjs-ng 0.18.0, p1 | Post-v1. 1.5 MB wasm, no preopens needed for `-e`; `Lang` stays extensible. |
| Rust | Not in v1 | Would need `rustc` to wasm32-wasip1; run compile step on host in a locked-down way, or skip. |

Assets are fetched by `scripts/fetch-assets.sh` from pinned release URLs into a gitignored `assets/`; the script verifies the archive SHA-256 and writes `checksums.txt` listing a SHA-256 for every file in the image (`python.wasm` plus all of `lib/python3.15/**`, image-relative paths). Startup reads `checksums.txt`, requires and verifies `python.wasm` and every listed file, and refuses — naming the mismatched file and the fix command — when a file is missing, a hash differs, or the checksums file is missing or corrupt. `python.wasm` also has an embedded pin, so a tampered checksums file cannot bless a tampered Guest. The whole tree is hashed once per startup. The default root is `./assets/` (falling back to the crate's assets directory); `DGR_ASSETS=<dir>` overrides it for tests and for deployments that keep the image elsewhere. Wasmtime's on-disk module cache (`Config::cache` with a directory) caches the compiled Guest under `<cwd>/.agent/cache/`, keyed by the wasm bytes and the Wasmtime version; the safe cache is used instead of unsafe `Module::deserialize`.

## Sandbox Limits (defaults)

- Memory: 256 MB via a `MemoryLimiter` that delegates to `StoreLimitsBuilder::memory_size` for every other limit. A `memory.grow` past the cap returns an error from `ResourceLimiter::memory_growing`, which Wasmtime turns into a trap the host maps to exit 137; a plain `StoreLimits` denial would surface as an ordinary Python `MemoryError` (exit 1). CPython's linear memory starts at 40 MiB, so smaller caps fail at instantiation; the cap is per linear memory.
- CPU: epoch interruption (`Config::epoch_interruption(true)`) with a ticker thread; 10 s wall-clock timeout. Fuel is not used: an open Wasmtime bug (PR #14647) loses accounting when fuel and epoch are combined, and v1 needs no deterministic cap.
- Output: stdout/stderr captured through a custom `OutputStream` that retains 64 KB per stream, drains and counts the remainder, and appends `[sandbox: output truncated; N bytes discarded]`; retained bytes decoded with `from_utf8_lossy`
- Filesystem: Guest gets a read-only preopen of the runtime dist at `/` (stdlib) and of `./workspace` at `/workspace`; all durable writes go through host tools (`cap-std`); 50 MB total quota enforced host-side
- Network: p1 sockets must be linked (CPython imports `sock_*`); `add_to_linker_sync` registers all of them and in 50.0.1 they are nonfunctional stubs. `WasiCtxBuilder` defaults deny TCP/UDP/DNS; omitting the imports makes CPython fail at instantiation.
- Env/args: no env; argv is `python.wasm -u -c <code>`
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
// tools.rs: one typed argument struct per tool, parsed from the wire arguments
#[derive(JsonSchema, Deserialize)]
pub struct RunCodeArgs { pub code: String, pub stdin: Option<String> }
// language is fixed to Python in v1; the tool schema does not expose it.

#[derive(JsonSchema, Deserialize)]
pub struct ReadFileArgs { pub path: String }

#[derive(JsonSchema, Deserialize)]
pub struct WriteFileArgs { pub path: String, pub content: String }

#[derive(JsonSchema, Deserialize)]
pub struct ListFilesArgs { pub path: Option<String> }

// llm.rs: the wire type, shared by request and response bodies. It carries a
// tool name and raw JSON arguments; the Agent Loop dispatches on the name and
// parses the arguments into the matching struct above. There is no typed
// `ToolCall` enum in v1.
pub struct ToolCall {
    pub id: String,
    pub kind: String,                 // "function"
    pub function: FunctionCall,       // name + JSON-string arguments
}

// sandbox.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang { Python }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunResult {
    pub exit_code: i32,
    pub stdout: String,               // retained output, may be truncated
    pub stderr: String,               // retained output, may be truncated
    pub timed_out: bool,
    pub truncated: bool,
    pub duration_ms: u64,
    pub stdout_bytes: u64,            // total produced, before truncation
    pub stderr_bytes: u64,            // total produced, before truncation
}
```

Tool schemas are generated with `schemars` and must stay flat: Ollama renders them into the prompt without resolving `$ref`/`$defs` and without validating arguments. Inline subschemas and avoid `anyOf`; invalid arguments from the model become error observations.

```rust
// agent.rs: the loop reports progress through this trait; the CLI passes
// `noop_observer()`, the Web Demo streams the events to the browser.
pub trait Observer: Send + Sync {
    fn step_started(&self, step: usize);
    fn assistant_text(&self, step: usize, text: &str);
    fn call_started(&self, step: usize, call: &ToolCall);
    fn observation(&self, step: usize, call: &ToolCall, observation: &Observation);
}

// sandbox.rs (shape)
pub struct Sandbox {
    engine: Engine,
    module: Module,
    runtime_image: PathBuf,
    cfg: SandboxCfg,
}

impl Sandbox {
    pub fn new(runtime_image: &Path) -> Result<Self>;
    pub fn with_cfg(runtime_image: &Path, cfg: SandboxCfg) -> Result<Self>;
    pub fn run(
        &self,
        lang: Lang,
        code: &str,
        stdin: Option<&str>,
        workspace: Option<&Path>,
    ) -> Result<RunResult, SandboxError>;
}
```

`Lang` selects the interpreter and argv (`python.wasm -u -c <code>`); v1 has only
`Lang::Python`, and the Sandbox compiles that module once at construction. `run`
is synchronous because the Agent Loop drives it from `spawn_blocking` (the
Sandbox owns its own epoch ticker thread).

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
- A failed run is `run_code` returning a non-zero `exit_code` (including traps). Three consecutive failed runs pause the loop; the CLI asks `continue? [y/N]`. Yes resets the counter; no, or a non-interactive session, ends the session with the last stderr. `--yes` answers the pause with yes, so unattended runs never block on it.
- A policy denial becomes an error observation; it never aborts the loop.
- The agent must run its code before claiming it works.
- `write_file` prompts only when the target existed before the session; a file the agent created this session overwrites silently. A pre-existing path prompts on *every* overwrite: only creation by the agent (`WriteInfo::existed == false`) makes a path a session path. `--yes` disables prompts; without a TTY a required confirmation becomes an error observation.
- Two budgets: capture keeps 64 KB per stream; before each LLM call, compact the message list to `context_budget_chars` from policy — newest results verbatim until the budget is spent, older ones stubbed to `{tool, exit_code, stdout_bytes, first stderr line}`; a newest result that alone exceeds the budget is sent as head+tail with a marker. Compaction is monotonic: once a newer Observation is stubbed or truncated, every older Observation is stubbed too, even if it would still fit. No LLM summarization.
- Invalid tool arguments become an error observation; the loop continues.
- Progress goes to an `Observer` (see `docs/adr/0004-agent-observer-seam.md`): `step_started` before each LLM request, `assistant_text` when the reply carries content, `call_started` before execution, and `observation` after the Audit Event is written. Observers are synchronous and must return promptly; they run on the Agent's worker thread.

## LLM Client

Non-streaming: one `/v1/chat/completions` request per Step, `tools` built from the `schemars` schemas, 120 s timeout, one retry on a connection error or 5xx only. A request timeout is surfaced without a retry (it is a slow model, not a broken connection), and 4xx is surfaced verbatim. `base_url` and `model` come from policy; no capability preflight.

## CLI

`dont-get-rusty "task"` — one-shot. Flags: `--workspace <dir>` (overrides `[fs].root`), `--policy <file>`, `--yes`. No subcommands in v1.

## Web Demo

`dont-get-rusty-web` (cargo feature `web`) serves the same Agent Loop to a localhost page. Scope is fixed by `docs/adr/0003-web-demo-scope.md`: loopback only, no auth, no CORS, one Task at a time.

```sh
cargo run --features web --bin dont-get-rusty-web [--workspace DIR] [--policy FILE] [--port 7878]
```

Each Task builds a fresh `Agent` (fresh Workspace handle, fresh Audit Log session) over the shared `Sandbox` and `Chat`. A second submission while one runs gets `409`. There is no cancel button: a Task ends at its answer, its Step limit, or a dropped connection only when it reaches one.

| Route | Meaning |
|---|---|
| `GET /`, `/app.js`, `/style.css` | Embedded assets (`include_str!`), no build step |
| `GET /api/state` | `model`, `workspace`, `running`, retained events (page reload) |
| `GET /api/events?after=N` | SSE: replay retained events from `N`, then live |
| `POST /api/task` | `{"task": "..."}`; `202`, `409` while running |
| `POST /api/decision/{id}` | `{"allow": bool}` answering a Confirmation |
| `GET /api/workspace?path=` | Read-only Workspace listing |
| `GET /api/workspace/file?path=` | Read-only file view, capped at 64 KiB |
| `GET /api/audit` | Last 200 Audit Events |

The SSE stream is driven by a `Notify`-woken replay loop, not a broadcast channel: events are retained per Task, so a reconnect or a page loaded mid-Task replays from index 0 and cannot lose or reorder events. The cursor resets when a new Task clears the transcript.

Confirmations reuse the core's synchronous prompts: the Web Demo blocks the Agent's worker thread (`tokio::task::block_in_place`) on a channel until the page answers, with a 10-minute timeout answered `deny`. Every Step, Tool Call, and Observation is sent to the browser with its arguments and captured output; the Audit Log stays redacted.

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

Every value has a documented default, so a partial file is valid. `--policy <file>` must name an existing file; without it `./policy.toml` is loaded when present, otherwise the built-in defaults apply. Unknown keys are rejected so a typo cannot silently fall back to a weaker default. The Policy file must live outside the Workspace (startup refuses otherwise) so the model can neither read nor modify it. `network = true` is rejected: v1 grants the Guest no network capability.

## Audit

Append-only JSONL at `<cwd>/.agent/audit.jsonl`, outside `[fs].root` so the Guest cannot read it. One event per Tool Call:

```json
{"ts":"2026-11-03T12:00:00Z","session":"…","step":3,"tool":"write_file","args_sha256":"…","args_prefix":"…","decision":"allow","error":"workspace quota exceeded: …","exit_code":null,"stdout_bytes":0,"stderr_bytes":0,"truncated":false,"duration_ms":4}
```

Args are truncated to 2 KiB plus a SHA-256 of the full payload; code, file contents, and piped `stdin` are never stored.

`decision`/`denial_reason` are the Policy verdict only: `deny` with a reason for a Policy refusal, `allow` otherwise. A call the Policy allowed but that failed on the host (quota, escape, overwrite decline, malformed arguments, Sandbox setup) stays `allow` and records the host-side message in the optional `error` field; a Policy denial has `denial_reason` and no `error`. A Guest Run records `allow` even when its exit code is non-zero: the Guest ran, and the exit code is the result. Output byte counts are totals produced before truncation.

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
| web | scripted fake `Chat` over the real routes | one Task streams to `/api/state` and SSE, workspace and audit views, real CPython Guest compiled but no Run |

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
│   ├── lib/python3.15/
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
│   ├── audit.rs       # JSONL audit log
│   ├── web/           # Web Demo server (feature `web`)
│   │   ├── hub.rs     # retained events, Confirmations, Observer
│   │   ├── decisions.rs # prompts -> browser Confirmations
│   │   ├── server.rs  # routes, AppState, one Task at a time
│   │   └── assets.rs  # include_str! of web/
│   └── bin/web.rs     # dont-get-rusty-web entry point
├── web/               # index.html, app.js, style.css (no build step)
└── tests/
    ├── escape.rs      # traversal, symlink, absolute path, fd tricks
    ├── limits.rs      # timeout, memory bomb, output flood
    ├── agent_loop.rs  # fake Llm end-to-end
    └── web.rs         # Web Demo routes over a fake Chat (feature `web`)
```

## Security Checklist

- [ ] `#![forbid(unsafe_code)]` in the agent crate
- [ ] No `std::process::Command` on model output anywhere
- [ ] All model-reachable host file access goes through `cap-std` ≥ 4.0.3 `Dir`; the Guest preopen is read-only. Host-controlled setup paths (Runtime Image verification, the Policy file, the Audit Log) use `std::fs`, because the model never supplies them
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
- The Web Demo runs one Task at a time, has no cancel button, and is loopback-only by design (`docs/adr/0003-web-demo-scope.md`).

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
- A public or hosted browser deployment (the loopback Web Demo is in scope)

## Open Questions

- Opt-in host-side proxied network mode later, or stay fully offline?
