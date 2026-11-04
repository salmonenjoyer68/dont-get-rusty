# Sandboxed Local Coding Agent

A local coding agent whose model-generated code runs only inside a WebAssembly sandbox, with host capabilities granted explicitly.

## Language

**Agent Loop**:
The host-side loop that alternates one LLM turn with the execution of the tool calls it requests, stopping at a final answer or the step limit.
_Avoid_: conversation, session, chat

**Task**:
The user's natural-language instruction that drives one Agent Loop.
_Avoid_: prompt, query, job, request

**Confirmation**:
A question the host asks the user before an action the Policy alone cannot authorise: overwriting a file that existed before the Task, or continuing after repeated failed Runs.
_Avoid_: prompt, approval, dialog

**Step**:
One LLM turn in the Agent Loop, together with the execution of every Tool Call it returns.
_Avoid_: iteration, turn, round

**Tool Call**:
A structured request from the model naming one tool and its arguments. It is the only channel through which the model acts on the host.
_Avoid_: action, command, function call

**Observation**:
The host-produced result of one Tool Call, as fed back to the model.
_Avoid_: tool result, tool output, response

**Workspace**:
The single host directory the agent may read and write.
_Avoid_: project, cwd, root, sandbox directory

**Guest**:
The untrusted WebAssembly module that executes model-generated code inside the sandbox.
_Avoid_: runtime, VM, container

**Sandbox**:
The Wasmtime environment — engine, limits, and capability grants — in which a Guest runs.
_Avoid_: container, VM, jail

**Runtime Image**:
The read-only host directory holding a Guest's language runtime and standard library, mounted at `/` for the duration of a Run.
_Avoid_: stdlib, sysroot, asset

**Run**:
One execution of a Guest in a fresh sandbox, producing a single result: exit code, captured output, timeout flag, duration.
_Avoid_: session, job, invocation

**Web Demo**:
The localhost browser front-end that submits a Task to the Agent Loop and observes it as it runs.
_Avoid_: UI, dashboard, playground, web app

**Capability**:
An explicit grant from the host to a Guest, scoped to one Run. v1 grants read-only access to the Runtime Image and the Workspace.
_Avoid_: permission, access right

**Policy**:
Host-side rules limiting what the model may do. The model can neither read nor modify them.
_Avoid_: config, permissions, guardrails

**Audit Event**:
A host-written record of one tool call, its policy decision, and its result.
_Avoid_: log line, trace
