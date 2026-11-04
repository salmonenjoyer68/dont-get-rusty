# The Agent Loop reports progress through a synchronous `Observer` trait

`Agent::run` used to report nothing between Steps, which is fine for a one-shot CLI but makes any live view impossible: the Audit Log deliberately omits assistant text and argument bodies. The loop now takes an `Arc<dyn Observer>` in `Agent::new` and calls `step_started`, `assistant_text`, `call_started`, and `observation` synchronously as it advances; `noop_observer()` keeps the CLI silent and the `run` signature unchanged.

Rejected: emitting into a `tokio::mpsc` channel (makes the core depend on runtime/event-bus semantics, and forces every caller to drain it); tailing the Audit Log in the front-end (cannot carry assistant text or arguments, by design); a callback closure parameter on `run` (would not fit the existing collaborator-injection style).

Consequences: an Observer must return promptly and must not block, because it runs on the Agent's worker thread; anything slow is the implementation's job to move elsewhere. The trait is public, so adding a method later is a breaking change for external implementors.
