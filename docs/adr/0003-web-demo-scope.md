# The Web Demo is loopback-only, single-Task, and shares the CLI's Policy surface

The Web Demo exists to make the sandbox's guarantees visible, not to host the agent. It therefore serves on `127.0.0.1` with no authentication, no CORS, no per-visitor Workspaces, and no hosted model; it reuses the same Policy file, Workspace, Runtime Image, and Audit Log the CLI uses. One Task runs at a time and a second submission is refused with `409`.

Rejected: a public or LAN deployment (needs tenancy, a hosted model, per-user isolation, and abuse limits — a different product); per-Task temporary Workspaces (breaks the "one Workspace, one Audit Log" invariant and hides the audit trail a demo is meant to show); queueing submissions (hides contention and makes the demo feel laggy).

Consequences: the demo inherits the CLI's trust model — the operator's machine, model endpoint, and Workspace. It can write to the same Workspace as the CLI, so running both at once is the operator's problem. Any future public deployment is a new decision, not a config flag.
