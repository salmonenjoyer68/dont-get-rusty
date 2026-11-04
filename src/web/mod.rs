//! The Web Demo: a localhost browser front-end over the same Agent Loop.
//!
//! The demo exists to make the sandbox's guarantees visible: the page shows
//! every Step, Tool Call, Observation (including exit codes, timeouts, and
//! truncation), the Workspace, and the Audit Log. It adds no capability the
//! CLI lacks — the Policy, Workspace, and Audit Log are the same ones.
//!
//! Scope, deliberately: loopback only, one Task at a time, no authentication,
//! no CORS, and no deployment story. See `docs/adr/0003-web-demo-scope.md`.

pub mod assets;
pub mod decisions;
pub mod hub;
pub mod server;

pub use hub::{DecisionKind, Event, Hub, HubObserver};
pub use server::{AppState, router, serve, serve_on};
