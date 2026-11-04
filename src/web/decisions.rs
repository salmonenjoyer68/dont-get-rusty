//! Bridges the Agent Loop's blocking prompts to browser Confirmations.
//!
//! The core prompts are synchronous `Fn -> bool`, which is what a terminal
//! wants. The Web Demo keeps that shape and blocks the Agent's worker thread
//! on the Hub until the page answers; see [`Hub::ask`].

use std::sync::Arc;

use crate::agent::{OverwritePrompt, PausePrompt};
use crate::web::hub::{DecisionKind, Hub};

/// Builds the two prompts the Agent asks a human about, wired to `hub`.
pub fn hub_prompts(hub: Arc<Hub>) -> (PausePrompt, OverwritePrompt) {
    let pause_hub = Arc::clone(&hub);
    let pause: PausePrompt = Arc::new(move |stderr: &str| {
        pause_hub.ask(
            DecisionKind::Continue,
            format!("Continue after repeated failed Runs?\n\nLast stderr:\n{stderr}"),
        )
    });
    let overwrite: OverwritePrompt = Arc::new(move |path: &str| {
        hub.ask(
            DecisionKind::Overwrite,
            format!("Overwrite `{path}`? The file existed before this Task."),
        )
    });
    (pause, overwrite)
}
