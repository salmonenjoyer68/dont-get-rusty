//! The event hub: one in-memory transcript plus the pending Confirmation.
//!
//! One Task runs at a time, so the Hub is also the admission control: starting
//! a Task clears the transcript and refuses a second one. Every event is kept
//! for the life of the Task so a page that loads mid-Task, or reconnects its
//! SSE stream, can replay from any index.

use std::sync::mpsc::{RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::Notify;

use crate::agent::Observer;
use crate::llm::ToolCall;

/// How long the browser has to answer a Confirmation before it is answered
/// `false`. Ten minutes: long enough to read the question, short enough that a
/// forgotten tab cannot pin the one Task slot forever.
pub const DECISION_TIMEOUT: Duration = Duration::from_secs(600);

/// The most bytes of any single field sent to the browser. Tool arguments and
/// Observation text can be as large as the Workspace quota; the page does not
/// need all of it.
pub const MAX_FIELD_BYTES: usize = 64 * 1024;

/// Which question a Confirmation asks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionKind {
    /// `write_file` would overwrite a file that existed before the Task.
    Overwrite,
    /// Consecutive Runs failed; continue or stop the Agent Loop.
    Continue,
}

/// One thing the browser is told, in order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    TaskStarted {
        task: String,
    },
    StepStarted {
        step: usize,
    },
    AssistantText {
        step: usize,
        text: String,
    },
    CallStarted {
        step: usize,
        tool: String,
        arguments: String,
    },
    Observation {
        step: usize,
        tool: String,
        content: String,
        exit_code: Option<i32>,
        timed_out: bool,
        truncated: bool,
        duration_ms: u64,
    },
    DecisionAsked {
        id: String,
        kind: DecisionKind,
        question: String,
    },
    DecisionAnswered {
        id: String,
        allow: bool,
        timed_out: bool,
    },
    TaskFinished {
        answer: Option<String>,
        error: Option<String>,
        steps: usize,
        duration_ms: u64,
    },
}

/// Cuts `text` to [`MAX_FIELD_BYTES`] on a character boundary and marks the cut.
fn capped(text: String) -> String {
    if text.len() <= MAX_FIELD_BYTES {
        return text;
    }
    let mut end = MAX_FIELD_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n[web demo: display truncated; {} bytes discarded]",
        &text[..end],
        text.len() - end
    )
}

struct PendingDecision {
    id: String,
    reply: Sender<bool>,
}

struct Inner {
    running: bool,
    /// Bumped whenever a Task clears the transcript, so a subscriber can tell
    /// its cursor belongs to a previous Task.
    epoch: u64,
    events: Vec<Event>,
    pending: Option<PendingDecision>,
    next_decision: u64,
}

/// The Web Demo's shared state for events and Confirmations.
pub struct Hub {
    inner: Mutex<Inner>,
    changed: Notify,
}

impl Default for Hub {
    fn default() -> Self {
        Self::new()
    }
}

impl Hub {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                running: false,
                epoch: 0,
                events: Vec::new(),
                pending: None,
                next_decision: 0,
            }),
            changed: Notify::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn push(&self, event: Event) {
        {
            let mut inner = self.lock();
            inner.events.push(event);
        }
        self.changed.notify_one();
    }

    /// Claims the single Task slot, clearing the previous transcript.
    ///
    /// Returns `None` when a Task is already running. The guard must be kept
    /// for the whole Task: dropping it without [`TaskGuard::finish`] ends the
    /// Task with an error, so a panicking worker cannot wedge the slot.
    pub fn start(self: &Arc<Self>, task: &str) -> Option<TaskGuard> {
        let mut inner = self.lock();
        if inner.running {
            return None;
        }
        inner.running = true;
        inner.epoch += 1;
        inner.events.clear();
        inner.pending = None;
        inner.events.push(Event::TaskStarted {
            task: task.to_owned(),
        });
        drop(inner);
        self.changed.notify_one();
        Some(TaskGuard {
            hub: Arc::clone(self),
            started: Instant::now(),
            finished: false,
        })
    }

    /// Ends the Task and publishes its outcome.
    fn finish(&self, steps: usize, duration: Duration, outcome: Result<String, String>) {
        let mut inner = self.lock();
        inner.running = false;
        inner.pending = None;
        inner.events.push(Event::TaskFinished {
            answer: outcome.as_ref().ok().cloned(),
            error: outcome.as_ref().err().cloned(),
            steps,
            duration_ms: duration.as_millis() as u64,
        });
        drop(inner);
        self.changed.notify_one();
    }

    /// Asks the browser a Confirmation and blocks until it answers or the
    /// timeout expires.
    ///
    /// The Agent Loop calls the returned `bool` from a synchronous prompt
    /// inside its async body, so this function deliberately blocks. It runs
    /// under [`tokio::task::block_in_place`], which lets the multi-threaded
    /// runtime keep serving the `POST` that answers it.
    pub fn ask(&self, kind: DecisionKind, question: String) -> bool {
        let (reply, answer) = std::sync::mpsc::channel();
        let id = {
            let mut inner = self.lock();
            inner.next_decision += 1;
            let id = format!("d{}", inner.next_decision);
            inner.pending = Some(PendingDecision {
                id: id.clone(),
                reply,
            });
            inner.events.push(Event::DecisionAsked {
                id: id.clone(),
                kind,
                question,
            });
            id
        };
        self.changed.notify_one();

        let received = tokio::task::block_in_place(|| answer.recv_timeout(DECISION_TIMEOUT));
        let (allow, timed_out) = match received {
            Ok(allow) => (allow, false),
            Err(RecvTimeoutError::Timeout) => (false, true),
            // The sender is dropped only when the Task ends; deny.
            Err(RecvTimeoutError::Disconnected) => (false, true),
        };

        let mut inner = self.lock();
        let ours = inner
            .pending
            .as_ref()
            .is_some_and(|pending| pending.id == id);
        if ours {
            inner.pending = None;
            inner.events.push(Event::DecisionAnswered {
                id,
                allow,
                timed_out,
            });
        }
        drop(inner);
        self.changed.notify_one();
        allow
    }

    /// Answers the pending Confirmation. `false` means no such Confirmation is
    /// pending, which the HTTP layer reports as `404`.
    pub fn answer(&self, id: &str, allow: bool) -> bool {
        let mut inner = self.lock();
        if inner
            .pending
            .as_ref()
            .is_none_or(|pending| pending.id != id)
        {
            return false;
        }
        let pending = inner.pending.take().expect("checked above");
        // A send failure means the blocked Agent already gave up; the answer
        // is consumed either way.
        let _ = pending.reply.send(allow);
        inner.events.push(Event::DecisionAnswered {
            id: id.to_owned(),
            allow,
            timed_out: false,
        });
        drop(inner);
        self.changed.notify_one();
        true
    }

    /// The current transcript epoch, used to seed a new SSE subscriber.
    pub fn epoch(&self) -> u64 {
        self.lock().epoch
    }

    /// Reads events from `after`, unless the transcript belongs to a different
    /// Task than the caller's `epoch`; then it restarts at the beginning of the
    /// new transcript.
    ///
    /// Returns `(epoch, next_cursor, events)` read under one lock, so a Task
    /// starting between the epoch check and the slice cannot lose events.
    pub fn events_since(&self, after: usize, epoch: u64) -> (u64, usize, Vec<Event>) {
        let inner = self.lock();
        if inner.epoch != epoch {
            return (inner.epoch, 0, inner.events.clone());
        }
        let events: Vec<Event> = inner.events.iter().skip(after).cloned().collect();
        (inner.epoch, after + events.len(), events)
    }

    /// Whether a Task's reply is pending.
    pub fn is_running(&self) -> bool {
        self.lock().running
    }

    /// A consistent `(running, events)` pair for `GET /api/state`.
    pub fn snapshot(&self) -> (bool, Vec<Event>) {
        let inner = self.lock();
        (inner.running, inner.events.clone())
    }

    /// Resolves when any event is pushed.
    pub async fn changed(&self) {
        self.changed.notified().await;
    }
}

/// Keeps a Task's slot until the Task reports its outcome.
pub struct TaskGuard {
    hub: Arc<Hub>,
    started: Instant,
    finished: bool,
}

impl TaskGuard {
    /// Publishes the outcome and releases the slot.
    pub fn finish(mut self, steps: usize, outcome: Result<String, String>) {
        self.finished = true;
        self.hub.finish(steps, self.started.elapsed(), outcome);
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.hub.finish(
                0,
                self.started.elapsed(),
                Err("the Task ended without reporting an outcome".to_owned()),
            );
        }
    }
}

/// Streams the Agent Loop's progress into a [`Hub`].
pub struct HubObserver {
    hub: Arc<Hub>,
}

impl HubObserver {
    pub fn new(hub: Arc<Hub>) -> Self {
        Self { hub }
    }
}

impl Observer for HubObserver {
    fn step_started(&self, step: usize) {
        self.hub.push(Event::StepStarted { step });
    }

    fn assistant_text(&self, step: usize, text: &str) {
        self.hub.push(Event::AssistantText {
            step,
            text: text.to_owned(),
        });
    }

    fn call_started(&self, step: usize, call: &ToolCall) {
        self.hub.push(Event::CallStarted {
            step,
            tool: call.function.name.clone(),
            arguments: capped(call.function.arguments.clone()),
        });
    }

    fn observation(&self, step: usize, _call: &ToolCall, observation: &crate::agent::Observation) {
        self.hub.push(Event::Observation {
            step,
            tool: observation.tool.clone(),
            content: capped(observation.to_content()),
            exit_code: observation.exit_code,
            timed_out: observation.timed_out,
            truncated: observation.truncated,
            duration_ms: observation.duration_ms,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_new_task_invalidates_an_old_cursor() {
        let hub = Arc::new(Hub::new());
        let guard = hub.start("first").expect("first Task");
        let epoch = hub.epoch();
        let (same, cursor, events) = hub.events_since(0, epoch);
        assert_eq!(same, epoch);
        assert_eq!(
            events,
            vec![Event::TaskStarted {
                task: "first".to_owned()
            }]
        );
        drop(guard);

        let guard = hub.start("second").expect("second Task");
        let (new_epoch, restart, events) = hub.events_since(cursor, epoch);
        assert_ne!(new_epoch, epoch);
        assert_eq!(restart, 0);
        assert_eq!(
            events,
            vec![Event::TaskStarted {
                task: "second".to_owned()
            }]
        );
        drop(guard);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_confirmation_blocks_until_the_browser_answers() {
        let hub = Arc::new(Hub::new());
        let asker = tokio::spawn({
            let hub = Arc::clone(&hub);
            async move { hub.ask(DecisionKind::Overwrite, "overwrite README.md?".to_owned()) }
        });

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let id = loop {
            let (_, _, events) = hub.events_since(0, hub.epoch());
            if let Some(Event::DecisionAsked { id, .. }) = events.last() {
                break id.clone();
            }
            assert!(tokio::time::Instant::now() < deadline, "no Confirmation");
            tokio::time::sleep(Duration::from_millis(5)).await;
        };

        assert!(
            hub.answer(&id, true),
            "the pending Confirmation is answerable"
        );
        assert!(!hub.answer(&id, true), "a Confirmation is answered once");
        assert!(asker.await.expect("ask task"));

        let (_, _, events) = hub.events_since(0, hub.epoch());
        let answers: Vec<&Event> = events
            .iter()
            .filter(|event| matches!(event, Event::DecisionAnswered { .. }))
            .collect();
        assert_eq!(
            answers.len(),
            1,
            "the answer is published exactly once: {answers:?}"
        );
    }
}
