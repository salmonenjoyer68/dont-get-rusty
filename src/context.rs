//! Deterministic context compaction for the Agent Loop.
//!
//! Before every LLM call the loop sends a *compacted view* of its
//! conversation instead of the conversation itself, so a long session cannot
//! overflow the model's context window. Compaction is pure: it never calls the
//! model, never writes, and returns the same output for the same input.
//!
//! # What counts toward the budget
//!
//! `budget_chars` bounds the sum of [`char`]s in the content of every message
//! of the compacted view:
//!
//! - `system` and `user` content, always counted;
//! - `assistant` content, when the model produced text;
//! - the full rendered text of every Observation (`Message::tool`).
//!
//! `ToolCall` names and arguments and the tool schemas are **not** counted;
//! they are preserved verbatim. Two consequences follow from never rewriting
//! non-Observation messages and never dropping a Tool Call's matching message:
//!
//! - if the system prompt and task alone exceed the budget, every Observation
//!   is stubbed and the view still exceeds the budget;
//! - stubs are emitted even after the budget is spent, so the view may exceed
//!   the budget by at most the size of those stubs.
//!
//! # Selection, newest first
//!
//! Observations are visited newest to oldest, against the budget left after
//! the fixed messages:
//!
//! 1. an Observation is sent verbatim while its rendered text fits;
//! 2. the newest Observation, when it does not fit, is sent as its head and
//!    tail around a `characters omitted` marker;
//! 3. every other Observation that does not fit is stubbed to its tool name,
//!    exit code, stdout/stderr byte counts, first stderr line, and — for a
//!    failed Tool Call — its host-side error message;
//! 4. stubs keep being emitted after the budget is spent.

use crate::agent::Observation;
use crate::llm::Message;

/// One conversation entry as the Agent Loop holds it.
///
/// Only Observations may be rewritten by [`compact`]; every other message is
/// passed through unchanged, byte for byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    /// A message compaction must not touch: the system prompt, the task, or an
    /// assistant reply (including its `tool_calls`).
    Message(Message),
    /// The Observation answering one assistant `tool_call_id`.
    Observation {
        tool_call_id: String,
        observation: Observation,
    },
}

/// Returns the messages to send as one LLM request.
///
/// See the module documentation for the budget rules.
pub fn compact(entries: &[Entry], budget_chars: usize) -> Vec<Message> {
    let fixed_chars: usize = entries
        .iter()
        .map(|entry| match entry {
            Entry::Message(message) => message_chars(message),
            Entry::Observation { .. } => 0,
        })
        .sum();

    let observation_indices: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| match entry {
            Entry::Observation { .. } => Some(index),
            Entry::Message(_) => None,
        })
        .collect();
    let newest = observation_indices.last().copied();

    let mut contents: Vec<Option<String>> = entries
        .iter()
        .map(|entry| match entry {
            Entry::Observation { observation, .. } => Some(observation.to_content()),
            Entry::Message(_) => None,
        })
        .collect();

    let mut remaining = budget_chars.saturating_sub(fixed_chars);

    for &index in observation_indices.iter().rev() {
        let content = contents[index]
            .take()
            .expect("an Observation entry always has rendered content");
        let length = content.chars().count();
        if length <= remaining {
            remaining -= length;
            contents[index] = Some(content);
            continue;
        }

        if Some(index) == newest {
            contents[index] = Some(head_and_tail(&content, remaining));
            remaining = 0;
            continue;
        }

        let Entry::Observation { observation, .. } = &entries[index] else {
            unreachable!("the observation index came from an Observation entry");
        };
        let stub = stub(observation, length);
        remaining = remaining.saturating_sub(stub.chars().count());
        contents[index] = Some(stub);
    }

    entries
        .iter()
        .zip(contents)
        .map(|(entry, content)| match entry {
            Entry::Message(message) => message.clone(),
            Entry::Observation { tool_call_id, .. } => Message::tool(
                tool_call_id.clone(),
                content.expect("every Observation entry was assigned content"),
            ),
        })
        .collect()
}

/// Characters this message contributes to the budget.
fn message_chars(message: &Message) -> usize {
    match message {
        Message::System { content } | Message::User { content } => content.chars().count(),
        Message::Assistant { content, .. } => {
            content.as_deref().map_or(0, |text| text.chars().count())
        }
        Message::Tool { content, .. } => content.chars().count(),
    }
}

/// The compact form of one Observation: the useful signal without the output.
///
/// `rendered_chars` is the length of [`Observation::to_content`], used only for
/// the omitted-character marker.
fn stub(observation: &Observation, rendered_chars: usize) -> String {
    let mut out = String::new();
    out.push_str(&format!("tool: {}\n", observation.tool));
    if let Some(error) = &observation.error {
        out.push_str(&format!("error: {error}\n"));
    }
    match observation.exit_code {
        Some(code) => out.push_str(&format!("exit_code: {code}\n")),
        None => out.push_str("exit_code: none\n"),
    }
    out.push_str(&format!("stdout_bytes: {}\n", observation.stdout_bytes));
    out.push_str(&format!("stderr_bytes: {}\n", observation.stderr_bytes));
    match &observation.first_stderr_line {
        Some(line) => out.push_str(&format!("first_stderr_line: {line}\n")),
        None => out.push_str("first_stderr_line: none\n"),
    }
    out.push_str(&format!(
        "--- [compacted Observation: {rendered_chars} characters omitted] ---\n"
    ));
    out
}

/// Keeps the first and last characters of `content` around a marker naming how
/// many characters were dropped.
///
/// The result is at most `target_chars` characters, unless that is smaller than
/// the marker itself, in which case the marker alone is returned.
fn head_and_tail(content: &str, target_chars: usize) -> String {
    let total = content.chars().count();
    if total <= target_chars {
        return content.to_owned();
    }

    // The omitted count is padded to the width of `total`, so the marker's
    // length is fixed; one probe determines the room left for kept text.
    let digits = total.to_string().len();
    let probe = format!("\n--- [compacted: {total:>digits$} characters omitted] ---\n");
    let keep = target_chars.saturating_sub(probe.chars().count());
    let omitted = total - keep;
    let marker = format!("\n--- [compacted: {omitted:>digits$} characters omitted] ---\n");

    let head_len = keep / 2;
    let tail_len = keep - head_len;
    let chars: Vec<char> = content.chars().collect();
    let head: String = chars[..head_len].iter().collect();
    let tail: String = chars[total - tail_len..].iter().collect();

    format!("{head}{marker}{tail}")
}
