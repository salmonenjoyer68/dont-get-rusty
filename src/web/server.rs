//! HTTP surface of the Web Demo.
//!
//! One Task at a time, one origin, no authentication: the listener is bound to
//! loopback by the binary and the page is served from the same origin, so there
//! is no CORS surface. The Agent Loop is the same one the CLI drives; this
//! module only translates it to HTTP.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_stream::stream;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use crate::agent::{Agent, AgentResult, Observer};
use crate::audit::AuditLog;
use crate::fs::Workspace;
use crate::llm::Chat;
use crate::policy::Policy;
use crate::sandbox::Sandbox;
use crate::web::assets;
use crate::web::decisions;
use crate::web::hub::{Event, Hub, HubObserver, MAX_FIELD_BYTES};

/// Everything the routes need. Built once by the binary; tests build it
/// directly with a fake [`Chat`].
pub struct AppState {
    pub hub: Arc<Hub>,
    pub sandbox: Arc<Sandbox>,
    pub chat: Arc<dyn Chat>,
    /// Opened once for browsing; every Task opens its own handle so the
    /// Agent's per-Task write tracking starts empty.
    pub workspace: Arc<Workspace>,
    pub workspace_root: PathBuf,
    pub policy: Policy,
    /// Directory holding `.agent/audit.jsonl`, outside the Workspace.
    pub audit_cwd: PathBuf,
    pub audit_path: PathBuf,
    pub model: String,
}

/// Serves the Web Demo on `addr`. The caller binds loopback.
pub async fn serve(addr: SocketAddr, state: Arc<AppState>) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    serve_on(listener, state).await
}

/// Serves an already-bound listener; tests bind port `0` with this.
pub async fn serve_on(listener: TcpListener, state: Arc<AppState>) -> std::io::Result<()> {
    axum::serve(listener, router(state)).await
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/app.js", get(script))
        .route("/style.css", get(stylesheet))
        .route("/api/state", get(snapshot))
        .route("/api/events", get(events))
        .route("/api/task", post(submit))
        .route("/api/decision/{id}", post(answer))
        .route("/api/workspace", get(list_workspace))
        .route("/api/workspace/file", get(read_workspace_file))
        .route("/api/audit", get(audit))
        .with_state(state)
}

async fn index() -> Response {
    ([(header::CONTENT_TYPE, assets::HTML)], assets::INDEX_HTML).into_response()
}

async fn script() -> Response {
    ([(header::CONTENT_TYPE, assets::JAVASCRIPT)], assets::APP_JS).into_response()
}

async fn stylesheet() -> Response {
    ([(header::CONTENT_TYPE, assets::CSS)], assets::STYLE_CSS).into_response()
}

#[derive(Serialize)]
struct StateView {
    model: String,
    workspace: String,
    running: bool,
    events: Vec<Event>,
}

async fn snapshot(State(state): State<Arc<AppState>>) -> Json<StateView> {
    let (running, events) = state.hub.snapshot();
    Json(StateView {
        model: state.model.clone(),
        workspace: state.workspace_root.display().to_string(),
        running,
        events,
    })
}

#[derive(Deserialize)]
struct EventsQuery {
    after: Option<usize>,
}

/// Replays the transcript from `after`, then streams new events.
///
/// The cursor resets when a new Task clears the transcript (the retained
/// events are then shorter than the requested index), so a page connected
/// across a Task boundary still sees the new one from the start.
async fn events(
    State(state): State<Arc<AppState>>,
    Query(query): Query<EventsQuery>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let hub = Arc::clone(&state.hub);
    let mut cursor = query.after.unwrap_or_default();
    let mut epoch = hub.epoch();
    let stream = stream! {
        loop {
            let (current, next_cursor, batch) = hub.events_since(cursor, epoch);
            if current != epoch {
                // A new Task cleared the transcript; replay it from the start.
                epoch = current;
                cursor = 0;
                continue;
            }
            cursor = next_cursor;
            if batch.is_empty() {
                hub.changed().await;
                continue;
            }
            for event in batch {
                let data = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_owned());
                yield Ok(SseEvent::default().data(data));
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[derive(Deserialize)]
struct TaskRequest {
    task: String,
}

async fn submit(State(state): State<Arc<AppState>>, Json(request): Json<TaskRequest>) -> Response {
    let task = request.task.trim();
    if task.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "task must not be empty");
    }
    let Some(guard) = state.hub.start(task) else {
        return api_error(StatusCode::CONFLICT, "a Task is already running");
    };
    let task = task.to_owned();
    let state = Arc::clone(&state);
    tokio::spawn(async move {
        match run_task(&state, &task).await {
            Ok(result) => guard.finish(result.steps, Ok(result.answer)),
            Err(message) => guard.finish(0, Err(message)),
        }
    });
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "started": true })),
    )
        .into_response()
}

/// One Task: a fresh Agent (so per-Task write tracking starts empty), a fresh
/// Workspace handle, and a fresh Audit Log session — all against the same
/// Policy and Sandbox the CLI uses.
async fn run_task(state: &AppState, task: &str) -> Result<AgentResult, String> {
    let workspace =
        Workspace::open_with_quota(&state.workspace_root, state.policy.workspace_quota_bytes())
            .map_err(|error| format!("failed to open the Workspace: {error}"))?;
    let audit = AuditLog::open(&state.audit_cwd, workspace.root_path())
        .map_err(|error| format!("refusing to start without an Audit Log: {error}"))?;
    let (pause, overwrite) = decisions::hub_prompts(Arc::clone(&state.hub));
    let observer: Arc<dyn Observer> = Arc::new(HubObserver::new(Arc::clone(&state.hub)));
    let agent = Agent::new(
        Arc::clone(&state.chat),
        Arc::clone(&state.sandbox),
        workspace,
        state.policy.agent_config(),
        state.policy.clone(),
        audit,
        pause,
        overwrite,
        observer,
    );
    agent.run(task).await.map_err(|error| error.to_string())
}

#[derive(Deserialize)]
struct AnswerRequest {
    allow: bool,
}

async fn answer(
    State(state): State<Arc<AppState>>,
    UrlPath(id): UrlPath<String>,
    Json(request): Json<AnswerRequest>,
) -> Response {
    if state.hub.answer(&id, request.allow) {
        StatusCode::NO_CONTENT.into_response()
    } else {
        api_error(StatusCode::NOT_FOUND, "no such Confirmation is pending")
    }
}

#[derive(Deserialize)]
struct PathQuery {
    path: Option<String>,
}

#[derive(Serialize)]
struct EntryView {
    name: String,
    kind: String,
    size: u64,
}

async fn list_workspace(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PathQuery>,
) -> Response {
    let path = query.path.as_deref().filter(|path| !path.is_empty());
    match state.workspace.list_files(path.map(Path::new)) {
        Ok(entries) => {
            let entries: Vec<EntryView> = entries
                .into_iter()
                .map(|entry| EntryView {
                    name: entry.name,
                    kind: entry.kind.to_string(),
                    size: entry.size,
                })
                .collect();
            Json(entries).into_response()
        }
        Err(error) => api_error(StatusCode::BAD_REQUEST, error.to_string()),
    }
}

async fn read_workspace_file(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PathQuery>,
) -> Response {
    let Some(path) = query.path.filter(|path| !path.is_empty()) else {
        return api_error(StatusCode::BAD_REQUEST, "`path` is required");
    };
    match state.workspace.read_file(&path) {
        Ok(bytes) => {
            let text = String::from_utf8_lossy(&bytes).into_owned();
            let text = if text.len() > MAX_FIELD_BYTES {
                let mut end = MAX_FIELD_BYTES;
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                format!(
                    "{}\n[web demo: display truncated; {} bytes discarded]",
                    &text[..end],
                    text.len() - end
                )
            } else {
                text
            };
            ([(header::CONTENT_TYPE, assets::TEXT)], text).into_response()
        }
        Err(error) => api_error(StatusCode::NOT_FOUND, error.to_string()),
    }
}

/// The last 200 Audit Events, in file order. The page shows the same JSONL the
/// CLI writes; the Guest can never read it.
async fn audit(State(state): State<Arc<AppState>>) -> Response {
    let text = match std::fs::read_to_string(&state.audit_path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()),
    };
    let events: Vec<serde_json::Value> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let start = events.len().saturating_sub(200);
    Json(events[start..].to_vec()).into_response()
}

fn api_error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": message.into() }))).into_response()
}
