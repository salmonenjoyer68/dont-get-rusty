//! Non-streaming OpenAI-compatible chat client with native tool calling.
//!
//! One call to [`Chat::chat`] is one Step's LLM turn: a single
//! `/chat/completions` request carrying the conversation and the tool
//! schemas. A reply with no `tool_calls` is the final answer; otherwise the
//! [`ToolCall`]s are executed by the Agent Loop and their Observations are fed
//! back as [`Message::tool`] messages.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Deadline for one Step, including reading the response body.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// A failure to obtain a model reply.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LlmError {
    /// Connection setup failed or the deadline expired; retried once.
    #[error("connection error: {0}")]
    Connect(String),
    /// Another transport failure, such as reading the response body.
    #[error("request error: {0}")]
    Request(String),
    /// A non-2xx response; the server's body is preserved verbatim.
    #[error("LLM returned HTTP {status}: {body}")]
    Status { status: u16, body: String },
    /// A 2xx response whose body does not parse as a chat completion.
    #[error("invalid LLM response: {0}")]
    Decode(String),
}

impl LlmError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Connect(_) => true,
            Self::Status { status, .. } => *status >= 500,
            Self::Request(_) | Self::Decode(_) => false,
        }
    }
}

/// One message in the conversation sent to the model.
///
/// `Assistant` with tool calls serializes without `content`, which the OpenAI
/// wire format accepts; `tool_calls` is omitted when empty.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    pub fn assistant(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self::Assistant {
            content,
            tool_calls,
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }
}

fn function_kind() -> String {
    "function".to_string()
}

/// A tool call, identical in request and response bodies.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Server-assigned id echoed back in the matching `tool` message. Empty
    /// when the server omitted it; the Agent Loop synthesizes one before echo.
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    pub function: FunctionCall,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: function_kind(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
}

/// The named function and its arguments.
///
/// `arguments` stays a JSON **string** on the wire, exactly as OpenAI returns
/// it; the Agent Loop parses it into the matching argument struct.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// The one request shape this client sends: one model, the conversation, the
/// tool schemas, and no streaming.
#[derive(Serialize, Debug)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    tools: &'a [Value],
    stream: bool,
}

/// A parsed chat-completions reply.
#[derive(Deserialize, Debug, Clone)]
pub struct ChatResponse {
    #[serde(default)]
    pub choices: Vec<Choice>,
}

/// One candidate reply.
#[derive(Deserialize, Debug, Clone)]
pub struct Choice {
    pub message: ResponseMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// The assistant turn returned by the model.
#[derive(Deserialize, Debug, Clone)]
pub struct ResponseMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

/// The seam the Agent Loop depends on, so tests can script replies.
#[async_trait]
pub trait Chat: Send + Sync {
    async fn chat(&self, messages: &[Message], tools: &[Value]) -> Result<ChatResponse, LlmError>;
}

/// Non-streaming client for an OpenAI-compatible `/v1` endpoint.
pub struct OpenAiClient {
    http: Client,
    url: String,
    model: String,
}

impl OpenAiClient {
    /// Builds a client for `base_url` (the API root including the version
    /// segment, e.g. `http://localhost:11434/v1`), appending
    /// `/chat/completions`.
    pub fn new(
        base_url: &str,
        model: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self, LlmError> {
        let http = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|error| LlmError::Request(error.to_string()))?;
        Ok(Self {
            http,
            url: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            model: model.into(),
        })
    }

    /// Sends the conversation and tool schemas, retrying at most once on a
    /// connection error, timeout, or 5xx. 4xx is surfaced verbatim.
    pub async fn chat(
        &self,
        messages: &[Message],
        tools: &[Value],
    ) -> Result<ChatResponse, LlmError> {
        let request = ChatRequest {
            model: &self.model,
            messages,
            tools,
            stream: false,
        };
        match self.send_once(&request).await {
            Err(error) if error.is_retryable() => self.send_once(&request).await,
            result => result,
        }
    }

    async fn send_once(&self, request: &ChatRequest<'_>) -> Result<ChatResponse, LlmError> {
        let response = self
            .http
            .post(&self.url)
            .json(request)
            .send()
            .await
            .map_err(|error| {
                if error.is_connect() || error.is_timeout() {
                    LlmError::Connect(error.to_string())
                } else {
                    LlmError::Request(error.to_string())
                }
            })?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| LlmError::Request(error.to_string()))?;

        if !status.is_success() {
            return Err(LlmError::Status {
                status: status.as_u16(),
                body,
            });
        }

        serde_json::from_str(&body)
            .map_err(|error| LlmError::Decode(format!("{error}; body: {body}")))
    }
}

#[async_trait]
impl Chat for OpenAiClient {
    async fn chat(&self, messages: &[Message], tools: &[Value]) -> Result<ChatResponse, LlmError> {
        // Type-qualified so this resolves to the inherent method rather than
        // recursing into the trait method.
        OpenAiClient::chat(self, messages, tools).await
    }
}
