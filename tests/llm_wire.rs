//! Wire-level tests for the LLM client against a dependency-free stub server.
//!
//! The stub binds an ephemeral port, serves one scripted response per TCP
//! connection, and captures the raw request body, so these tests assert what
//! the client actually sends and how it parses what comes back.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::Duration;

use dont_get_rusty::llm::{
    Chat, ChatResponse, Choice, DEFAULT_TIMEOUT, LlmError, Message, OpenAiClient, ResponseMessage,
    ToolCall,
};
use dont_get_rusty::tools::{RunCodeArgs, tool_definitions};
use serde_json::{Value, json};

mod stub {
    use super::*;

    pub struct StubServer {
        pub base_url: String,
        pub requests: Receiver<String>,
    }

    pub struct Response {
        pub status: u16,
        pub body: String,
    }

    impl Response {
        pub fn json(status: u16, body: impl Into<String>) -> Self {
            Self {
                status,
                body: body.into(),
            }
        }
    }

    impl StubServer {
        /// Serves exactly one connection per scripted response, in order.
        pub fn spawn(responses: Vec<Response>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let (tx, rx) = channel();

            thread::spawn(move || {
                for response in responses {
                    let (mut stream, _) = listener.accept().unwrap();
                    let body = read_http_request(&mut stream);
                    let _ = tx.send(body);

                    let reason = match response.status {
                        200 => "OK",
                        400 => "Bad Request",
                        500 => "Internal Server Error",
                        _ => "Status",
                    };
                    let head = format!(
                        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        response.status,
                        reason,
                        response.body.len()
                    );
                    stream.write_all(head.as_bytes()).unwrap();
                    stream.write_all(response.body.as_bytes()).unwrap();
                    stream.flush().unwrap();
                }
            });

            Self {
                base_url: format!("http://{addr}"),
                requests: rx,
            }
        }
    }

    fn read_http_request(stream: &mut TcpStream) -> String {
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            let read = stream.read(&mut byte).unwrap();
            assert_ne!(read, 0, "client closed before sending complete headers");
            head.push(byte[0]);
            if head.ends_with(b"\r\n\r\n") {
                break;
            }
        }
        let head = String::from_utf8(head).unwrap();
        let content_length = head
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap_or(0);

        let mut body = vec![0u8; content_length];
        stream.read_exact(&mut body).unwrap();
        String::from_utf8(body).unwrap()
    }
}

use stub::{Response, StubServer};

fn client(base_url: &str) -> OpenAiClient {
    OpenAiClient::new(base_url, "qwen3.5:4b", DEFAULT_TIMEOUT).unwrap()
}

const TEXT_REPLY: &str =
    r#"{"choices":[{"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}]}"#;

const TOOL_REPLY: &str = r#"{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"run_code","arguments":"{\"code\":\"print(1)\",\"stdin\":null}"}}]},"finish_reason":"tool_calls"}]}"#;

#[test]
fn tool_schemas_are_flat() {
    let tools = tool_definitions();
    assert_eq!(tools.len(), 4);

    let encoded = serde_json::to_string(&tools).unwrap();
    for forbidden in ["$ref", "$defs", "anyOf", "oneOf", "$schema"] {
        assert!(
            !encoded.contains(forbidden),
            "tool schemas must not contain {forbidden}: {encoded}"
        );
    }

    let run_code = &tools[0];
    assert_eq!(run_code["type"], "function");
    assert_eq!(run_code["function"]["name"], "run_code");
    assert_eq!(
        run_code["function"]["parameters"],
        json!({
            "type": "object",
            "description": "Arguments of the `run_code` tool. `language` is fixed to Python in v1.",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to execute in the Sandbox.",
                },
                "stdin": {
                    "type": "string",
                    "description": "Optional text piped to the program's stdin. Omit it to send no input.",
                },
            },
            "required": ["code"],
        })
    );

    let names: Vec<_> = tools
        .iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["run_code", "read_file", "write_file", "list_files"]);

    for tool in &tools {
        let parameters = &tool["function"]["parameters"];
        assert_eq!(parameters["type"], "object");
        let properties = parameters["properties"].as_object().unwrap();
        assert!(!properties.is_empty());
        for (name, property) in properties {
            assert_eq!(
                property["type"], "string",
                "property {name} must be a flat string"
            );
            assert!(
                property["description"]
                    .as_str()
                    .is_some_and(|description| !description.is_empty()),
                "property {name} needs a description for the model"
            );
        }
    }

    // Optional fields are absent from `required`, which is the only signal the
    // model gets that they may be omitted.
    assert_eq!(tools[3]["function"]["parameters"].get("required"), None);
}

#[test]
fn messages_serialize_like_openai() {
    let call = ToolCall::new("call_1", "read_file", r#"{"path":"a.txt"}"#);
    let messages = vec![
        Message::system("You are a coding agent running locally."),
        Message::user("read a.txt"),
        Message::assistant(None, vec![call]),
        Message::tool("call_1", r#"{"path":"a.txt","content":"hi"}"#),
    ];

    assert_eq!(
        serde_json::to_value(&messages).unwrap(),
        json!([
            { "role": "system", "content": "You are a coding agent running locally." },
            { "role": "user", "content": "read a.txt" },
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "read_file", "arguments": "{\"path\":\"a.txt\"}" }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{\"path\":\"a.txt\",\"content\":\"hi\"}"
            }
        ])
    );
}

#[tokio::test]
async fn sends_exact_request_shape_and_parses_text_reply() {
    let server = StubServer::spawn(vec![Response::json(200, TEXT_REPLY)]);
    let client = client(&server.base_url);
    let tools = tool_definitions();
    let messages = vec![
        Message::system("You are a coding agent running locally."),
        Message::user("say hello"),
    ];

    let response = client.chat(&messages, &tools).await.unwrap();
    let message = &response.choices[0].message;
    assert_eq!(message.content.as_deref(), Some("Hello"));
    assert!(message.tool_calls.is_empty());
    assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));

    let body: Value = serde_json::from_str(&server.requests.recv().unwrap()).unwrap();
    assert_eq!(body["model"], "qwen3.5:4b");
    assert_eq!(body["stream"], false);
    assert_eq!(body["messages"], serde_json::to_value(&messages).unwrap());
    assert_eq!(body["tools"], serde_json::to_value(&tools).unwrap());
    assert_eq!(body["tools"].as_array().unwrap().len(), 4);
    for tool in body["tools"].as_array().unwrap() {
        assert_eq!(tool["type"], "function");
        assert!(tool["function"]["parameters"]["properties"].is_object());
    }

    // Exactly one request: no retry on success.
    assert!(server.requests.try_recv().is_err());
}

#[tokio::test]
async fn retries_once_on_5xx_and_parses_tool_calls() {
    let server = StubServer::spawn(vec![
        Response::json(500, r#"{"error":"temporarily unavailable"}"#),
        Response::json(200, TOOL_REPLY),
    ]);
    let client = client(&server.base_url);
    let messages = vec![Message::user("compute something")];

    let response = client.chat(&messages, &tool_definitions()).await.unwrap();
    let message = &response.choices[0].message;
    assert_eq!(message.content.as_deref(), Some(""));
    assert_eq!(message.tool_calls.len(), 1);

    let call = &message.tool_calls[0];
    assert_eq!(call.id, "call_1");
    assert_eq!(call.kind, "function");
    assert_eq!(call.function.name, "run_code");
    assert_eq!(
        call.function.arguments,
        r#"{"code":"print(1)","stdin":null}"#
    );

    let args: RunCodeArgs = serde_json::from_str(&call.function.arguments).unwrap();
    assert_eq!(args.code, "print(1)");
    assert_eq!(args.stdin, None);

    // The retry resent the identical body.
    let first = server.requests.recv().unwrap();
    let second = server.requests.recv().unwrap();
    assert_eq!(first, second);
    assert!(server.requests.try_recv().is_err());
}

#[tokio::test]
async fn surfaces_4xx_verbatim_without_retry() {
    let body = r#"{"error":"registry.ollama.ai/library/qwen3.5:4b does not support tools"}"#;
    let server = StubServer::spawn(vec![Response::json(400, body)]);
    let client = client(&server.base_url);

    let error = client
        .chat(&[Message::user("go")], &tool_definitions())
        .await
        .unwrap_err();

    assert_eq!(
        error,
        LlmError::Status {
            status: 400,
            body: body.to_string(),
        }
    );
    let error = error.to_string();
    assert!(error.contains("HTTP 400"));
    assert!(error.contains("does not support tools"));

    // 4xx is never retried: the stub scripted one response, so a retry would
    // panic inside the server thread and never satisfy a second recv.
    server.requests.recv().unwrap();
    assert!(server.requests.try_recv().is_err());
}

#[tokio::test]
async fn connection_errors_surface_as_connect() {
    // Bind and immediately release a port so nothing is listening.
    let address = {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap()
    };
    let client = OpenAiClient::new(
        &format!("http://{address}"),
        "m",
        Duration::from_millis(500),
    )
    .unwrap();

    let error = client
        .chat(&[Message::user("go")], &tool_definitions())
        .await
        .unwrap_err();
    assert!(matches!(error, LlmError::Connect(_)), "{error:?}");
}

#[tokio::test]
async fn timeout_is_not_retried() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let accepts = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&accepts);

    thread::spawn(move || {
        let mut held = Vec::new();
        while let Ok((stream, _)) = listener.accept() {
            counter.fetch_add(1, Ordering::SeqCst);
            held.push(stream); // Keep the socket open and never respond.
        }
    });

    let client = OpenAiClient::new(
        &format!("http://{address}"),
        "m",
        Duration::from_millis(150),
    )
    .unwrap();

    let error = client
        .chat(&[Message::user("go")], &tool_definitions())
        .await
        .unwrap_err();
    assert!(matches!(error, LlmError::Request(_)), "{error:?}");
    assert_eq!(
        accepts.load(Ordering::SeqCst),
        1,
        "a request timeout is not retried"
    );
}

struct FakeChat;

#[async_trait::async_trait]
impl Chat for FakeChat {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[Value],
    ) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            choices: vec![Choice {
                message: ResponseMessage {
                    content: Some("from the fake".to_string()),
                    tool_calls: Vec::new(),
                },
                finish_reason: Some("stop".to_string()),
            }],
        })
    }
}

#[tokio::test]
async fn chat_trait_is_object_safe_and_fakeable() {
    let chat: Box<dyn Chat> = Box::new(FakeChat);

    let response = chat.chat(&[], &[]).await.unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("from the fake")
    );
}
