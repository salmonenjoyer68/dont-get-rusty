//! Tool argument structs and the flat JSON Schema tool definitions.
//!
//! Ollama renders these schemas into the prompt without resolving `$ref` or
//! `$defs`, so every schema stays flat: plain object properties with scalar
//! types. Optional fields are declared as `Option<T>` for parsing but are
//! advertised as plain strings via [`schemars`]'s post-processing extension.
//! They remain absent from `required`, which is how the model sees them as
//! optional.

use schemars::{JsonSchema, SchemaGenerator};
use serde::Deserialize;
use serde_json::{Value, json};

/// Arguments of the `run_code` tool. `language` is fixed to Python in v1.
#[derive(JsonSchema, Deserialize, Debug, PartialEq, Eq)]
pub struct RunCodeArgs {
    /// Python source to execute in the Sandbox.
    pub code: String,
    /// Optional text piped to the program's stdin. Omit it to send no input.
    #[schemars(extend("type" = "string"))]
    pub stdin: Option<String>,
}

/// Arguments of the `read_file` tool.
#[derive(JsonSchema, Deserialize, Debug, PartialEq, Eq)]
pub struct ReadFileArgs {
    /// Workspace-relative path of the file to read.
    pub path: String,
}

/// Arguments of the `write_file` tool.
#[derive(JsonSchema, Deserialize, Debug, PartialEq, Eq)]
pub struct WriteFileArgs {
    /// Workspace-relative path of the file to write.
    pub path: String,
    /// Full file contents; replaces any existing file.
    pub content: String,
}

/// Arguments of the `list_files` tool.
#[derive(JsonSchema, Deserialize, Debug, PartialEq, Eq)]
pub struct ListFilesArgs {
    /// Optional workspace-relative directory. Omit it to list the root.
    #[schemars(extend("type" = "string"))]
    pub path: Option<String>,
}

/// Root JSON Schema for `T`, stripped of the metadata schemars adds.
///
/// OpenAI's `parameters` is a bare schema, and Ollama prints it verbatim, so
/// `$schema` and `title` are noise.
pub fn parameters_for<T: JsonSchema>() -> Value {
    let mut schema = SchemaGenerator::default()
        .into_root_schema_for::<T>()
        .to_value();
    if let Some(object) = schema.as_object_mut() {
        object.remove("$schema");
        object.remove("title");
    }
    schema
}

/// One OpenAI-style function tool definition for argument type `T`.
pub fn tool_def<T: JsonSchema>(name: &str, description: &str) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters_for::<T>(),
        }
    })
}

/// The `run_code` tool definition, advertised by the v1 Agent Loop.
pub fn run_code_tool() -> Value {
    tool_def::<RunCodeArgs>(
        "run_code",
        "Run Python code in an isolated WebAssembly sandbox with no network and no access \
         to host files. Returns the exit code, stdout, stderr, whether it timed out, and \
         how long it took. The Python standard library is available. /workspace is the \
         user's directory and is readable but read-only; write files with write_file.",
    )
}

/// The complete tool surface, in the order advertised to the model.
pub fn tool_definitions() -> Vec<Value> {
    vec![
        run_code_tool(),
        tool_def::<ReadFileArgs>(
            "read_file",
            "Read a UTF-8 text file from the user's workspace. `path` is relative to the \
             workspace root.",
        ),
        tool_def::<WriteFileArgs>(
            "write_file",
            "Write a UTF-8 text file in the user's workspace, creating parent directories as \
             needed. `path` is relative to the workspace root. Writing replaces the whole file; \
             read an existing file first if you only want to change part of it.",
        ),
        tool_def::<ListFilesArgs>(
            "list_files",
            "List the entries of a directory in the user's workspace. Omit `path` to list the \
             workspace root.",
        ),
    ]
}
