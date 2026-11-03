//! A local coding agent that runs model-generated code inside a WebAssembly sandbox.

#![forbid(unsafe_code)]

pub mod agent;
pub mod assets;
pub mod fs;
pub mod llm;
pub mod policy;
pub mod sandbox;
pub mod tools;
