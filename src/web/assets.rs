//! The Web Demo's static files, embedded at compile time.
//!
//! There is no front-end build step: editing a file in `web/` and rebuilding
//! the binary is the whole workflow.

pub const INDEX_HTML: &str = include_str!("../../web/index.html");
pub const APP_JS: &str = include_str!("../../web/app.js");
pub const STYLE_CSS: &str = include_str!("../../web/style.css");

pub const HTML: &str = "text/html; charset=utf-8";
pub const JAVASCRIPT: &str = "text/javascript; charset=utf-8";
pub const CSS: &str = "text/css; charset=utf-8";
pub const TEXT: &str = "text/plain; charset=utf-8";
