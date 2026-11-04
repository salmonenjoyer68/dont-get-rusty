//! Web Demo entry point: serves the localhost front-end over the same Agent
//! Loop the CLI drives.

#![forbid(unsafe_code)]

use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use dont_get_rusty::assets::RuntimeImage;
use dont_get_rusty::audit::{AuditLog, DIR as AGENT_DIR};
use dont_get_rusty::fs::Workspace;
use dont_get_rusty::llm::OpenAiClient;
use dont_get_rusty::policy::{Policy, ensure_outside_workspace};
use dont_get_rusty::sandbox::{CACHE_DIR, Sandbox};
use dont_get_rusty::web::{AppState, Hub, serve};

#[derive(Debug, Parser)]
#[command(
    name = "dont-get-rusty-web",
    version,
    about = "Web Demo: drive the sandboxed agent from a localhost browser page"
)]
struct Cli {
    /// Workspace directory the agent may read and write; overrides [fs].root
    #[arg(long, value_name = "DIR")]
    workspace: Option<PathBuf>,

    /// Policy file to load instead of ./policy.toml or the built-in defaults
    #[arg(long, value_name = "FILE")]
    policy: Option<PathBuf>,

    /// Loopback port to serve on
    #[arg(long, value_name = "PORT", default_value_t = 7878)]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Load the Policy before anything else so a missing or invalid file is
    // reported before the Sandbox is compiled.
    let (policy, policy_file) = Policy::load(cli.policy.as_deref())?;
    let image =
        RuntimeImage::discover().context("refusing to run without a verified Runtime Image")?;

    let workspace_root = cli
        .workspace
        .unwrap_or_else(|| policy.workspace_root().to_path_buf());
    std::fs::create_dir_all(&workspace_root)
        .with_context(|| format!("failed to create workspace `{}`", workspace_root.display()))?;
    ensure_outside_workspace(policy_file.as_deref(), &workspace_root)?;
    let workspace = Arc::new(
        Workspace::open_with_quota(&workspace_root, policy.workspace_quota_bytes())
            .with_context(|| format!("failed to open workspace `{}`", workspace_root.display()))?,
    );

    let cwd = std::env::current_dir().context("failed to read the working directory")?;
    let audit_path = AuditLog::open(&cwd, workspace.root_path())
        .context("refusing to start without an Audit Log outside the Workspace")?
        .path()
        .to_path_buf();

    let config = policy.agent_config();
    // Same compiled-module cache as the CLI: the first Run pays the compile
    // once, later Runs do not.
    let mut sandbox_cfg = policy.sandbox_cfg();
    sandbox_cfg.cache_dir = Some(cwd.join(AGENT_DIR).join(CACHE_DIR));
    let sandbox = Arc::new(
        Sandbox::with_cfg(image.root(), sandbox_cfg).context("failed to initialize the Sandbox")?,
    );
    let chat = Arc::new(
        OpenAiClient::new(&config.base_url, &config.model, config.request_timeout)
            .context("failed to build the LLM client")?,
    );

    let state = Arc::new(AppState {
        hub: Arc::new(Hub::new()),
        sandbox,
        chat,
        workspace,
        workspace_root,
        policy,
        audit_cwd: cwd,
        audit_path,
        model: config.model.clone(),
    });

    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, cli.port));
    println!(
        "Web Demo: http://{}\nModel:     {}\nWorkspace: {}",
        addr,
        state.model,
        state.workspace_root.display()
    );
    serve(addr, state).await.context("web server failed")
}
