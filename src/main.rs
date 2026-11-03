//! CLI wiring: one-shot `dont-get-rusty "task"`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use dont_get_rusty::agent::{Agent, AgentConfig};
use dont_get_rusty::assets::RuntimeImage;
use dont_get_rusty::fs::Workspace;
use dont_get_rusty::llm::OpenAiClient;
use dont_get_rusty::sandbox::Sandbox;

/// Workspace root used when neither `--workspace` nor (later) the Policy file
/// supplies one.
const DEFAULT_WORKSPACE: &str = "./workspace";

#[derive(Debug, Parser)]
#[command(
    name = "dont-get-rusty",
    version,
    about = "Runs model-generated code inside a WebAssembly sandbox"
)]
struct Cli {
    /// Task for the agent to perform
    #[arg(value_name = "task")]
    task: String,

    /// Workspace directory the agent may read and write
    #[arg(long, value_name = "DIR")]
    workspace: Option<PathBuf>,

    /// Policy file to load instead of the defaults
    #[arg(long, value_name = "FILE")]
    policy: Option<PathBuf>,

    /// Answer yes to every confirmation prompt
    #[arg(long)]
    yes: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let image =
        RuntimeImage::discover().context("refusing to run without a verified Runtime Image")?;

    run_agent(cli, image).await
}

async fn run_agent(cli: Cli, image: RuntimeImage) -> anyhow::Result<()> {
    let Cli {
        task,
        workspace,
        policy,
        yes,
    } = cli;
    // The Policy module arrives in a later ticket; until then `--workspace`
    // overrides the built-in default only.
    let workspace = workspace.unwrap_or_else(|| PathBuf::from(DEFAULT_WORKSPACE));
    std::fs::create_dir_all(&workspace)
        .with_context(|| format!("failed to create workspace `{}`", workspace.display()))?;
    let _workspace = Workspace::open(&workspace)
        .with_context(|| format!("failed to open workspace `{}`", workspace.display()))?;
    let _ = (policy, yes);

    let config = AgentConfig::default();
    let sandbox = Arc::new(Sandbox::new(image.root()).context("failed to initialize the Sandbox")?);
    let client = OpenAiClient::new(&config.base_url, &config.model, config.request_timeout)
        .context("failed to build the LLM client")?;
    let agent = Agent::new(Arc::new(client), sandbox, workspace, config);

    let result = agent.run(&task).await?;
    println!("{}", result.answer);
    Ok(())
}
