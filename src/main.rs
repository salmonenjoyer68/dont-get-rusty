//! CLI wiring: one-shot `dont-get-rusty "task"`.

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use dont_get_rusty::assets::RuntimeImage;

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

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    RuntimeImage::discover().context("refusing to run without a verified Runtime Image")?;

    run_agent(cli)
}

fn run_agent(cli: Cli) -> anyhow::Result<()> {
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
    let _ = (policy, yes);

    anyhow::bail!("the agent loop is not implemented yet; cannot execute task {task:?}")
}
