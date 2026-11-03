//! CLI wiring: one-shot `dont-get-rusty "task"`.

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use dont_get_rusty::agent::{Agent, AgentConfig, PausePrompt};
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

    /// Answer yes to every confirmation prompt, including the pause after
    /// repeated failed Runs
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
    let _ = policy;

    let config = AgentConfig::default();
    let sandbox = Arc::new(Sandbox::new(image.root()).context("failed to initialize the Sandbox")?);
    let client = OpenAiClient::new(&config.base_url, &config.model, config.request_timeout)
        .context("failed to build the LLM client")?;
    let agent = Agent::new(
        Arc::new(client),
        sandbox,
        workspace,
        config,
        pause_prompt(yes),
    );

    let result = agent.run(&task).await?;
    println!("{}", result.answer);
    Ok(())
}

/// The failed-Run pause decision.
///
/// `--yes` answers every pause with yes, so unattended runs never block on the
/// prompt. Otherwise the question is asked only when both stdin and stdout are
/// terminals; a non-interactive session answers no, ending the loop with the
/// last stderr. For security, a partial answer (anything but `y`/`yes`) is no.
fn pause_prompt(yes: bool) -> PausePrompt {
    if yes {
        return Arc::new(|_last_stderr| true);
    }

    Arc::new(|last_stderr: &str| {
        if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
            return false;
        }
        if !last_stderr.trim().is_empty() {
            eprintln!("{last_stderr}");
        }
        print!("continue? [y/N] ");
        if std::io::stdout().flush().is_err() {
            return false;
        }
        let mut answer = String::new();
        match std::io::stdin().read_line(&mut answer) {
            Ok(0) | Err(_) => false,
            Ok(_) => matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes"),
        }
    })
}
