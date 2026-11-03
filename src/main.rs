//! CLI wiring: one-shot `dont-get-rusty "task"`.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use dont_get_rusty::agent::{Agent, PausePrompt};
use dont_get_rusty::assets::RuntimeImage;
use dont_get_rusty::fs::Workspace;
use dont_get_rusty::llm::OpenAiClient;
use dont_get_rusty::policy::Policy;
use dont_get_rusty::sandbox::Sandbox;

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

    /// Workspace directory the agent may read and write; overrides [fs].root
    #[arg(long, value_name = "DIR")]
    workspace: Option<PathBuf>,

    /// Policy file to load instead of ./policy.toml or the defaults
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

    // Load the Policy before anything else so a missing or invalid file is
    // reported before the Sandbox is compiled.
    let (policy, policy_file) = Policy::load(cli.policy.as_deref())?;
    let image =
        RuntimeImage::discover().context("refusing to run without a verified Runtime Image")?;

    run_agent(cli, policy, policy_file, image).await
}

async fn run_agent(
    cli: Cli,
    policy: Policy,
    policy_file: Option<PathBuf>,
    image: RuntimeImage,
) -> anyhow::Result<()> {
    let Cli {
        task,
        workspace,
        policy: _,
        yes,
    } = cli;
    let workspace_root = workspace.unwrap_or_else(|| policy.workspace_root().to_path_buf());
    std::fs::create_dir_all(&workspace_root)
        .with_context(|| format!("failed to create workspace `{}`", workspace_root.display()))?;
    ensure_policy_outside_workspace(policy_file.as_deref(), &workspace_root)?;
    let workspace = Workspace::open_with_quota(&workspace_root, policy.workspace_quota_bytes())
        .with_context(|| format!("failed to open workspace `{}`", workspace_root.display()))?;

    let config = policy.agent_config();
    let sandbox = Arc::new(
        Sandbox::with_cfg(image.root(), policy.sandbox_cfg())
            .context("failed to initialize the Sandbox")?,
    );
    let client = OpenAiClient::new(&config.base_url, &config.model, config.request_timeout)
        .context("failed to build the LLM client")?;
    let agent = Agent::new(
        Arc::new(client),
        sandbox,
        workspace,
        config,
        policy,
        pause_prompt(yes),
    );

    let result = agent.run(&task).await?;
    println!("{}", result.answer);
    Ok(())
}

/// Refuses to start when the Policy file is inside the Workspace.
///
/// The model reaches files only through the Workspace, so a Policy file there
/// would be readable and writable through `read_file`/`write_file`. Keeping the
/// file outside guarantees the model can neither read nor modify it.
fn ensure_policy_outside_workspace(
    policy_file: Option<&Path>,
    workspace_root: &Path,
) -> anyhow::Result<()> {
    let Some(policy_file) = policy_file else {
        return Ok(());
    };
    let policy_path = policy_file
        .canonicalize()
        .with_context(|| format!("failed to resolve policy file `{}`", policy_file.display()))?;
    let workspace_path = workspace_root
        .canonicalize()
        .with_context(|| format!("failed to resolve workspace `{}`", workspace_root.display()))?;
    if policy_path.starts_with(&workspace_path) {
        anyhow::bail!(
            "policy file `{}` is inside the Workspace `{}`; keep the Policy outside the \
             Workspace so the model cannot read or modify it",
            policy_path.display(),
            workspace_path.display()
        );
    }
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
