//! NexusZero CLI - Privacy-Preserving Transaction Interface
//!
//! A comprehensive command-line tool for interacting with the NexusZero protocol,
//! enabling privacy-preserving transactions, proof generation, and compliance operations.

use clap::{Parser, Subcommand};
use colored::Colorize;
use tracing::Level;
use tracing_subscriber::{fmt, EnvFilter};

mod commands;
mod config;
mod error;
mod client;
mod wallet;
mod output;

use commands::{privacy, proof, bridge, compliance, wallet as wallet_cmd, config as config_cmd};
use error::CliResult;

/// NexusZero CLI - Privacy-Preserving Blockchain Transactions
#[derive(Parser)]
#[command(name = "nexuszero")]
#[command(author = "NexusZero Team")]
#[command(version = "0.1.0")]
#[command(about = "Privacy-preserving transaction protocol CLI", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// API endpoint URL
    #[arg(long, env = "NEXUSZERO_API_URL", default_value = "http://localhost:8080")]
    api_url: String,

    /// Output format (text, json, yaml)
    #[arg(long, short, default_value = "text")]
    output: String,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Config file path
    #[arg(long, short)]
    config: Option<String>,

    /// Enable quiet mode (minimal output)
    #[arg(long, short)]
    quiet: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Privacy operations (shield, unshield, transfer)
    #[command(subcommand)]
    Privacy(privacy::PrivacyCommands),

    /// Zero-knowledge proof operations
    #[command(subcommand)]
    Proof(proof::ProofCommands),

    /// Cross-chain bridge operations
    #[command(subcommand)]
    Bridge(bridge::BridgeCommands),

    /// Compliance and attestation operations
    #[command(subcommand)]
    Compliance(compliance::ComplianceCommands),

    /// Wallet management
    #[command(subcommand)]
    Wallet(wallet_cmd::WalletCommands),

    /// Configuration management
    #[command(subcommand)]
    Config(config_cmd::ConfigCommands),

    /// Display system status and health
    Status {
        /// Show detailed status
        #[arg(long, short)]
        detailed: bool,
    },

    /// Display version and build information
    Version,
}

#[tokio::main]
async fn main() -> CliResult<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.verbose, cli.quiet);

    // Load configuration
    let config = config::Config::load(cli.config.as_deref())?;

    // Create API client
    let api_url = if cli.api_url != "http://localhost:8080" {
        cli.api_url.clone()
    } else {
        config.api_url.clone().unwrap_or(cli.api_url.clone())
    };

    let client = client::NexusZeroClient::new(&api_url)?;

    // Execute command
    let result = match cli.command {
        Commands::Privacy(cmd) => privacy::execute(cmd, &client, &config).await,
        Commands::Proof(cmd) => proof::execute(cmd, &client, &config).await,
        Commands::Bridge(cmd) => bridge::execute(cmd, &client, &config).await,
        Commands::Compliance(cmd) => compliance::execute(cmd, &client, &config).await,
        Commands::Wallet(cmd) => wallet_cmd::execute(cmd, &config).await,
        Commands::Config(cmd) => config_cmd::execute(cmd, &config).await,
        Commands::Status { detailed } => execute_status(&client, detailed).await,
        Commands::Version => execute_version(),
    };

    // Handle output format
    match result {
        Ok(output) => {
            if !cli.quiet {
                output::print_output(&output, &cli.output)?;
            }
            Ok(())
        }
        Err(e) => {
            if !cli.quiet {
                eprintln!("{}: {}", "Error".red().bold(), e);
            }
            std::process::exit(1);
        }
    }
}

fn init_logging(verbosity: u8, quiet: bool) {
    if quiet {
        return;
    }

    let level = match verbosity {
        0 => Level::WARN,
        1 => Level::INFO,
        2 => Level::DEBUG,
        _ => Level::TRACE,
    };

    let filter = EnvFilter::from_default_env()
        .add_directive(level.into());

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(verbosity >= 3)
        .with_file(verbosity >= 3)
        .with_line_number(verbosity >= 3)
        .init();
}

async fn execute_status(client: &client::NexusZeroClient, detailed: bool) -> CliResult<output::Output> {
    let health = client.health_check().await?;
    
    let mut status = output::Output::new();
    let api_status = if health.healthy { "✓ Healthy" } else { "✗ Unhealthy" };
    status.add_field("API Status", api_status);
    status.add_field("API Version", &health.version);
    status.add_field("Network", &health.network);

    if detailed {
        status.add_field("Prover Nodes", &health.prover_nodes.to_string());
        status.add_field("Pending Proofs", &health.pending_proofs.to_string());
        status.add_field("Supported Chains", &health.supported_chains.join(", "));
    }

    Ok(status)
}

fn execute_version() -> CliResult<output::Output> {
    let mut output = output::Output::new();
    output.add_field("Name", "NexusZero CLI");
    output.add_field("Version", env!("CARGO_PKG_VERSION"));
    output.add_field("Build", option_env!("BUILD_SHA").unwrap_or("dev"));
    output.add_field("Rust Version", env!("CARGO_PKG_RUST_VERSION").to_string().as_str());
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn verify_cli() {
        Cli::command().debug_assert();
    }
}
