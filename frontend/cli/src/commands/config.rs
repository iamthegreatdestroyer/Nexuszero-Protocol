//! Configuration Management Commands

use clap::Subcommand;
use colored::Colorize;
use dialoguer::{Input, Select};

use crate::config::Config;
use crate::error::CliResult;
use crate::output::Output;

#[derive(Subcommand)]
pub enum ConfigCommands {
    /// Show current configuration
    Show,

    /// Set a configuration value
    Set {
        /// Configuration key
        #[arg(long, short)]
        key: String,

        /// Configuration value
        #[arg(long, short)]
        value: String,
    },

    /// Get a configuration value
    Get {
        /// Configuration key
        #[arg(long, short)]
        key: String,
    },

    /// Reset configuration to defaults
    Reset {
        /// Skip confirmation
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Interactive configuration wizard
    Init,

    /// Show configuration file path
    Path,
}

pub async fn execute(cmd: ConfigCommands, config: &Config) -> CliResult<Output> {
    match cmd {
        ConfigCommands::Show => {
            let mut output = Output::new();
            output.set_message("Current Configuration");

            output.add_field("API URL", &config.get_api_url());
            output.add_field("Network", &config.get_network());
            
            if let Some(ref path) = config.wallet_path {
                output.add_field("Wallet Path", path.to_str().unwrap_or(""));
            }

            if let Some(ref gas) = config.gas {
                output.add_field("Gas Limit Multiplier", &gas.limit_multiplier.to_string());
                output.add_field("Max Gas Price (gwei)", &gas.max_price_gwei.to_string());
            }

            if let Some(ref proof) = config.proof {
                output.add_field("Prover", &proof.prover);
                output.add_field("Proof Timeout (s)", &proof.timeout_secs.to_string());
            }

            Ok(output)
        }

        ConfigCommands::Set { key, value } => {
            let mut new_config = config.clone();

            match key.as_str() {
                "api_url" | "api-url" => {
                    new_config.api_url = Some(value.clone());
                }
                "network" => {
                    new_config.network = Some(value.clone());
                }
                "prover" => {
                    let proof = new_config.proof.get_or_insert_with(Default::default);
                    proof.prover = value.clone();
                }
                "proof_timeout" | "proof-timeout" => {
                    let timeout: u64 = value.parse()
                        .map_err(|_| crate::error::CliError::InvalidInput("Invalid timeout value".to_string()))?;
                    let proof = new_config.proof.get_or_insert_with(Default::default);
                    proof.timeout_secs = timeout;
                }
                "gas_multiplier" | "gas-multiplier" => {
                    let multiplier: f64 = value.parse()
                        .map_err(|_| crate::error::CliError::InvalidInput("Invalid multiplier value".to_string()))?;
                    let gas = new_config.gas.get_or_insert_with(Default::default);
                    gas.limit_multiplier = multiplier;
                }
                "max_gas_price" | "max-gas-price" => {
                    let price: u64 = value.parse()
                        .map_err(|_| crate::error::CliError::InvalidInput("Invalid gas price value".to_string()))?;
                    let gas = new_config.gas.get_or_insert_with(Default::default);
                    gas.max_price_gwei = price;
                }
                _ => {
                    return Err(crate::error::CliError::InvalidInput(format!("Unknown config key: {}", key)));
                }
            }

            new_config.save(None)?;

            let mut output = Output::new();
            output.add_field("Key", &key);
            output.add_field("Value", &value);
            output.set_message(&format!("{} Configuration updated", "✓".green()));
            Ok(output)
        }

        ConfigCommands::Get { key } => {
            let value = match key.as_str() {
                "api_url" | "api-url" => config.get_api_url(),
                "network" => config.get_network(),
                "prover" => config.proof.as_ref().map_or("auto".to_string(), |p| p.prover.clone()),
                "proof_timeout" | "proof-timeout" => config.proof.as_ref().map_or("300".to_string(), |p| p.timeout_secs.to_string()),
                "gas_multiplier" | "gas-multiplier" => config.gas.as_ref().map_or("1.2".to_string(), |g| g.limit_multiplier.to_string()),
                "max_gas_price" | "max-gas-price" => config.gas.as_ref().map_or("500".to_string(), |g| g.max_price_gwei.to_string()),
                _ => {
                    return Err(crate::error::CliError::InvalidInput(format!("Unknown config key: {}", key)));
                }
            };

            let mut output = Output::new();
            output.add_field(&key, &value);
            Ok(output)
        }

        ConfigCommands::Reset { yes } => {
            if !yes {
                let options = vec!["No, cancel", "Yes, reset to defaults"];
                let selection = Select::new()
                    .with_prompt("Reset configuration to defaults?")
                    .items(&options)
                    .default(0)
                    .interact()
                    .map_err(|e| crate::error::CliError::Other(e.to_string()))?;

                if selection == 0 {
                    return Ok(Output::with_message("Operation cancelled"));
                }
            }

            let default_config = Config::default();
            default_config.save(None)?;

            Ok(Output::with_message(&format!("{} Configuration reset to defaults", "✓".green())))
        }

        ConfigCommands::Init => {
            println!("{}", "NexusZero CLI Configuration Wizard".bold().cyan());
            println!();

            // API URL
            let api_url: String = Input::new()
                .with_prompt("API URL")
                .default("http://localhost:8080".to_string())
                .interact_text()
                .map_err(|e| crate::error::CliError::Other(e.to_string()))?;

            // Network
            let networks = vec!["mainnet", "testnet", "devnet"];
            let network_idx = Select::new()
                .with_prompt("Network")
                .items(&networks)
                .default(0)
                .interact()
                .map_err(|e| crate::error::CliError::Other(e.to_string()))?;
            let network = networks[network_idx].to_string();

            // Prover
            let provers = vec!["auto", "local", "network"];
            let prover_idx = Select::new()
                .with_prompt("Preferred prover")
                .items(&provers)
                .default(0)
                .interact()
                .map_err(|e| crate::error::CliError::Other(e.to_string()))?;
            let prover = provers[prover_idx].to_string();

            // Create config
            let mut new_config = Config::default();
            new_config.api_url = Some(api_url);
            new_config.network = Some(network);
            new_config.proof = Some(crate::config::ProofConfig {
                prover,
                ..Default::default()
            });

            new_config.save(None)?;

            println!();
            Ok(Output::with_message(&format!("{} Configuration saved!", "✓".green())))
        }

        ConfigCommands::Path => {
            let path = Config::default_config_path()?;

            let mut output = Output::new();
            output.add_field("Config Path", path.to_str().unwrap_or(""));
            output.add_field("Exists", &path.exists().to_string());
            Ok(output)
        }
    }
}
