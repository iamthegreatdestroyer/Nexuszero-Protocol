//! Bridge Operations Commands

use clap::Subcommand;
use colored::Colorize;

use crate::client::NexusZeroClient;
use crate::config::Config;
use crate::error::CliResult;
use crate::output::{Output, Progress};

#[derive(Subcommand)]
pub enum BridgeCommands {
    /// Transfer tokens across chains
    Transfer {
        /// Source chain (ethereum, polygon, solana, etc.)
        #[arg(long, short)]
        source: String,

        /// Destination chain
        #[arg(long, short)]
        dest: String,

        /// Amount to transfer
        #[arg(long, short)]
        amount: String,

        /// Token to transfer
        #[arg(long, short)]
        token: String,

        /// Recipient address on destination chain
        #[arg(long, short)]
        recipient: String,

        /// Preserve privacy during transfer
        #[arg(long)]
        private: bool,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Check bridge transfer status
    Status {
        /// Transfer ID
        #[arg(long, short)]
        transfer_id: String,

        /// Watch for completion
        #[arg(long, short)]
        watch: bool,
    },

    /// List supported chains
    Chains,

    /// List pending transfers
    Pending {
        /// Filter by source chain
        #[arg(long)]
        source: Option<String>,

        /// Filter by destination chain
        #[arg(long)]
        dest: Option<String>,
    },

    /// Estimate bridge transfer cost
    Estimate {
        /// Source chain
        #[arg(long, short)]
        source: String,

        /// Destination chain
        #[arg(long, short)]
        dest: String,

        /// Amount to transfer
        #[arg(long, short)]
        amount: String,

        /// Token to transfer
        #[arg(long, short)]
        token: String,
    },
}

pub async fn execute(
    cmd: BridgeCommands,
    client: &NexusZeroClient,
    _config: &Config,
) -> CliResult<Output> {
    match cmd {
        BridgeCommands::Transfer {
            source,
            dest,
            amount,
            token,
            recipient,
            private,
            yes,
        } => {
            if !yes {
                println!("{}", "Cross-Chain Bridge Transfer".bold().cyan());
                println!("Source Chain:  {}", source.yellow());
                println!("Dest Chain:    {}", dest.yellow());
                println!("Amount:        {} {}", amount.green(), token);
                println!("Recipient:     {}", recipient);
                println!("Privacy:       {}", if private { "Enabled".green() } else { "Disabled".dimmed() });
                println!();

                if !confirm_action("Proceed with bridge transfer?")? {
                    return Ok(Output::with_message("Operation cancelled"));
                }
            }

            let progress = Progress::spinner("Initiating bridge transfer...");

            let response = client.bridge_transfer(crate::client::BridgeRequest {
                source_chain: source.clone(),
                dest_chain: dest.clone(),
                amount,
                token,
                recipient,
                preserve_privacy: private,
            }).await;

            match response {
                Ok(result) => {
                    progress.finish_success("Bridge transfer initiated");
                    
                    let mut output = Output::new();
                    output.add_field("Transfer ID", &result.transfer_id);
                    output.add_field("Source Chain", &source);
                    output.add_field("Dest Chain", &dest);
                    output.add_field("Status", &result.status);
                    output.add_field("Confirmations", &format!("{}/{}", result.confirmations, result.required_confirmations));
                    
                    if let Some(ref tx) = result.source_tx_hash {
                        output.add_field("Source TX", tx);
                    }
                    
                    output.set_message(&format!("{} Bridge transfer initiated!", "✓".green()));
                    Ok(output)
                }
                Err(e) => {
                    progress.finish_error("Bridge transfer failed");
                    Err(e)
                }
            }
        }

        BridgeCommands::Status { transfer_id, watch } => {
            let progress = Progress::spinner("Fetching transfer status...");

            if watch {
                loop {
                    match client.bridge_status(&transfer_id).await {
                        Ok(result) => {
                            progress.set_message(&format!(
                                "Status: {} ({}/{} confirmations)",
                                result.status,
                                result.confirmations,
                                result.required_confirmations
                            ));

                            if result.status == "completed" {
                                progress.finish_success("Transfer completed");
                                
                                let mut output = Output::new();
                                output.add_field("Transfer ID", &result.transfer_id);
                                output.add_field("Status", &result.status.green().to_string());
                                
                                if let Some(ref tx) = result.source_tx_hash {
                                    output.add_field("Source TX", tx);
                                }
                                if let Some(ref tx) = result.dest_tx_hash {
                                    output.add_field("Dest TX", tx);
                                }
                                
                                return Ok(output);
                            } else if result.status == "failed" {
                                progress.finish_error("Transfer failed");
                                return Err(crate::error::CliError::Bridge("Transfer failed".to_string()));
                            }

                            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                        }
                        Err(e) => {
                            progress.finish_error("Failed to fetch status");
                            return Err(e);
                        }
                    }
                }
            } else {
                let response = client.bridge_status(&transfer_id).await;

                match response {
                    Ok(result) => {
                        progress.finish_clear();
                        
                        let status_display = match result.status.as_str() {
                            "completed" => result.status.green().to_string(),
                            "pending" => result.status.yellow().to_string(),
                            "failed" => result.status.red().to_string(),
                            _ => result.status.clone(),
                        };

                        let mut output = Output::new();
                        output.add_field("Transfer ID", &result.transfer_id);
                        output.add_field("Status", &status_display);
                        output.add_field("Confirmations", &format!("{}/{}", result.confirmations, result.required_confirmations));
                        
                        if let Some(ref tx) = result.source_tx_hash {
                            output.add_field("Source TX", tx);
                        }
                        if let Some(ref tx) = result.dest_tx_hash {
                            output.add_field("Dest TX", tx);
                        }
                        
                        Ok(output)
                    }
                    Err(e) => {
                        progress.finish_error("Failed to fetch status");
                        Err(e)
                    }
                }
            }
        }

        BridgeCommands::Chains => {
            let progress = Progress::spinner("Fetching supported chains...");

            let response = client.list_supported_chains().await;

            match response {
                Ok(chains) => {
                    progress.finish_clear();
                    
                    let mut output = Output::new();
                    output.set_message("Supported Chains");
                    
                    let rows: Vec<Vec<String>> = chains.iter()
                        .enumerate()
                        .map(|(i, chain)| vec![(i + 1).to_string(), chain.clone()])
                        .collect();
                    
                    output.set_table(vec!["#", "Chain"], rows);
                    Ok(output)
                }
                Err(e) => {
                    progress.finish_error("Failed to fetch chains");
                    Err(e)
                }
            }
        }

        BridgeCommands::Pending { source, dest } => {
            let mut output = Output::new();
            output.set_message(&format!(
                "Pending transfers (source: {:?}, dest: {:?})",
                source, dest
            ));
            output.add_field("Status", "Feature pending API implementation");
            Ok(output)
        }

        BridgeCommands::Estimate { source, dest, amount, token } => {
            let mut output = Output::new();
            output.set_message("Bridge Transfer Estimate");
            output.add_field("Source", &source);
            output.add_field("Destination", &dest);
            output.add_field("Amount", &format!("{} {}", amount, token));
            output.add_field("Estimated Fee", "~0.001 ETH");
            output.add_field("Estimated Time", "~15 minutes");
            output.add_field("Status", "Feature pending API implementation");
            Ok(output)
        }
    }
}

fn confirm_action(prompt: &str) -> CliResult<bool> {
    use dialoguer::Confirm;

    Confirm::new()
        .with_prompt(prompt)
        .default(false)
        .interact()
        .map_err(|e| crate::error::CliError::Other(e.to_string()))
}
