//! Privacy Operations Commands

use clap::Subcommand;
use colored::Colorize;

use crate::client::NexusZeroClient;
use crate::config::Config;
use crate::error::CliResult;
use crate::output::{Output, Progress};

#[derive(Subcommand)]
pub enum PrivacyCommands {
    /// Shield tokens (convert public to private)
    Shield {
        /// Amount to shield
        #[arg(long, short)]
        amount: String,

        /// Token address or symbol
        #[arg(long, short)]
        token: String,

        /// Recipient address (optional, defaults to self)
        #[arg(long, short)]
        recipient: Option<String>,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Unshield tokens (convert private to public)
    Unshield {
        /// Private note to spend
        #[arg(long, short)]
        note: String,

        /// Recipient address
        #[arg(long, short)]
        recipient: String,

        /// Amount to unshield
        #[arg(long, short)]
        amount: String,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Private transfer between shielded addresses
    Transfer {
        /// Input notes (comma-separated)
        #[arg(long, short)]
        inputs: String,

        /// Output amounts (format: amount:recipient, comma-separated)
        #[arg(long, short)]
        outputs: String,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Check shielded balance
    Balance {
        /// Commitment or note to check
        #[arg(long, short)]
        commitment: String,
    },

    /// List all notes owned by wallet
    Notes {
        /// Show spent notes too
        #[arg(long)]
        include_spent: bool,

        /// Filter by token
        #[arg(long, short)]
        token: Option<String>,
    },
}

pub async fn execute(
    cmd: PrivacyCommands,
    client: &NexusZeroClient,
    _config: &Config,
) -> CliResult<Output> {
    match cmd {
        PrivacyCommands::Shield { amount, token, recipient, yes } => {
            if !yes {
                println!("{}", "Shield Operation".bold().cyan());
                println!("Amount: {} {}", amount.yellow(), token);
                if let Some(ref r) = recipient {
                    println!("Recipient: {}", r);
                }
                println!();

                if !confirm_action("Proceed with shield operation?")? {
                    return Ok(Output::with_message("Operation cancelled"));
                }
            }

            let progress = Progress::spinner("Shielding tokens...");

            let response = client.shield(crate::client::ShieldRequest {
                amount,
                token,
                recipient_address: recipient,
            }).await;

            match response {
                Ok(result) => {
                    progress.finish_success("Shield complete");
                    
                    let mut output = Output::new();
                    output.add_field("Transaction Hash", &result.tx_hash);
                    output.add_field("Commitment", &result.commitment);
                    output.add_field("Note", &result.note);
                    output.add_field("Status", &result.status);
                    output.set_message(&format!("{} Tokens shielded successfully!", "✓".green()));
                    Ok(output)
                }
                Err(e) => {
                    progress.finish_error("Shield failed");
                    Err(e)
                }
            }
        }

        PrivacyCommands::Unshield { note, recipient, amount, yes } => {
            if !yes {
                println!("{}", "Unshield Operation".bold().cyan());
                println!("Amount: {}", amount.yellow());
                println!("Recipient: {}", recipient);
                println!();

                if !confirm_action("Proceed with unshield operation?")? {
                    return Ok(Output::with_message("Operation cancelled"));
                }
            }

            let progress = Progress::spinner("Unshielding tokens...");

            let response = client.unshield(crate::client::UnshieldRequest {
                note,
                recipient,
                amount,
            }).await;

            match response {
                Ok(result) => {
                    progress.finish_success("Unshield complete");
                    
                    let mut output = Output::new();
                    output.add_field("Transaction Hash", &result.tx_hash);
                    output.add_field("Nullifier", &result.nullifier);
                    output.add_field("Status", &result.status);
                    output.set_message(&format!("{} Tokens unshielded successfully!", "✓".green()));
                    Ok(output)
                }
                Err(e) => {
                    progress.finish_error("Unshield failed");
                    Err(e)
                }
            }
        }

        PrivacyCommands::Transfer { inputs, outputs, yes } => {
            let input_notes: Vec<String> = inputs.split(',').map(|s| s.trim().to_string()).collect();
            
            let transfer_outputs: Vec<crate::client::TransferOutput> = outputs
                .split(',')
                .map(|s| {
                    let parts: Vec<&str> = s.trim().split(':').collect();
                    crate::client::TransferOutput {
                        amount: parts.get(0).unwrap_or(&"0").to_string(),
                        recipient: parts.get(1).unwrap_or(&"").to_string(),
                    }
                })
                .collect();

            if !yes {
                println!("{}", "Private Transfer".bold().cyan());
                println!("Inputs: {} notes", input_notes.len());
                println!("Outputs: {} recipients", transfer_outputs.len());
                println!();

                if !confirm_action("Proceed with private transfer?")? {
                    return Ok(Output::with_message("Operation cancelled"));
                }
            }

            let progress = Progress::spinner("Executing private transfer...");

            let response = client.transfer(crate::client::TransferRequest {
                input_notes,
                outputs: transfer_outputs,
            }).await;

            match response {
                Ok(result) => {
                    progress.finish_success("Transfer complete");
                    
                    let mut output = Output::new();
                    output.add_field("Transaction Hash", &result.tx_hash);
                    output.add_field("Output Notes", &result.output_notes.join(", "));
                    output.add_field("Status", &result.status);
                    output.set_message(&format!("{} Private transfer successful!", "✓".green()));
                    Ok(output)
                }
                Err(e) => {
                    progress.finish_error("Transfer failed");
                    Err(e)
                }
            }
        }

        PrivacyCommands::Balance { commitment } => {
            let progress = Progress::spinner("Checking balance...");

            let response = client.get_balance(&commitment).await;

            match response {
                Ok(result) => {
                    progress.finish_clear();
                    
                    let mut output = Output::new();
                    output.set_message("Shielded Balance");
                    output.data = Some(result);
                    Ok(output)
                }
                Err(e) => {
                    progress.finish_error("Failed to check balance");
                    Err(e)
                }
            }
        }

        PrivacyCommands::Notes { include_spent, token } => {
            let mut output = Output::new();
            output.set_message(&format!(
                "Notes listing (include_spent: {}, token: {:?})",
                include_spent,
                token
            ));
            // TODO: Implement actual notes listing
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
