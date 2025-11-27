//! Proof Operations Commands

use clap::Subcommand;
use colored::Colorize;

use crate::client::NexusZeroClient;
use crate::config::Config;
use crate::error::CliResult;
use crate::output::{Output, Progress};

#[derive(Subcommand)]
pub enum ProofCommands {
    /// Generate a zero-knowledge proof
    Generate {
        /// Proof type (transfer, shield, unshield, compliance)
        #[arg(long, short)]
        proof_type: String,

        /// Public inputs as JSON
        #[arg(long)]
        public_inputs: String,

        /// Private inputs as JSON
        #[arg(long)]
        private_inputs: String,

        /// Wait for proof completion
        #[arg(long, short)]
        wait: bool,
    },

    /// Verify a proof
    Verify {
        /// Proof ID to verify
        #[arg(long, short)]
        proof_id: String,
    },

    /// Check proof generation status
    Status {
        /// Proof ID
        #[arg(long, short)]
        proof_id: String,
    },

    /// List recent proofs
    List {
        /// Maximum number to list
        #[arg(long, short, default_value = "10")]
        limit: u32,

        /// Filter by status (pending, completed, failed)
        #[arg(long)]
        status: Option<String>,
    },

    /// Export proof data
    Export {
        /// Proof ID
        #[arg(long, short)]
        proof_id: String,

        /// Output file path
        #[arg(long, short)]
        output: String,

        /// Format (json, hex, base64)
        #[arg(long, short, default_value = "json")]
        format: String,
    },
}

pub async fn execute(
    cmd: ProofCommands,
    client: &NexusZeroClient,
    _config: &Config,
) -> CliResult<Output> {
    match cmd {
        ProofCommands::Generate { proof_type, public_inputs, private_inputs, wait } => {
            let public: serde_json::Value = serde_json::from_str(&public_inputs)
                .map_err(|e| crate::error::CliError::InvalidInput(format!("Invalid public inputs: {}", e)))?;

            let private: serde_json::Value = serde_json::from_str(&private_inputs)
                .map_err(|e| crate::error::CliError::InvalidInput(format!("Invalid private inputs: {}", e)))?;

            let progress = Progress::spinner("Generating proof...");

            let response = client.generate_proof(crate::client::ProofRequest {
                proof_type: proof_type.clone(),
                public_inputs: public,
                private_inputs: private,
            }).await;

            match response {
                Ok(result) => {
                    if wait && result.status != "completed" {
                        progress.set_message("Waiting for proof completion...");
                        
                        // Poll for completion
                        loop {
                            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                            
                            match client.get_proof_status(&result.proof_id).await {
                                Ok(status) => {
                                    if status.status == "completed" {
                                        progress.finish_success("Proof generated");
                                        
                                        let mut output = Output::new();
                                        output.add_field("Proof ID", &status.proof_id);
                                        output.add_field("Status", &status.status);
                                        output.add_field("Public Signals", &status.public_signals.join(", "));
                                        output.set_message(&format!("{} Proof ready!", "✓".green()));
                                        return Ok(output);
                                    } else if status.status == "failed" {
                                        progress.finish_error("Proof generation failed");
                                        return Err(crate::error::CliError::ProofGeneration("Proof generation failed".to_string()));
                                    }
                                }
                                Err(e) => {
                                    progress.finish_error("Failed to check status");
                                    return Err(e);
                                }
                            }
                        }
                    } else {
                        progress.finish_success("Proof submitted");
                        
                        let mut output = Output::new();
                        output.add_field("Proof ID", &result.proof_id);
                        output.add_field("Proof Type", &proof_type);
                        output.add_field("Status", &result.status);
                        output.set_message(&format!("{} Proof generation initiated", "✓".green()));
                        Ok(output)
                    }
                }
                Err(e) => {
                    progress.finish_error("Proof generation failed");
                    Err(e)
                }
            }
        }

        ProofCommands::Verify { proof_id } => {
            let progress = Progress::spinner("Verifying proof...");

            let response = client.verify_proof(&proof_id).await;

            match response {
                Ok(result) => {
                    progress.finish_success("Verification complete");
                    
                    let mut output = Output::new();
                    output.add_field("Proof ID", &proof_id);
                    output.data = Some(result);
                    Ok(output)
                }
                Err(e) => {
                    progress.finish_error("Verification failed");
                    Err(e)
                }
            }
        }

        ProofCommands::Status { proof_id } => {
            let progress = Progress::spinner("Fetching proof status...");

            let response = client.get_proof_status(&proof_id).await;

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
                    output.add_field("Proof ID", &result.proof_id);
                    output.add_field("Status", &status_display);
                    if !result.public_signals.is_empty() {
                        output.add_field("Public Signals", &result.public_signals.join(", "));
                    }
                    Ok(output)
                }
                Err(e) => {
                    progress.finish_error("Failed to fetch status");
                    Err(e)
                }
            }
        }

        ProofCommands::List { limit, status } => {
            let mut output = Output::new();
            output.set_message(&format!(
                "Proof listing (limit: {}, status: {:?})",
                limit, status
            ));
            // TODO: Implement actual proof listing
            output.add_field("Status", "Feature pending API implementation");
            Ok(output)
        }

        ProofCommands::Export { proof_id, output: output_path, format } => {
            use base64::{Engine as _, engine::general_purpose::STANDARD};
            
            let progress = Progress::spinner("Fetching proof data...");

            let response = client.get_proof_status(&proof_id).await;

            match response {
                Ok(result) => {
                    progress.set_message("Exporting proof...");

                    let content = match format.as_str() {
                        "hex" => hex::encode(&result.proof),
                        "base64" => STANDARD.encode(&result.proof),
                        _ => serde_json::to_string_pretty(&result)?,
                    };

                    std::fs::write(&output_path, content)?;
                    progress.finish_success("Proof exported");
                    
                    let mut output = Output::new();
                    output.add_field("Proof ID", &proof_id);
                    output.add_field("Format", &format);
                    output.add_field("Output", &output_path);
                    output.set_message(&format!("{} Proof exported successfully!", "✓".green()));
                    Ok(output)
                }
                Err(e) => {
                    progress.finish_error("Export failed");
                    Err(e)
                }
            }
        }
    }
}
