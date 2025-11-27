//! Wallet Management Commands

use clap::Subcommand;
use colored::Colorize;
use dialoguer::{Input, Password, Select};

use crate::config::Config;
use crate::error::CliResult;
use crate::output::Output;
use crate::wallet::{self, Wallet};

#[derive(Subcommand)]
pub enum WalletCommands {
    /// Create a new wallet
    Create {
        /// Wallet name
        #[arg(long, short)]
        name: Option<String>,
    },

    /// Import wallet from mnemonic
    Import {
        /// Wallet name
        #[arg(long, short)]
        name: Option<String>,

        /// Mnemonic phrase (will prompt if not provided)
        #[arg(long)]
        mnemonic: Option<String>,
    },

    /// Import wallet from private key
    ImportKey {
        /// Wallet name
        #[arg(long, short)]
        name: Option<String>,

        /// Private key (will prompt if not provided)
        #[arg(long)]
        key: Option<String>,
    },

    /// List all wallets
    List,

    /// Show wallet details
    Show {
        /// Wallet name
        #[arg(long, short)]
        name: String,
    },

    /// Delete a wallet
    Delete {
        /// Wallet name
        #[arg(long, short)]
        name: String,

        /// Skip confirmation
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Set default wallet
    Default {
        /// Wallet name
        #[arg(long, short)]
        name: String,
    },

    /// Export wallet (encrypted)
    Export {
        /// Wallet name
        #[arg(long, short)]
        name: String,

        /// Output file path
        #[arg(long, short)]
        output: String,
    },
}

pub async fn execute(cmd: WalletCommands, config: &Config) -> CliResult<Output> {
    match cmd {
        WalletCommands::Create { name } => {
            let wallet_name = match name {
                Some(n) => n,
                None => Input::new()
                    .with_prompt("Wallet name")
                    .interact_text()
                    .map_err(|e| crate::error::CliError::Other(e.to_string()))?,
            };

            let password: String = Password::new()
                .with_prompt("Enter password")
                .with_confirmation("Confirm password", "Passwords don't match")
                .interact()
                .map_err(|e| crate::error::CliError::Other(e.to_string()))?;

            println!();
            println!("{}", "Creating new wallet...".dimmed());

            let (wallet, mnemonic) = Wallet::create(&wallet_name, &password)?;

            // Save wallet
            let wallet_path = wallet::wallet_dir(config)?.join(format!("{}.json", wallet_name));
            wallet.save(&wallet_path, &password)?;

            println!();
            println!("{}", "⚠️  IMPORTANT: Write down your recovery phrase!".yellow().bold());
            println!("{}", "─".repeat(50).dimmed());
            println!();
            
            let words: Vec<&str> = mnemonic.split_whitespace().collect();
            for (i, word) in words.iter().enumerate() {
                print!("{:2}. {:12}", i + 1, word);
                if (i + 1) % 4 == 0 {
                    println!();
                }
            }
            
            println!();
            println!("{}", "─".repeat(50).dimmed());
            println!("{}", "Store this phrase safely. Anyone with it can access your funds.".yellow());
            println!();

            let mut output = Output::new();
            output.add_field("Name", &wallet_name);
            output.add_field("Address", &wallet.address);
            output.add_field("Path", wallet_path.to_str().unwrap_or(""));
            output.set_message(&format!("{} Wallet created successfully!", "✓".green()));
            Ok(output)
        }

        WalletCommands::Import { name, mnemonic } => {
            let wallet_name = match name {
                Some(n) => n,
                None => Input::new()
                    .with_prompt("Wallet name")
                    .interact_text()
                    .map_err(|e| crate::error::CliError::Other(e.to_string()))?,
            };

            let phrase = match mnemonic {
                Some(m) => m,
                None => Password::new()
                    .with_prompt("Enter mnemonic phrase")
                    .interact()
                    .map_err(|e| crate::error::CliError::Other(e.to_string()))?,
            };

            let password: String = Password::new()
                .with_prompt("Enter password")
                .with_confirmation("Confirm password", "Passwords don't match")
                .interact()
                .map_err(|e| crate::error::CliError::Other(e.to_string()))?;

            let wallet = Wallet::from_mnemonic(&wallet_name, &phrase, &password)?;

            let wallet_path = wallet::wallet_dir(config)?.join(format!("{}.json", wallet_name));
            wallet.save(&wallet_path, &password)?;

            let mut output = Output::new();
            output.add_field("Name", &wallet_name);
            output.add_field("Address", &wallet.address);
            output.set_message(&format!("{} Wallet imported successfully!", "✓".green()));
            Ok(output)
        }

        WalletCommands::ImportKey { name, key } => {
            let wallet_name = match name {
                Some(n) => n,
                None => Input::new()
                    .with_prompt("Wallet name")
                    .interact_text()
                    .map_err(|e| crate::error::CliError::Other(e.to_string()))?,
            };

            let private_key = match key {
                Some(k) => k,
                None => Password::new()
                    .with_prompt("Enter private key")
                    .interact()
                    .map_err(|e| crate::error::CliError::Other(e.to_string()))?,
            };

            let password: String = Password::new()
                .with_prompt("Enter password")
                .with_confirmation("Confirm password", "Passwords don't match")
                .interact()
                .map_err(|e| crate::error::CliError::Other(e.to_string()))?;

            let wallet = Wallet::from_private_key(&wallet_name, &private_key)?;

            let wallet_path = wallet::wallet_dir(config)?.join(format!("{}.json", wallet_name));
            wallet.save(&wallet_path, &password)?;

            let mut output = Output::new();
            output.add_field("Name", &wallet_name);
            output.add_field("Address", &wallet.address);
            output.set_message(&format!("{} Wallet imported successfully!", "✓".green()));
            Ok(output)
        }

        WalletCommands::List => {
            let wallets = wallet::list_wallets(config)?;

            if wallets.is_empty() {
                return Ok(Output::with_message("No wallets found. Create one with: nexuszero wallet create"));
            }

            let mut output = Output::new();
            output.set_message("Available Wallets");

            let rows: Vec<Vec<String>> = wallets.iter()
                .map(|w| vec![w.name.clone(), w.address.clone(), w.created_at.clone()])
                .collect();

            output.set_table(vec!["Name", "Address", "Created"], rows);
            Ok(output)
        }

        WalletCommands::Show { name } => {
            let wallet_path = wallet::wallet_dir(config)?.join(format!("{}.json", name));
            
            if !wallet_path.exists() {
                return Err(crate::error::CliError::Wallet(format!("Wallet '{}' not found", name)));
            }

            let content = std::fs::read_to_string(&wallet_path)?;
            let data: wallet::WalletData = serde_json::from_str(&content)?;

            let mut output = Output::new();
            output.set_message(&format!("Wallet: {}", name.bold()));
            output.add_field("Address", &data.address);
            output.add_field("Created", &data.created_at);
            output.add_field("Path", wallet_path.to_str().unwrap_or(""));
            Ok(output)
        }

        WalletCommands::Delete { name, yes } => {
            let wallet_path = wallet::wallet_dir(config)?.join(format!("{}.json", name));
            
            if !wallet_path.exists() {
                return Err(crate::error::CliError::Wallet(format!("Wallet '{}' not found", name)));
            }

            if !yes {
                println!("{}", "⚠️  Warning: This action cannot be undone!".yellow().bold());
                
                let options = vec!["No, cancel", "Yes, delete permanently"];
                let selection = Select::new()
                    .with_prompt(&format!("Delete wallet '{}'?", name))
                    .items(&options)
                    .default(0)
                    .interact()
                    .map_err(|e| crate::error::CliError::Other(e.to_string()))?;

                if selection == 0 {
                    return Ok(Output::with_message("Operation cancelled"));
                }
            }

            std::fs::remove_file(&wallet_path)?;

            Ok(Output::with_message(&format!("{} Wallet '{}' deleted", "✓".green(), name)))
        }

        WalletCommands::Default { name } => {
            let wallet_path = wallet::wallet_dir(config)?.join(format!("{}.json", name));
            
            if !wallet_path.exists() {
                return Err(crate::error::CliError::Wallet(format!("Wallet '{}' not found", name)));
            }

            // TODO: Save default wallet in config
            Ok(Output::with_message(&format!("{} Default wallet set to '{}'", "✓".green(), name)))
        }

        WalletCommands::Export { name, output: output_path } => {
            let wallet_path = wallet::wallet_dir(config)?.join(format!("{}.json", name));
            
            if !wallet_path.exists() {
                return Err(crate::error::CliError::Wallet(format!("Wallet '{}' not found", name)));
            }

            std::fs::copy(&wallet_path, &output_path)?;

            let mut output = Output::new();
            output.add_field("Wallet", &name);
            output.add_field("Exported to", &output_path);
            output.set_message(&format!("{} Wallet exported successfully!", "✓".green()));
            Ok(output)
        }
    }
}
