//! Output formatting for CLI

use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::CliResult;

/// Structured output container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Output {
    #[serde(flatten)]
    pub fields: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table: Option<Table>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Table data for structured output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Output {
    /// Create new empty output
    pub fn new() -> Self {
        Self {
            fields: BTreeMap::new(),
            table: None,
            message: None,
            data: None,
        }
    }

    /// Create output with a message
    pub fn with_message(message: &str) -> Self {
        Self {
            fields: BTreeMap::new(),
            table: None,
            message: Some(message.to_string()),
            data: None,
        }
    }

    /// Create output with raw data
    pub fn with_data(data: serde_json::Value) -> Self {
        Self {
            fields: BTreeMap::new(),
            table: None,
            message: None,
            data: Some(data),
        }
    }

    /// Add a key-value field
    pub fn add_field(&mut self, key: &str, value: &str) {
        self.fields.insert(key.to_string(), value.to_string());
    }

    /// Set table data
    pub fn set_table(&mut self, headers: Vec<&str>, rows: Vec<Vec<String>>) {
        self.table = Some(Table {
            headers: headers.into_iter().map(String::from).collect(),
            rows,
        });
    }

    /// Set message
    pub fn set_message(&mut self, message: &str) {
        self.message = Some(message.to_string());
    }
}

impl Default for Output {
    fn default() -> Self {
        Self::new()
    }
}

/// Print output in specified format
pub fn print_output(output: &Output, format: &str) -> CliResult<()> {
    match format {
        "json" => print_json(output),
        "yaml" => print_yaml(output),
        _ => print_text(output),
    }
}

fn print_text(output: &Output) -> CliResult<()> {
    // Print message if present
    if let Some(ref message) = output.message {
        println!("{}", message);
        println!();
    }

    // Print fields
    if !output.fields.is_empty() {
        let max_key_len = output.fields.keys().map(|k| k.len()).max().unwrap_or(0);
        for (key, value) in &output.fields {
            println!("{:width$}  {}", format!("{}:", key).bold(), value, width = max_key_len + 1);
        }
    }

    // Print table if present
    if let Some(ref table) = output.table {
        println!();
        print_table(table);
    }

    // Print raw data if present
    if let Some(ref data) = output.data {
        if !output.fields.is_empty() || output.table.is_some() {
            println!();
        }
        println!("{}", serde_json::to_string_pretty(data)?);
    }

    Ok(())
}

fn print_table(table: &Table) {
    if table.rows.is_empty() {
        println!("(no data)");
        return;
    }

    // Calculate column widths
    let mut widths: Vec<usize> = table.headers.iter().map(|h| h.len()).collect();
    for row in &table.rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(cell.len());
            }
        }
    }

    // Print header
    let header_line: Vec<String> = table.headers.iter()
        .enumerate()
        .map(|(i, h)| format!("{:width$}", h.bold(), width = widths[i]))
        .collect();
    println!("{}", header_line.join("  "));

    // Print separator
    let separator: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    println!("{}", separator.join("  ").dimmed());

    // Print rows
    for row in &table.rows {
        let formatted: Vec<String> = row.iter()
            .enumerate()
            .map(|(i, cell)| {
                let width = widths.get(i).copied().unwrap_or(cell.len());
                format!("{:width$}", cell, width = width)
            })
            .collect();
        println!("{}", formatted.join("  "));
    }
}

fn print_json(output: &Output) -> CliResult<()> {
    println!("{}", serde_json::to_string_pretty(output)?);
    Ok(())
}

fn print_yaml(output: &Output) -> CliResult<()> {
    // Simple YAML-like output
    if let Some(ref message) = output.message {
        println!("message: \"{}\"", message);
    }
    for (key, value) in &output.fields {
        println!("{}: \"{}\"", key, value);
    }
    if let Some(ref data) = output.data {
        println!("data:");
        println!("{}", serde_json::to_string_pretty(data)?);
    }
    Ok(())
}

/// Progress indicator wrapper
pub struct Progress {
    bar: indicatif::ProgressBar,
}

impl Progress {
    /// Create a new spinner progress indicator
    pub fn spinner(message: &str) -> Self {
        let bar = indicatif::ProgressBar::new_spinner();
        bar.set_style(
            indicatif::ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
        bar.set_message(message.to_string());
        bar.enable_steady_tick(std::time::Duration::from_millis(100));
        Self { bar }
    }

    /// Create a progress bar
    pub fn bar(total: u64, message: &str) -> Self {
        let bar = indicatif::ProgressBar::new(total);
        bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{msg}\n{bar:40.cyan/blue} {pos}/{len} ({percent}%)")
                .unwrap()
                .progress_chars("█▓▒░"),
        );
        bar.set_message(message.to_string());
        Self { bar }
    }

    /// Update progress
    pub fn set(&self, pos: u64) {
        self.bar.set_position(pos);
    }

    /// Increment progress
    pub fn inc(&self, delta: u64) {
        self.bar.inc(delta);
    }

    /// Update message
    pub fn set_message(&self, message: &str) {
        self.bar.set_message(message.to_string());
    }

    /// Finish with success message
    pub fn finish_success(&self, message: &str) {
        self.bar.finish_with_message(format!("{} {}", "✓".green(), message));
    }

    /// Finish with error message
    pub fn finish_error(&self, message: &str) {
        self.bar.finish_with_message(format!("{} {}", "✗".red(), message));
    }

    /// Finish and clear
    pub fn finish_clear(&self) {
        self.bar.finish_and_clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_creation() {
        let output = Output::new();
        assert!(output.fields.is_empty());
    }

    #[test]
    fn test_output_with_fields() {
        let mut output = Output::new();
        output.add_field("key1", "value1");
        output.add_field("key2", "value2");
        assert_eq!(output.fields.len(), 2);
    }

    #[test]
    fn test_output_with_message() {
        let output = Output::with_message("Hello, World!");
        assert_eq!(output.message, Some("Hello, World!".to_string()));
    }
}
