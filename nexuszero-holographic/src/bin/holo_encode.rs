use nexuszero_holographic::cli::run_cli;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match run_cli(&args) {
        Ok(()) => Ok(()),
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    }
}

// Main is intentionally tiny; unit and integration tests are in the top-level crate

#[cfg(test)]
mod tests {
    use super::*;
    use nexuszero_holographic::cli::run_cli;
    
    #[test]
    fn test_main_runs_help() {
        // Simulate calling with --help; run_cli returns Ok(()) for help
        let args = vec!["holo_encode".to_string(), "--help".to_string()];
        let res = run_cli(&args);
        // Ensure run_cli handles help without error
        assert!(res.is_ok());
    }
}