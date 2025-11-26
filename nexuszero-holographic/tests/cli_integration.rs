use nexuszero_holographic::cli::{run_cli, encode_path, decode_path};
use std::fs::{File, remove_file};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_path(suffix: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("nexuszero_test_{}{}", nanos, suffix));
    p
}

#[test]
fn integration_encode_and_decode_roundtrip() {
    let input_path = unique_temp_path("_cli_in.bin");
    let out_path = unique_temp_path("_cli_out.mps");
    let mut f = File::create(&input_path).unwrap();
    f.write_all(&[7u8; 16]).unwrap();
    f.flush().unwrap();

    // encode
    let res = encode_path(input_path.to_str().unwrap(), 8, Some(out_path.to_str().unwrap()), true, true);
    assert!(res.is_ok());
    assert!(out_path.exists());

    // Now test decode
    let res2 = decode_path(out_path.to_str().unwrap(), true);
    assert!(res2.is_ok());

    let _ = remove_file(input_path);
    let _ = remove_file(out_path);
}

#[test]
fn integration_run_cli_help() {
    let args = vec!["holo_encode".to_string(), "--help".to_string()];
    let res = run_cli(&args);
    assert!(res.is_ok());
}

#[test]
fn integration_run_cli_missing_input_returns_err() {
    let args = vec!["holo_encode".to_string()];
    let res = run_cli(&args);
    assert!(res.is_err());
}

#[test]
fn integration_run_cli_encode_and_decode_via_run_cli() {
    let input_path = unique_temp_path("_cli_run_in.bin");
    let out_path = unique_temp_path("_cli_run_out.mps");
    let mut f = File::create(&input_path).unwrap();
    f.write_all(&[9u8; 32]).unwrap();
    f.flush().unwrap();

    let args = vec!["holo_encode".to_string(), input_path.to_str().unwrap().to_string(), "--bond".to_string(), "8".to_string(), "--out".to_string(), out_path.to_str().unwrap().to_string(), "--stats".to_string(), "--lossless".to_string()];
    let res = run_cli(&args);
    assert!(res.is_ok());
    assert!(out_path.exists());

    let args2 = vec!["holo_encode".to_string(), "--decode".to_string(), out_path.to_str().unwrap().to_string(), "--stats".to_string()];
    let res2 = run_cli(&args2);
    assert!(res2.is_ok());

    let _ = remove_file(input_path);
    let _ = remove_file(out_path);
}

#[test]
fn integration_run_cli_encode_invalid_bond_parses_default() {
    let input_path = unique_temp_path("_cli_run_in_badbond.bin");
    let out_path = unique_temp_path("_cli_run_out_badbond.mps");
    let mut f = File::create(&input_path).unwrap();
    f.write_all(&[1u8; 16]).unwrap();
    f.flush().unwrap();

    let args = vec!["holo_encode".to_string(), input_path.to_str().unwrap().to_string(), "--bond".to_string(), "notanumber".to_string(), "--out".to_string(), out_path.to_str().unwrap().to_string()];
    let res = run_cli(&args);
    assert!(res.is_ok());
    assert!(out_path.exists());

    let _ = remove_file(input_path);
    let _ = remove_file(out_path);
}

#[test]
fn integration_run_cli_decode_nonexistent_file_returns_err() {
    let args = vec!["holo_encode".to_string(), "--decode".to_string(), "__file_does_not_exist.mps".to_string()];
    let res = run_cli(&args);
    assert!(res.is_err());
}

#[test]
fn integration_run_cli_with_missing_values() {
    let args1 = vec!["holo_encode".to_string(), "some.bin".to_string(), "--bond".to_string()];
    assert!(run_cli(&args1).is_err());
    let args2 = vec!["holo_encode".to_string(), "some.bin".to_string(), "--out".to_string()];
    assert!(run_cli(&args2).is_err());
    let args3 = vec!["holo_encode".to_string(), "--decode".to_string()];
    assert!(run_cli(&args3).is_err());
}
