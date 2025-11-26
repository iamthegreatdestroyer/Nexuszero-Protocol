use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;
use crate::compression::encoder::{encode_proof, encode_proof_lossless};
use crate::MPS;

pub fn print_usage() {
    eprintln!("Usage: holo_encode <input_file> [--bond <dim>] [--out <mps.bin>] [--decode <mps.bin>] [--stats]\n\nCommands:\n  Encode: holo_encode input.bin --bond 8 --out compressed.mps --stats\n  Decode (inspect): holo_encode --decode compressed.mps --stats\n");
}

pub fn encode_path(input: &str, bond: usize, out: Option<&str>, stats: bool, lossless: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut data = Vec::new();
    File::open(input)?.read_to_end(&mut data)?;
    let start = Instant::now();
    let mps = if lossless { encode_proof_lossless(&data, bond)? } else { encode_proof(&data, bond)? };
    let elapsed = start.elapsed();
    if let Some(out_path) = out {
        let encoded = bincode::serialize(&mps)?;
        File::create(out_path)?.write_all(&encoded)?;
        println!("Written MPS to {} ({} bytes)", out_path, encoded.len());
    }
    println!("Encode complete.");
    println!("  Input bytes          : {}", data.len());
    println!("  Bond dimension       : {}", bond);
    println!("  MPS length           : {}", mps.len());
    println!("  Compression ratio    : {:.4}", mps.compression_ratio());
    println!("  Approx serialized sz : {} bytes", mps.approx_serialized_size());
    println!("  Time                 : {:.3} ms", elapsed.as_secs_f64()*1000.0);
    if stats {
        let total_elems: usize = mps.len() * bond * 4 * bond; // rough upper bound
        println!("  Rough tensor elems   : {}", total_elems);
    }
    Ok(())
}

pub fn decode_path(input: &str, stats: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut buf = Vec::new();
    File::open(input)?.read_to_end(&mut buf)?;
    let start = Instant::now();
    let mps: MPS = bincode::deserialize(&buf)?;
    let elapsed = start.elapsed();
    println!("Loaded MPS from {} ({} bytes)", input, buf.len());
    println!("  Length            : {}", mps.len());
    println!("  Compression ratio : {:.4}", mps.compression_ratio());
    println!("  Approx size       : {} bytes", mps.approx_serialized_size());
    println!("  Load time         : {:.3} ms", elapsed.as_secs_f64()*1000.0);
    if stats {
        println!("  (Decode placeholder: original bytes not reconstructed)");
    }
    Ok(())
}

pub fn run_cli(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| a == "--help" || a == "-h") { print_usage(); return Ok(()); }
    // Decode mode
    if let Some(pos) = args.iter().position(|a| a == "--decode") {
        if pos+1 >= args.len() { print_usage(); return Err(Box::<dyn std::error::Error>::from("missing decode path")); }
        let file = &args[pos+1];
        let stats = args.iter().any(|a| a == "--stats");
        return decode_path(file, stats);
    }
    if args.len() < 2 { print_usage(); return Err(Box::<dyn std::error::Error>::from("missing input file")); }
    let input = &args[1];
    if !Path::new(input).exists() { eprintln!("Input file not found: {}", input); return Err(Box::<dyn std::error::Error>::from("input not found")); }
    let mut bond: usize = 8;
    let mut out: Option<&str> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--bond" => { if i+1 >= args.len() { print_usage(); return Err(Box::<dyn std::error::Error>::from("missing bond value")); } bond = args[i+1].parse().unwrap_or(8); i += 2; },
            "--out" => { if i+1 >= args.len() { print_usage(); return Err(Box::<dyn std::error::Error>::from("missing out path")); } out = Some(args[i+1].as_str()); i += 2; },
            _ => { i += 1; }
        }
    }
    let stats = args.iter().any(|a| a == "--stats");
    let lossless = args.iter().any(|a| a == "--lossless");
    encode_path(input, bond, out, stats, lossless)
}
