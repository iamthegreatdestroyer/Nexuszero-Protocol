use nexuszero_integration::NexuszeroAPI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut api = NexuszeroAPI::new();
    let g = vec![2u8; 32];
    let x = vec![42u8; 32];
    // Placeholder: public value should be g^x; reuse bytes for demo.
    let h = g.clone();
    println!("Generating proof...");
    let proof = api.prove_discrete_log(&g, &h, &x)?;
    println!("Proof size: {} bytes", proof.metrics.proof_size_bytes);
    println!("Compression ratio: {:.2}", proof.metrics.compression_ratio);
    println!("Generation time: {:.2} ms", proof.metrics.generation_time_ms);
    let valid = api.verify(&proof)?;
    println!("Verified: {valid}");
    Ok(())
}
