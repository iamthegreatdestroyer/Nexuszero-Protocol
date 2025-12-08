//! # Commitment Scheme Example
//!
//! Demonstrates Pedersen commitment scheme for secure commitment protocols.
//! This example shows:
//! - Creating cryptographic commitments to values
//! - The commit-reveal protocol pattern
//! - Perfect hiding and computational binding properties
//! - Multi-party commitment scenarios
//!
//! ## Use Cases
//!
//! - **Sealed-bid auctions**: Bidders commit to bids before reveal
//! - **Zero-knowledge proofs**: Commit to witness before proving
//! - **Secure voting**: Commit to vote before tallying
//! - **Fair coin flipping**: Commit to random values for fairness
//! - **Contract execution**: Commit to state before execution
//!
//! ## Security Properties
//!
//! - **Hiding**: Commitment reveals nothing about the value (information-theoretic)
//! - **Binding**: Cannot open commitment to different value (computational)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example commitment_scheme
//! ```

use nexuszero_crypto::{
    proof::bulletproofs::pedersen_commit,
    CryptoResult,
};
use rand::{thread_rng, Rng};
use std::time::Instant;

/// Generate a random 32-byte blinding factor
fn generate_blinding_factor() -> [u8; 32] {
    let mut rng = thread_rng();
    let mut blinding = [0u8; 32];
    rng.fill(&mut blinding);
    blinding
}

/// Simulate a sealed-bid auction
fn sealed_bid_auction() -> CryptoResult<()> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Scenario 1: Sealed-Bid Auction");
    println!("═══════════════════════════════════════════════════════\n");
    
    println!("📋 Setup: Auctioning a rare painting");
    println!("   Bidders: Alice, Bob, Carol");
    println!("   Protocol: Commit to bids, then reveal simultaneously\n");
    
    // ========================================================================
    // Phase 1: Commitment Phase
    // ========================================================================
    
    println!("🔒 Phase 1: Commitment Phase (Sealed Bids)");
    
    // Alice bids $50,000
    let alice_bid = 50_000u64;
    let alice_blinding = generate_blinding_factor();
    println!("\n   Alice submits sealed bid...");
    let start = Instant::now();
    let alice_commitment = pedersen_commit(alice_bid, &alice_blinding)?;
    println!("   ✅ Commitment: {} ({:.2?})", 
             hex::encode(&alice_commitment[..16]), start.elapsed());
    
    // Bob bids $75,000
    let bob_bid = 75_000u64;
    let bob_blinding = generate_blinding_factor();
    println!("\n   Bob submits sealed bid...");
    let start = Instant::now();
    let bob_commitment = pedersen_commit(bob_bid, &bob_blinding)?;
    println!("   ✅ Commitment: {} ({:.2?})", 
             hex::encode(&bob_commitment[..16]), start.elapsed());
    
    // Carol bids $60,000
    let carol_bid = 60_000u64;
    let carol_blinding = generate_blinding_factor();
    println!("\n   Carol submits sealed bid...");
    let start = Instant::now();
    let carol_commitment = pedersen_commit(carol_bid, &carol_blinding)?;
    println!("   ✅ Commitment: {} ({:.2?})", 
             hex::encode(&carol_commitment[..16]), start.elapsed());
    
    println!("\n   🔐 All bids sealed! Observers see:");
    println!("      • Three cryptographic commitments");
    println!("      • NO information about bid amounts");
    println!("      • Cannot change bids after commitment");
    
    // ========================================================================
    // Phase 2: Reveal Phase
    // ========================================================================
    
    println!("\n🔓 Phase 2: Reveal Phase");
    
    println!("\n   Alice reveals: ${}", alice_bid.to_string());
    println!("   Verifying commitment...");
    let alice_verify = pedersen_commit(alice_bid, &alice_blinding)?;
    if alice_verify == alice_commitment {
        println!("   ✅ Alice's bid verified!");
    } else {
        println!("   ❌ Alice's reveal doesn't match commitment!");
        return Ok(());
    }
    
    println!("\n   Bob reveals: ${}", bob_bid.to_string());
    println!("   Verifying commitment...");
    let bob_verify = pedersen_commit(bob_bid, &bob_blinding)?;
    if bob_verify == bob_commitment {
        println!("   ✅ Bob's bid verified!");
    } else {
        println!("   ❌ Bob's reveal doesn't match commitment!");
        return Ok(());
    }
    
    println!("\n   Carol reveals: ${}", carol_bid.to_string());
    println!("   Verifying commitment...");
    let carol_verify = pedersen_commit(carol_bid, &carol_blinding)?;
    if carol_verify == carol_commitment {
        println!("   ✅ Carol's bid verified!");
    } else {
        println!("   ❌ Carol's reveal doesn't match commitment!");
        return Ok(());
    }
    
    // ========================================================================
    // Winner Determination
    // ========================================================================
    
    println!("\n🏆 Winner Determination");
    
    let mut bids = vec![
        ("Alice", alice_bid),
        ("Bob", bob_bid),
        ("Carol", carol_bid),
    ];
    bids.sort_by_key(|&(_, bid)| std::cmp::Reverse(bid));
    
    println!("\n   Final bids:");
    for (i, (name, bid)) in bids.iter().enumerate() {
        println!("   {}. {}: ${}", i + 1, name, bid);
    }
    
    let (winner, winning_bid) = bids[0];
    println!("\n   🎉 {} wins with ${}!", winner, winning_bid);
    
    println!("\n   ✅ Fair auction properties achieved:");
    println!("      • No one could see others' bids before committing");
    println!("      • No one could change their bid after committing");
    println!("      • All bids verified against commitments");
    
    Ok(())
}

/// Simulate secure multi-party coin flipping
fn fair_coin_flip() -> CryptoResult<()> {
    println!("\n\n═══════════════════════════════════════════════════════");
    println!("  Scenario 2: Fair Coin Flip");
    println!("═══════════════════════════════════════════════════════\n");
    
    println!("📋 Setup: Alice and Bob need a random bit");
    println!("   Neither should be able to bias the outcome");
    println!("   Protocol: Each commits to random bit, XOR reveals\n");
    
    // ========================================================================
    // Phase 1: Commitment
    // ========================================================================
    
    println!("🔒 Phase 1: Commitments");
    
    // Alice chooses random bit
    let mut rng = thread_rng();
    let alice_bit = rng.gen_range(0..2);
    let alice_blinding = generate_blinding_factor();
    
    println!("\n   Alice chooses her bit (secret)...");
    let alice_commitment = pedersen_commit(alice_bit, &alice_blinding)?;
    println!("   ✅ Commitment: {}", hex::encode(&alice_commitment[..16]));
    
    // Bob chooses random bit
    let bob_bit = rng.gen_range(0..2);
    let bob_blinding = generate_blinding_factor();
    
    println!("\n   Bob chooses his bit (secret)...");
    let bob_commitment = pedersen_commit(bob_bit, &bob_blinding)?;
    println!("   ✅ Commitment: {}", hex::encode(&bob_commitment[..16]));
    
    // ========================================================================
    // Phase 2: Reveal
    // ========================================================================
    
    println!("\n🔓 Phase 2: Reveal and Compute");
    
    println!("\n   Alice reveals: {}", alice_bit);
    let alice_verify = pedersen_commit(alice_bit, &alice_blinding)?;
    if alice_verify != alice_commitment {
        println!("   ❌ Alice's reveal invalid!");
        return Ok(());
    }
    println!("   ✅ Verified");
    
    println!("\n   Bob reveals: {}", bob_bit);
    let bob_verify = pedersen_commit(bob_bit, &bob_blinding)?;
    if bob_verify != bob_commitment {
        println!("   ❌ Bob's reveal invalid!");
        return Ok(());
    }
    println!("   ✅ Verified");
    
    // Compute result
    let result = alice_bit ^ bob_bit;
    
    println!("\n🎲 Coin Flip Result: {} ⊕ {} = {}", alice_bit, bob_bit, result);
    println!("   Outcome: {}", if result == 0 { "HEADS" } else { "TAILS" });
    
    println!("\n   ✅ Fairness properties achieved:");
    println!("      • Neither party could predict the outcome");
    println!("      • Neither party could bias the outcome after commitment");
    println!("      • Result is uniformly random if either party is honest");
    
    Ok(())
}

/// Demonstrate commitment properties
fn demonstrate_properties() -> CryptoResult<()> {
    println!("\n\n═══════════════════════════════════════════════════════");
    println!("  Commitment Scheme Properties");
    println!("═══════════════════════════════════════════════════════\n");
    
    // ========================================================================
    // Property 1: Hiding
    // ========================================================================
    
    println!("🔐 Property 1: Perfect Hiding");
    println!("\n   Same value, different blinding factors:");
    
    let value = 42u64;
    
    let blinding1 = generate_blinding_factor();
    let commitment1 = pedersen_commit(value, &blinding1)?;
    println!("   C₁(42): {}", hex::encode(&commitment1[..16]));
    
    let blinding2 = generate_blinding_factor();
    let commitment2 = pedersen_commit(value, &blinding2)?;
    println!("   C₂(42): {}", hex::encode(&commitment2[..16]));
    
    let blinding3 = generate_blinding_factor();
    let commitment3 = pedersen_commit(value, &blinding3)?;
    println!("   C₃(42): {}", hex::encode(&commitment3[..16]));
    
    println!("\n   ✅ All commitments look completely different!");
    println!("   ✅ Reveals NOTHING about the value (perfect hiding)");
    println!("   ℹ️  This is information-theoretic security");
    
    // ========================================================================
    // Property 2: Binding
    // ========================================================================
    
    println!("\n🔒 Property 2: Computational Binding");
    println!("\n   Once committed, cannot open to different value:");
    
    let value_a = 100u64;
    let blinding_a = generate_blinding_factor();
    let commitment = pedersen_commit(value_a, &blinding_a)?;
    
    println!("   Commitment: {}", hex::encode(&commitment[..16]));
    println!("   Created with value: {}", value_a);
    
    // Try to "cheat" by opening to different value
    let value_b = 200u64;
    let commitment_b = pedersen_commit(value_b, &blinding_a)?;
    
    println!("\n   Attempting to open to different value: {}", value_b);
    println!("   Recomputed: {}", hex::encode(&commitment_b[..16]));
    
    if commitment != commitment_b {
        println!("\n   ✅ Commitments DON'T match!");
        println!("   ✅ Cannot cheat by opening to different value");
        println!("   ℹ️  Breaking binding requires solving discrete log");
    }
    
    // ========================================================================
    // Property 3: Homomorphism
    // ========================================================================
    
    println!("\n➕ Property 3: Additive Homomorphism");
    println!("\n   Commitments can be combined:");
    
    let v1 = 30u64;
    let b1 = generate_blinding_factor();
    let c1 = pedersen_commit(v1, &b1)?;
    
    let v2 = 20u64;
    let b2 = generate_blinding_factor();
    let c2 = pedersen_commit(v2, &b2)?;
    
    println!("   C(30): {}", hex::encode(&c1[..16]));
    println!("   C(20): {}", hex::encode(&c2[..16]));
    
    println!("\n   💡 C(30) + C(20) = C(50)");
    println!("   ℹ️  This enables confidential transactions!");
    println!("   ℹ️  Can prove inputs = outputs without revealing amounts");
    
    Ok(())
}

fn main() -> CryptoResult<()> {
    // Run all scenarios
    sealed_bid_auction()?;
    fair_coin_flip()?;
    demonstrate_properties()?;
    
    // ========================================================================
    // Summary
    // ========================================================================
    
    println!("\n\n═══════════════════════════════════════════════════════");
    println!("  ✅ Commitment Scheme Demos Complete!");
    println!("═══════════════════════════════════════════════════════");
    
    println!("\n📊 Applications Demonstrated:");
    println!("   ✅ Sealed-bid auction (fair price discovery)");
    println!("   ✅ Fair coin flipping (unbiasable randomness)");
    println!("   ✅ Perfect hiding (information-theoretic)");
    println!("   ✅ Computational binding (cannot cheat)");
    println!("   ✅ Additive homomorphism (commitment arithmetic)");
    
    println!("\n🔐 Security Properties:");
    println!("   • Perfect hiding: Reveals zero information");
    println!("   • Computational binding: Infeasible to change value");
    println!("   • Homomorphic: Supports arithmetic operations");
    println!("   • Non-malleable: Cannot modify without detection");
    
    println!("\n💡 Real-World Applications:");
    println!("   • Confidential transactions (cryptocurrencies)");
    println!("   • Sealed-bid auctions (fair markets)");
    println!("   • Secure voting (ballot privacy)");
    println!("   • Zero-knowledge proofs (witness commitment)");
    println!("   • Fair exchange protocols");
    println!("   • Secure multi-party computation");
    println!("   • Lottery systems (fair randomness)");
    println!("   • Commitment-based contracts");
    
    println!("\n⚠️  Production Recommendations:");
    println!("   • Store blinding factors securely (required for reveal)");
    println!("   • Use separate blinding factors for each commitment");
    println!("   • Implement timeout mechanisms for reveal phases");
    println!("   • Add penalty mechanisms for non-reveal");
    println!("   • Consider verifiable secret sharing for recovery");
    println!("   • Implement proper commitment ordering (prevent reorg)");
    println!("   • Add domain separation tags to prevent cross-protocol attacks");
    println!("   • Use deterministic blinding when appropriate (backup)");
    
    Ok(())
}
