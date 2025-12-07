//! PSBT (Partially Signed Bitcoin Transaction) utilities.

use bitcoin::psbt::Psbt;
use bitcoin::{Transaction, TxIn, TxOut, OutPoint, Sequence, ScriptBuf, Amount};
use bitcoin::transaction::Version;
use bitcoin::absolute::LockTime;
use bitcoin::script::PushBytesBuf;

use crate::error::BitcoinError;
use crate::utxo::Utxo;

/// Builder for constructing PSBTs for NexusZero transactions.
pub struct PsbtBuilder {
    inputs: Vec<TxIn>,
    outputs: Vec<TxOut>,
    witness_utxos: Vec<Option<TxOut>>,
}

impl PsbtBuilder {
    /// Create a new PSBT builder.
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            witness_utxos: Vec::new(),
        }
    }

    /// Add an input from a UTXO.
    pub fn add_input(mut self, utxo: &Utxo) -> Self {
        let txin = TxIn {
            previous_output: utxo.outpoint,
            script_sig: ScriptBuf::new(),
            sequence: Sequence::ENABLE_RBF_NO_LOCKTIME,
            witness: bitcoin::Witness::new(),
        };

        let txout = TxOut {
            value: Amount::from_sat(utxo.value),
            script_pubkey: utxo.script_pubkey.clone(),
        };

        self.inputs.push(txin);
        self.witness_utxos.push(Some(txout));
        self
    }

    /// Add an output.
    pub fn add_output(mut self, txout: TxOut) -> Self {
        self.outputs.push(txout);
        self
    }

    /// Add an OP_RETURN output for proof embedding.
    pub fn add_op_return(mut self, data: &[u8]) -> Result<Self, BitcoinError> {
        use bitcoin::opcodes::all::*;

        if data.len() > 80 {
            return Err(BitcoinError::TransactionBuildError("OP_RETURN data too large".to_string()));
        }

        let push_bytes = PushBytesBuf::try_from(data.to_vec())
            .map_err(|_| BitcoinError::ScriptError("Invalid push bytes".to_string()))?;

        let script = ScriptBuf::builder()
            .push_opcode(OP_RETURN)
            .push_slice(push_bytes)
            .into_script();

        let txout = TxOut {
            value: Amount::ZERO,
            script_pubkey: script,
        };

        self.outputs.push(txout);
        Ok(self)
    }

    /// Build the PSBT.
    pub fn build(self) -> Result<Psbt, BitcoinError> {
        if self.inputs.is_empty() {
            return Err(BitcoinError::TransactionBuildError("No inputs provided".to_string()));
        }

        if self.outputs.is_empty() {
            return Err(BitcoinError::TransactionBuildError("No outputs provided".to_string()));
        }

        let tx = Transaction {
            version: Version::TWO,
            lock_time: LockTime::ZERO,
            input: self.inputs,
            output: self.outputs,
        };

        let mut psbt = Psbt::from_unsigned_tx(tx)
            .map_err(|e| BitcoinError::PsbtError(e.to_string()))?;

        // Add witness UTXOs to inputs
        for (i, witness_utxo) in self.witness_utxos.into_iter().enumerate() {
            if let Some(utxo) = witness_utxo {
                psbt.inputs[i].witness_utxo = Some(utxo);
            }
        }

        Ok(psbt)
    }
}

impl Default for PsbtBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Finalize a PSBT and extract the transaction.
pub fn finalize_psbt(psbt: Psbt) -> Result<Transaction, BitcoinError> {
    psbt.extract_tx()
        .map_err(|e| BitcoinError::PsbtError(format!("Failed to extract tx: {:?}", e)))
}

/// Serialize PSBT to base64.
pub fn psbt_to_base64(psbt: &Psbt) -> String {
    use base64::Engine;
    let bytes = psbt.serialize();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Deserialize PSBT from base64.
pub fn psbt_from_base64(s: &str) -> Result<Psbt, BitcoinError> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| BitcoinError::PsbtError(e.to_string()))?;
    Psbt::deserialize(&bytes)
        .map_err(|e| BitcoinError::PsbtError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitcoin::hashes::Hash;
    use bitcoin::Txid;

    fn mock_utxo(value: u64) -> Utxo {
        let txid = Txid::from_byte_array([1u8; 32]);
        let outpoint = OutPoint::new(txid, 0);
        
        Utxo {
            outpoint,
            value,
            script_pubkey: ScriptBuf::new(),
            confirmations: 6,
            is_taproot: true,
            has_proof: false,
            proof_hash: None,
        }
    }

    #[test]
    fn test_psbt_builder_empty() {
        let builder = PsbtBuilder::new();
        let result = builder.build();
        assert!(result.is_err());
    }

    // ===== HARDENING TESTS =====

    #[test]
    fn test_psbt_builder_no_inputs() {
        let output = TxOut {
            value: Amount::from_sat(50_000),
            script_pubkey: ScriptBuf::new(),
        };

        let builder = PsbtBuilder::new().add_output(output);
        let result = builder.build();
        
        assert!(result.is_err());
        if let Err(BitcoinError::TransactionBuildError(msg)) = result {
            assert!(msg.contains("No inputs"));
        }
    }

    #[test]
    fn test_psbt_builder_no_outputs() {
        let utxo = mock_utxo(100_000);
        
        let builder = PsbtBuilder::new().add_input(&utxo);
        let result = builder.build();
        
        assert!(result.is_err());
        if let Err(BitcoinError::TransactionBuildError(msg)) = result {
            assert!(msg.contains("No outputs"));
        }
    }

    #[test]
    fn test_psbt_builder_valid() {
        let utxo = mock_utxo(100_000);
        let output = TxOut {
            value: Amount::from_sat(50_000),
            script_pubkey: ScriptBuf::new(),
        };

        let builder = PsbtBuilder::new()
            .add_input(&utxo)
            .add_output(output);
        
        let result = builder.build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_psbt_builder_multiple_inputs() {
        let utxo1 = mock_utxo(50_000);
        let utxo2 = mock_utxo(60_000);
        let output = TxOut {
            value: Amount::from_sat(100_000),
            script_pubkey: ScriptBuf::new(),
        };

        let builder = PsbtBuilder::new()
            .add_input(&utxo1)
            .add_input(&utxo2)
            .add_output(output);
        
        let result = builder.build();
        assert!(result.is_ok());
        
        let psbt = result.unwrap();
        assert_eq!(psbt.inputs.len(), 2);
    }

    #[test]
    fn test_psbt_builder_multiple_outputs() {
        let utxo = mock_utxo(100_000);
        let output1 = TxOut {
            value: Amount::from_sat(30_000),
            script_pubkey: ScriptBuf::new(),
        };
        let output2 = TxOut {
            value: Amount::from_sat(40_000),
            script_pubkey: ScriptBuf::new(),
        };

        let builder = PsbtBuilder::new()
            .add_input(&utxo)
            .add_output(output1)
            .add_output(output2);
        
        let result = builder.build();
        assert!(result.is_ok());
        
        let psbt = result.unwrap();
        assert_eq!(psbt.unsigned_tx.output.len(), 2);
    }

    #[test]
    fn test_psbt_builder_op_return() {
        let utxo = mock_utxo(100_000);
        let data = b"NexusZero proof commitment";

        let builder = PsbtBuilder::new()
            .add_input(&utxo)
            .add_op_return(data)
            .unwrap();
        
        // Need at least one value output too
        let output = TxOut {
            value: Amount::from_sat(50_000),
            script_pubkey: ScriptBuf::new(),
        };
        
        let result = builder.add_output(output).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_psbt_builder_op_return_max_size() {
        let utxo = mock_utxo(100_000);
        let data = vec![0xab; 80]; // Max allowed size

        let builder = PsbtBuilder::new()
            .add_input(&utxo)
            .add_op_return(&data);
        
        assert!(builder.is_ok());
    }

    #[test]
    fn test_psbt_builder_op_return_too_large() {
        let utxo = mock_utxo(100_000);
        let data = vec![0xab; 81]; // Exceeds 80 byte limit

        let builder = PsbtBuilder::new()
            .add_input(&utxo)
            .add_op_return(&data);
        
        assert!(builder.is_err());
        if let Err(BitcoinError::TransactionBuildError(msg)) = builder {
            assert!(msg.contains("too large"));
        }
    }

    #[test]
    fn test_psbt_builder_op_return_empty() {
        let utxo = mock_utxo(100_000);
        let data: &[u8] = &[];

        let builder = PsbtBuilder::new()
            .add_input(&utxo)
            .add_op_return(data);
        
        assert!(builder.is_ok());
    }

    #[test]
    fn test_psbt_builder_default() {
        let builder = PsbtBuilder::default();
        let result = builder.build();
        assert!(result.is_err()); // No inputs
    }

    #[test]
    fn test_psbt_base64_roundtrip() {
        let utxo = mock_utxo(100_000);
        let output = TxOut {
            value: Amount::from_sat(50_000),
            script_pubkey: ScriptBuf::new(),
        };

        let psbt = PsbtBuilder::new()
            .add_input(&utxo)
            .add_output(output)
            .build()
            .unwrap();

        let base64 = psbt_to_base64(&psbt);
        let parsed = psbt_from_base64(&base64).unwrap();
        
        assert_eq!(psbt.unsigned_tx.input.len(), parsed.unsigned_tx.input.len());
        assert_eq!(psbt.unsigned_tx.output.len(), parsed.unsigned_tx.output.len());
    }

    #[test]
    fn test_psbt_from_invalid_base64() {
        let result = psbt_from_base64("not-valid-base64!!!");
        assert!(result.is_err());
    }

    #[test]
    fn test_psbt_from_valid_base64_invalid_psbt() {
        use base64::Engine;
        let invalid_data = base64::engine::general_purpose::STANDARD.encode(b"not a psbt");
        
        let result = psbt_from_base64(&invalid_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_psbt_witness_utxo_populated() {
        let utxo = mock_utxo(100_000);
        let output = TxOut {
            value: Amount::from_sat(50_000),
            script_pubkey: ScriptBuf::new(),
        };

        let psbt = PsbtBuilder::new()
            .add_input(&utxo)
            .add_output(output)
            .build()
            .unwrap();

        // Witness UTXO should be populated
        assert!(psbt.inputs[0].witness_utxo.is_some());
        assert_eq!(psbt.inputs[0].witness_utxo.as_ref().unwrap().value, Amount::from_sat(100_000));
    }

    #[test]
    fn test_psbt_version() {
        let utxo = mock_utxo(100_000);
        let output = TxOut {
            value: Amount::from_sat(50_000),
            script_pubkey: ScriptBuf::new(),
        };

        let psbt = PsbtBuilder::new()
            .add_input(&utxo)
            .add_output(output)
            .build()
            .unwrap();

        // Should use Version::TWO for Taproot
        assert_eq!(psbt.unsigned_tx.version, Version::TWO);
    }

    #[test]
    fn test_psbt_sequence_rbf() {
        let utxo = mock_utxo(100_000);
        let output = TxOut {
            value: Amount::from_sat(50_000),
            script_pubkey: ScriptBuf::new(),
        };

        let psbt = PsbtBuilder::new()
            .add_input(&utxo)
            .add_output(output)
            .build()
            .unwrap();

        // Should enable RBF
        assert_eq!(psbt.unsigned_tx.input[0].sequence, Sequence::ENABLE_RBF_NO_LOCKTIME);
    }
}
