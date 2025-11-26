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

    #[test]
    fn test_psbt_builder_empty() {
        let builder = PsbtBuilder::new();
        let result = builder.build();
        assert!(result.is_err());
    }
}

