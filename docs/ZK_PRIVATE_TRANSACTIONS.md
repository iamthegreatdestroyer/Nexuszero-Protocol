# ZK Private Transactions

> Privacy-Preserving Financial Transactions with Zero-Knowledge Proofs

## Table of Contents

- [ZK Private Transactions](#zk-private-transactions)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Privacy Transaction Architecture](#privacy-transaction-architecture)
  - [UTXO Model Private Transfers](#utxo-model-private-transfers)
    - [Note Structure](#note-structure)
    - [Private Transfer Implementation](#private-transfer-implementation)
  - [Account Model Private Transfers](#account-model-private-transfers)
  - [Confidential Amounts](#confidential-amounts)
    - [Range Proofs for Positive Amounts](#range-proofs-for-positive-amounts)
  - [Multi-Asset Private Swaps](#multi-asset-private-swaps)
    - [Atomic Swap with Hidden Amounts](#atomic-swap-with-hidden-amounts)
  - [Private DeFi Interactions](#private-defi-interactions)
    - [Private Liquidity Pool](#private-liquidity-pool)
  - [Compliance-Ready Privacy](#compliance-ready-privacy)
    - [Selective Disclosure for Regulators](#selective-disclosure-for-regulators)
  - [Cross-Chain Private Bridges](#cross-chain-private-bridges)
    - [Private Bridge Protocol](#private-bridge-protocol)
  - [Quick Reference](#quick-reference)
    - [Common Imports](#common-imports)
    - [Transaction Types](#transaction-types)
    - [Security Properties](#security-properties)
  - [See Also](#see-also)

---

## Overview

Zero-knowledge private transactions enable:

- **Amount Privacy**: Transaction values are hidden
- **Sender/Receiver Privacy**: Addresses are not linkable
- **Asset Privacy**: Which assets are transferred is hidden
- **Compliance Options**: Selective disclosure for regulators
- **Auditability**: Proof of reserves without revealing balances

### Privacy Transaction Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                  Private Transaction Flow                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Sender                          Network                     Receiver  │
│  ──────                          ───────                     ────────  │
│                                                                        │
│  ┌─────────────┐                                      ┌─────────────┐  │
│  │ Private Key │                                      │ Public Key  │  │
│  │   + UTXOs   │                                      │(Stealth Addr)│  │
│  └──────┬──────┘                                      └──────┬──────┘  │
│         │                                                    │         │
│         ▼                                                    │         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │         │
│  │  Create     │────▶│  Generate   │────▶│  Broadcast  │    │         │
│  │  Notes      │     │  ZK Proof   │     │    TX       │    │         │
│  └─────────────┘     └─────────────┘     └──────┬──────┘    │         │
│                                                 │            │         │
│                                                 ▼            │         │
│                                          ┌─────────────┐    │         │
│                                          │   Verify    │    │         │
│                                          │   Proof     │    │         │
│                                          └──────┬──────┘    │         │
│                                                 │            │         │
│                                                 ▼            ▼         │
│                                          ┌─────────────────────┐      │
│                                          │  Update Commitment  │      │
│                                          │       Tree          │      │
│                                          └─────────────────────┘      │
│                                                                        │
│  Private Information:                     Public Information:          │
│  • Sender identity                        • Nullifiers                 │
│  • Receiver identity                      • Note commitments           │
│  • Amount transferred                     • ZK proof                   │
│  • Asset type                             • Transaction hash           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## UTXO Model Private Transfers

### Note Structure

```typescript
import {
  NexuszeroClient,
  ProofBuilder,
  StatementType,
  SecurityLevel,
} from "nexuszero-sdk";
import { createHash, randomBytes } from "crypto";

// Core Types for Private UTXO System
interface Note {
  // Public (appears on-chain)
  commitment: string; // Pedersen commitment to note contents
  encryptedData: Uint8Array; // Encrypted note for receiver

  // Private (only known to owner)
  owner: string; // Owner's viewing key
  amount: bigint; // Asset amount
  asset: string; // Asset identifier
  blinding: Uint8Array; // Randomness for commitment
  nullifier: string; // Unique identifier for spending
}

interface NoteCommitment {
  // commitment = H(owner || amount || asset || blinding)
  hash: Uint8Array;
  // Pedersen commitment for amount: C = amount * G + blinding * H
  pedersenCommitment: Uint8Array;
}

// Note Creation
class NoteFactory {
  constructor(private client: NexuszeroClient) {}

  async createNote(
    ownerViewKey: Uint8Array,
    amount: bigint,
    asset: string
  ): Promise<Note> {
    // Generate random blinding factor
    const blinding = randomBytes(32);

    // Compute nullifier = H(ownerSpendKey || noteIndex)
    const nullifier = await this.computeNullifier(ownerViewKey, blinding);

    // Compute commitment = H(owner || amount || asset || blinding)
    const commitment = await this.computeCommitment(
      ownerViewKey,
      amount,
      asset,
      blinding
    );

    // Encrypt note data for receiver
    const encryptedData = await this.encryptNote(ownerViewKey, {
      amount,
      asset,
      blinding,
    });

    return {
      commitment,
      encryptedData,
      owner: Buffer.from(ownerViewKey).toString("hex"),
      amount,
      asset,
      blinding,
      nullifier,
    };
  }

  private async computeCommitment(
    owner: Uint8Array,
    amount: bigint,
    asset: string,
    blinding: Uint8Array
  ): Promise<string> {
    const data = Buffer.concat([
      owner,
      Buffer.from(amount.toString()),
      Buffer.from(asset),
      blinding,
    ]);
    return createHash("sha256").update(data).digest("hex");
  }

  private async computeNullifier(
    spendKey: Uint8Array,
    blinding: Uint8Array
  ): Promise<string> {
    const data = Buffer.concat([spendKey, blinding]);
    return createHash("sha256").update(data).digest("hex");
  }

  private async encryptNote(
    viewKey: Uint8Array,
    noteData: { amount: bigint; asset: string; blinding: Uint8Array }
  ): Promise<Uint8Array> {
    // In production: use ECIES or similar
    const plaintext = JSON.stringify({
      amount: noteData.amount.toString(),
      asset: noteData.asset,
      blinding: Buffer.from(noteData.blinding).toString("hex"),
    });
    return Buffer.from(plaintext); // Simplified
  }
}
```

### Private Transfer Implementation

```typescript
interface PrivateTransfer {
  // Inputs (notes being spent)
  inputNullifiers: string[];
  inputProofs: Uint8Array[];

  // Outputs (new notes being created)
  outputCommitments: string[];
  encryptedOutputs: Uint8Array[];

  // ZK Proof that:
  // 1. Input notes exist in commitment tree
  // 2. Sender knows secrets for input notes
  // 3. Input sum >= output sum
  // 4. All amounts are positive
  transferProof: Uint8Array;
}

class PrivateTransferBuilder {
  private client: NexuszeroClient;
  private noteFactory: NoteFactory;
  private inputNotes: Note[] = [];
  private outputSpecs: {
    recipient: Uint8Array;
    amount: bigint;
    asset: string;
  }[] = [];

  constructor(client: NexuszeroClient) {
    this.client = client;
    this.noteFactory = new NoteFactory(client);
  }

  // Add an input note (one you own and want to spend)
  addInput(note: Note, spendKey: Uint8Array): this {
    this.inputNotes.push(note);
    return this;
  }

  // Add an output (recipient and amount)
  addOutput(recipient: Uint8Array, amount: bigint, asset: string): this {
    this.outputSpecs.push({ recipient, amount, asset });
    return this;
  }

  async build(): Promise<PrivateTransfer> {
    // Validate: input sum >= output sum
    const inputSum = this.inputNotes.reduce((sum, n) => sum + n.amount, 0n);
    const outputSum = this.outputSpecs.reduce((sum, o) => sum + o.amount, 0n);

    if (inputSum < outputSum) {
      throw new Error("Insufficient input amount");
    }

    // Create output notes
    const outputNotes: Note[] = [];
    for (const spec of this.outputSpecs) {
      const note = await this.noteFactory.createNote(
        spec.recipient,
        spec.amount,
        spec.asset
      );
      outputNotes.push(note);
    }

    // If there's change, create change note
    const change = inputSum - outputSum;
    if (change > 0n) {
      const senderKey = this.inputNotes[0].owner;
      const changeNote = await this.noteFactory.createNote(
        Buffer.from(senderKey, "hex"),
        change,
        this.inputNotes[0].asset
      );
      outputNotes.push(changeNote);
    }

    // Generate ZK proof
    const transferProof = await this.generateTransferProof(
      this.inputNotes,
      outputNotes
    );

    return {
      inputNullifiers: this.inputNotes.map((n) => n.nullifier),
      inputProofs: await this.generateMembershipProofs(this.inputNotes),
      outputCommitments: outputNotes.map((n) => n.commitment),
      encryptedOutputs: outputNotes.map((n) => n.encryptedData),
      transferProof,
    };
  }

  private async generateTransferProof(
    inputs: Note[],
    outputs: Note[]
  ): Promise<Uint8Array> {
    const builder = new ProofBuilder(this.client).setSecurityLevel(
      SecurityLevel.High
    );

    // Prove: sum(inputs) >= sum(outputs)
    const inputSum = inputs.reduce((sum, n) => sum + n.amount, 0n);
    const outputSum = outputs.reduce((sum, n) => sum + n.amount, 0n);

    // Add conservation of value constraint
    builder
      .setStatementType(StatementType.Custom)
      .addCustomConstraint("value_conservation", {
        input_sum_commitment: this.computeSumCommitment(inputs),
        output_sum_commitment: this.computeSumCommitment(outputs),
      });

    // Add range proofs for all outputs (amounts are positive)
    for (let i = 0; i < outputs.length; i++) {
      builder.addRangeProof(`output_${i}_amount`, 0n, 2n ** 64n - 1n);
    }

    // Add nullifier validity proofs
    for (let i = 0; i < inputs.length; i++) {
      builder.addCustomConstraint(`nullifier_${i}`, {
        nullifier: inputs[i].nullifier,
        commitment: inputs[i].commitment,
      });
    }

    return builder.buildProof();
  }

  private computeSumCommitment(notes: Note[]): string {
    // Homomorphic sum of Pedersen commitments
    // C_total = C_1 + C_2 + ... + C_n
    let sumBlinding = Buffer.alloc(32);
    let sumAmount = 0n;

    for (const note of notes) {
      sumAmount += note.amount;
      // XOR blindings (simplified; real impl uses elliptic curve addition)
      for (let i = 0; i < 32; i++) {
        sumBlinding[i] ^= note.blinding[i];
      }
    }

    return createHash("sha256")
      .update(Buffer.concat([Buffer.from(sumAmount.toString()), sumBlinding]))
      .digest("hex");
  }

  private async generateMembershipProofs(notes: Note[]): Promise<Uint8Array[]> {
    // For each input note, prove it exists in the commitment tree
    const proofs: Uint8Array[] = [];

    for (const note of notes) {
      const merkleProof = await this.client.getMerkleProof(note.commitment);

      const builder = new ProofBuilder(this.client)
        .setStatementType(StatementType.MerkleMembership)
        .setMerkleRoot(merkleProof.root)
        .setMerklePath(merkleProof.path)
        .setLeaf(note.commitment);

      proofs.push(await builder.buildProof());
    }

    return proofs;
  }
}

// Usage Example
async function privateTransferExample() {
  const client = new NexuszeroClient({
    endpoint: "https://api.nexuszero.io",
    apiKey: process.env.NEXUSZERO_API_KEY!,
  });

  // Alice's spending key and viewing key
  const aliceSpendKey = randomBytes(32);
  const aliceViewKey = randomBytes(32);

  // Bob's viewing key (public)
  const bobViewKey = randomBytes(32);

  // Alice has a note worth 100 tokens
  const noteFactory = new NoteFactory(client);
  const aliceNote = await noteFactory.createNote(aliceViewKey, 100n, "USDC");

  // Alice sends 30 USDC to Bob
  const transfer = await new PrivateTransferBuilder(client)
    .addInput(aliceNote, aliceSpendKey)
    .addOutput(bobViewKey, 30n, "USDC") // To Bob
    // Change (70 USDC) automatically sent back to Alice
    .build();

  // Submit to network
  const txHash = await client.submitPrivateTransfer(transfer);

  console.log("Private transfer submitted:", txHash);
  console.log("Nullifiers:", transfer.inputNullifiers);
  console.log("New commitments:", transfer.outputCommitments);

  // What's visible on-chain:
  // - Nullifiers (prevents double spend)
  // - New commitments (for new notes)
  // - Encrypted data (only recipients can decrypt)
  // - ZK proof

  // What's NOT visible:
  // - Sender identity
  // - Receiver identity
  // - Amount transferred
  // - Asset type
}
```

---

## Account Model Private Transfers

For blockchains using account model (like Ethereum):

```typescript
interface PrivateAccountTransfer {
  // Sender's encrypted balance update
  senderBalanceCommitment: string;
  senderUpdateProof: Uint8Array;

  // Receiver's encrypted balance update
  receiverBalanceCommitment: string;

  // Transfer details (encrypted)
  encryptedAmount: Uint8Array;

  // ZK Proof
  transferProof: Uint8Array;
}

class PrivateAccountSystem {
  private client: NexuszeroClient;

  constructor(client: NexuszeroClient) {
    this.client = client;
  }

  // Initialize private balance for an account
  async initializeAccount(
    accountKey: Uint8Array
  ): Promise<{ commitment: string; proof: Uint8Array }> {
    // Initial balance = 0, but committed with randomness
    const blinding = randomBytes(32);
    const commitment = this.computeBalanceCommitment(0n, blinding);

    // Prove knowledge of the commitment opening
    const proof = await new ProofBuilder(this.client)
      .setStatementType(StatementType.Commitment)
      .setCommitment(commitment)
      .setPrivateValue(0n)
      .setBlinding(blinding)
      .buildProof();

    return { commitment, proof };
  }

  // Private deposit (convert public to private)
  async deposit(
    accountKey: Uint8Array,
    currentCommitment: string,
    currentBalance: bigint,
    currentBlinding: Uint8Array,
    depositAmount: bigint
  ): Promise<{ newCommitment: string; proof: Uint8Array }> {
    const newBalance = currentBalance + depositAmount;
    const newBlinding = randomBytes(32);
    const newCommitment = this.computeBalanceCommitment(
      newBalance,
      newBlinding
    );

    // Prove: new_commitment = old_commitment + deposit_amount
    const proof = await new ProofBuilder(this.client)
      .setStatementType(StatementType.Custom)
      .addCustomConstraint("balance_update", {
        old_commitment: currentCommitment,
        new_commitment: newCommitment,
        public_delta: depositAmount.toString(),
      })
      .buildProof();

    return { newCommitment, proof };
  }

  // Private transfer between accounts
  async transfer(
    senderKey: Uint8Array,
    senderBalance: bigint,
    senderBlinding: Uint8Array,
    senderCommitment: string,
    receiverKey: Uint8Array,
    receiverBalance: bigint,
    receiverBlinding: Uint8Array,
    receiverCommitment: string,
    amount: bigint
  ): Promise<PrivateAccountTransfer> {
    // Update sender balance
    const newSenderBalance = senderBalance - amount;
    const newSenderBlinding = randomBytes(32);
    const newSenderCommitment = this.computeBalanceCommitment(
      newSenderBalance,
      newSenderBlinding
    );

    // Update receiver balance
    const newReceiverBalance = receiverBalance + amount;
    const newReceiverBlinding = randomBytes(32);
    const newReceiverCommitment = this.computeBalanceCommitment(
      newReceiverBalance,
      newReceiverBlinding
    );

    // Encrypt amount for receiver
    const encryptedAmount = await this.encryptForReceiver(receiverKey, amount);

    // Generate transfer proof
    const transferProof = await this.generateTransferProof(
      senderCommitment,
      newSenderCommitment,
      receiverCommitment,
      newReceiverCommitment,
      senderBalance,
      senderBlinding,
      newSenderBlinding,
      receiverBalance,
      receiverBlinding,
      newReceiverBlinding,
      amount
    );

    // Generate sender update proof (proves sender authorized this)
    const senderUpdateProof = await this.generateSenderProof(
      senderKey,
      senderCommitment,
      newSenderCommitment
    );

    return {
      senderBalanceCommitment: newSenderCommitment,
      senderUpdateProof,
      receiverBalanceCommitment: newReceiverCommitment,
      encryptedAmount,
      transferProof,
    };
  }

  private computeBalanceCommitment(
    balance: bigint,
    blinding: Uint8Array
  ): string {
    // Pedersen commitment: C = balance * G + blinding * H
    return createHash("sha256")
      .update(Buffer.concat([Buffer.from(balance.toString()), blinding]))
      .digest("hex");
  }

  private async generateTransferProof(
    oldSenderCommitment: string,
    newSenderCommitment: string,
    oldReceiverCommitment: string,
    newReceiverCommitment: string,
    senderBalance: bigint,
    senderBlinding: Uint8Array,
    newSenderBlinding: Uint8Array,
    receiverBalance: bigint,
    receiverBlinding: Uint8Array,
    newReceiverBlinding: Uint8Array,
    amount: bigint
  ): Promise<Uint8Array> {
    return (
      new ProofBuilder(this.client)
        .setStatementType(StatementType.Custom)
        .addCustomConstraint("transfer_validity", {
          // Prove: old_sender - amount = new_sender
          old_sender_commitment: oldSenderCommitment,
          new_sender_commitment: newSenderCommitment,
          // Prove: old_receiver + amount = new_receiver
          old_receiver_commitment: oldReceiverCommitment,
          new_receiver_commitment: newReceiverCommitment,
          // Prove: amount is the same in both operations
          // (Using commitment arithmetic)
        })
        // Prove sender has sufficient balance
        .addRangeProof("sender_balance", 0n, 2n ** 64n - 1n)
        // Prove amount is positive
        .addRangeProof("amount", 0n, 2n ** 64n - 1n)
        .buildProof()
    );
  }

  private async generateSenderProof(
    senderKey: Uint8Array,
    oldCommitment: string,
    newCommitment: string
  ): Promise<Uint8Array> {
    return new ProofBuilder(this.client)
      .setStatementType(StatementType.Authorization)
      .setSignerKey(senderKey)
      .setMessage(
        Buffer.concat([Buffer.from(oldCommitment), Buffer.from(newCommitment)])
      )
      .buildProof();
  }

  private async encryptForReceiver(
    receiverKey: Uint8Array,
    amount: bigint
  ): Promise<Uint8Array> {
    // ECIES encryption for receiver
    return Buffer.from(amount.toString()); // Simplified
  }
}
```

---

## Confidential Amounts

### Range Proofs for Positive Amounts

```typescript
class ConfidentialAmount {
  private client: NexuszeroClient;

  constructor(client: NexuszeroClient) {
    this.client = client;
  }

  // Create a confidential amount with range proof
  async createConfidentialAmount(
    amount: bigint,
    minBits: number = 64
  ): Promise<{
    commitment: string;
    rangeProof: Uint8Array;
    blinding: Uint8Array;
  }> {
    const blinding = randomBytes(32);
    const commitment = this.pedersenCommit(amount, blinding);

    // Range proof: 0 <= amount < 2^minBits
    const rangeProof = await new ProofBuilder(this.client)
      .setStatementType(StatementType.Range)
      .setRange(0n, 2n ** BigInt(minBits) - 1n)
      .setCommitment(commitment)
      .setPrivateValue(amount)
      .setBlinding(blinding)
      .buildProof();

    return { commitment, rangeProof, blinding };
  }

  // Verify a confidential amount is in valid range
  async verifyConfidentialAmount(
    commitment: string,
    rangeProof: Uint8Array,
    minBits: number = 64
  ): Promise<boolean> {
    return this.client.verifyProof(rangeProof, {
      commitment,
      range: { min: 0n, max: 2n ** BigInt(minBits) - 1n },
    });
  }

  // Homomorphic addition of confidential amounts
  addCommitments(
    commitment1: string,
    blinding1: Uint8Array,
    commitment2: string,
    blinding2: Uint8Array
  ): { sumCommitment: string; sumBlinding: Uint8Array } {
    // C1 + C2 = (v1 * G + r1 * H) + (v2 * G + r2 * H)
    //         = (v1 + v2) * G + (r1 + r2) * H
    // In practice, use elliptic curve point addition

    // For now, simplified representation
    const sumBlinding = Buffer.alloc(32);
    for (let i = 0; i < 32; i++) {
      sumBlinding[i] = (blinding1[i] + blinding2[i]) % 256;
    }

    const sumCommitment = this.xorHex(commitment1, commitment2);
    return { sumCommitment, sumBlinding };
  }

  // Prove two commitments sum to a third
  async proveSum(
    commitment1: string,
    blinding1: Uint8Array,
    amount1: bigint,
    commitment2: string,
    blinding2: Uint8Array,
    amount2: bigint,
    sumCommitment: string
  ): Promise<Uint8Array> {
    return new ProofBuilder(this.client)
      .setStatementType(StatementType.Custom)
      .addCustomConstraint("sum_proof", {
        c1: commitment1,
        c2: commitment2,
        sum: sumCommitment,
      })
      .setPrivateInputs({
        v1: amount1,
        r1: blinding1,
        v2: amount2,
        r2: blinding2,
      })
      .buildProof();
  }

  private pedersenCommit(value: bigint, blinding: Uint8Array): string {
    // Pedersen commitment: C = value * G + blinding * H
    return createHash("sha256")
      .update(Buffer.concat([Buffer.from(value.toString()), blinding]))
      .digest("hex");
  }

  private xorHex(hex1: string, hex2: string): string {
    const buf1 = Buffer.from(hex1, "hex");
    const buf2 = Buffer.from(hex2, "hex");
    const result = Buffer.alloc(buf1.length);
    for (let i = 0; i < buf1.length; i++) {
      result[i] = buf1[i] ^ buf2[i];
    }
    return result.toString("hex");
  }
}

// Usage: Confidential Transaction
async function confidentialTransactionExample() {
  const client = new NexuszeroClient({
    endpoint: "https://api.nexuszero.io",
    apiKey: process.env.NEXUSZERO_API_KEY!,
  });

  const confidentialAmount = new ConfidentialAmount(client);

  // Create confidential input (100 tokens)
  const input = await confidentialAmount.createConfidentialAmount(100n);
  console.log("Input commitment:", input.commitment);

  // Create confidential outputs
  const output1 = await confidentialAmount.createConfidentialAmount(60n);
  const output2 = await confidentialAmount.createConfidentialAmount(40n);

  // Prove conservation of value: input = output1 + output2
  const { sumCommitment } = confidentialAmount.addCommitments(
    output1.commitment,
    output1.blinding,
    output2.commitment,
    output2.blinding
  );

  // In a real implementation:
  // Prove: input.commitment == sumCommitment
  // This proves 100 = 60 + 40 without revealing any values

  console.log("Output 1 commitment:", output1.commitment);
  console.log("Output 2 commitment:", output2.commitment);
  console.log("Values hidden but conservation proven!");
}
```

---

## Multi-Asset Private Swaps

### Atomic Swap with Hidden Amounts

```typescript
interface PrivateSwap {
  // Party A's side
  partyA: {
    inputNullifiers: string[];
    outputCommitments: string[];
  };

  // Party B's side
  partyB: {
    inputNullifiers: string[];
    outputCommitments: string[];
  };

  // Combined proof that:
  // 1. A receives asset B in correct amount
  // 2. B receives asset A in correct amount
  // 3. No value is created or destroyed
  swapProof: Uint8Array;

  // Time-lock for atomic execution
  timelock: number;
  hashlock: string;
}

class PrivateSwapProtocol {
  private client: NexuszeroClient;

  constructor(client: NexuszeroClient) {
    this.client = client;
  }

  // Create a private swap proposal
  async proposeSwap(
    myNotes: Note[], // Notes I'm offering
    myAsset: string,
    myAmount: bigint,
    wantAsset: string,
    wantAmount: bigint,
    counterpartyViewKey: Uint8Array
  ): Promise<{
    proposal: SwapProposal;
    secret: Uint8Array;
  }> {
    // Generate hashlock secret
    const secret = randomBytes(32);
    const hashlock = createHash("sha256").update(secret).digest("hex");

    // Create time-locked output for counterparty
    const counterpartyNote = await new NoteFactory(this.client).createNote(
      counterpartyViewKey,
      myAmount,
      myAsset
    );

    // Create proposal
    const proposal: SwapProposal = {
      offeredAsset: myAsset,
      offeredAmount: myAmount, // This would be committed/hidden in production
      wantedAsset: wantAsset,
      wantedAmount: wantAmount,
      hashlock,
      timelock: Date.now() + 24 * 60 * 60 * 1000, // 24 hours
      offeredNoteCommitment: counterpartyNote.commitment,
      inputNullifiers: myNotes.map((n) => n.nullifier),
    };

    return { proposal, secret };
  }

  // Accept a swap proposal
  async acceptSwap(
    proposal: SwapProposal,
    myNotes: Note[],
    myViewKey: Uint8Array,
    counterpartyViewKey: Uint8Array
  ): Promise<PrivateSwap> {
    // Validate I have enough of the wanted asset
    const myTotal = myNotes.reduce((sum, n) => sum + n.amount, 0n);
    if (myTotal < proposal.wantedAmount) {
      throw new Error("Insufficient balance for swap");
    }

    // Create note for proposer
    const proposerNote = await new NoteFactory(this.client).createNote(
      counterpartyViewKey,
      proposal.wantedAmount,
      proposal.wantedAsset
    );

    // Create change note for myself if needed
    const changeNotes: Note[] = [];
    if (myTotal > proposal.wantedAmount) {
      const changeNote = await new NoteFactory(this.client).createNote(
        myViewKey,
        myTotal - proposal.wantedAmount,
        proposal.wantedAsset
      );
      changeNotes.push(changeNote);
    }

    // Generate swap proof
    const swapProof = await this.generateSwapProof(
      proposal,
      myNotes,
      proposerNote,
      changeNotes
    );

    return {
      partyA: {
        inputNullifiers: proposal.inputNullifiers,
        outputCommitments: [proposal.offeredNoteCommitment],
      },
      partyB: {
        inputNullifiers: myNotes.map((n) => n.nullifier),
        outputCommitments: [
          proposerNote.commitment,
          ...changeNotes.map((n) => n.commitment),
        ],
      },
      swapProof,
      timelock: proposal.timelock,
      hashlock: proposal.hashlock,
    };
  }

  // Complete swap by revealing secret
  async completeSwap(swap: PrivateSwap, secret: Uint8Array): Promise<string> {
    // Verify secret matches hashlock
    const hashlock = createHash("sha256").update(secret).digest("hex");
    if (hashlock !== swap.hashlock) {
      throw new Error("Invalid secret");
    }

    // Submit to network
    return this.client.submitAtomicSwap({
      ...swap,
      secret: Buffer.from(secret).toString("hex"),
    });
  }

  private async generateSwapProof(
    proposal: SwapProposal,
    acceptorInputs: Note[],
    proposerOutput: Note,
    changeNotes: Note[]
  ): Promise<Uint8Array> {
    return (
      new ProofBuilder(this.client)
        .setStatementType(StatementType.Custom)
        .addCustomConstraint("atomic_swap", {
          // Prove: proposer gives correct amount of offered asset
          proposer_output_commitment: proposal.offeredNoteCommitment,
          // Prove: acceptor gives correct amount of wanted asset
          acceptor_output_commitment: proposerOutput.commitment,
          // Prove: change is correctly computed
          change_commitments: changeNotes.map((n) => n.commitment),
          // Prove: hashlock is correct
          hashlock: proposal.hashlock,
        })
        // Range proofs for all amounts
        .addRangeProof("proposer_amount", 0n, 2n ** 64n - 1n)
        .addRangeProof("acceptor_amount", 0n, 2n ** 64n - 1n)
        .buildProof()
    );
  }
}

interface SwapProposal {
  offeredAsset: string;
  offeredAmount: bigint;
  wantedAsset: string;
  wantedAmount: bigint;
  hashlock: string;
  timelock: number;
  offeredNoteCommitment: string;
  inputNullifiers: string[];
}
```

---

## Private DeFi Interactions

### Private Liquidity Pool

```typescript
interface PrivatePool {
  poolId: string;
  tokenA: string;
  tokenB: string;
  reserveCommitmentA: string; // Hidden reserve of token A
  reserveCommitmentB: string; // Hidden reserve of token B
  lpTokenSupply: string; // Total LP tokens (can be hidden too)
}

class PrivateDeFi {
  private client: NexuszeroClient;

  constructor(client: NexuszeroClient) {
    this.client = client;
  }

  // Private swap through AMM
  async privateSwap(
    pool: PrivatePool,
    inputNote: Note,
    inputAsset: "A" | "B",
    minOutputAmount: bigint,
    recipientKey: Uint8Array
  ): Promise<{
    outputNote: Note;
    swapProof: Uint8Array;
    newPoolState: PrivatePool;
  }> {
    // In private AMM, we prove:
    // 1. Input is valid (exists in tree, we own it)
    // 2. Output follows x*y=k formula
    // 3. Slippage is within tolerance
    // All without revealing amounts!

    // Generate output note
    const noteFactory = new NoteFactory(this.client);
    const outputAsset = inputAsset === "A" ? pool.tokenB : pool.tokenA;

    // Calculate output amount (hidden computation)
    // Real implementation would do this in ZK circuit
    const outputAmount = await this.calculateSwapOutput(
      pool,
      inputNote.amount,
      inputAsset
    );

    if (outputAmount < minOutputAmount) {
      throw new Error("Slippage exceeded");
    }

    const outputNote = await noteFactory.createNote(
      recipientKey,
      outputAmount,
      outputAsset
    );

    // Generate swap proof
    const swapProof = await this.generateSwapProof(
      pool,
      inputNote,
      outputNote,
      inputAsset
    );

    // Update pool state
    const newPoolState = await this.updatePoolState(
      pool,
      inputNote.amount,
      outputAmount,
      inputAsset
    );

    return { outputNote, swapProof, newPoolState };
  }

  // Private liquidity provision
  async addLiquidity(
    pool: PrivatePool,
    noteA: Note,
    noteB: Note,
    lpRecipientKey: Uint8Array
  ): Promise<{
    lpNote: Note;
    changeNoteA: Note | null;
    changeNoteB: Note | null;
    liquidityProof: Uint8Array;
    newPoolState: PrivatePool;
  }> {
    // Calculate LP tokens to mint
    const lpAmount = await this.calculateLpTokens(
      pool,
      noteA.amount,
      noteB.amount
    );

    const noteFactory = new NoteFactory(this.client);

    // Create LP token note
    const lpNote = await noteFactory.createNote(
      lpRecipientKey,
      lpAmount,
      `${pool.poolId}-LP`
    );

    // Calculate change (if deposits aren't perfectly balanced)
    const { changeA, changeB } = await this.calculateChange(
      pool,
      noteA.amount,
      noteB.amount
    );

    const changeNoteA =
      changeA > 0n
        ? await noteFactory.createNote(lpRecipientKey, changeA, pool.tokenA)
        : null;

    const changeNoteB =
      changeB > 0n
        ? await noteFactory.createNote(lpRecipientKey, changeB, pool.tokenB)
        : null;

    // Generate liquidity proof
    const liquidityProof = await new ProofBuilder(this.client)
      .setStatementType(StatementType.Custom)
      .addCustomConstraint("add_liquidity", {
        pool_id: pool.poolId,
        input_commitment_a: noteA.commitment,
        input_commitment_b: noteB.commitment,
        lp_commitment: lpNote.commitment,
        change_commitment_a: changeNoteA?.commitment ?? null,
        change_commitment_b: changeNoteB?.commitment ?? null,
      })
      .buildProof();

    // Update pool state
    const newPoolState = await this.updatePoolReserves(
      pool,
      noteA.amount - (changeA ?? 0n),
      noteB.amount - (changeB ?? 0n),
      lpAmount,
      "add"
    );

    return {
      lpNote,
      changeNoteA,
      changeNoteB,
      liquidityProof,
      newPoolState,
    };
  }

  // Private lending/borrowing
  async privateBorrow(
    collateralNote: Note,
    collateralPrice: bigint, // Oracle price (public or committed)
    borrowAmount: bigint,
    borrowAsset: string,
    recipientKey: Uint8Array,
    collateralRatio: bigint = 150n // 150% collateralization
  ): Promise<{
    borrowNote: Note;
    positionCommitment: string;
    borrowProof: Uint8Array;
  }> {
    // Validate collateralization
    const collateralValue = collateralNote.amount * collateralPrice;
    const requiredCollateral = (borrowAmount * collateralRatio) / 100n;

    if (collateralValue < requiredCollateral) {
      throw new Error("Insufficient collateral");
    }

    // Create borrow note
    const noteFactory = new NoteFactory(this.client);
    const borrowNote = await noteFactory.createNote(
      recipientKey,
      borrowAmount,
      borrowAsset
    );

    // Create position commitment (tracks the loan)
    const positionCommitment = createHash("sha256")
      .update(
        Buffer.concat([
          Buffer.from(collateralNote.commitment),
          Buffer.from(borrowNote.commitment),
          Buffer.from(collateralRatio.toString()),
        ])
      )
      .digest("hex");

    // Generate borrow proof
    const borrowProof = await new ProofBuilder(this.client)
      .setStatementType(StatementType.Custom)
      .addCustomConstraint("private_borrow", {
        collateral_commitment: collateralNote.commitment,
        borrow_commitment: borrowNote.commitment,
        position_commitment: positionCommitment,
        // Public oracle price (or commitment to it)
        price_commitment: this.commitPrice(collateralPrice),
      })
      // Prove collateralization ratio is met
      .addRangeProof("collateral_value", requiredCollateral, 2n ** 128n - 1n)
      .buildProof();

    return { borrowNote, positionCommitment, borrowProof };
  }

  private async calculateSwapOutput(
    pool: PrivatePool,
    inputAmount: bigint,
    inputAsset: "A" | "B"
  ): Promise<bigint> {
    // x * y = k constant product formula
    // Would be done in ZK circuit in production
    // Output = (y * input) / (x + input) * (1 - fee)
    return (inputAmount * 99n) / 100n; // Simplified
  }

  private async calculateLpTokens(
    pool: PrivatePool,
    amountA: bigint,
    amountB: bigint
  ): Promise<bigint> {
    // LP tokens = sqrt(amountA * amountB) for initial deposit
    // Or proportional to existing liquidity for subsequent
    return BigInt(Math.floor(Math.sqrt(Number(amountA * amountB))));
  }

  private async calculateChange(
    pool: PrivatePool,
    amountA: bigint,
    amountB: bigint
  ): Promise<{ changeA: bigint; changeB: bigint }> {
    // Return excess tokens that don't match pool ratio
    return { changeA: 0n, changeB: 0n }; // Simplified
  }

  private async generateSwapProof(
    pool: PrivatePool,
    input: Note,
    output: Note,
    inputAsset: "A" | "B"
  ): Promise<Uint8Array> {
    return new ProofBuilder(this.client)
      .setStatementType(StatementType.Custom)
      .addCustomConstraint("amm_swap", {
        pool_id: pool.poolId,
        input_commitment: input.commitment,
        output_commitment: output.commitment,
        reserve_a: pool.reserveCommitmentA,
        reserve_b: pool.reserveCommitmentB,
      })
      .buildProof();
  }

  private async updatePoolState(
    pool: PrivatePool,
    inputAmount: bigint,
    outputAmount: bigint,
    inputAsset: "A" | "B"
  ): Promise<PrivatePool> {
    // Update reserve commitments
    return pool; // Simplified
  }

  private async updatePoolReserves(
    pool: PrivatePool,
    deltaA: bigint,
    deltaB: bigint,
    lpDelta: bigint,
    operation: "add" | "remove"
  ): Promise<PrivatePool> {
    return pool; // Simplified
  }

  private commitPrice(price: bigint): string {
    return createHash("sha256")
      .update(Buffer.from(price.toString()))
      .digest("hex");
  }
}
```

---

## Compliance-Ready Privacy

### Selective Disclosure for Regulators

```typescript
interface ComplianceViewKey {
  regulatorId: string;
  viewKey: Uint8Array;
  permissions: (
    | "view_amount"
    | "view_sender"
    | "view_receiver"
    | "view_asset"
  )[];
  validUntil: Date;
}

class CompliancePrivateTransactions {
  private client: NexuszeroClient;

  constructor(client: NexuszeroClient) {
    this.client = client;
  }

  // Create transaction with compliance hooks
  async createCompliantTransfer(
    senderKey: Uint8Array,
    recipientKey: Uint8Array,
    amount: bigint,
    asset: string,
    complianceViewKeys: ComplianceViewKey[]
  ): Promise<{
    transfer: PrivateTransfer;
    complianceDisclosures: Map<string, Uint8Array>;
  }> {
    // Create standard private transfer
    const noteFactory = new NoteFactory(this.client);
    const senderNote = await noteFactory.createNote(senderKey, amount, asset);

    const transferBuilder = new PrivateTransferBuilder(this.client)
      .addInput(senderNote, senderKey)
      .addOutput(recipientKey, amount, asset);

    const transfer = await transferBuilder.build();

    // Create encrypted disclosures for each regulator
    const complianceDisclosures = new Map<string, Uint8Array>();

    for (const viewKey of complianceViewKeys) {
      const disclosure = await this.createDisclosure(
        viewKey,
        senderKey,
        recipientKey,
        amount,
        asset
      );
      complianceDisclosures.set(viewKey.regulatorId, disclosure);
    }

    return { transfer, complianceDisclosures };
  }

  // Regulator decrypts their view
  async regulatorView(
    transfer: PrivateTransfer,
    disclosure: Uint8Array,
    viewKey: ComplianceViewKey
  ): Promise<{
    sender?: string;
    receiver?: string;
    amount?: bigint;
    asset?: string;
  }> {
    const decrypted = await this.decryptDisclosure(disclosure, viewKey.viewKey);

    // Only return fields the regulator is permitted to see
    const result: Record<string, any> = {};

    if (viewKey.permissions.includes("view_sender")) {
      result.sender = decrypted.sender;
    }
    if (viewKey.permissions.includes("view_receiver")) {
      result.receiver = decrypted.receiver;
    }
    if (viewKey.permissions.includes("view_amount")) {
      result.amount = decrypted.amount;
    }
    if (viewKey.permissions.includes("view_asset")) {
      result.asset = decrypted.asset;
    }

    return result;
  }

  // Prove transaction is compliant without revealing details
  async proveCompliance(
    transfer: PrivateTransfer,
    rules: ComplianceRules
  ): Promise<Uint8Array> {
    const proofBuilder = new ProofBuilder(this.client).setStatementType(
      StatementType.Custom
    );

    // Add relevant compliance proofs
    if (rules.maxAmount) {
      proofBuilder.addRangeProof("amount", 0n, rules.maxAmount);
    }

    if (rules.senderNotOnSanctionsList) {
      proofBuilder.addCustomConstraint("not_sanctioned", {
        sanctions_root: rules.sanctionsListRoot,
        // Prove sender is NOT in Merkle tree (non-membership proof)
      });
    }

    if (rules.receiverInAllowedJurisdictions) {
      proofBuilder.addCustomConstraint("jurisdiction_ok", {
        allowed_jurisdictions_root: rules.allowedJurisdictionsRoot,
        // Prove receiver is in allowed set
      });
    }

    if (rules.sourceOfFundsVerified) {
      proofBuilder.addCustomConstraint("source_verified", {
        verified_sources_root: rules.verifiedSourcesRoot,
        // Prove input notes trace to verified sources
      });
    }

    return proofBuilder.buildProof();
  }

  // Travel Rule compliance: encrypted sender/receiver info
  async travelRuleDisclosure(
    senderVASP: string,
    senderInfo: VASPCustomerInfo,
    receiverVASP: string,
    receiverInfo: VASPCustomerInfo,
    amount: bigint,
    receiverVASPKey: Uint8Array
  ): Promise<{
    encryptedPayload: Uint8Array;
    proof: Uint8Array;
  }> {
    // Encrypt travel rule data for receiving VASP
    const payload = {
      senderVASP,
      senderName: senderInfo.name,
      senderAddress: senderInfo.address,
      senderAccountId: senderInfo.accountId,
      receiverVASP,
      receiverName: receiverInfo.name,
      receiverAddress: receiverInfo.address,
      receiverAccountId: receiverInfo.accountId,
      amount: amount.toString(),
      timestamp: Date.now(),
    };

    const encryptedPayload = await this.encryptForVASP(
      receiverVASPKey,
      JSON.stringify(payload)
    );

    // Proof that disclosure matches transaction
    const proof = await new ProofBuilder(this.client)
      .setStatementType(StatementType.Custom)
      .addCustomConstraint("travel_rule", {
        encrypted_payload_hash: createHash("sha256")
          .update(encryptedPayload)
          .digest("hex"),
        sender_commitment: createHash("sha256")
          .update(Buffer.from(JSON.stringify(senderInfo)))
          .digest("hex"),
        receiver_commitment: createHash("sha256")
          .update(Buffer.from(JSON.stringify(receiverInfo)))
          .digest("hex"),
      })
      .buildProof();

    return { encryptedPayload, proof };
  }

  private async createDisclosure(
    viewKey: ComplianceViewKey,
    senderKey: Uint8Array,
    recipientKey: Uint8Array,
    amount: bigint,
    asset: string
  ): Promise<Uint8Array> {
    const data = {
      sender: Buffer.from(senderKey).toString("hex"),
      receiver: Buffer.from(recipientKey).toString("hex"),
      amount: amount.toString(),
      asset,
    };

    // Encrypt with regulator's view key
    return Buffer.from(JSON.stringify(data)); // Simplified
  }

  private async decryptDisclosure(
    disclosure: Uint8Array,
    viewKey: Uint8Array
  ): Promise<any> {
    return JSON.parse(Buffer.from(disclosure).toString()); // Simplified
  }

  private async encryptForVASP(
    vaspKey: Uint8Array,
    data: string
  ): Promise<Uint8Array> {
    return Buffer.from(data); // Simplified
  }
}

interface ComplianceRules {
  maxAmount?: bigint;
  senderNotOnSanctionsList?: boolean;
  sanctionsListRoot?: string;
  receiverInAllowedJurisdictions?: boolean;
  allowedJurisdictionsRoot?: string;
  sourceOfFundsVerified?: boolean;
  verifiedSourcesRoot?: string;
}

interface VASPCustomerInfo {
  name: string;
  address: string;
  accountId: string;
}
```

---

## Cross-Chain Private Bridges

### Private Bridge Protocol

```typescript
interface PrivateBridgeDeposit {
  sourceChain: string;
  depositCommitment: string;
  nullifier: string;
  depositProof: Uint8Array;
}

interface PrivateBridgeWithdrawal {
  targetChain: string;
  withdrawalCommitment: string;
  bridgeProof: Uint8Array;
}

class PrivateCrossChainBridge {
  private client: NexuszeroClient;

  constructor(client: NexuszeroClient) {
    this.client = client;
  }

  // Deposit into bridge (source chain)
  async deposit(
    sourceChain: string,
    note: Note,
    targetChain: string,
    recipientKey: Uint8Array
  ): Promise<{
    deposit: PrivateBridgeDeposit;
    claimSecret: Uint8Array;
  }> {
    // Generate claim secret for withdrawal on target chain
    const claimSecret = randomBytes(32);
    const claimCommitment = createHash("sha256")
      .update(claimSecret)
      .digest("hex");

    // Create deposit commitment linking to claim
    const depositCommitment = createHash("sha256")
      .update(
        Buffer.concat([
          Buffer.from(note.commitment),
          Buffer.from(targetChain),
          Buffer.from(claimCommitment),
        ])
      )
      .digest("hex");

    // Generate deposit proof
    const depositProof = await new ProofBuilder(this.client)
      .setStatementType(StatementType.Custom)
      .addCustomConstraint("bridge_deposit", {
        source_chain: sourceChain,
        target_chain: targetChain,
        note_commitment: note.commitment,
        deposit_commitment: depositCommitment,
        claim_commitment: claimCommitment,
      })
      .buildProof();

    return {
      deposit: {
        sourceChain,
        depositCommitment,
        nullifier: note.nullifier,
        depositProof,
      },
      claimSecret,
    };
  }

  // Withdraw from bridge (target chain)
  async withdraw(
    deposit: PrivateBridgeDeposit,
    claimSecret: Uint8Array,
    targetChain: string,
    recipientKey: Uint8Array,
    amount: bigint,
    asset: string
  ): Promise<{
    withdrawal: PrivateBridgeWithdrawal;
    recipientNote: Note;
  }> {
    // Verify claim secret matches deposit
    const claimCommitment = createHash("sha256")
      .update(claimSecret)
      .digest("hex");

    // Create recipient note on target chain
    const noteFactory = new NoteFactory(this.client);
    const recipientNote = await noteFactory.createNote(
      recipientKey,
      amount,
      asset
    );

    // Generate withdrawal proof
    const bridgeProof = await new ProofBuilder(this.client)
      .setStatementType(StatementType.Custom)
      .addCustomConstraint("bridge_withdrawal", {
        deposit_commitment: deposit.depositCommitment,
        claim_commitment: claimCommitment,
        recipient_commitment: recipientNote.commitment,
        target_chain: targetChain,
      })
      // Prove: claimSecret hashes to claimCommitment
      .addPreimageProof(claimCommitment, "sha256")
      .buildProof();

    return {
      withdrawal: {
        targetChain,
        withdrawalCommitment: recipientNote.commitment,
        bridgeProof,
      },
      recipientNote,
    };
  }

  // Prove deposit occurred on source chain (for relayers)
  async proveDeposit(
    deposit: PrivateBridgeDeposit,
    sourceChainBlockHeader: Uint8Array,
    inclusionProof: Uint8Array[]
  ): Promise<Uint8Array> {
    return new ProofBuilder(this.client)
      .setStatementType(StatementType.Custom)
      .addCustomConstraint("deposit_inclusion", {
        deposit_commitment: deposit.depositCommitment,
        block_header: Buffer.from(sourceChainBlockHeader).toString("hex"),
        merkle_path: inclusionProof.map((p) => Buffer.from(p).toString("hex")),
      })
      .buildProof();
  }
}

// Usage: Private Cross-Chain Transfer
async function privateBridgeExample() {
  const client = new NexuszeroClient({
    endpoint: "https://api.nexuszero.io",
    apiKey: process.env.NEXUSZERO_API_KEY!,
  });

  const bridge = new PrivateCrossChainBridge(client);
  const noteFactory = new NoteFactory(client);

  // User's keys
  const userKey = randomBytes(32);

  // Create note on Ethereum
  const ethNote = await noteFactory.createNote(userKey, 1000n, "USDC");

  // Deposit to bridge (locks on Ethereum)
  const { deposit, claimSecret } = await bridge.deposit(
    "ethereum",
    ethNote,
    "solana",
    userKey
  );

  console.log("Deposited on Ethereum:", deposit.depositCommitment);

  // Wait for deposit confirmation and relayer to post proof to Solana...

  // Withdraw on Solana (mints equivalent)
  const { withdrawal, recipientNote } = await bridge.withdraw(
    deposit,
    claimSecret,
    "solana",
    userKey,
    1000n,
    "USDC"
  );

  console.log("Withdrawn on Solana:", withdrawal.withdrawalCommitment);
  console.log("Recipient note:", recipientNote.commitment);

  // Privacy preserved:
  // - No one knows who bridged
  // - No one knows how much was bridged
  // - Only commitment and proofs are public
}
```

---

## Quick Reference

### Common Imports

```typescript
import {
  NexuszeroClient,
  ProofBuilder,
  StatementType,
  SecurityLevel,
  PrivacyLevel,
} from "nexuszero-sdk";
import { createHash, randomBytes } from "crypto";
```

### Transaction Types

| Type                     | Privacy     | Use Case             |
| ------------------------ | ----------- | -------------------- |
| UTXO Private Transfer    | Full        | Zcash-style shielded |
| Account Private Transfer | Partial     | Tornado Cash-style   |
| Confidential Transaction | Amount only | CT/RingCT            |
| Private Swap             | Full        | DEX trades           |
| Private Lending          | Full        | DeFi borrowing       |
| Compliant Private        | Selective   | Regulated transfers  |
| Private Bridge           | Full        | Cross-chain          |

### Security Properties

| Property                | Description      | Achieved By            |
| ----------------------- | ---------------- | ---------------------- |
| Amount Privacy          | Values hidden    | Pedersen commitments   |
| Sender Privacy          | Sender unknown   | Note nullifiers        |
| Receiver Privacy        | Receiver unknown | Stealth addresses      |
| Unlinkability           | Can't link txs   | Ring signatures/mixing |
| Conservation            | No inflation     | Sum proofs             |
| Double-Spend Prevention | Can't reuse      | Nullifier set          |

---

## See Also

- [ZK_PROOF_API.md](./ZK_PROOF_API.md) - Core proof API reference
- [ZK_IDENTITY.md](./ZK_IDENTITY.md) - Identity verification examples
- [ZK_VOTING.md](./ZK_VOTING.md) - Voting system examples
- [SDK_DEVELOPER_EXPERIENCE.md](./SDK_DEVELOPER_EXPERIENCE.md) - SDK evaluation
- [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) - Integration guide

---

_These examples demonstrate privacy-preserving financial transactions. For production use, ensure proper security audits and regulatory compliance._
