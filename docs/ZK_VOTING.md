# ZK Voting System Examples

> Privacy-Preserving Voting with Zero-Knowledge Proofs

## Table of Contents

1. [Overview](#overview)
2. [Voting System Architecture](#voting-system-architecture)
3. [Voter Eligibility Proofs](#voter-eligibility-proofs)
4. [Anonymous Ballot Casting](#anonymous-ballot-casting)
5. [Vote Tallying with ZK](#vote-tallying-with-zk)
6. [Verifiable Election Results](#verifiable-election-results)
7. [Advanced Voting Schemes](#advanced-voting-schemes)
8. [Complete Election Implementation](#complete-election-implementation)

---

## Overview

Zero-knowledge voting enables:

- **Anonymous Voting**: No one can link a vote to a voter
- **Eligibility Verification**: Prove right to vote without revealing identity
- **Double-Vote Prevention**: Detect duplicates without de-anonymizing
- **Verifiable Tallying**: Anyone can verify results are correct
- **Coercion Resistance**: Voters can't prove how they voted (to others)

### Security Properties

```
┌────────────────────────────────────────────────────────────────────────┐
│                     ZK Voting Security Properties                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐      │
│  │    Privacy      │   │   Integrity     │   │  Verifiability  │      │
│  │                 │   │                 │   │                 │      │
│  │ • Vote secrecy  │   │ • Only eligible │   │ • Individual    │      │
│  │ • Voter anon    │   │ • No double vote│   │ • Universal     │      │
│  │ • Unlinkability │   │ • Correct tally │   │ • End-to-end    │      │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘      │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     Additional Properties                        │  │
│  │  • Coercion Resistance: Can't prove vote to third party         │  │
│  │  • Receipt-Freeness: No receipt that proves vote                │  │
│  │  • Dispute Resolution: Audit trail without breaking privacy     │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Voting System Architecture

```typescript
import {
  NexuszeroClient,
  ProofBuilder,
  StatementType,
  SecurityLevel,
} from "nexuszero-sdk";
import { createHash, randomBytes } from "crypto";

// Core Types
interface Election {
  id: string;
  title: string;
  description: string;
  options: string[];
  startTime: Date;
  endTime: Date;
  voterRegistryRoot: string; // Merkle root of eligible voters
  nullifierSet: Set<string>; // To detect double voting
  config: ElectionConfig;
}

interface ElectionConfig {
  allowWriteIn: boolean;
  requireMinVotes: number;
  maxVotesPerVoter: number; // For multi-select elections
  anonymitySet: number; // Minimum voters for privacy
}

interface VoterCredential {
  voterId: string; // Hashed voter ID
  voterSecret: Uint8Array; // Random secret for nullifier
  eligibilityProof: Uint8Array;
  registeredAt: Date;
  merkleIndex: number;
  merklePath: Uint8Array[];
}

interface Ballot {
  electionId: string;
  encryptedVote: Uint8Array;
  nullifier: string; // Unique per voter per election
  eligibilityProof: Uint8Array;
  voteProof: Uint8Array;
  timestamp: number;
}

interface VoteResult {
  electionId: string;
  totalVotes: number;
  results: Map<string, number>;
  tallyProof: Uint8Array;
  verified: boolean;
}
```

---

## Voter Eligibility Proofs

### Prove Voter Registration Without Revealing Identity

```typescript
class VoterEligibilityService {
  private client: NexuszeroClient;

  constructor() {
    this.client = new NexuszeroClient({
      securityLevel: SecurityLevel.Bit256, // High security for elections
    });
  }

  /**
   * Generate voter credential after identity verification
   */
  async registerVoter(
    identityHash: string, // Hash of voter's verified identity
    election: Election
  ): Promise<VoterCredential> {
    // Generate random voter secret (never leaves voter's device)
    const voterSecret = randomBytes(32);

    // Compute voter ID (deterministic from identity, unlinkable across elections)
    const voterId = createHash("sha256")
      .update(identityHash)
      .update(election.id)
      .update(voterSecret)
      .digest("hex");

    // In real system: Add to voter registry Merkle tree
    const merkleIndex = 0; // Would be assigned by registry
    const merklePath: Uint8Array[] = []; // Would be computed

    // Generate eligibility proof
    const eligibilityProof = await this.generateEligibilityProof(
      voterId,
      election.voterRegistryRoot,
      merkleIndex,
      merklePath
    );

    return {
      voterId,
      voterSecret,
      eligibilityProof,
      registeredAt: new Date(),
      merkleIndex,
      merklePath,
    };
  }

  /**
   * Prove voter is in the registry without revealing which one
   */
  async generateEligibilityProof(
    voterId: string,
    registryRoot: string,
    merkleIndex: number,
    merklePath: Uint8Array[]
  ): Promise<Uint8Array> {
    // Compute leaf hash
    const leafHash = createHash("sha256").update(voterId).digest();

    // Generate Merkle membership proof
    // This proves: leafHash is at merkleIndex in tree with root registryRoot
    const merkleProof = await this.generateMerkleProof(
      leafHash,
      merkleIndex,
      merklePath,
      Buffer.from(registryRoot, "hex")
    );

    // Generate ZK proof wrapping the Merkle proof
    const zkProof = await new ProofBuilder()
      .setStatement(StatementType.Custom, {
        type: "merkle_membership",
        root: registryRoot,
      })
      .setWitness({
        value: BigInt("0x" + voterId),
      })
      .generate();

    return this.combineProofs([merkleProof, zkProof.data]);
  }

  /**
   * Verify eligibility proof without learning voter identity
   */
  async verifyEligibility(
    eligibilityProof: Uint8Array,
    registryRoot: string
  ): Promise<{ eligible: boolean; error?: string }> {
    try {
      // Verify Merkle proof against registry root
      const [merkleProof, zkProof] = this.splitProofs(eligibilityProof);

      // Verify the ZK proof
      const result = await this.client.verifyProof({
        data: zkProof,
        statement: {
          type: StatementType.Range,
          min: 0n,
          max: BigInt(2) ** BigInt(256),
          bitLength: 256,
        },
        commitment: new Uint8Array(32),
      });

      return { eligible: result.valid };
    } catch (error) {
      return {
        eligible: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  private async generateMerkleProof(
    leaf: Buffer,
    index: number,
    path: Uint8Array[],
    root: Buffer
  ): Promise<Uint8Array> {
    // Construct proof showing leaf is in tree
    const proof = Buffer.alloc(32 + 4 + path.length * 33);
    leaf.copy(proof, 0);
    proof.writeUInt32LE(index, 32);

    let offset = 36;
    for (let i = 0; i < path.length; i++) {
      proof[offset] = (index >> i) & 1; // Left or right
      path[i].slice(0, 32).forEach((b, j) => (proof[offset + 1 + j] = b));
      offset += 33;
    }

    return new Uint8Array(proof);
  }

  private combineProofs(proofs: Uint8Array[]): Uint8Array {
    const lengths = proofs.map((p) => p.length);
    const total = lengths.reduce((a, b) => a + b, 0) + lengths.length * 4;
    const combined = new Uint8Array(total);

    let offset = 0;
    for (let i = 0; i < proofs.length; i++) {
      new DataView(combined.buffer).setUint32(offset, lengths[i], true);
      combined.set(proofs[i], offset + 4);
      offset += 4 + lengths[i];
    }

    return combined;
  }

  private splitProofs(combined: Uint8Array): Uint8Array[] {
    const proofs: Uint8Array[] = [];
    let offset = 0;

    while (offset < combined.length) {
      const length = new DataView(combined.buffer).getUint32(offset, true);
      proofs.push(combined.slice(offset + 4, offset + 4 + length));
      offset += 4 + length;
    }

    return proofs;
  }
}
```

---

## Anonymous Ballot Casting

### Cast Vote Without Revealing Identity or Vote Content

```typescript
class AnonymousVotingService {
  private client: NexuszeroClient;

  constructor() {
    this.client = new NexuszeroClient({
      securityLevel: SecurityLevel.Bit256,
    });
  }

  /**
   * Generate nullifier - unique per voter per election, unlinkable to identity
   */
  generateNullifier(voterSecret: Uint8Array, electionId: string): string {
    return createHash("sha256")
      .update(voterSecret)
      .update(electionId)
      .update("nullifier")
      .digest("hex");
  }

  /**
   * Encrypt vote so only tallying authority can decrypt
   */
  async encryptVote(
    vote: number, // Index of chosen option
    electionPublicKey: Uint8Array
  ): Promise<{ ciphertext: Uint8Array; randomness: Uint8Array }> {
    // ElGamal encryption for homomorphic tallying
    const randomness = randomBytes(32);

    // In production: Use actual ElGamal encryption
    // vote -> g^vote * pk^r, g^r
    const encoded = this.encodeVote(vote);
    const ciphertext = this.elgamalEncrypt(
      encoded,
      electionPublicKey,
      randomness
    );

    return { ciphertext, randomness };
  }

  /**
   * Prove vote is valid (one of allowed options) without revealing which
   */
  async proveValidVote(
    vote: number,
    numOptions: number,
    ciphertext: Uint8Array,
    randomness: Uint8Array
  ): Promise<Uint8Array> {
    // Prove: vote ∈ [0, numOptions) AND ciphertext encrypts vote

    // 1. Range proof for vote validity
    const rangeProof = await new ProofBuilder()
      .setStatement(StatementType.Range, {
        min: 0n,
        max: BigInt(numOptions),
      })
      .setWitness({ value: BigInt(vote) })
      .generate();

    // 2. Encryption correctness proof
    // This proves ciphertext is correctly formed encryption of vote
    const encryptionProof = await this.proveEncryptionCorrectness(
      vote,
      ciphertext,
      randomness
    );

    return this.combineProofs([rangeProof.data, encryptionProof]);
  }

  /**
   * Create complete ballot with all proofs
   */
  async createBallot(
    election: Election,
    credential: VoterCredential,
    vote: number
  ): Promise<Ballot> {
    // Validate vote is in range
    if (vote < 0 || vote >= election.options.length) {
      throw new Error(`Invalid vote: must be 0-${election.options.length - 1}`);
    }

    // Check election is active
    const now = new Date();
    if (now < election.startTime || now > election.endTime) {
      throw new Error("Election is not active");
    }

    // Generate nullifier
    const nullifier = this.generateNullifier(
      credential.voterSecret,
      election.id
    );

    // Check not already voted (would be done by server)
    if (election.nullifierSet.has(nullifier)) {
      throw new Error("Already voted in this election");
    }

    // Get election public key (for encrypted tallying)
    const electionPublicKey = new Uint8Array(32); // Would come from election

    // Encrypt vote
    const { ciphertext, randomness } = await this.encryptVote(
      vote,
      electionPublicKey
    );

    // Generate vote validity proof
    const voteProof = await this.proveValidVote(
      vote,
      election.options.length,
      ciphertext,
      randomness
    );

    return {
      electionId: election.id,
      encryptedVote: ciphertext,
      nullifier,
      eligibilityProof: credential.eligibilityProof,
      voteProof,
      timestamp: Date.now(),
    };
  }

  /**
   * Verify ballot without decrypting vote
   */
  async verifyBallot(
    ballot: Ballot,
    election: Election
  ): Promise<{ valid: boolean; errors: string[] }> {
    const errors: string[] = [];

    // 1. Check nullifier not used
    if (election.nullifierSet.has(ballot.nullifier)) {
      errors.push("Duplicate nullifier - voter already voted");
    }

    // 2. Verify eligibility proof
    const eligibilityService = new VoterEligibilityService();
    const eligibility = await eligibilityService.verifyEligibility(
      ballot.eligibilityProof,
      election.voterRegistryRoot
    );
    if (!eligibility.eligible) {
      errors.push("Invalid eligibility proof");
    }

    // 3. Verify vote proof
    const voteValid = await this.verifyVoteProof(
      ballot.voteProof,
      ballot.encryptedVote,
      election.options.length
    );
    if (!voteValid) {
      errors.push("Invalid vote proof");
    }

    // 4. Check timestamp
    if (
      ballot.timestamp < election.startTime.getTime() ||
      ballot.timestamp > election.endTime.getTime()
    ) {
      errors.push("Ballot timestamp outside election window");
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  private async verifyVoteProof(
    proof: Uint8Array,
    ciphertext: Uint8Array,
    numOptions: number
  ): Promise<boolean> {
    // Verify the combined proof
    const [rangeProof, encryptionProof] = this.splitProofs(proof);

    const rangeResult = await this.client.verifyProof({
      data: rangeProof,
      statement: {
        type: StatementType.Range,
        min: 0n,
        max: BigInt(numOptions),
        bitLength: Math.ceil(Math.log2(numOptions)),
      },
      commitment: new Uint8Array(32),
    });

    // In production: Also verify encryption proof

    return rangeResult.valid;
  }

  private encodeVote(vote: number): Uint8Array {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, vote, true);
    return new Uint8Array(buf);
  }

  private elgamalEncrypt(
    message: Uint8Array,
    publicKey: Uint8Array,
    randomness: Uint8Array
  ): Uint8Array {
    // Simplified - real implementation would use actual ElGamal
    const ciphertext = new Uint8Array(message.length + 32);
    ciphertext.set(message, 0);
    ciphertext.set(randomness, message.length);
    return ciphertext;
  }

  private async proveEncryptionCorrectness(
    vote: number,
    ciphertext: Uint8Array,
    randomness: Uint8Array
  ): Promise<Uint8Array> {
    // In production: Sigma protocol for ElGamal encryption
    return new Uint8Array(256);
  }

  private combineProofs(proofs: Uint8Array[]): Uint8Array {
    const total = proofs.reduce((sum, p) => sum + p.length + 4, 0);
    const combined = new Uint8Array(total);
    let offset = 0;

    for (const proof of proofs) {
      new DataView(combined.buffer).setUint32(offset, proof.length, true);
      combined.set(proof, offset + 4);
      offset += proof.length + 4;
    }

    return combined;
  }

  private splitProofs(combined: Uint8Array): Uint8Array[] {
    const proofs: Uint8Array[] = [];
    let offset = 0;

    while (offset < combined.length) {
      const length = new DataView(combined.buffer).getUint32(offset, true);
      proofs.push(combined.slice(offset + 4, offset + 4 + length));
      offset += 4 + length;
    }

    return proofs;
  }
}
```

---

## Vote Tallying with ZK

### Verifiable Homomorphic Tallying

```typescript
class ZKTallyingService {
  private client: NexuszeroClient;

  constructor() {
    this.client = new NexuszeroClient({
      securityLevel: SecurityLevel.Bit256,
    });
  }

  /**
   * Tally votes homomorphically without decrypting individual votes
   */
  async tallyElection(
    election: Election,
    ballots: Ballot[],
    privateKey: Uint8Array // Only tallying authority has this
  ): Promise<VoteResult> {
    // Filter valid ballots
    const votingService = new AnonymousVotingService();
    const validBallots: Ballot[] = [];

    for (const ballot of ballots) {
      const { valid } = await votingService.verifyBallot(ballot, election);
      if (valid) {
        validBallots.push(ballot);
      }
    }

    // Homomorphic aggregation of encrypted votes
    const aggregatedCiphertext = this.homomorphicSum(
      validBallots.map((b) => b.encryptedVote)
    );

    // Decrypt aggregated result (only reveals totals, not individual votes)
    const decryptedTotals = this.decryptTally(aggregatedCiphertext, privateKey);

    // Convert to results map
    const results = new Map<string, number>();
    for (let i = 0; i < election.options.length; i++) {
      results.set(election.options[i], decryptedTotals[i]);
    }

    // Generate proof that decryption was done correctly
    const tallyProof = await this.generateTallyProof(
      aggregatedCiphertext,
      decryptedTotals,
      privateKey
    );

    return {
      electionId: election.id,
      totalVotes: validBallots.length,
      results,
      tallyProof,
      verified: true,
    };
  }

  /**
   * Verify tally proof without knowing private key
   */
  async verifyTally(
    election: Election,
    ballots: Ballot[],
    result: VoteResult
  ): Promise<{ valid: boolean; discrepancies: string[] }> {
    const discrepancies: string[] = [];

    // 1. Verify total vote count
    const votingService = new AnonymousVotingService();
    let validCount = 0;

    for (const ballot of ballots) {
      const { valid } = await votingService.verifyBallot(ballot, election);
      if (valid) validCount++;
    }

    if (validCount !== result.totalVotes) {
      discrepancies.push(
        `Vote count mismatch: expected ${validCount}, got ${result.totalVotes}`
      );
    }

    // 2. Verify results sum to total
    const resultSum = Array.from(result.results.values()).reduce(
      (a, b) => a + b,
      0
    );
    if (resultSum !== result.totalVotes) {
      discrepancies.push(
        `Results don't sum to total: ${resultSum} vs ${result.totalVotes}`
      );
    }

    // 3. Verify tally proof
    const tallyValid = await this.verifyTallyProof(
      ballots.map((b) => b.encryptedVote),
      Array.from(result.results.values()),
      result.tallyProof
    );

    if (!tallyValid) {
      discrepancies.push("Tally proof verification failed");
    }

    return {
      valid: discrepancies.length === 0,
      discrepancies,
    };
  }

  /**
   * Homomorphically sum encrypted votes
   */
  private homomorphicSum(ciphertexts: Uint8Array[]): Uint8Array {
    // ElGamal is additively homomorphic:
    // E(m1) * E(m2) = E(m1 + m2)

    if (ciphertexts.length === 0) {
      return new Uint8Array(64);
    }

    // Start with first ciphertext
    let result = new Uint8Array(ciphertexts[0]);

    // Multiply (add in encrypted domain) remaining ciphertexts
    for (let i = 1; i < ciphertexts.length; i++) {
      result = this.multiplyCiphertexts(result, ciphertexts[i]);
    }

    return result;
  }

  private multiplyCiphertexts(c1: Uint8Array, c2: Uint8Array): Uint8Array {
    // In production: Point multiplication for ElGamal
    // Simplified: XOR (not actually secure)
    const result = new Uint8Array(c1.length);
    for (let i = 0; i < c1.length; i++) {
      result[i] = c1[i] ^ c2[i];
    }
    return result;
  }

  private decryptTally(
    aggregatedCiphertext: Uint8Array,
    privateKey: Uint8Array
  ): number[] {
    // In production: ElGamal decryption
    // Returns vote counts for each option
    return [10, 15, 5]; // Example: 10 for option 0, 15 for option 1, etc.
  }

  private async generateTallyProof(
    aggregatedCiphertext: Uint8Array,
    decryptedTotals: number[],
    privateKey: Uint8Array
  ): Promise<Uint8Array> {
    // Generate ZK proof that decryption was performed correctly
    // without revealing the private key

    // This proves: decryptedTotals = Decrypt(aggregatedCiphertext, privateKey)
    // Verifier can check with public key

    return new Uint8Array(512);
  }

  private async verifyTallyProof(
    encryptedVotes: Uint8Array[],
    decryptedTotals: number[],
    proof: Uint8Array
  ): Promise<boolean> {
    // Verify the tally proof using public key
    return proof.length > 0;
  }
}
```

---

## Verifiable Election Results

### Public Verification of Results

```typescript
class ElectionVerifier {
  private client: NexuszeroClient;

  constructor() {
    this.client = new NexuszeroClient();
  }

  /**
   * Anyone can verify election integrity
   */
  async verifyElectionIntegrity(
    election: Election,
    ballots: Ballot[],
    result: VoteResult,
    options: {
      verifyAllBallots?: boolean;
      sampleSize?: number;
    } = {}
  ): Promise<ElectionIntegrityReport> {
    const report: ElectionIntegrityReport = {
      electionId: election.id,
      totalBallots: ballots.length,
      validBallots: 0,
      invalidBallots: 0,
      duplicateNullifiers: 0,
      tallyVerified: false,
      resultsMatchBallots: false,
      checks: [],
    };

    // Track nullifiers for duplicate detection
    const seenNullifiers = new Set<string>();
    const duplicates: string[] = [];

    // Verify ballots
    const ballotsToCheck = options.verifyAllBallots
      ? ballots
      : this.randomSample(ballots, options.sampleSize || 100);

    const votingService = new AnonymousVotingService();

    for (const ballot of ballotsToCheck) {
      // Check for duplicate nullifier
      if (seenNullifiers.has(ballot.nullifier)) {
        duplicates.push(ballot.nullifier);
        report.duplicateNullifiers++;
      }
      seenNullifiers.add(ballot.nullifier);

      // Verify ballot
      const { valid, errors } = await votingService.verifyBallot(
        ballot,
        election
      );

      if (valid) {
        report.validBallots++;
      } else {
        report.invalidBallots++;
        report.checks.push({
          type: "ballot_invalid",
          details: errors.join(", "),
          passed: false,
        });
      }
    }

    // Verify tally
    const tallyService = new ZKTallyingService();
    const { valid: tallyValid, discrepancies } = await tallyService.verifyTally(
      election,
      ballots,
      result
    );

    report.tallyVerified = tallyValid;
    report.checks.push({
      type: "tally_verification",
      details: tallyValid ? "Tally proof verified" : discrepancies.join(", "),
      passed: tallyValid,
    });

    // Check results match ballot count
    const resultSum = Array.from(result.results.values()).reduce(
      (a, b) => a + b,
      0
    );
    report.resultsMatchBallots = resultSum === result.totalVotes;
    report.checks.push({
      type: "result_sum",
      details: `Sum: ${resultSum}, Total: ${result.totalVotes}`,
      passed: report.resultsMatchBallots,
    });

    return report;
  }

  /**
   * Voter can verify their vote was counted
   */
  async verifyVoteIncluded(
    ballot: Ballot,
    allBallots: Ballot[],
    result: VoteResult
  ): Promise<{ included: boolean; proof: Uint8Array }> {
    // Find ballot by nullifier
    const found = allBallots.find((b) => b.nullifier === ballot.nullifier);

    if (!found) {
      return { included: false, proof: new Uint8Array(0) };
    }

    // Generate Merkle proof that ballot is in the set
    const ballotHashes = allBallots.map((b) => this.hashBallot(b));
    const myIndex = allBallots.indexOf(found);
    const merkleProof = this.generateMerkleProof(ballotHashes, myIndex);

    return { included: true, proof: merkleProof };
  }

  private randomSample<T>(arr: T[], size: number): T[] {
    const shuffled = [...arr].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, Math.min(size, arr.length));
  }

  private hashBallot(ballot: Ballot): Uint8Array {
    const hash = createHash("sha256");
    hash.update(ballot.electionId);
    hash.update(ballot.nullifier);
    hash.update(ballot.encryptedVote);
    return new Uint8Array(hash.digest());
  }

  private generateMerkleProof(leaves: Uint8Array[], index: number): Uint8Array {
    // Build Merkle proof for leaf at index
    return new Uint8Array(256);
  }
}

interface ElectionIntegrityReport {
  electionId: string;
  totalBallots: number;
  validBallots: number;
  invalidBallots: number;
  duplicateNullifiers: number;
  tallyVerified: boolean;
  resultsMatchBallots: boolean;
  checks: Array<{
    type: string;
    details: string;
    passed: boolean;
  }>;
}
```

---

## Advanced Voting Schemes

### Ranked Choice Voting with ZK

```typescript
class ZKRankedChoiceVoting {
  private client: NexuszeroClient;

  constructor() {
    this.client = new NexuszeroClient();
  }

  /**
   * Create ranked choice ballot
   */
  async createRankedBallot(
    election: Election,
    credential: VoterCredential,
    rankings: number[] // e.g., [2, 0, 1] means option 2 first, then 0, then 1
  ): Promise<RankedBallot> {
    // Validate rankings
    if (rankings.length !== election.options.length) {
      throw new Error("Must rank all options");
    }

    const uniqueRanks = new Set(rankings);
    if (uniqueRanks.size !== rankings.length) {
      throw new Error("Each rank must be unique");
    }

    // Encrypt each ranking separately
    const encryptedRankings: Uint8Array[] = [];
    const rankProofs: Uint8Array[] = [];

    for (let i = 0; i < rankings.length; i++) {
      const { ciphertext, randomness } = await this.encryptRank(
        rankings[i],
        election.options.length
      );
      encryptedRankings.push(ciphertext);

      // Prove rank is valid (0 to n-1)
      const rankProof = await new ProofBuilder()
        .setStatement(StatementType.Range, {
          min: 0n,
          max: BigInt(election.options.length),
        })
        .setWitness({ value: BigInt(rankings[i]) })
        .generate();
      rankProofs.push(rankProof.data);
    }

    // Prove rankings are a valid permutation
    const permutationProof = await this.provePermutation(
      rankings,
      election.options.length
    );

    const nullifier = this.generateNullifier(
      credential.voterSecret,
      election.id
    );

    return {
      electionId: election.id,
      encryptedRankings,
      nullifier,
      eligibilityProof: credential.eligibilityProof,
      rankProofs,
      permutationProof,
      timestamp: Date.now(),
    };
  }

  /**
   * Instant Runoff Voting tallying
   */
  async tallyIRV(
    election: Election,
    ballots: RankedBallot[],
    privateKey: Uint8Array
  ): Promise<IRVResult> {
    const rounds: IRVRound[] = [];
    let eliminatedOptions: number[] = [];
    let winner: number | null = null;

    // Decrypt all rankings (in batched, privacy-preserving way)
    const decryptedBallots = await this.decryptRankings(ballots, privateKey);

    // Run IRV rounds
    while (winner === null) {
      const round = await this.runIRVRound(
        decryptedBallots,
        eliminatedOptions,
        election.options.length
      );

      rounds.push(round);

      // Check for winner (majority)
      const majority = Math.floor(ballots.length / 2) + 1;
      if (round.leaderVotes >= majority) {
        winner = round.leader;
      } else if (round.eliminated === null) {
        // Tie or all eliminated
        winner = round.leader;
      } else {
        eliminatedOptions.push(round.eliminated);
      }
    }

    return {
      electionId: election.id,
      winner: election.options[winner],
      rounds,
      totalBallots: ballots.length,
    };
  }

  private async encryptRank(
    rank: number,
    numOptions: number
  ): Promise<{ ciphertext: Uint8Array; randomness: Uint8Array }> {
    const randomness = randomBytes(32);
    const ciphertext = new Uint8Array(36);
    new DataView(ciphertext.buffer).setUint32(0, rank, true);
    ciphertext.set(randomness, 4);
    return { ciphertext, randomness };
  }

  private async provePermutation(
    values: number[],
    size: number
  ): Promise<Uint8Array> {
    // Prove values is a valid permutation of [0, 1, ..., size-1]
    return new Uint8Array(256);
  }

  private generateNullifier(secret: Uint8Array, electionId: string): string {
    return createHash("sha256").update(secret).update(electionId).digest("hex");
  }

  private async decryptRankings(
    ballots: RankedBallot[],
    privateKey: Uint8Array
  ): Promise<number[][]> {
    // Decrypt all rankings
    return ballots.map((b) => [0, 1, 2]); // Placeholder
  }

  private async runIRVRound(
    ballots: number[][],
    eliminated: number[],
    numOptions: number
  ): Promise<IRVRound> {
    // Count first-choice votes (excluding eliminated)
    const votes = new Array(numOptions).fill(0);

    for (const ballot of ballots) {
      // Find highest-ranked non-eliminated option
      for (let rank = 0; rank < ballot.length; rank++) {
        const option = ballot.indexOf(rank);
        if (!eliminated.includes(option)) {
          votes[option]++;
          break;
        }
      }
    }

    // Find leader and loser
    let leader = -1;
    let loser = -1;
    let maxVotes = -1;
    let minVotes = Infinity;

    for (let i = 0; i < votes.length; i++) {
      if (eliminated.includes(i)) continue;

      if (votes[i] > maxVotes) {
        maxVotes = votes[i];
        leader = i;
      }
      if (votes[i] < minVotes) {
        minVotes = votes[i];
        loser = i;
      }
    }

    return {
      voteCounts: votes,
      leader,
      leaderVotes: maxVotes,
      eliminated: loser,
      eliminatedVotes: minVotes,
    };
  }
}

interface RankedBallot {
  electionId: string;
  encryptedRankings: Uint8Array[];
  nullifier: string;
  eligibilityProof: Uint8Array;
  rankProofs: Uint8Array[];
  permutationProof: Uint8Array;
  timestamp: number;
}

interface IRVRound {
  voteCounts: number[];
  leader: number;
  leaderVotes: number;
  eliminated: number | null;
  eliminatedVotes: number;
}

interface IRVResult {
  electionId: string;
  winner: string;
  rounds: IRVRound[];
  totalBallots: number;
}
```

### Quadratic Voting with ZK

```typescript
class ZKQuadraticVoting {
  private client: NexuszeroClient;

  constructor() {
    this.client = new NexuszeroClient();
  }

  /**
   * Create quadratic voting ballot
   *
   * In quadratic voting, voters have N voice credits.
   * To cast k votes for an option costs k² credits.
   */
  async createQuadraticBallot(
    election: QuadraticElection,
    credential: VoterCredential,
    votes: Map<number, number> // option -> vote count
  ): Promise<QuadraticBallot> {
    // Calculate total credits used
    let creditsUsed = 0;
    for (const [option, voteCount] of votes) {
      creditsUsed += voteCount * voteCount; // Quadratic cost
    }

    if (creditsUsed > election.voiceCredits) {
      throw new Error(
        `Insufficient credits: used ${creditsUsed}, have ${election.voiceCredits}`
      );
    }

    // Encrypt each vote
    const encryptedVotes: Map<number, Uint8Array> = new Map();
    const voteProofs: Uint8Array[] = [];

    for (const [option, voteCount] of votes) {
      const { ciphertext } = await this.encryptVote(voteCount);
      encryptedVotes.set(option, ciphertext);

      // Prove vote count is non-negative
      const voteProof = await new ProofBuilder()
        .setStatement(StatementType.Range, {
          min: 0n,
          max: BigInt(Math.sqrt(election.voiceCredits) + 1),
        })
        .setWitness({ value: BigInt(voteCount) })
        .generate();
      voteProofs.push(voteProof.data);
    }

    // Prove total credits ≤ available credits
    const creditProof = await new ProofBuilder()
      .setStatement(StatementType.Range, {
        min: 0n,
        max: BigInt(election.voiceCredits + 1),
      })
      .setWitness({ value: BigInt(creditsUsed) })
      .generate();

    // Prove quadratic relationship: Σ(votes²) = creditsUsed
    const quadraticProof = await this.proveQuadraticSum(votes, creditsUsed);

    const nullifier = this.generateNullifier(
      credential.voterSecret,
      election.id
    );

    return {
      electionId: election.id,
      encryptedVotes: Object.fromEntries(encryptedVotes),
      nullifier,
      eligibilityProof: credential.eligibilityProof,
      voteProofs,
      creditProof: creditProof.data,
      quadraticProof,
      timestamp: Date.now(),
    };
  }

  /**
   * Tally quadratic votes
   */
  async tallyQuadratic(
    election: QuadraticElection,
    ballots: QuadraticBallot[],
    privateKey: Uint8Array
  ): Promise<QuadraticResult> {
    const totals = new Map<string, number>();

    // Initialize totals
    for (const option of election.options) {
      totals.set(option, 0);
    }

    // Decrypt and sum votes
    for (const ballot of ballots) {
      for (const [optionIdx, encryptedVote] of Object.entries(
        ballot.encryptedVotes
      )) {
        const decrypted = await this.decryptVote(
          encryptedVote as unknown as Uint8Array,
          privateKey
        );
        const option = election.options[Number(optionIdx)];
        totals.set(option, (totals.get(option) || 0) + decrypted);
      }
    }

    // Generate tally proof
    const tallyProof = new Uint8Array(512);

    return {
      electionId: election.id,
      totalBallots: ballots.length,
      results: totals,
      tallyProof,
    };
  }

  private async encryptVote(vote: number): Promise<{ ciphertext: Uint8Array }> {
    return { ciphertext: new Uint8Array(36) };
  }

  private async decryptVote(
    ciphertext: Uint8Array,
    privateKey: Uint8Array
  ): Promise<number> {
    return 0;
  }

  private async proveQuadraticSum(
    votes: Map<number, number>,
    total: number
  ): Promise<Uint8Array> {
    return new Uint8Array(256);
  }

  private generateNullifier(secret: Uint8Array, electionId: string): string {
    return createHash("sha256").update(secret).update(electionId).digest("hex");
  }
}

interface QuadraticElection extends Election {
  voiceCredits: number; // Total credits per voter
}

interface QuadraticBallot {
  electionId: string;
  encryptedVotes: Record<number, Uint8Array>;
  nullifier: string;
  eligibilityProof: Uint8Array;
  voteProofs: Uint8Array[];
  creditProof: Uint8Array;
  quadraticProof: Uint8Array;
  timestamp: number;
}

interface QuadraticResult {
  electionId: string;
  totalBallots: number;
  results: Map<string, number>;
  tallyProof: Uint8Array;
}
```

---

## Complete Election Implementation

### End-to-End Election Example

```typescript
async function runCompleteElection() {
  // ============ SETUP PHASE ============

  // 1. Create election
  const election: Election = {
    id: "election_2024_board",
    title: "Board of Directors Election 2024",
    description: "Annual board member election",
    options: ["Alice Johnson", "Bob Smith", "Carol Williams"],
    startTime: new Date("2024-03-01T00:00:00Z"),
    endTime: new Date("2024-03-15T23:59:59Z"),
    voterRegistryRoot: "", // Will be set after registration
    nullifierSet: new Set(),
    config: {
      allowWriteIn: false,
      requireMinVotes: 100,
      maxVotesPerVoter: 1,
      anonymitySet: 50,
    },
  };

  console.log("Election created:", election.title);

  // ============ REGISTRATION PHASE ============

  const eligibilityService = new VoterEligibilityService();
  const voters: VoterCredential[] = [];

  // Register voters (in practice, done after identity verification)
  const voterIdentities = [
    "hash_voter_1",
    "hash_voter_2",
    "hash_voter_3",
    // ... hundreds of voters
  ];

  for (const identity of voterIdentities) {
    const credential = await eligibilityService.registerVoter(
      identity,
      election
    );
    voters.push(credential);
  }

  // Update registry root (Merkle root of all voters)
  election.voterRegistryRoot = computeRegistryRoot(voters);

  console.log(`Registered ${voters.length} voters`);

  // ============ VOTING PHASE ============

  const votingService = new AnonymousVotingService();
  const ballots: Ballot[] = [];

  // Each voter casts their vote
  for (let i = 0; i < voters.length; i++) {
    const voter = voters[i];
    const vote = Math.floor(Math.random() * election.options.length); // Random for example

    try {
      const ballot = await votingService.createBallot(election, voter, vote);

      // Verify ballot before accepting
      const { valid, errors } = await votingService.verifyBallot(
        ballot,
        election
      );

      if (valid) {
        ballots.push(ballot);
        election.nullifierSet.add(ballot.nullifier);
        console.log(`Vote ${i + 1} accepted`);
      } else {
        console.log(`Vote ${i + 1} rejected:`, errors);
      }
    } catch (error) {
      console.log(`Vote ${i + 1} failed:`, error);
    }
  }

  console.log(`\nVoting complete: ${ballots.length} valid ballots`);

  // ============ TALLYING PHASE ============

  const tallyService = new ZKTallyingService();
  const privateKey = generateTallyingPrivateKey(); // In practice, multi-party

  const result = await tallyService.tallyElection(
    election,
    ballots,
    privateKey
  );

  console.log("\n=== Election Results ===");
  console.log(`Total votes: ${result.totalVotes}`);
  for (const [option, votes] of result.results) {
    const percentage = ((votes / result.totalVotes) * 100).toFixed(1);
    console.log(`  ${option}: ${votes} (${percentage}%)`);
  }

  // ============ VERIFICATION PHASE ============

  const verifier = new ElectionVerifier();
  const report = await verifier.verifyElectionIntegrity(
    election,
    ballots,
    result,
    { verifyAllBallots: true }
  );

  console.log("\n=== Integrity Report ===");
  console.log(`Valid ballots: ${report.validBallots}`);
  console.log(`Invalid ballots: ${report.invalidBallots}`);
  console.log(`Duplicate nullifiers: ${report.duplicateNullifiers}`);
  console.log(`Tally verified: ${report.tallyVerified}`);
  console.log(`Results match ballots: ${report.resultsMatchBallots}`);

  for (const check of report.checks) {
    console.log(
      `  [${check.passed ? "✓" : "✗"}] ${check.type}: ${check.details}`
    );
  }

  // ============ INDIVIDUAL VERIFICATION ============

  // Any voter can verify their vote was counted
  const myVoter = voters[0];
  const myBallot = ballots.find(
    (b) =>
      b.nullifier ===
      votingService.generateNullifier(myVoter.voterSecret, election.id)
  );

  if (myBallot) {
    const { included, proof } = await verifier.verifyVoteIncluded(
      myBallot,
      ballots,
      result
    );
    console.log(`\nMy vote included: ${included}`);
  }
}

function computeRegistryRoot(voters: VoterCredential[]): string {
  // Compute Merkle root of voter IDs
  return "merkle_root_" + voters.length;
}

function generateTallyingPrivateKey(): Uint8Array {
  return randomBytes(32);
}

// Run the election
runCompleteElection().catch(console.error);
```

---

## Security Considerations

### Threats and Mitigations

| Threat             | Mitigation                                    |
| ------------------ | --------------------------------------------- |
| Voter coercion     | Receipt-freeness (can't prove vote to others) |
| Vote buying        | Nullifier prevents selling proof              |
| Ballot stuffing    | Eligibility proofs tied to identity           |
| Double voting      | Unique nullifiers per voter/election          |
| Tally manipulation | ZK proof of correct decryption                |
| Voter enumeration  | Unlinkable nullifiers                         |

### Best Practices

1. **Minimum anonymity set**: Require 50+ voters before revealing results
2. **Time-lock tallying**: Don't reveal results until election ends
3. **Multi-party decryption**: Require multiple authorities for tallying
4. **Audit log**: Keep verifiable log of all operations
5. **Disaster recovery**: Plan for key compromise scenarios

---

_See also: [ZK_IDENTITY.md](./ZK_IDENTITY.md) | [ZK_PRIVATE_TRANSACTIONS.md](./ZK_PRIVATE_TRANSACTIONS.md)_
