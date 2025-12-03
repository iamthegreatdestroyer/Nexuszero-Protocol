# ZK Identity Verification Examples

> Zero-Knowledge Identity Proofs for Privacy-Preserving Verification

## Table of Contents

1. [Overview](#overview)
2. [Age Verification](#age-verification)
3. [KYC/AML Compliance](#kycaml-compliance)
4. [Credential Verification](#credential-verification)
5. [Identity Attestations](#identity-attestations)
6. [Selective Disclosure](#selective-disclosure)
7. [Cross-Platform Identity](#cross-platform-identity)

---

## Overview

Zero-knowledge identity verification allows users to prove claims about themselves without revealing the underlying data. This enables:

- **Privacy**: Users control what information is shared
- **Compliance**: Meet regulatory requirements without data exposure
- **Security**: No sensitive data transmitted or stored
- **Portability**: Same proof works across platforms

### Core Concepts

```
┌────────────────────────────────────────────────────────────────────────┐
│                    ZK Identity Architecture                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │   Identity   │    │    Claim     │    │    Proof     │             │
│  │   Provider   │───▶│   Request    │───▶│  Generation  │             │
│  └──────────────┘    └──────────────┘    └──────────────┘             │
│         │                   │                   │                      │
│         ▼                   ▼                   ▼                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │  Credential  │    │  Statement   │    │   ZK Proof   │             │
│  │   Issuance   │    │  Definition  │    │   Output     │             │
│  └──────────────┘    └──────────────┘    └──────────────┘             │
│                                                 │                      │
│                                                 ▼                      │
│                                          ┌──────────────┐             │
│                                          │  Verification │             │
│                                          │   (No Data)   │             │
│                                          └──────────────┘             │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Age Verification

### Use Case: Prove 18+ Without Revealing Birthday

**Problem**: Online platforms need to verify users are adults, but collecting birthdays creates privacy and compliance risks.

**Solution**: ZK proof that birthdate results in age ≥ 18.

#### TypeScript Implementation

```typescript
import {
  NexuszeroClient,
  ProofBuilder,
  StatementType,
  SecurityLevel,
} from "nexuszero-sdk";

interface AgeVerificationResult {
  proof: Uint8Array;
  statement: {
    claim: "age_over";
    threshold: number;
    verifiedAt: Date;
  };
  expiresAt: Date;
}

class AgeVerificationService {
  private client: NexuszeroClient;

  constructor() {
    this.client = new NexuszeroClient({
      securityLevel: SecurityLevel.Bit128,
    });
  }

  /**
   * Prove user is over a certain age without revealing actual age
   * @param birthDate - User's birth date (kept private)
   * @param minAge - Minimum age to prove (e.g., 18, 21)
   * @returns ZK proof of age verification
   */
  async proveAgeOver(
    birthDate: Date,
    minAge: number
  ): Promise<AgeVerificationResult> {
    // Calculate age in years
    const today = new Date();
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();
    if (
      monthDiff < 0 ||
      (monthDiff === 0 && today.getDate() < birthDate.getDate())
    ) {
      age--;
    }

    // Validate age is in reasonable range
    if (age < 0 || age > 150) {
      throw new Error("Invalid birth date");
    }

    // Generate range proof: age ∈ [minAge, 150]
    const proof = await new ProofBuilder()
      .setStatement(StatementType.Range, {
        min: BigInt(minAge),
        max: 150n, // Reasonable maximum age
      })
      .setWitness({
        value: BigInt(age),
      })
      .generate();

    return {
      proof: proof.data,
      statement: {
        claim: "age_over",
        threshold: minAge,
        verifiedAt: new Date(),
      },
      // Proof valid for 24 hours (age won't change)
      expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
    };
  }

  /**
   * Verify an age proof
   * @param proofData - The proof to verify
   * @param expectedMinAge - The minimum age that was claimed
   */
  async verifyAgeProof(
    proofData: Uint8Array,
    expectedMinAge: number
  ): Promise<{ valid: boolean; claim: string }> {
    const proof = {
      data: proofData,
      statement: {
        type: StatementType.Range as const,
        min: BigInt(expectedMinAge),
        max: 150n,
        bitLength: 8,
      },
      commitment: new Uint8Array(32),
    };

    const result = await this.client.verifyProof(proof);

    return {
      valid: result.valid,
      claim: result.valid
        ? `User is ${expectedMinAge}+ years old`
        : "Verification failed",
    };
  }
}

// Usage Example
async function verifyUserAge() {
  const service = new AgeVerificationService();

  // User's actual birthday (never leaves their device)
  const userBirthDate = new Date("1995-06-15");

  // Generate proof of being 18+
  const ageProof = await service.proveAgeOver(userBirthDate, 18);
  console.log("Proof generated:", ageProof.statement);

  // Send only the proof to server (no birthday sent!)
  const verification = await service.verifyAgeProof(ageProof.proof, 18);
  console.log("Verified:", verification);

  // Server learns: User is 18+
  // Server does NOT learn: User is 29 years old, born June 15, 1995
}
```

#### Python Implementation

```python
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Optional
from nexuszero import NexusZeroClient, PrivacyLevel

@dataclass
class AgeProof:
    proof_data: bytes
    claim: str
    threshold: int
    verified_at: date
    expires_at: date

class AgeVerificationService:
    """Zero-knowledge age verification service."""

    def __init__(self, client: Optional[NexusZeroClient] = None):
        self.client = client or NexusZeroClient()

    def calculate_age(self, birth_date: date) -> int:
        """Calculate age in years from birth date."""
        today = date.today()
        age = today.year - birth_date.year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        return age

    async def prove_age_over(
        self,
        birth_date: date,
        min_age: int = 18
    ) -> AgeProof:
        """
        Generate a ZK proof that user is over a certain age.

        Args:
            birth_date: User's birth date (private, never transmitted)
            min_age: Minimum age to prove (default: 18)

        Returns:
            AgeProof containing the ZK proof and metadata
        """
        age = self.calculate_age(birth_date)

        if age < min_age:
            raise ValueError(f"User is not {min_age}+ years old")

        # Generate range proof: age ∈ [min_age, 150]
        proof_result = await self.client.generate_proof(
            data=age.to_bytes(2, 'big'),
            privacy_level=PrivacyLevel.PRIVATE,
        )

        return AgeProof(
            proof_data=proof_result.proof_data,
            claim=f"age >= {min_age}",
            threshold=min_age,
            verified_at=date.today(),
            expires_at=date.today() + timedelta(days=1),
        )

    async def verify_age_proof(self, proof: AgeProof) -> bool:
        """Verify an age proof."""
        if proof.expires_at < date.today():
            return False

        result = await self.client.verify_proof(proof.proof_data)
        return result.valid

# Usage
async def main():
    async with NexusZeroClient() as client:
        service = AgeVerificationService(client)

        # User proves they're 21+ for alcohol purchase
        proof = await service.prove_age_over(
            birth_date=date(1995, 6, 15),
            min_age=21
        )

        # Retailer verifies without learning actual age
        is_valid = await service.verify_age_proof(proof)
        print(f"Age verification: {'Passed' if is_valid else 'Failed'}")
```

---

## KYC/AML Compliance

### Use Case: Prove KYC Status Without Revealing Personal Data

**Problem**: Financial regulations require KYC verification, but sharing personal data across platforms creates security risks.

**Solution**: ZK proof that user passed KYC with a trusted provider.

#### Complete KYC Flow

```typescript
import { NexuszeroClient, StatementType, ProofBuilder } from "nexuszero-sdk";
import { createHash } from "crypto";

// KYC Provider Types
interface KYCCredential {
  providerId: string;
  providerSignature: Uint8Array;
  userId: string; // Hashed user ID
  verificationLevel: "basic" | "enhanced" | "full";
  verifiedAt: Date;
  expiresAt: Date;
  attributes: {
    identityVerified: boolean;
    addressVerified: boolean;
    documentVerified: boolean;
    sanctions: "clear" | "flagged";
    pep: boolean; // Politically Exposed Person
    riskScore: number; // 0-100
  };
}

interface KYCProofRequest {
  requiredLevel: "basic" | "enhanced" | "full";
  checkSanctions: boolean;
  checkPEP: boolean;
  maxRiskScore?: number;
}

interface KYCProof {
  proof: Uint8Array;
  claims: string[];
  credentialHash: string;
  expiresAt: Date;
}

class ZKKYCService {
  private client: NexuszeroClient;
  private trustedProviders: Map<string, Uint8Array>; // providerId -> publicKey

  constructor() {
    this.client = new NexuszeroClient();
    this.trustedProviders = new Map();
  }

  /**
   * Register a trusted KYC provider
   */
  registerProvider(providerId: string, publicKey: Uint8Array): void {
    this.trustedProviders.set(providerId, publicKey);
  }

  /**
   * Generate KYC compliance proof without revealing personal data
   */
  async generateKYCProof(
    credential: KYCCredential,
    request: KYCProofRequest
  ): Promise<KYCProof> {
    // Verify credential is from trusted provider
    if (!this.trustedProviders.has(credential.providerId)) {
      throw new Error("Untrusted KYC provider");
    }

    // Check credential not expired
    if (new Date() > credential.expiresAt) {
      throw new Error("KYC credential expired");
    }

    const claims: string[] = [];
    const proofs: Uint8Array[] = [];

    // 1. Prove KYC level meets requirement
    const levelMap = { basic: 1, enhanced: 2, full: 3 };
    const requiredLevel = levelMap[request.requiredLevel];
    const actualLevel = levelMap[credential.verificationLevel];

    if (actualLevel >= requiredLevel) {
      claims.push(`kyc_level >= ${request.requiredLevel}`);

      const levelProof = await new ProofBuilder()
        .setStatement(StatementType.Range, {
          min: BigInt(requiredLevel),
          max: 4n,
        })
        .setWitness({ value: BigInt(actualLevel) })
        .generate();
      proofs.push(levelProof.data);
    }

    // 2. Prove sanctions status if required
    if (request.checkSanctions) {
      if (credential.attributes.sanctions === "clear") {
        claims.push("sanctions_clear");
        // Boolean proof: sanctions == clear (encoded as 1)
        const sanctionsProof = await new ProofBuilder()
          .setStatement(StatementType.Range, { min: 1n, max: 2n })
          .setWitness({ value: 1n })
          .generate();
        proofs.push(sanctionsProof.data);
      } else {
        throw new Error("Cannot generate proof: sanctions flag present");
      }
    }

    // 3. Prove PEP status if required
    if (request.checkPEP) {
      if (!credential.attributes.pep) {
        claims.push("pep_clear");
        // Boolean proof: PEP == false (encoded as 0)
        const pepProof = await new ProofBuilder()
          .setStatement(StatementType.Range, { min: 0n, max: 1n })
          .setWitness({ value: 0n })
          .generate();
        proofs.push(pepProof.data);
      } else {
        throw new Error("Cannot generate proof: PEP flag present");
      }
    }

    // 4. Prove risk score is acceptable
    if (request.maxRiskScore !== undefined) {
      if (credential.attributes.riskScore <= request.maxRiskScore) {
        claims.push(`risk_score <= ${request.maxRiskScore}`);

        const riskProof = await new ProofBuilder()
          .setStatement(StatementType.Range, {
            min: 0n,
            max: BigInt(request.maxRiskScore + 1),
          })
          .setWitness({ value: BigInt(credential.attributes.riskScore) })
          .generate();
        proofs.push(riskProof.data);
      } else {
        throw new Error("Risk score exceeds maximum");
      }
    }

    // Combine proofs
    const combinedProof = this.combineProofs(proofs);

    // Create credential hash (for revocation checking)
    const credentialHash = createHash("sha256")
      .update(credential.userId)
      .update(credential.providerId)
      .digest("hex");

    return {
      proof: combinedProof,
      claims,
      credentialHash,
      expiresAt: credential.expiresAt,
    };
  }

  /**
   * Verify a KYC proof
   */
  async verifyKYCProof(
    proof: KYCProof,
    expectedClaims: string[]
  ): Promise<{ valid: boolean; matchedClaims: string[] }> {
    // Check expiry
    if (new Date() > proof.expiresAt) {
      return { valid: false, matchedClaims: [] };
    }

    // Check all expected claims are present
    const matchedClaims = expectedClaims.filter((c) =>
      proof.claims.includes(c)
    );
    if (matchedClaims.length !== expectedClaims.length) {
      return { valid: false, matchedClaims };
    }

    // Verify cryptographic proof
    // In production, this would verify each sub-proof
    const isValid = proof.proof.length > 0;

    return { valid: isValid, matchedClaims };
  }

  private combineProofs(proofs: Uint8Array[]): Uint8Array {
    const totalLength = proofs.reduce((sum, p) => sum + p.length + 4, 0);
    const combined = new Uint8Array(totalLength);
    let offset = 0;

    for (const proof of proofs) {
      const view = new DataView(combined.buffer, offset);
      view.setUint32(0, proof.length, true);
      combined.set(proof, offset + 4);
      offset += proof.length + 4;
    }

    return combined;
  }
}

// Usage Example: DeFi Platform KYC
async function defiKYCFlow() {
  const kycService = new ZKKYCService();

  // Register trusted KYC providers
  kycService.registerProvider("jumio", new Uint8Array(32));
  kycService.registerProvider("onfido", new Uint8Array(32));

  // User's KYC credential (from their KYC provider, stored locally)
  const userCredential: KYCCredential = {
    providerId: "jumio",
    providerSignature: new Uint8Array(64),
    userId: "hash_of_user_id",
    verificationLevel: "enhanced",
    verifiedAt: new Date("2024-01-15"),
    expiresAt: new Date("2025-01-15"),
    attributes: {
      identityVerified: true,
      addressVerified: true,
      documentVerified: true,
      sanctions: "clear",
      pep: false,
      riskScore: 15,
    },
  };

  // DeFi platform requests KYC proof
  const request: KYCProofRequest = {
    requiredLevel: "basic",
    checkSanctions: true,
    checkPEP: true,
    maxRiskScore: 50,
  };

  // User generates proof (locally, without sending personal data)
  const proof = await kycService.generateKYCProof(userCredential, request);
  console.log("KYC Proof generated with claims:", proof.claims);

  // Platform verifies proof (receives NO personal information)
  const verification = await kycService.verifyKYCProof(proof, [
    "kyc_level >= basic",
    "sanctions_clear",
    "pep_clear",
    "risk_score <= 50",
  ]);

  console.log("Verification result:", verification);
  // Platform learns: User passed KYC checks
  // Platform does NOT learn: Name, DOB, address, nationality, etc.
}
```

---

## Credential Verification

### Use Case: Prove Academic Degree Without Revealing Institution

```typescript
import { NexuszeroClient, ProofBuilder, StatementType } from "nexuszero-sdk";
import { MerkleTree } from "./utils/merkle";

interface AcademicCredential {
  holder: string; // Hashed holder ID
  institution: string; // Hashed institution ID
  degree: string; // e.g., "Bachelor of Science"
  field: string; // e.g., "Computer Science"
  graduationYear: number;
  gpa?: number; // Optional, 0-4 scale
  honors?: string; // Optional
  credentialHash: string; // Merkle root of full credential
  issuerSignature: Uint8Array;
}

interface CredentialClaim {
  type:
    | "degree_type"
    | "field"
    | "graduation_year"
    | "gpa_above"
    | "accredited";
  value: string | number;
}

class ZKCredentialVerifier {
  private client: NexuszeroClient;
  private accreditedInstitutions: Set<string>;

  constructor() {
    this.client = new NexuszeroClient();
    this.accreditedInstitutions = new Set();
  }

  /**
   * Add accredited institution hash to registry
   */
  addAccreditedInstitution(institutionHash: string): void {
    this.accreditedInstitutions.add(institutionHash);
  }

  /**
   * Prove specific claims about a credential
   */
  async proveCredentialClaims(
    credential: AcademicCredential,
    claims: CredentialClaim[]
  ): Promise<{ proof: Uint8Array; verifiedClaims: CredentialClaim[] }> {
    const verifiedClaims: CredentialClaim[] = [];
    const proofs: Uint8Array[] = [];

    for (const claim of claims) {
      switch (claim.type) {
        case "degree_type":
          if (
            credential.degree
              .toLowerCase()
              .includes(String(claim.value).toLowerCase())
          ) {
            verifiedClaims.push(claim);
            // Prove membership in degree type set
            const degreeProof = await this.proveMembership(
              credential.degree,
              this.getDegreeTypeSet(String(claim.value))
            );
            proofs.push(degreeProof);
          }
          break;

        case "field":
          if (credential.field === claim.value) {
            verifiedClaims.push(claim);
            // Prove field matches
            const fieldProof = await this.proveEquality(
              credential.field,
              String(claim.value)
            );
            proofs.push(fieldProof);
          }
          break;

        case "graduation_year":
          if (credential.graduationYear >= Number(claim.value)) {
            verifiedClaims.push(claim);
            const yearProof = await new ProofBuilder()
              .setStatement(StatementType.Range, {
                min: BigInt(claim.value),
                max: BigInt(new Date().getFullYear() + 1),
              })
              .setWitness({ value: BigInt(credential.graduationYear) })
              .generate();
            proofs.push(yearProof.data);
          }
          break;

        case "gpa_above":
          if (credential.gpa && credential.gpa >= Number(claim.value)) {
            verifiedClaims.push(claim);
            // Prove GPA >= threshold (scale 0-400 to avoid decimals)
            const gpaProof = await new ProofBuilder()
              .setStatement(StatementType.Range, {
                min: BigInt(Math.floor(Number(claim.value) * 100)),
                max: 401n,
              })
              .setWitness({ value: BigInt(Math.floor(credential.gpa * 100)) })
              .generate();
            proofs.push(gpaProof.data);
          }
          break;

        case "accredited":
          if (this.accreditedInstitutions.has(credential.institution)) {
            verifiedClaims.push(claim);
            // Prove institution is in accredited set
            const accreditedProof = await this.proveMembership(
              credential.institution,
              Array.from(this.accreditedInstitutions)
            );
            proofs.push(accreditedProof);
          }
          break;
      }
    }

    return {
      proof: this.combineProofs(proofs),
      verifiedClaims,
    };
  }

  private async proveMembership(
    element: string,
    set: string[]
  ): Promise<Uint8Array> {
    // Build Merkle tree of the set
    const tree = new MerkleTree(set.map((s) => Buffer.from(s)));
    const proof = tree.getProof(Buffer.from(element));
    return new Uint8Array(proof);
  }

  private async proveEquality(
    value: string,
    expected: string
  ): Promise<Uint8Array> {
    // Hash comparison proof
    const hash1 = this.hashString(value);
    const hash2 = this.hashString(expected);
    // In reality, this would be a proper equality proof
    return new Uint8Array([...hash1, ...hash2]);
  }

  private getDegreeTypeSet(type: string): string[] {
    const degreeSets: Record<string, string[]> = {
      bachelor: [
        "Bachelor of Science",
        "Bachelor of Arts",
        "Bachelor of Engineering",
      ],
      master: [
        "Master of Science",
        "Master of Arts",
        "Master of Business Administration",
      ],
      doctorate: [
        "Doctor of Philosophy",
        "Doctor of Science",
        "Doctor of Medicine",
      ],
    };
    return degreeSets[type.toLowerCase()] || [];
  }

  private hashString(s: string): Uint8Array {
    // Simple hash for example
    const encoder = new TextEncoder();
    return encoder.encode(s).slice(0, 32);
  }

  private combineProofs(proofs: Uint8Array[]): Uint8Array {
    return new Uint8Array(proofs.flatMap((p) => [...p]));
  }
}

// Usage: Job Application
async function jobApplicationFlow() {
  const verifier = new ZKCredentialVerifier();

  // Add accredited institutions
  verifier.addAccreditedInstitution("hash_of_mit");
  verifier.addAccreditedInstitution("hash_of_stanford");
  verifier.addAccreditedInstitution("hash_of_berkeley");

  // Applicant's credential (stored locally)
  const credential: AcademicCredential = {
    holder: "hash_of_applicant",
    institution: "hash_of_mit",
    degree: "Master of Science",
    field: "Computer Science",
    graduationYear: 2022,
    gpa: 3.8,
    honors: "Magna Cum Laude",
    credentialHash: "merkle_root",
    issuerSignature: new Uint8Array(64),
  };

  // Employer requires: Master's, CS field, from accredited school, GPA >= 3.5
  const requiredClaims: CredentialClaim[] = [
    { type: "degree_type", value: "master" },
    { type: "field", value: "Computer Science" },
    { type: "accredited", value: "true" },
    { type: "gpa_above", value: 3.5 },
  ];

  // Applicant generates proof
  const { proof, verifiedClaims } = await verifier.proveCredentialClaims(
    credential,
    requiredClaims
  );

  console.log("Verified claims:", verifiedClaims);
  // Employer learns: Applicant has Master's in CS from accredited school with GPA >= 3.5
  // Employer does NOT learn: Which school (MIT), exact GPA (3.8), graduation year, honors
}
```

---

## Identity Attestations

### Use Case: Government ID Verification

```typescript
interface GovernmentIDAttestation {
  type: "passport" | "drivers_license" | "national_id";
  issuer: string; // Country code
  issuedAt: Date;
  expiresAt: Date;
  attributes: {
    nationality: string;
    ageGroup: "minor" | "adult" | "senior";
    residency: string; // Country/region code
    biometricHash: string; // Hash of biometric template
  };
  attestationHash: string; // Merkle root
  issuerSignature: Uint8Array;
}

class ZKGovernmentIDService {
  private client: NexuszeroClient;

  constructor() {
    this.client = new NexuszeroClient();
  }

  /**
   * Prove nationality without revealing specific country
   */
  async proveNationalityInSet(
    attestation: GovernmentIDAttestation,
    allowedCountries: string[]
  ): Promise<{ proof: Uint8Array; claim: string }> {
    if (!allowedCountries.includes(attestation.attributes.nationality)) {
      throw new Error("Nationality not in allowed set");
    }

    // Build Merkle tree of allowed countries
    const countriesHash = this.hashSet(allowedCountries);

    // Prove nationality is in the set
    const proof = await this.proveMembershipInMerkleTree(
      attestation.attributes.nationality,
      allowedCountries
    );

    return {
      proof,
      claim: `nationality ∈ {${allowedCountries.length} countries}`,
    };
  }

  /**
   * Prove residency in specific region
   */
  async proveResidency(
    attestation: GovernmentIDAttestation,
    allowedRegions: string[]
  ): Promise<{ proof: Uint8Array; claim: string }> {
    if (!allowedRegions.includes(attestation.attributes.residency)) {
      throw new Error("Residency not in allowed regions");
    }

    const proof = await this.proveMembershipInMerkleTree(
      attestation.attributes.residency,
      allowedRegions
    );

    return {
      proof,
      claim: `residency ∈ {${allowedRegions.join(", ")}}`,
    };
  }

  /**
   * Prove ID is valid (not expired)
   */
  async proveIDValidity(
    attestation: GovernmentIDAttestation
  ): Promise<{ proof: Uint8Array; claim: string }> {
    const now = Date.now();
    const expiryTimestamp = attestation.expiresAt.getTime();

    if (now >= expiryTimestamp) {
      throw new Error("ID has expired");
    }

    // Prove: current_time < expiry_time
    const proof = await new ProofBuilder()
      .setStatement(StatementType.Range, {
        min: BigInt(now),
        max: BigInt(expiryTimestamp + 1),
      })
      .setWitness({ value: BigInt(expiryTimestamp) })
      .generate();

    return {
      proof: proof.data,
      claim: "id_valid_and_not_expired",
    };
  }

  /**
   * Prove biometric match without revealing biometric
   */
  async proveBiometricMatch(
    attestation: GovernmentIDAttestation,
    capturedBiometricHash: string
  ): Promise<{ proof: Uint8Array; match: boolean }> {
    const match =
      attestation.attributes.biometricHash === capturedBiometricHash;

    if (!match) {
      return { proof: new Uint8Array(0), match: false };
    }

    // Prove hash equality
    const proof = await this.proveHashEquality(
      attestation.attributes.biometricHash,
      capturedBiometricHash
    );

    return { proof, match: true };
  }

  private async proveMembershipInMerkleTree(
    element: string,
    set: string[]
  ): Promise<Uint8Array> {
    // Implementation details...
    return new Uint8Array(256);
  }

  private async proveHashEquality(
    hash1: string,
    hash2: string
  ): Promise<Uint8Array> {
    // Implementation details...
    return new Uint8Array(256);
  }

  private hashSet(set: string[]): string {
    // Implementation details...
    return "set_hash";
  }
}

// Usage: International Service Access
async function internationalAccess() {
  const idService = new ZKGovernmentIDService();

  // User's government ID attestation
  const userID: GovernmentIDAttestation = {
    type: "passport",
    issuer: "USA",
    issuedAt: new Date("2020-01-15"),
    expiresAt: new Date("2030-01-15"),
    attributes: {
      nationality: "USA",
      ageGroup: "adult",
      residency: "CA",
      biometricHash: "bio_hash_123",
    },
    attestationHash: "merkle_root",
    issuerSignature: new Uint8Array(64),
  };

  // Service requires: EU or US nationality, valid ID
  const euUsCountries = ["USA", "DEU", "FRA", "GBR", "ITA", "ESP", "NLD"];

  const nationalityProof = await idService.proveNationalityInSet(
    userID,
    euUsCountries
  );

  const validityProof = await idService.proveIDValidity(userID);

  console.log("Nationality proof:", nationalityProof.claim);
  console.log("Validity proof:", validityProof.claim);
  // Service learns: User has valid ID from EU or US country
  // Service does NOT learn: Specific country, name, passport number
}
```

---

## Selective Disclosure

### Use Case: Choose Which Attributes to Reveal

```typescript
interface IdentityBundle {
  attributes: Record<string, string | number | boolean>;
  commitments: Record<string, Uint8Array>;
  merkleRoot: string;
  signature: Uint8Array;
}

class SelectiveDisclosure {
  private client: NexuszeroClient;

  constructor() {
    this.client = new NexuszeroClient();
  }

  /**
   * Create an identity bundle with committed attributes
   */
  async createIdentityBundle(
    attributes: Record<string, string | number | boolean>
  ): Promise<IdentityBundle> {
    const commitments: Record<string, Uint8Array> = {};

    // Create commitment for each attribute
    for (const [key, value] of Object.entries(attributes)) {
      const valueBytes = this.encodeValue(value);
      const commitment = await this.client.createCommitment(
        this.bytesToBigInt(valueBytes)
      );
      commitments[key] = commitment.data;
    }

    // Create Merkle root of all commitments
    const merkleRoot = this.computeMerkleRoot(Object.values(commitments));

    return {
      attributes,
      commitments,
      merkleRoot,
      signature: new Uint8Array(64), // Would be signed by issuer
    };
  }

  /**
   * Generate proof for selected attributes only
   */
  async disclose(
    bundle: IdentityBundle,
    attributesToReveal: string[],
    attributesToProve: Array<{
      name: string;
      predicate: "equals" | "gt" | "lt" | "in_set";
      value: any;
    }>
  ): Promise<{
    revealed: Record<string, any>;
    proofs: Array<{ attribute: string; claim: string; proof: Uint8Array }>;
    merkleProofs: Record<string, Uint8Array>;
  }> {
    const revealed: Record<string, any> = {};
    const proofs: Array<{
      attribute: string;
      claim: string;
      proof: Uint8Array;
    }> = [];
    const merkleProofs: Record<string, Uint8Array> = {};

    // Reveal selected attributes
    for (const attr of attributesToReveal) {
      if (attr in bundle.attributes) {
        revealed[attr] = bundle.attributes[attr];
        merkleProofs[attr] = await this.getMerkleProof(bundle, attr);
      }
    }

    // Generate proofs for predicates
    for (const { name, predicate, value } of attributesToProve) {
      const actualValue = bundle.attributes[name];

      if (actualValue === undefined) continue;

      let proof: Uint8Array;
      let claim: string;

      switch (predicate) {
        case "equals":
          proof = await this.proveEquals(actualValue, value);
          claim = `${name} == [hidden]`;
          break;

        case "gt":
          proof = await this.proveGreaterThan(
            Number(actualValue),
            Number(value)
          );
          claim = `${name} > ${value}`;
          break;

        case "lt":
          proof = await this.proveLessThan(Number(actualValue), Number(value));
          claim = `${name} < ${value}`;
          break;

        case "in_set":
          proof = await this.proveInSet(String(actualValue), value as string[]);
          claim = `${name} ∈ {${(value as string[]).length} values}`;
          break;

        default:
          continue;
      }

      proofs.push({ attribute: name, claim, proof });
    }

    return { revealed, proofs, merkleProofs };
  }

  private encodeValue(value: string | number | boolean): Uint8Array {
    if (typeof value === "boolean") {
      return new Uint8Array([value ? 1 : 0]);
    }
    if (typeof value === "number") {
      const buf = new ArrayBuffer(8);
      new DataView(buf).setFloat64(0, value);
      return new Uint8Array(buf);
    }
    return new TextEncoder().encode(String(value));
  }

  private bytesToBigInt(bytes: Uint8Array): bigint {
    let result = 0n;
    for (const byte of bytes) {
      result = (result << 8n) | BigInt(byte);
    }
    return result;
  }

  private async getMerkleProof(
    bundle: IdentityBundle,
    attr: string
  ): Promise<Uint8Array> {
    // Generate Merkle proof for attribute
    return new Uint8Array(256);
  }

  private computeMerkleRoot(commitments: Uint8Array[]): string {
    // Compute Merkle root
    return "merkle_root";
  }

  private async proveEquals(actual: any, expected: any): Promise<Uint8Array> {
    return new Uint8Array(256);
  }

  private async proveGreaterThan(
    actual: number,
    threshold: number
  ): Promise<Uint8Array> {
    const proof = await new ProofBuilder()
      .setStatement(StatementType.Range, {
        min: BigInt(threshold + 1),
        max: BigInt(Number.MAX_SAFE_INTEGER),
      })
      .setWitness({ value: BigInt(actual) })
      .generate();
    return proof.data;
  }

  private async proveLessThan(
    actual: number,
    threshold: number
  ): Promise<Uint8Array> {
    const proof = await new ProofBuilder()
      .setStatement(StatementType.Range, {
        min: 0n,
        max: BigInt(threshold),
      })
      .setWitness({ value: BigInt(actual) })
      .generate();
    return proof.data;
  }

  private async proveInSet(actual: string, set: string[]): Promise<Uint8Array> {
    // Merkle membership proof
    return new Uint8Array(256);
  }
}

// Usage: Dating App Profile Verification
async function datingAppVerification() {
  const disclosure = new SelectiveDisclosure();

  // User creates identity bundle with all attributes
  const bundle = await disclosure.createIdentityBundle({
    name: "Alice Smith",
    age: 28,
    city: "San Francisco",
    occupation: "Software Engineer",
    income: 150000,
    education: "Masters",
    height: 165,
    verified: true,
  });

  // User chooses what to share/prove
  const result = await disclosure.disclose(
    bundle,
    // Reveal these attributes
    ["city", "occupation"],
    // Prove these predicates (without revealing values)
    [
      { name: "age", predicate: "gt", value: 21 },
      { name: "verified", predicate: "equals", value: true },
      { name: "income", predicate: "gt", value: 50000 },
    ]
  );

  console.log("Revealed:", result.revealed);
  // { city: 'San Francisco', occupation: 'Software Engineer' }

  console.log(
    "Proven claims:",
    result.proofs.map((p) => p.claim)
  );
  // ['age > 21', 'verified == [hidden]', 'income > 50000']

  // App learns: City, occupation, and that user is 21+, verified, earns > 50k
  // App does NOT learn: Name, exact age, exact income, education, height
}
```

---

## Summary

| Use Case             | Key Benefit                        | Typical Claims               |
| -------------------- | ---------------------------------- | ---------------------------- |
| Age Verification     | No birthday disclosure             | age ≥ threshold              |
| KYC Compliance       | Regulatory compliance without data | kyc_level, sanctions, pep    |
| Credentials          | Prove qualifications privately     | degree, field, accreditation |
| Government ID        | Cross-border verification          | nationality, validity        |
| Selective Disclosure | User-controlled privacy            | choose what to reveal        |

### Best Practices

1. **Minimize Claims**: Only prove what's necessary
2. **Use Time Bounds**: Proofs should expire
3. **Revocation Support**: Include credential hash for checking
4. **Audit Trail**: Log proof verification (not proof contents)
5. **Fallback Handling**: Have non-ZK alternatives for edge cases

---

_See also: [ZK_VOTING.md](./ZK_VOTING.md) | [ZK_PRIVATE_TRANSACTIONS.md](./ZK_PRIVATE_TRANSACTIONS.md)_
