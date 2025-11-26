"""
Tests for NexusZero Python SDK.
"""

import pytest

from nexuszero import (
    NexusZeroClient,
    PrivacyEngine,
    PrivacyLevel,
    ProofGenerator,
    ProofVerifier,
    CrossChainBridge,
    ComplianceProver,
    ComplianceProofType,
)
from nexuszero.privacy import TransactionContext


class TestPrivacyLevel:
    """Tests for PrivacyLevel enum."""

    def test_privacy_levels_exist(self):
        """Test all 6 privacy levels are defined."""
        assert PrivacyLevel.TRANSPARENT == 0
        assert PrivacyLevel.PSEUDONYMOUS == 1
        assert PrivacyLevel.CONFIDENTIAL == 2
        assert PrivacyLevel.PRIVATE == 3
        assert PrivacyLevel.ANONYMOUS == 4
        assert PrivacyLevel.SOVEREIGN == 5

    def test_privacy_level_descriptions(self):
        """Test privacy levels have descriptions."""
        for level in PrivacyLevel:
            assert len(level.description) > 0

    def test_privacy_level_security_bits(self):
        """Test security bits increase with level."""
        for i, level in enumerate(PrivacyLevel):
            if i > 0:
                assert level.security_bits >= PrivacyLevel(i - 1).security_bits


class TestPrivacyEngine:
    """Tests for PrivacyEngine."""

    @pytest.fixture
    def engine(self):
        return PrivacyEngine()

    def test_get_parameters(self, engine):
        """Test getting parameters for each level."""
        for level in range(6):
            params = engine.get_parameters(level)
            assert params is not None
            assert params.security_bits >= 0

    def test_get_parameters_invalid_level(self, engine):
        """Test invalid level raises error."""
        with pytest.raises(ValueError):
            engine.get_parameters(6)

    def test_recommend_default(self, engine):
        """Test default recommendation is PRIVATE."""
        context = TransactionContext(value_usd=100.0)
        rec = engine.recommend(context)
        assert rec.level == PrivacyLevel.PRIVATE

    def test_recommend_high_value(self, engine):
        """Test high value transactions get higher privacy."""
        context = TransactionContext(value_usd=50000.0)
        rec = engine.recommend(context)
        assert rec.level >= PrivacyLevel.ANONYMOUS

    def test_recommend_compliance(self, engine):
        """Test compliance limits max privacy."""
        context = TransactionContext(
            value_usd=100000.0,
            requires_compliance=True,
        )
        rec = engine.recommend(context)
        assert rec.level <= PrivacyLevel.PRIVATE

    def test_recommend_high_risk(self, engine):
        """Test high risk score limits privacy."""
        context = TransactionContext(
            value_usd=1000.0,
            risk_score=0.8,
        )
        rec = engine.recommend(context)
        assert rec.level <= PrivacyLevel.CONFIDENTIAL

    def test_morph_path_increase(self, engine):
        """Test morphing to higher privacy is one step."""
        path = engine.get_morph_path(2, 5)
        assert path == [5]

    def test_morph_path_decrease(self, engine):
        """Test morphing to lower privacy is incremental."""
        path = engine.get_morph_path(5, 2)
        assert len(path) == 3  # 4, 3, 2


class TestProofGenerator:
    """Tests for ProofGenerator."""

    @pytest.fixture
    def generator(self):
        return ProofGenerator()

    def test_generate_proof(self, generator):
        """Test basic proof generation."""
        data = b"test circuit data"
        result = generator.generate(data, PrivacyLevel.PRIVATE)

        assert result.proof_id is not None
        assert len(result.proof_data) > 0
        assert result.privacy_level == 3
        assert result.quality_score > 0

    def test_generate_proof_levels(self, generator):
        """Test proof size varies with privacy level."""
        data = b"test data"

        sizes = []
        for level in PrivacyLevel:
            result = generator.generate(data, level)
            sizes.append(len(result.proof_data))

        # Higher levels should produce larger proofs
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1]


class TestProofVerifier:
    """Tests for ProofVerifier."""

    @pytest.fixture
    def verifier(self):
        return ProofVerifier()

    def test_verify_valid_proof(self, verifier):
        """Test verifying a valid proof."""
        proof = b"\x00" * 64  # 64 bytes
        result = verifier.verify(proof, privacy_level=PrivacyLevel.PSEUDONYMOUS)
        assert result is True

    def test_verify_too_short(self, verifier):
        """Test proof too short fails."""
        proof = b"\x00" * 16
        result = verifier.verify(proof)
        assert result is False

    def test_verify_batch(self, verifier):
        """Test batch verification."""
        proofs = [b"\x00" * 64 for _ in range(5)]
        results = verifier.verify_batch(proofs)
        assert len(results) == 5
        assert all(results)


class TestCrossChainBridge:
    """Tests for CrossChainBridge."""

    @pytest.fixture
    def bridge(self):
        return CrossChainBridge()

    def test_supported_routes(self, bridge):
        """Test checking supported routes."""
        assert bridge.is_route_supported("ethereum", "polygon")
        assert bridge.is_route_supported("polygon", "ethereum")
        assert not bridge.is_route_supported("bitcoin", "solana")

    def test_get_quote(self, bridge):
        """Test getting a bridge quote."""
        quote = bridge.get_quote("ethereum", "polygon", 1000000)

        assert quote.source_chain == "ethereum"
        assert quote.target_chain == "polygon"
        assert quote.amount == 1000000
        assert quote.total_fee > 0
        assert quote.estimated_time_seconds > 0

    def test_quote_privacy_affects_fees(self, bridge):
        """Test higher privacy levels cost more."""
        quote_low = bridge.get_quote("ethereum", "polygon", 1000000, privacy_level=1)
        quote_high = bridge.get_quote("ethereum", "polygon", 1000000, privacy_level=5)

        assert quote_high.fee_source_chain > quote_low.fee_source_chain


class TestComplianceProver:
    """Tests for ComplianceProver."""

    @pytest.fixture
    def prover(self):
        return ComplianceProver()

    def test_prove_age(self, prover):
        """Test age verification proof."""
        encrypted_birthdate = b"\x00" * 32
        proof = prover.prove_age(encrypted_birthdate, minimum_age=18)

        assert proof.proof_type == ComplianceProofType.AGE_VERIFICATION
        assert proof.verified
        assert proof.metadata["minimum_age"] == 18

    def test_prove_accredited_investor(self, prover):
        """Test accredited investor proof."""
        proof = prover.prove_accredited_investor(
            encrypted_net_worth=b"\x00" * 32,
            encrypted_income=b"\x00" * 32,
            jurisdiction="US",
        )

        assert proof.proof_type == ComplianceProofType.ACCREDITED_INVESTOR
        assert proof.verified
        assert proof.metadata["jurisdiction"] == "US"

    def test_prove_not_sanctioned(self, prover):
        """Test sanctions compliance proof."""
        proof = prover.prove_not_sanctioned(
            encrypted_identity_hash=b"\x00" * 32,
            sanctions_list_hash=b"\x01" * 32,
        )

        assert proof.proof_type == ComplianceProofType.SANCTIONS_COMPLIANCE
        assert proof.verified

    def test_prove_transaction_limit(self, prover):
        """Test transaction limit proof."""
        proof = prover.prove_transaction_under_limit(
            encrypted_amount=b"\x00" * 32,
            limit_usd=10000.0,
        )

        assert proof.proof_type == ComplianceProofType.TRANSACTION_LIMIT
        assert proof.metadata["limit_usd"] == 10000.0

    def test_jurisdiction_profiles(self, prover):
        """Test jurisdiction profiles exist."""
        us_profile = prover.get_jurisdiction_profile("US")
        assert us_profile["requires_kyc"] is True
        assert us_profile["max_privacy_level"] == 4

        eu_profile = prover.get_jurisdiction_profile("EU")
        assert eu_profile["max_privacy_level"] == 5


class TestNexusZeroClient:
    """Tests for NexusZeroClient initialization."""

    def test_client_init(self):
        """Test client initialization."""
        client = NexusZeroClient(
            api_url="https://test.api.nexuszero.io",
            api_key="test-key",
        )
        assert client.api_url == "https://test.api.nexuszero.io"
        assert client.api_key == "test-key"

    def test_client_default_url(self):
        """Test client uses default URL."""
        client = NexusZeroClient()
        assert client.api_url == NexusZeroClient.DEFAULT_API_URL

    def test_recommend_privacy_local(self):
        """Test local privacy recommendation."""
        client = NexusZeroClient()
        rec = client.recommend_privacy(
            value_usd=5000.0,
            requires_compliance=False,
        )
        assert rec.level == PrivacyLevel.PRIVATE
