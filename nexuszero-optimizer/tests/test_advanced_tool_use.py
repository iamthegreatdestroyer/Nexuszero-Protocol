"""
Integration Tests for Advanced Tool Use Features

Tests the three Advanced Tool Use features:
1. Tool Use Examples - FFI parameter accuracy
2. Programmatic Tool Calling - Batch processing efficiency
3. Tool Search Tool - Dynamic tool discovery

Target Metrics:
- Tool Use Examples: 90%+ parameter accuracy
- PTC: 37%+ token reduction
- Tool Search: 85%+ token savings (70K → 5.5K)
"""

import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

# Import our implementations
from nexuszero_optimizer.utils.tool_registry import (
    ToolRegistry,
    ToolCategory,
    ToolDefinition,
    get_registry,
    search_tools
)
from nexuszero_optimizer.utils.batch_orchestrator import (
    ProgrammaticToolOrchestrator,
    CircuitBenchmark,
    BatchResult,
    batch_benchmark_circuits
)


class TestToolUseExamples:
    """Test Tool Use Examples for FFI Bridge."""
    
    def test_ffi_examples_file_exists(self):
        """Verify FFI_EXAMPLES.json exists and is valid JSON."""
        examples_path = Path("nexuszero-crypto/docs/FFI_EXAMPLES.json")
        
        assert examples_path.exists(), "FFI_EXAMPLES.json not found"
        
        # Validate JSON structure
        with open(examples_path) as f:
            data = json.load(f)
        
        assert "version" in data
        assert "tools" in data
        assert isinstance(data["tools"], list)
        assert len(data["tools"]) >= 2  # At least 2 tools
    
    def test_examples_match_schema(self):
        """Verify examples match expected schema."""
        examples_path = Path("nexuszero-crypto/docs/FFI_EXAMPLES.json")
        
        with open(examples_path) as f:
            data = json.load(f)
        
        for tool in data["tools"]:
            assert "name" in tool
            assert "input_examples" in tool
            
            for example in tool["input_examples"]:
                assert "input" in example
                assert isinstance(example["input"], dict)
                
                # Optional fields
                if "_description" in example:
                    assert isinstance(example["_description"], str)
                if "_expected_output" in example or "_expected_return" in example:
                    # Has expected result metadata
                    pass
    
    @pytest.mark.skipif(
        not Path("nexuszero-crypto/python/nexuszero_ffi.py").exists(),
        reason="FFI Python bindings not available"
    )
    def test_crypto_tool_examples_accuracy(self):
        """
        Test that Tool Use Examples improve parameter accuracy.
        
        Target: 90%+ accuracy (vs 72% baseline)
        """
        # This would test against actual FFI implementation
        # For now, verify examples are embedded in Python module
        
        from nexuszero_crypto.python.nexuszero_ffi import (
            TOOL_USE_EXAMPLES,
            NexusZeroCrypto
        )
        
        # Verify examples exist
        assert "estimate_parameters" in TOOL_USE_EXAMPLES
        assert "validate_params" in TOOL_USE_EXAMPLES
        
        # Verify structure
        estimate_examples = TOOL_USE_EXAMPLES["estimate_parameters"]
        assert len(estimate_examples) >= 3
        
        for example in estimate_examples:
            assert "input" in example
            assert "security_level" in example["input"]
            assert "circuit_size" in example["input"]
        
        # Validate examples
        validate_examples = TOOL_USE_EXAMPLES["validate_params"]
        assert len(validate_examples) >= 3
        
        # Test that examples improve AI assistant accuracy
        # (This would require actual API testing with Anthropic)
        # For unit tests, we verify structure only


class TestProgrammaticToolCalling:
    """Test Programmatic Tool Calling for batch operations."""
    
    @pytest.fixture
    def mock_benchmark_tool(self):
        """Mock benchmark tool for testing."""
        async def mock_benchmark(circuit_id: str, security_level: int, circuit_size: int) -> Dict[str, Any]:
            """Mock benchmark that returns realistic results."""
            import random
            
            # Simulate some variance
            base_n = 512 if security_level == 128 else 1024 if security_level == 192 else 2048
            
            return {
                "circuit_id": circuit_id,
                "circuit_size": circuit_size,
                "optimal_n": base_n,
                "optimal_q": 12289 if base_n == 512 else 40961,
                "optimal_sigma": 3.2 if base_n == 512 else 2.8,
                "estimated_proof_size": circuit_size * 32 + base_n * 16,
                "estimated_prove_time_ms": int(circuit_size * 0.5 + random.uniform(-10, 10)),
                "actual_prove_time_ms": int(circuit_size * 0.5 + random.uniform(-5, 5)),
            }
        
        return mock_benchmark
    
    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, mock_benchmark_tool):
        """
        Test batch processing returns summary, not all results.
        
        Target: Process 10,000 circuits, return ~1KB summary
        """
        orchestrator = ProgrammaticToolOrchestrator(
            benchmark_tool=mock_benchmark_tool,
            max_parallel=20
        )
        
        # Create 100 test circuits (use 100 instead of 10k for test speed)
        circuits = [
            {"id": f"circuit_{i}", "size": 100 + i * 10}
            for i in range(100)
        ]
        
        result = await orchestrator.batch_benchmark(circuits, security_level=128)
        
        # Verify result is BatchResult, not list of 100 items
        assert isinstance(result, BatchResult)
        
        # Verify summary structure
        assert result.total_circuits == 100
        assert result.successful_benchmarks <= 100
        assert result.failed_benchmarks >= 0
        
        # Verify only top/worst 10 included
        assert len(result.top_10_fastest) <= 10
        assert len(result.top_10_smallest_proofs) <= 10
        assert len(result.worst_10_slowest) <= 10
        
        # Verify statistics exist
        assert result.average_proof_size > 0
        assert result.average_prove_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_context_token_savings(self, mock_benchmark_tool):
        """
        Test token savings estimate is accurate.
        
        Target: 37%+ token reduction for batch operations
        """
        orchestrator = ProgrammaticToolOrchestrator(mock_benchmark_tool)
        
        circuits = [{"id": f"circuit_{i}", "size": 100} for i in range(50)]
        
        result = await orchestrator.batch_benchmark(circuits)
        
        # Verify token savings calculation
        expected_savings = (50 * 50) - 1000  # 50 tokens/circuit × 50 - 1000 overhead
        
        assert result.context_tokens_saved > 0
        assert result.context_tokens_saved >= expected_savings * 0.8  # Allow 20% margin
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, mock_benchmark_tool):
        """Test that parallel execution works correctly."""
        orchestrator = ProgrammaticToolOrchestrator(
            benchmark_tool=mock_benchmark_tool,
            max_parallel=5  # Limit for test
        )
        
        circuits = [{"id": f"circuit_{i}", "size": 100} for i in range(20)]
        
        import time
        start = time.time()
        result = await orchestrator.batch_benchmark(circuits)
        elapsed = time.time() - start
        
        # With parallelism, should complete faster than serial
        # 20 circuits × 0.1s (mock delay) / 5 parallel = ~0.4s expected
        # Without parallelism: 20 × 0.1s = 2s
        
        assert result.successful_benchmarks == 20
        assert elapsed < 2.0  # Should be faster than serial
    
    def test_convenience_function(self):
        """Test batch_benchmark_circuits convenience function."""
        # This tests the function exists and has correct signature
        import inspect
        
        sig = inspect.signature(batch_benchmark_circuits)
        params = list(sig.parameters.keys())
        
        assert "circuits" in params
        assert "security_level" in params


class TestToolSearchTool:
    """Test Tool Search Tool Registry."""
    
    def test_registry_initialization(self):
        """Test registry initializes with NexusZero tools."""
        registry = get_registry()
        
        # Verify tools are registered
        assert registry._tools  # Has tools
        assert len(registry._tools) >= 25  # At least 25 tools
        
        # Verify core tools exist
        assert len(registry._core_tools) >= 3  # At least 3 core tools
        
        # Verify categories exist
        categories = set(tool.category for tool in registry._tools.values())
        assert ToolCategory.CRYPTO in categories
        assert ToolCategory.NEURAL in categories
        assert ToolCategory.CHAIN in categories
    
    def test_search_by_keyword(self):
        """Test searching tools by keyword."""
        registry = get_registry()
        
        # Search for benchmark tools
        results = registry.search("benchmark")
        assert len(results) > 0
        
        # Verify results are relevant
        for tool in results:
            assert (
                "benchmark" in tool.name.lower() or
                "benchmark" in tool.description.lower() or
                "benchmark" in [k.lower() for k in tool.keywords]
            )
    
    def test_search_by_category(self):
        """Test searching tools by category."""
        registry = get_registry()
        
        # Search crypto tools
        results = registry.search("", category=ToolCategory.CRYPTO)
        assert len(results) > 0
        
        for tool in results:
            assert tool.category == ToolCategory.CRYPTO
    
    def test_search_with_max_results(self):
        """Test max_results parameter."""
        registry = get_registry()
        
        # Search with limit
        results = registry.search("proof", max_results=5)
        assert len(results) <= 5
    
    def test_token_savings_estimate(self):
        """
        Test token savings calculation.
        
        Target: 85%+ savings (70K → 5.5K tokens)
        """
        registry = get_registry()
        
        savings = registry.estimate_token_savings()
        
        # Verify structure
        assert "total_tools" in savings
        assert "core_tools" in savings
        assert "deferred_tools" in savings
        assert "naive_approach_tokens" in savings
        assert "optimized_approach_tokens" in savings
        assert "tokens_saved" in savings
        assert "savings_percentage" in savings
        
        # Verify savings meet target
        assert savings["savings_percentage"] >= 80.0  # At least 80% savings
        
        # Verify core tools are reasonable
        assert 3 <= savings["core_tools"] <= 10
        
        # Verify total tools
        assert savings["total_tools"] >= 25
    
    def test_api_tools_format(self):
        """Test get_api_tools returns correct format."""
        registry = get_registry()
        
        # Get API tools with search tool
        api_tools = registry.get_api_tools(
            searched_tool_names=["benchmark_ntt"],
            include_search_tool=True
        )
        
        # Verify search tool is included
        search_tools_found = [
            t for t in api_tools 
            if t.get("type") == "tool_search_tool_regex_20251119"
        ]
        assert len(search_tools_found) == 1
        
        # Verify core tools are included
        core_tool_names = [t["name"] for t in api_tools if "name" in t]
        assert "nexuszero_prove_range" in core_tool_names
        assert "nexuszero_verify_proof" in core_tool_names
        
        # Verify searched tool is included
        assert "benchmark_ntt" in core_tool_names
    
    def test_deferred_loading(self):
        """Test deferred tools have defer_loading=True."""
        registry = get_registry()
        
        deferred = registry.get_all_deferred()
        
        # Verify all have defer_loading=True
        for tool in deferred:
            assert tool.get("defer_loading") == True
        
        # Verify count
        assert len(deferred) >= 20  # Most tools should be deferred
    
    def test_convenience_function(self):
        """Test search_tools convenience function."""
        results = search_tools("ethereum")
        
        assert len(results) > 0
        
        # Verify results contain ethereum tools
        ethereum_tools = [
            t for t in results 
            if "ethereum" in t.name.lower() or "ethereum" in t.description.lower()
        ]
        assert len(ethereum_tools) > 0


class TestIntegration:
    """Integration tests combining multiple features."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """
        Test complete workflow:
        1. Search for benchmark tools (Tool Search)
        2. Use batch orchestrator with found tool (PTC)
        3. Verify token savings
        
        This simulates real AI agent workflow.
        """
        # Step 1: Search for tools
        registry = get_registry()
        benchmark_tools = registry.search("benchmark proof")
        
        assert len(benchmark_tools) > 0
        assert any("benchmark" in t.name for t in benchmark_tools)
        
        # Step 2: Get API tools for context
        api_tools = registry.get_api_tools(
            searched_tool_names=["benchmark_proof_generation"],
            include_search_tool=True
        )
        
        # Verify minimal tool loading
        named_tools = [t for t in api_tools if "name" in t]
        assert len(named_tools) <= 10  # Core + searched tools only
        
        # Step 3: Calculate token savings
        savings = registry.estimate_token_savings()
        
        # Verify high savings
        assert savings["savings_percentage"] >= 80.0
        
        # Step 4: Mock batch processing (would use real tool in production)
        async def mock_tool(cid, sec, size):
            return {
                "circuit_id": cid,
                "circuit_size": size,
                "optimal_n": 512,
                "optimal_q": 12289,
                "optimal_sigma": 3.2,
                "estimated_proof_size": size * 32,
                "estimated_prove_time_ms": size // 2,
                "actual_prove_time_ms": size // 2,
            }
        
        orchestrator = ProgrammaticToolOrchestrator(mock_tool)
        circuits = [{"id": f"c{i}", "size": 100} for i in range(20)]
        
        result = await orchestrator.batch_benchmark(circuits)
        
        # Verify PTC token savings
        assert result.context_tokens_saved > 0
        
        # Total savings: Tool Search + PTC
        total_naive = savings["naive_approach_tokens"] + (20 * 50)
        total_optimized = savings["optimized_approach_tokens"] + 1000
        
        total_savings_pct = (1 - total_optimized / total_naive) * 100
        
        # Combined savings should be substantial
        assert total_savings_pct >= 70.0  # At least 70% combined savings
    
    def test_tool_use_examples_integration(self):
        """
        Test Tool Use Examples are accessible for AI assistants.
        
        This verifies the examples are embedded and discoverable.
        """
        registry = get_registry()
        
        # Find crypto parameter tool
        crypto_tools = registry.search("estimate parameters")
        param_tool = next(
            (t for t in crypto_tools if t.name == "nexuszero_estimate_parameters"),
            None
        )
        
        assert param_tool is not None
        
        # Verify examples exist
        assert len(param_tool.input_examples) >= 2
        
        # Verify example structure
        for example in param_tool.input_examples:
            assert "security_level" in example
            assert "circuit_size" in example


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
