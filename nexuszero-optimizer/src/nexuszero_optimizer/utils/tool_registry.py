"""
Tool Search Tool Registry for NexusZero Protocol

Implements Anthropic's Tool Search Tool pattern:
- Register tools with defer_loading=True for on-demand discovery
- Search tools by name, description, or capability
- Load only relevant tools into context (85% token savings)

Target: 65+ tools across 7 categories with 85%+ token reduction
"""

import json
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories for tool organization."""
    CRYPTO = "cryptography"
    NEURAL = "neural_optimizer"
    COMPRESSION = "holographic_compression"
    BENCHMARK = "benchmarking"
    SECURITY = "security"
    CHAIN = "blockchain_connector"
    MONITORING = "monitoring"
    ADMIN = "administration"


@dataclass
class ToolDefinition:
    """
    Tool definition with defer_loading support.
    
    Matches Anthropic's Tool Use API structure.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    category: ToolCategory
    
    # Tool Search Tool configuration
    defer_loading: bool = True  # True = discovered on-demand
    
    # Search metadata
    keywords: List[str] = field(default_factory=list)
    agent_owner: Optional[str] = None  # Which agent owns this tool
    
    # Tool Use Examples (optional)
    input_examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Anthropic API format."""
        result = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "defer_loading": self.defer_loading
        }
        
        if self.input_examples:
            result["input_examples"] = self.input_examples
        
        return result


class ToolRegistry:
    """
    Central registry for all NexusZero tools.
    
    Implements Tool Search Tool pattern for efficient tool discovery.
    
    Usage:
    ------
    registry = ToolRegistry()
    
    # Register tools from all agents
    registry.register_crypto_tools()
    registry.register_neural_tools()
    registry.register_compression_tools()
    
    # Search for relevant tools (returns only matches)
    tools = registry.search("benchmark proof generation")
    
    # Get tools for API call (core + searched tools)
    api_tools = registry.get_api_tools(searched_tool_names=["benchmark_ntt"])
    
    Token Savings Example:
    ---------------------
    Without Tool Search:
    - All 65 tools loaded: ~70,000 tokens
    
    With Tool Search:
    - Core tools loaded: ~3,000 tokens
    - Search tool: ~500 tokens
    - On-demand tools: ~2,000 tokens (only what's needed)
    - Total: ~5,500 tokens (92% savings)
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._core_tools: Set[str] = set()  # Always loaded (defer_loading=False)
        self._search_index: Dict[str, Set[str]] = {}  # keyword -> tool names
    
    def register(self, tool: ToolDefinition) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
        
        if not tool.defer_loading:
            self._core_tools.add(tool.name)
        
        # Build search index
        self._index_tool(tool)
        
        logger.debug(f"Registered tool: {tool.name} (defer={tool.defer_loading})")
    
    def _index_tool(self, tool: ToolDefinition) -> None:
        """Index a tool for search."""
        # Index by name parts
        for word in tool.name.lower().split("_"):
            self._add_to_index(word, tool.name)
        
        # Index by description words
        for word in tool.description.lower().split():
            if len(word) > 3:  # Skip short words
                self._add_to_index(word, tool.name)
        
        # Index by keywords
        for keyword in tool.keywords:
            self._add_to_index(keyword.lower(), tool.name)
        
        # Index by category
        self._add_to_index(tool.category.value, tool.name)
    
    def _add_to_index(self, keyword: str, tool_name: str) -> None:
        """Add a keyword to the search index."""
        if keyword not in self._search_index:
            self._search_index[keyword] = set()
        self._search_index[keyword].add(tool_name)
    
    def search(
        self, 
        query: str,
        category: Optional[ToolCategory] = None,
        max_results: int = 10
    ) -> List[ToolDefinition]:
        """
        Search for tools matching a query.
        
        Implements regex-based search similar to Anthropic's
        tool_search_tool_regex_20251119.
        
        Args:
            query: Search query (keywords or regex pattern)
            category: Optional category filter
            max_results: Maximum number of results
        
        Returns:
            List of matching ToolDefinitions (not loaded yet)
        
        Example (for AI assistants):
        ---------------------------
        # Search for benchmark tools:
        tools = registry.search("benchmark proof")
        # Returns: [benchmark_ntt, benchmark_proof_gen, ...]
        
        # Search with category filter:
        tools = registry.search("optimize", category=ToolCategory.NEURAL)
        # Returns: [optimize_circuit, optimize_params, ...]
        """
        matches: Dict[str, int] = {}  # tool_name -> score
        
        # Tokenize query
        query_words = query.lower().split()
        
        for word in query_words:
            # Exact match in index
            if word in self._search_index:
                for tool_name in self._search_index[word]:
                    matches[tool_name] = matches.get(tool_name, 0) + 2
            
            # Partial match (prefix)
            for indexed_word, tool_names in self._search_index.items():
                if indexed_word.startswith(word) or word.startswith(indexed_word):
                    for tool_name in tool_names:
                        matches[tool_name] = matches.get(tool_name, 0) + 1
        
        # Apply category filter
        if category:
            matches = {
                name: score 
                for name, score in matches.items()
                if self._tools[name].category == category
            }
        
        # Sort by score and return top results
        sorted_matches = sorted(
            matches.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_results]
        
        return [self._tools[name] for name, _ in sorted_matches]
    
    def get_api_tools(
        self,
        searched_tool_names: Optional[List[str]] = None,
        include_search_tool: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get tools formatted for Anthropic API.
        
        Returns:
        - Core tools (always loaded)
        - Tool Search Tool (if enabled)
        - Searched tools (if provided)
        
        This is the final output sent to the API.
        """
        result = []
        
        # Add search tool if enabled
        if include_search_tool:
            result.append({
                "type": "tool_search_tool_regex_20251119",
                "name": "tool_search"
            })
        
        # Add core tools (defer_loading=False)
        for name in self._core_tools:
            result.append(self._tools[name].to_api_format())
        
        # Add searched tools
        if searched_tool_names:
            for name in searched_tool_names:
                if name in self._tools and name not in self._core_tools:
                    result.append(self._tools[name].to_api_format())
        
        return result
    
    def get_all_deferred(self) -> List[Dict[str, Any]]:
        """
        Get all deferred tools for initial API setup.
        
        These are NOT loaded into context - they're just registered
        for on-demand discovery.
        """
        return [
            tool.to_api_format()
            for tool in self._tools.values()
            if tool.defer_loading
        ]
    
    def estimate_token_savings(self) -> Dict[str, int]:
        """
        Estimate token savings from using Tool Search Tool.
        
        Returns:
            Dictionary with token estimates
        """
        # Estimate ~1000 tokens per tool definition
        tokens_per_tool = 1000
        
        total_tools = len(self._tools)
        core_tools = len(self._core_tools)
        deferred_tools = total_tools - core_tools
        
        naive_tokens = total_tools * tokens_per_tool
        optimized_tokens = core_tools * tokens_per_tool + 500  # +500 for search tool
        
        return {
            "total_tools": total_tools,
            "core_tools": core_tools,
            "deferred_tools": deferred_tools,
            "naive_approach_tokens": naive_tokens,
            "optimized_approach_tokens": optimized_tokens,
            "tokens_saved": naive_tokens - optimized_tokens,
            "savings_percentage": round(
                (1 - optimized_tokens / naive_tokens) * 100, 1
            ) if naive_tokens > 0 else 0
        }
    
    # =================================================================
    # PRE-REGISTERED TOOLS FOR NEXUSZERO PROTOCOL
    # =================================================================
    
    def register_all_nexuszero_tools(self) -> None:
        """Register all NexusZero Protocol tools."""
        self.register_crypto_tools()
        self.register_neural_tools()
        self.register_compression_tools()
        self.register_benchmark_tools()
        self.register_security_tools()
        self.register_chain_tools()
        self.register_monitoring_tools()
    
    def register_crypto_tools(self) -> None:
        """Register cryptography tools (nexuszero-crypto)."""
        
        # CORE TOOLS - Always loaded
        self.register(ToolDefinition(
            name="nexuszero_prove_range",
            description="Generate a zero-knowledge range proof for a value",
            input_schema={
                "type": "object",
                "properties": {
                    "value": {"type": "integer", "description": "Value to prove is in range"},
                    "min_value": {"type": "integer", "description": "Minimum allowed value"},
                    "max_value": {"type": "integer", "description": "Maximum allowed value"},
                    "blinding": {"type": "string", "description": "32-byte hex blinding factor"}
                },
                "required": ["value", "min_value", "max_value"]
            },
            category=ToolCategory.CRYPTO,
            defer_loading=False,  # CORE - always loaded
            keywords=["proof", "range", "zkp", "zero-knowledge", "bulletproof"],
            agent_owner="dr_alex_cipher",
            input_examples=[
                {"value": 50, "min_value": 0, "max_value": 100, "_description": "Prove 50 is in [0,100]"},
                {"value": 1000000, "min_value": 0, "max_value": 2000000, "_description": "Large range proof"}
            ]
        ))
        
        self.register(ToolDefinition(
            name="nexuszero_verify_proof",
            description="Verify a zero-knowledge proof",
            input_schema={
                "type": "object",
                "properties": {
                    "proof": {"type": "string", "description": "Hex-encoded proof"},
                    "statement": {"type": "object", "description": "Statement being proved"}
                },
                "required": ["proof", "statement"]
            },
            category=ToolCategory.CRYPTO,
            defer_loading=False,  # CORE - always loaded
            keywords=["verify", "proof", "zkp", "validation"],
            agent_owner="dr_alex_cipher"
        ))
        
        self.register(ToolDefinition(
            name="nexuszero_estimate_parameters",
            description="Estimate optimal lattice parameters for security level and circuit size",
            input_schema={
                "type": "object",
                "properties": {
                    "security_level": {"type": "integer", "enum": [128, 192, 256]},
                    "circuit_size": {"type": "integer", "minimum": 1, "maximum": 1000000}
                },
                "required": ["security_level", "circuit_size"]
            },
            category=ToolCategory.CRYPTO,
            defer_loading=False,  # CORE - always loaded
            keywords=["parameters", "lattice", "lwe", "security"],
            agent_owner="morgan_rustico",
            input_examples=[
                {"security_level": 128, "circuit_size": 1000, "_expected_n": 512},
                {"security_level": 256, "circuit_size": 50000, "_expected_n": 2048}
            ]
        ))
        
        # DEFERRED TOOLS - Discovered on demand
        self.register(ToolDefinition(
            name="generate_keypair",
            description="Generate a new lattice-based keypair",
            input_schema={
                "type": "object",
                "properties": {
                    "security_level": {"type": "integer", "enum": [128, 192, 256]}
                },
                "required": ["security_level"]
            },
            category=ToolCategory.CRYPTO,
            defer_loading=True,
            keywords=["key", "keypair", "generate", "lattice"],
            agent_owner="dr_alex_cipher"
        ))
        
        self.register(ToolDefinition(
            name="sign_message",
            description="Sign a message using lattice-based signature scheme",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "private_key": {"type": "string"}
                },
                "required": ["message", "private_key"]
            },
            category=ToolCategory.CRYPTO,
            defer_loading=True,
            keywords=["sign", "signature", "message", "authenticate"],
            agent_owner="dr_alex_cipher"
        ))
        
        self.register(ToolDefinition(
            name="verify_signature",
            description="Verify a lattice-based signature",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "signature": {"type": "string"},
                    "public_key": {"type": "string"}
                },
                "required": ["message", "signature", "public_key"]
            },
            category=ToolCategory.CRYPTO,
            defer_loading=True,
            keywords=["verify", "signature", "authenticate"],
            agent_owner="dr_alex_cipher"
        ))
    
    def register_neural_tools(self) -> None:
        """Register neural optimizer tools (nexuszero-optimizer)."""
        
        self.register(ToolDefinition(
            name="optimize_circuit",
            description="Optimize a proof circuit using neural network",
            input_schema={
                "type": "object",
                "properties": {
                    "circuit": {"type": "object", "description": "Circuit definition"},
                    "optimization_level": {"type": "string", "enum": ["fast", "balanced", "thorough"]}
                },
                "required": ["circuit"]
            },
            category=ToolCategory.NEURAL,
            defer_loading=True,
            keywords=["optimize", "circuit", "neural", "gnn", "ml"],
            agent_owner="dr_asha_neural"
        ))
        
        self.register(ToolDefinition(
            name="predict_optimal_params",
            description="Use trained model to predict optimal parameters",
            input_schema={
                "type": "object",
                "properties": {
                    "circuit_features": {"type": "object"},
                    "model_checkpoint": {"type": "string"}
                },
                "required": ["circuit_features"]
            },
            category=ToolCategory.NEURAL,
            defer_loading=True,
            keywords=["predict", "parameters", "model", "inference"],
            agent_owner="dr_asha_neural"
        ))
        
        self.register(ToolDefinition(
            name="batch_optimize_circuits",
            description="Optimize multiple circuits in batch using PTC pattern",
            input_schema={
                "type": "object",
                "properties": {
                    "circuits": {"type": "array", "items": {"type": "object"}},
                    "max_parallel": {"type": "integer", "default": 20}
                },
                "required": ["circuits"]
            },
            category=ToolCategory.NEURAL,
            defer_loading=True,
            keywords=["batch", "optimize", "parallel", "circuits"],
            agent_owner="dr_asha_neural"
        ))
    
    def register_compression_tools(self) -> None:
        """Register holographic compression tools (nexuszero-holographic)."""
        
        self.register(ToolDefinition(
            name="compress_state",
            description="Compress proof state using MPS/tensor network",
            input_schema={
                "type": "object",
                "properties": {
                    "state": {"type": "string", "description": "Hex-encoded state"},
                    "target_ratio": {"type": "number", "description": "Target compression ratio"}
                },
                "required": ["state"]
            },
            category=ToolCategory.COMPRESSION,
            defer_loading=True,
            keywords=["compress", "mps", "tensor", "holographic"],
            agent_owner="morgan_rustico"
        ))
        
        self.register(ToolDefinition(
            name="decompress_state",
            description="Decompress state from MPS representation",
            input_schema={
                "type": "object",
                "properties": {
                    "compressed": {"type": "string"}
                },
                "required": ["compressed"]
            },
            category=ToolCategory.COMPRESSION,
            defer_loading=True,
            keywords=["decompress", "restore", "state"],
            agent_owner="morgan_rustico"
        ))
    
    def register_benchmark_tools(self) -> None:
        """Register benchmarking tools."""
        
        self.register(ToolDefinition(
            name="benchmark_ntt",
            description="Benchmark NTT performance for various dimensions",
            input_schema={
                "type": "object",
                "properties": {
                    "dimensions": {"type": "array", "items": {"type": "integer"}},
                    "iterations": {"type": "integer", "default": 1000}
                },
                "required": ["dimensions"]
            },
            category=ToolCategory.BENCHMARK,
            defer_loading=True,
            keywords=["benchmark", "ntt", "performance", "fft"],
            agent_owner="quinn_quality"
        ))
        
        self.register(ToolDefinition(
            name="benchmark_proof_generation",
            description="Benchmark proof generation across parameter sets",
            input_schema={
                "type": "object",
                "properties": {
                    "parameter_sets": {"type": "array"},
                    "circuit_sizes": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["parameter_sets", "circuit_sizes"]
            },
            category=ToolCategory.BENCHMARK,
            defer_loading=True,
            keywords=["benchmark", "proof", "generation", "performance"],
            agent_owner="quinn_quality"
        ))
    
    def register_security_tools(self) -> None:
        """Register security audit tools."""
        
        self.register(ToolDefinition(
            name="audit_timing",
            description="Run timing analysis for side-channel vulnerabilities",
            input_schema={
                "type": "object",
                "properties": {
                    "function_name": {"type": "string"},
                    "input_variations": {"type": "array"}
                },
                "required": ["function_name"]
            },
            category=ToolCategory.SECURITY,
            defer_loading=True,
            keywords=["audit", "timing", "side-channel", "security"],
            agent_owner="sam_sentinel"
        ))
        
        self.register(ToolDefinition(
            name="fuzz_test",
            description="Run fuzz testing on cryptographic functions",
            input_schema={
                "type": "object",
                "properties": {
                    "target_function": {"type": "string"},
                    "iterations": {"type": "integer", "default": 10000}
                },
                "required": ["target_function"]
            },
            category=ToolCategory.SECURITY,
            defer_loading=True,
            keywords=["fuzz", "test", "security", "vulnerability"],
            agent_owner="sam_sentinel"
        ))
    
    def register_chain_tools(self) -> None:
        """Register blockchain connector tools."""
        
        chains = ["ethereum", "bitcoin", "solana", "polygon", "cosmos"]
        
        for chain in chains:
            self.register(ToolDefinition(
                name=f"submit_proof_{chain}",
                description=f"Submit a ZK proof to {chain.title()} network",
                input_schema={
                    "type": "object",
                    "properties": {
                        "proof": {"type": "string"},
                        "contract_address": {"type": "string"}
                    },
                    "required": ["proof", "contract_address"]
                },
                category=ToolCategory.CHAIN,
                defer_loading=True,
                keywords=["submit", "proof", chain, "blockchain"],
                agent_owner="casey_cloud"
            ))
            
            self.register(ToolDefinition(
                name=f"verify_proof_{chain}",
                description=f"Verify a ZK proof on {chain.title()} network",
                input_schema={
                    "type": "object",
                    "properties": {
                        "proof_hash": {"type": "string"},
                        "contract_address": {"type": "string"}
                    },
                    "required": ["proof_hash", "contract_address"]
                },
                category=ToolCategory.CHAIN,
                defer_loading=True,
                keywords=["verify", "proof", chain, "blockchain"],
                agent_owner="casey_cloud"
            ))
    
    def register_monitoring_tools(self) -> None:
        """Register monitoring and observability tools."""
        
        self.register(ToolDefinition(
            name="get_metrics",
            description="Get current system metrics (CPU, memory, proof rate)",
            input_schema={
                "type": "object",
                "properties": {
                    "metrics": {"type": "array", "items": {"type": "string"}}
                }
            },
            category=ToolCategory.MONITORING,
            defer_loading=True,
            keywords=["metrics", "monitoring", "prometheus", "grafana"],
            agent_owner="jordan_ops"
        ))
        
        self.register(ToolDefinition(
            name="alert_on_threshold",
            description="Set up alert for metric threshold",
            input_schema={
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "threshold": {"type": "number"},
                    "comparison": {"type": "string", "enum": ["gt", "lt", "eq"]}
                },
                "required": ["metric", "threshold", "comparison"]
            },
            category=ToolCategory.MONITORING,
            defer_loading=True,
            keywords=["alert", "threshold", "monitoring", "notification"],
            agent_owner="jordan_ops"
        ))


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry (singleton)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
        _global_registry.register_all_nexuszero_tools()
    return _global_registry


def search_tools(query: str) -> List[ToolDefinition]:
    """
    Search for tools matching a query.
    
    Tool Use Example (for AI assistants):
    ------------------------------------
    # Find benchmark tools:
    tools = search_tools("benchmark proof")
    
    # Find security audit tools:
    tools = search_tools("side-channel timing")
    
    # Find Ethereum tools:
    tools = search_tools("ethereum submit")
    """
    return get_registry().search(query)
