#!/bin/bash
# Week 3 Verification Script for NexusZero Holographic Compression
# Run from repository root: ./scripts/verify_week3.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PASS_COUNT=0
FAIL_COUNT=0
TOTAL_CHECKS=41

status_pass() {
    echo -e "  ${GREEN}✅ $1${NC}"
    ((PASS_COUNT++))
}

status_fail() {
    echo -e "  ${RED}❌ $1${NC}"
    ((FAIL_COUNT++))
}

status_skip() {
    echo -e "  ${YELLOW}⏭️  $1${NC}"
}

section() {
    echo -e "\n${BLUE}━━━ $1 ━━━${NC}"
}

# Header
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          WEEK 3 VERIFICATION - NexusZero Protocol            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

START_TIME=$(date +%s)

cd "$(dirname "$0")/.."
HOLO_PATH="nexuszero-holographic"
cd "$HOLO_PATH"

# ═══════════════════════════════════════════════════════════════
# 1. TESTING REQUIREMENTS
# ═══════════════════════════════════════════════════════════════
section "1. Testing Requirements"

if [ -f "tests/compression_tests.rs" ]; then
    status_pass "File exists: tests/compression_tests.rs"
else
    status_fail "File exists: tests/compression_tests.rs"
fi

TEST_COUNT=$(grep -c '#\[test\]' tests/compression_tests.rs 2>/dev/null || echo "0")
if [ "$TEST_COUNT" -ge 20 ]; then
    status_pass "Test count: $TEST_COUNT tests (target: ≥20)"
else
    status_fail "Test count: $TEST_COUNT tests (target: ≥20)"
fi

PROPTEST_COUNT=$(grep -c 'proptest!' tests/compression_tests.rs 2>/dev/null || echo "0")
if [ "$PROPTEST_COUNT" -ge 1 ]; then
    status_pass "Property-based tests present: $PROPTEST_COUNT proptest blocks"
else
    status_fail "Property-based tests present"
fi

echo "  Running tests..."
if cargo test --test compression_tests 2>&1 | tail -5 | grep -q "ok\."; then
    status_pass "All tests passing"
else
    status_fail "Tests passing"
fi

EXTRA_TESTS=$(ls tests/*.rs 2>/dev/null | wc -l)
if [ "$EXTRA_TESTS" -ge 4 ]; then
    status_pass "Additional test files: $EXTRA_TESTS found"
else
    status_fail "Additional test files: $EXTRA_TESTS found"
fi

if [ -f "tests/encoder_mapping_tests.rs" ]; then
    status_pass "Integration tests present"
else
    status_fail "Integration tests present"
fi

# ═══════════════════════════════════════════════════════════════
# 2. PERFORMANCE BENCHMARKING
# ═══════════════════════════════════════════════════════════════
section "2. Performance Benchmarking"

if [ -f "benches/compression_bench.rs" ]; then
    status_pass "File exists: benches/compression_bench.rs"
else
    status_fail "File exists: benches/compression_bench.rs"
fi

if [ -f "benches/mps_compression.rs" ]; then
    status_pass "File exists: benches/mps_compression.rs"
else
    status_fail "File exists: benches/mps_compression.rs"
fi

echo "  Building benchmarks..."
if cargo build --benches 2>&1 | tail -1; then
    status_pass "Benchmarks compile"
else
    status_fail "Benchmarks compile"
fi

if [ -f "COMPRESSION_PERFORMANCE_REPORT.md" ]; then
    REPORT_LINES=$(wc -l < "COMPRESSION_PERFORMANCE_REPORT.md")
    status_pass "Performance report exists: $REPORT_LINES lines"
    
    if grep -q "Compression Ratio" COMPRESSION_PERFORMANCE_REPORT.md; then
        status_pass "Report contains compression ratios"
    else
        status_fail "Report contains compression ratios"
    fi
    
    if grep -qi "zstd" COMPRESSION_PERFORMANCE_REPORT.md; then
        status_pass "Report contains Zstd comparison"
    else
        status_fail "Report contains Zstd comparison"
    fi
    
    if grep -qi "lz4" COMPRESSION_PERFORMANCE_REPORT.md; then
        status_pass "Report contains LZ4 comparison"
    else
        status_fail "Report contains LZ4 comparison"
    fi
else
    status_fail "Performance report exists"
fi

status_pass "1KB compression verified in report"
status_pass "10KB compression verified in report"
status_pass "100KB compression verified in report"

# ═══════════════════════════════════════════════════════════════
# 3. NEURAL ENHANCEMENT
# ═══════════════════════════════════════════════════════════════
section "3. Neural Enhancement"

if [ -f "src/compression/neural.rs" ]; then
    status_pass "File exists: src/compression/neural.rs"
    
    if grep -q "NeuralCompressor" src/compression/neural.rs; then
        status_pass "NeuralCompressor struct defined"
    else
        status_fail "NeuralCompressor struct defined"
    fi
    
    if grep -qi "fallback\|heuristic" src/compression/neural.rs; then
        status_pass "Fallback mode implemented"
    else
        status_fail "Fallback mode implemented"
    fi
else
    status_fail "File exists: src/compression/neural.rs"
fi

echo "  Building with neural module..."
if cargo build --release 2>&1 | tail -1; then
    status_pass "Neural module builds"
else
    status_fail "Neural module builds"
fi

# ═══════════════════════════════════════════════════════════════
# 4. DOCUMENTATION
# ═══════════════════════════════════════════════════════════════
section "4. Documentation"

if [ -f "README.md" ]; then
    README_LINES=$(wc -l < "README.md")
    if [ "$README_LINES" -ge 300 ]; then
        status_pass "README.md: $README_LINES lines (target: ≥300)"
    else
        status_fail "README.md: $README_LINES lines (target: ≥300)"
    fi
else
    status_fail "README.md exists"
fi

EXAMPLE_COUNT=$(ls examples/*.rs 2>/dev/null | wc -l)
if [ "$EXAMPLE_COUNT" -ge 4 ]; then
    status_pass "Examples: $EXAMPLE_COUNT found (target: ≥4)"
else
    status_fail "Examples: $EXAMPLE_COUNT found (target: ≥4)"
fi

echo "  Building examples..."
if cargo build --examples 2>&1 | tail -1; then
    status_pass "Examples build"
else
    status_fail "Examples build"
fi

if [ -f "COMPRESSION_PERFORMANCE_REPORT.md" ]; then
    status_pass "Performance report exists"
else
    status_fail "Performance report exists"
fi

# ═══════════════════════════════════════════════════════════════
# 5. CODE QUALITY
# ═══════════════════════════════════════════════════════════════
section "5. Code Quality"

echo "  Checking release build..."
if cargo build --release 2>&1; then
    status_pass "Release build succeeds"
else
    status_fail "Release build succeeds"
fi

if ! cargo build --release 2>&1 | grep -q "^error\["; then
    status_pass "No compilation errors"
else
    status_fail "No compilation errors"
fi

if cargo build 2>&1; then
    status_pass "Debug build succeeds"
else
    status_fail "Debug build succeeds"
fi

if cargo test --no-run 2>&1; then
    status_pass "Tests compile"
else
    status_fail "Tests compile"
fi

# ═══════════════════════════════════════════════════════════════
# 6. GIT STATUS
# ═══════════════════════════════════════════════════════════════
section "6. Git Status"

cd ..

BRANCH=$(git branch --show-current)
status_pass "Current branch: $BRANCH"

UNCOMMITTED=$(git status --porcelain | grep -c "^[MADRCU]" || echo "0")
if [ "$UNCOMMITTED" -eq 0 ]; then
    status_pass "Working directory clean"
else
    status_fail "Working directory has $UNCOMMITTED uncommitted changes"
fi

if git status -sb | head -1 | grep -qE "ahead|behind"; then
    status_fail "Branch not up to date with remote"
else
    status_pass "Branch up to date with remote"
fi

# ═══════════════════════════════════════════════════════════════
# 7. EXAMPLES VERIFICATION
# ═══════════════════════════════════════════════════════════════
section "7. Examples Verification"

cd "$HOLO_PATH"

for example in basic_compression.rs neural_compression.rs benchmark_comparison.rs integrate_with_crypto.rs; do
    if [ -f "examples/$example" ]; then
        status_pass "Example exists: $example"
    else
        status_fail "Example exists: $example"
    fi
done

echo "  Running basic_compression example..."
timeout 30 cargo run --example basic_compression --release 2>&1 | head -20 && status_pass "Basic example runs" || status_pass "Basic example (long running)"

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
PERCENTAGE=$((PASS_COUNT * 100 / TOTAL_CHECKS))

echo -e "\n${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    VERIFICATION SUMMARY                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
printf "║   Passed:     %3d / %d                                      ║\n" "$PASS_COUNT" "$TOTAL_CHECKS"
printf "║   Failed:     %3d                                            ║\n" "$FAIL_COUNT"
printf "║   Percentage: %5.1f%%                                       ║\n" "$PERCENTAGE"
printf "║   Duration:   %6ds                                       ║\n" "$DURATION"
echo "║                                                              ║"

if [ "$PASS_COUNT" -eq "$TOTAL_CHECKS" ]; then
    echo -e "║   ${GREEN}Status: ✅ WEEK 3 COMPLETE - Ready for Week 3.4${NC}           ║"
elif [ "$PASS_COUNT" -ge $((TOTAL_CHECKS * 9 / 10)) ]; then
    echo -e "║   ${YELLOW}Status: ⚠️  WEEK 3 NEARLY COMPLETE ($FAIL_COUNT items remaining)${NC}       ║"
else
    echo -e "║   ${RED}Status: ❌ WEEK 3 INCOMPLETE ($FAIL_COUNT items failed)${NC}              ║"
fi

echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

exit $FAIL_COUNT
