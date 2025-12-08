use nexuszero_crypto::benchmark::{PerformanceBenchmarker, BenchmarkConfig};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 NexusZero Performance Benchmarking Demo");
    println!("==========================================");

    // Create benchmark configuration
    let config = BenchmarkConfig {
        iterations: 10,
        warmup_iterations: 2,
        max_execution_time_secs: 30,
        memory_sample_interval_ms: 10,
        enable_memory_profiling: true,
        enable_hardware_benchmarks: true,
        concurrency_limit: 2,
    };

    // Create benchmarker
    let benchmarker = PerformanceBenchmarker::with_config(config.clone());

    println!("📊 Running comprehensive performance benchmarks...");
    println!("   - Iterations per operation: {}", config.iterations);
    println!("   - Warmup iterations: {}", config.warmup_iterations);
    println!("   - Concurrency limit: {}", config.concurrency_limit);
    println!();

    // Run benchmarks
    let results = benchmarker.run_comprehensive_benchmarks().await?;

    println!("✅ Benchmarking completed successfully!");
    println!();
    println!("📈 Results Summary:");
    println!("==================");
    println!("Timestamp: {}", results.timestamp);
    println!("System: {} {}", results.system_info.os, results.system_info.cpu_model);
    println!("CPU Cores: {}", results.system_info.cpu_cores);
    println!("Total Memory: {} MB", results.system_info.total_memory / 1024 / 1024);
    println!();

    println!("🔍 Operation Performance:");
    println!("========================");
    for (op_name, metrics) in &results.operation_results {
        println!("{}:", op_name);
        println!("  - Average Time: {:.2} ms", metrics.avg_time.as_secs_f64() * 1000.0);
        println!("  - Min Time: {:.2} ms", metrics.min_time.as_secs_f64() * 1000.0);
        println!("  - Max Time: {:.2} ms", metrics.max_time.as_secs_f64() * 1000.0);
        println!("  - Operations/sec: {:.0}", metrics.ops_per_sec);
        println!("  - Memory Usage: {} bytes", metrics.memory_stats.as_ref().map_or(0, |m| m.peak_usage));
        println!();
    }

    // Check for regressions (no baseline, so none expected)
    if results.regressions.is_empty() {
        println!("🎉 No performance regressions detected!");
    } else {
        println!("⚠️  Performance regressions detected:");
        for regression in &results.regressions {
            println!("  - {}: {:.1}% slower ({:.0} → {:.0} ops/sec)",
                regression.operation,
                regression.regression_percent,
                regression.previous_ops_per_sec,
                regression.current_ops_per_sec);
        }
    }

    println!();
    println!("🏆 Benchmarking framework successfully implemented!");
    println!("   - ✅ Async execution with concurrency control");
    println!("   - ✅ Detailed performance metrics collection");
    println!("   - ✅ Memory usage tracking");
    println!("   - ✅ Regression detection capabilities");
    println!("   - ✅ System information gathering");
    println!("   - ✅ JSON serialization for persistence");

    Ok(())
}