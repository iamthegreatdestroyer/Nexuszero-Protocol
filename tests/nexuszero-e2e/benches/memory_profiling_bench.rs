//! Memory Profiling Benchmarks
//!
//! Comprehensive memory profiling benchmarks to identify allocation patterns,
//! memory leaks, and optimization opportunities in the NexusZero protocol.
//!
//! Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
//! Licensed under AGPL-3.0.

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BatchSize, BenchmarkGroup,
    BenchmarkId, Criterion, Throughput,
};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Duration;

// =============================================================================
// ALLOCATION PATTERN BENCHMARKS
// =============================================================================

/// Benchmark different allocation patterns and their performance characteristics
fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");
    group.sample_size(50);

    let sizes = [1024, 4096, 16384, 65536, 262144, 1048576];

    for size in sizes.iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        // Single large allocation
        group.bench_with_input(
            BenchmarkId::new("single_large_alloc", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || (),
                    |_| {
                        let vec: Vec<u8> = vec![0u8; size];
                        black_box(vec)
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        // Many small allocations totaling same size
        group.bench_with_input(
            BenchmarkId::new("many_small_allocs", size),
            size,
            |b, &size| {
                let chunk_size = 64;
                let num_chunks = size / chunk_size;

                b.iter_batched(
                    || (),
                    |_| {
                        let vecs: Vec<Vec<u8>> =
                            (0..num_chunks).map(|_| vec![0u8; chunk_size]).collect();
                        black_box(vecs)
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        // Pre-allocated with capacity
        group.bench_with_input(
            BenchmarkId::new("preallocated_capacity", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || (),
                    |_| {
                        let mut vec = Vec::with_capacity(size);
                        vec.resize(size, 0u8);
                        black_box(vec)
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        // Incremental growth
        group.bench_with_input(
            BenchmarkId::new("incremental_growth", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || (),
                    |_| {
                        let mut vec = Vec::new();
                        for i in 0..size {
                            vec.push((i & 0xFF) as u8);
                        }
                        black_box(vec)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// DATA STRUCTURE MEMORY BENCHMARKS
// =============================================================================

/// Benchmark memory usage of different data structures
fn bench_data_structure_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_structure_memory");
    group.sample_size(50);

    let element_counts = [100, 1000, 10000, 100000];

    for count in element_counts.iter() {
        group.throughput(Throughput::Elements(*count as u64));

        // Vec<u64>
        group.bench_with_input(BenchmarkId::new("vec_u64", count), count, |b, &count| {
            b.iter_batched(
                || (),
                |_| {
                    let vec: Vec<u64> = (0..count as u64).collect();
                    black_box(vec)
                },
                BatchSize::SmallInput,
            );
        });

        // HashMap<u64, u64>
        group.bench_with_input(BenchmarkId::new("hashmap_u64", count), count, |b, &count| {
            b.iter_batched(
                || (),
                |_| {
                    let map: HashMap<u64, u64> = (0..count as u64).map(|i| (i, i * 2)).collect();
                    black_box(map)
                },
                BatchSize::SmallInput,
            );
        });

        // BTreeMap<u64, u64>
        group.bench_with_input(
            BenchmarkId::new("btreemap_u64", count),
            count,
            |b, &count| {
                b.iter_batched(
                    || (),
                    |_| {
                        let map: BTreeMap<u64, u64> =
                            (0..count as u64).map(|i| (i, i * 2)).collect();
                        black_box(map)
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        // VecDeque<u64>
        group.bench_with_input(
            BenchmarkId::new("vecdeque_u64", count),
            count,
            |b, &count| {
                b.iter_batched(
                    || (),
                    |_| {
                        let deque: VecDeque<u64> = (0..count as u64).collect();
                        black_box(deque)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// BUFFER REUSE BENCHMARKS
// =============================================================================

/// Benchmark buffer reuse strategies
fn bench_buffer_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_reuse");
    group.sample_size(100);

    let iterations = 100;
    let buffer_size = 65536;

    // Fresh buffer each iteration
    group.bench_function("fresh_buffer_per_iter", |b| {
        b.iter(|| {
            for i in 0..iterations {
                let mut buffer = vec![0u8; buffer_size];
                for (j, byte) in buffer.iter_mut().enumerate() {
                    *byte = ((i + j) & 0xFF) as u8;
                }
                black_box(&buffer);
            }
        });
    });

    // Single reused buffer
    group.bench_function("reused_buffer", |b| {
        b.iter(|| {
            let mut buffer = vec![0u8; buffer_size];
            for i in 0..iterations {
                for (j, byte) in buffer.iter_mut().enumerate() {
                    *byte = ((i + j) & 0xFF) as u8;
                }
                black_box(&buffer);
            }
        });
    });

    // Buffer pool (simulated)
    group.bench_function("buffer_pool", |b| {
        b.iter(|| {
            let mut pool: Vec<Vec<u8>> = (0..4).map(|_| vec![0u8; buffer_size]).collect();
            let mut pool_index = 0;

            for i in 0..iterations {
                let buffer = &mut pool[pool_index % 4];
                for (j, byte) in buffer.iter_mut().enumerate() {
                    *byte = ((i + j) & 0xFF) as u8;
                }
                black_box(&buffer);
                pool_index += 1;
            }
        });
    });

    // Clear and reuse
    group.bench_function("clear_and_reuse", |b| {
        b.iter(|| {
            let mut buffer = Vec::with_capacity(buffer_size);
            for i in 0..iterations {
                buffer.clear();
                buffer.extend((0..buffer_size).map(|j| ((i + j) & 0xFF) as u8));
                black_box(&buffer);
            }
        });
    });

    group.finish();
}

// =============================================================================
// ARENA ALLOCATION PATTERNS
// =============================================================================

/// Benchmark arena-style allocation patterns
fn bench_arena_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena_patterns");
    group.sample_size(50);

    // Simulated arena allocator
    struct SimpleArena {
        buffer: Vec<u8>,
        offset: usize,
    }

    impl SimpleArena {
        fn new(size: usize) -> Self {
            Self {
                buffer: vec![0u8; size],
                offset: 0,
            }
        }

        fn alloc(&mut self, size: usize) -> Option<&mut [u8]> {
            if self.offset + size <= self.buffer.len() {
                let slice = &mut self.buffer[self.offset..self.offset + size];
                self.offset += size;
                Some(slice)
            } else {
                None
            }
        }

        fn reset(&mut self) {
            self.offset = 0;
        }
    }

    // Many small allocations - standard
    group.bench_function("standard_small_allocs", |b| {
        let alloc_size = 256;
        let num_allocs = 1000;

        b.iter(|| {
            let allocations: Vec<Vec<u8>> = (0..num_allocs)
                .map(|i| {
                    let mut v = vec![0u8; alloc_size];
                    v[0] = (i & 0xFF) as u8;
                    v
                })
                .collect();
            black_box(allocations)
        });
    });

    // Many small allocations - arena style
    group.bench_function("arena_small_allocs", |b| {
        let alloc_size = 256;
        let num_allocs = 1000;
        let arena_size = alloc_size * num_allocs;

        b.iter(|| {
            let mut arena = SimpleArena::new(arena_size);
            for i in 0..num_allocs {
                if let Some(slice) = arena.alloc(alloc_size) {
                    slice[0] = (i & 0xFF) as u8;
                }
            }
            black_box(arena.offset)
        });
    });

    // Arena with reset
    group.bench_function("arena_with_reset", |b| {
        let alloc_size = 256;
        let num_allocs = 100;
        let arena_size = alloc_size * num_allocs;
        let rounds = 10;

        b.iter(|| {
            let mut arena = SimpleArena::new(arena_size);
            for _round in 0..rounds {
                for i in 0..num_allocs {
                    if let Some(slice) = arena.alloc(alloc_size) {
                        slice[0] = (i & 0xFF) as u8;
                    }
                }
                arena.reset();
            }
            black_box(arena.offset)
        });
    });

    group.finish();
}

// =============================================================================
// STRING MEMORY BENCHMARKS
// =============================================================================

/// Benchmark string allocation and manipulation patterns
fn bench_string_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_memory");
    group.sample_size(100);

    // String concatenation patterns
    let parts: Vec<&str> = vec!["hello", " ", "world", "!", " ", "this", " ", "is", " ", "a", " ", "test"];

    group.bench_function("string_concat_push", |b| {
        b.iter(|| {
            let mut result = String::new();
            for part in parts.iter() {
                result.push_str(part);
            }
            black_box(result)
        });
    });

    group.bench_function("string_concat_capacity", |b| {
        let total_len: usize = parts.iter().map(|s| s.len()).sum();
        b.iter(|| {
            let mut result = String::with_capacity(total_len);
            for part in parts.iter() {
                result.push_str(part);
            }
            black_box(result)
        });
    });

    group.bench_function("string_collect", |b| {
        b.iter(|| {
            let result: String = parts.iter().copied().collect();
            black_box(result)
        });
    });

    group.bench_function("string_join", |b| {
        b.iter(|| {
            let result = parts.join("");
            black_box(result)
        });
    });

    // Large string building
    group.bench_function("large_string_build", |b| {
        b.iter(|| {
            let mut result = String::with_capacity(10000);
            for i in 0..1000 {
                result.push_str(&format!("item{:04} ", i));
            }
            black_box(result)
        });
    });

    group.finish();
}

// =============================================================================
// ZERO-COPY BENCHMARKS
// =============================================================================

/// Benchmark zero-copy operations
fn bench_zero_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy");
    group.sample_size(100);

    let data_size = 1024 * 1024; // 1MB
    let data: Vec<u8> = (0..data_size).map(|i| (i & 0xFF) as u8).collect();

    // Copy-based processing
    group.bench_function("copy_based", |b| {
        b.iter(|| {
            let copy = data.clone();
            let sum: u64 = copy.iter().map(|&b| b as u64).sum();
            black_box(sum)
        });
    });

    // Reference-based processing
    group.bench_function("reference_based", |b| {
        b.iter(|| {
            let sum: u64 = data.iter().map(|&b| b as u64).sum();
            black_box(sum)
        });
    });

    // Slice-based processing
    group.bench_function("slice_chunks", |b| {
        b.iter(|| {
            let sum: u64 = data
                .chunks(1024)
                .map(|chunk| chunk.iter().map(|&b| b as u64).sum::<u64>())
                .sum();
            black_box(sum)
        });
    });

    // Parallel chunk processing (simulated)
    group.bench_function("parallel_chunks_simulated", |b| {
        let num_chunks = 4;
        let chunk_size = data.len() / num_chunks;

        b.iter(|| {
            let sums: Vec<u64> = (0..num_chunks)
                .map(|i| {
                    let start = i * chunk_size;
                    let end = if i == num_chunks - 1 {
                        data.len()
                    } else {
                        (i + 1) * chunk_size
                    };
                    data[start..end].iter().map(|&b| b as u64).sum()
                })
                .collect();
            let total: u64 = sums.iter().sum();
            black_box(total)
        });
    });

    group.finish();
}

// =============================================================================
// MEMORY LAYOUT BENCHMARKS
// =============================================================================

/// Benchmark different memory layout patterns
fn bench_memory_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout");
    group.sample_size(50);

    let count = 10000;

    // Array of structs
    #[derive(Clone, Copy)]
    struct AoSPoint {
        x: f64,
        y: f64,
        z: f64,
        w: f64,
    }

    group.bench_function("aos_access", |b| {
        let points: Vec<AoSPoint> = (0..count)
            .map(|i| AoSPoint {
                x: i as f64,
                y: (i * 2) as f64,
                z: (i * 3) as f64,
                w: (i * 4) as f64,
            })
            .collect();

        b.iter(|| {
            let sum: f64 = points.iter().map(|p| p.x + p.y + p.z + p.w).sum();
            black_box(sum)
        });
    });

    // Struct of arrays
    struct SoAPoints {
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
        w: Vec<f64>,
    }

    group.bench_function("soa_access", |b| {
        let points = SoAPoints {
            x: (0..count).map(|i| i as f64).collect(),
            y: (0..count).map(|i| (i * 2) as f64).collect(),
            z: (0..count).map(|i| (i * 3) as f64).collect(),
            w: (0..count).map(|i| (i * 4) as f64).collect(),
        };

        b.iter(|| {
            let sum: f64 = (0..count)
                .map(|i| points.x[i] + points.y[i] + points.z[i] + points.w[i])
                .sum();
            black_box(sum)
        });
    });

    // Column-specific access (SoA advantage)
    group.bench_function("soa_column_access", |b| {
        let points = SoAPoints {
            x: (0..count).map(|i| i as f64).collect(),
            y: (0..count).map(|i| (i * 2) as f64).collect(),
            z: (0..count).map(|i| (i * 3) as f64).collect(),
            w: (0..count).map(|i| (i * 4) as f64).collect(),
        };

        b.iter(|| {
            let sum_x: f64 = points.x.iter().sum();
            let sum_y: f64 = points.y.iter().sum();
            black_box((sum_x, sum_y))
        });
    });

    group.finish();
}

// =============================================================================
// CRITERION GROUPS AND MAIN
// =============================================================================

criterion_group!(
    name = memory_benchmarks;
    config = Criterion::default()
        .significance_level(0.05)
        .noise_threshold(0.02)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));
    targets =
        bench_allocation_patterns,
        bench_data_structure_memory,
        bench_buffer_reuse,
        bench_arena_patterns,
        bench_string_memory,
        bench_zero_copy,
        bench_memory_layout
);

criterion_main!(memory_benchmarks);
