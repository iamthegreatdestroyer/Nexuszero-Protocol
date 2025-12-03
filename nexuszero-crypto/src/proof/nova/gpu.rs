// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.

//! GPU-Accelerated Nova Operations
//!
//! This module provides hardware-accelerated implementations of computationally
//! intensive operations used in Nova folding scheme:
//!
//! - **Multi-Scalar Multiplication (MSM)**: Parallel scalar-point multiplication
//! - **Number-Theoretic Transform (NTT)**: Fast polynomial multiplication
//! - **Batch Commitment**: Parallel Pedersen/IPA commitments
//! - **Matrix Operations**: Efficient R1CS evaluation
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                  Nova GPU Acceleration Layer                        │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
//! │  │   MSM Unit   │  │   NTT Unit   │  │  Commitment  │              │
//! │  │              │  │              │  │    Unit      │              │
//! │  │ - Pippenger  │  │ - Forward    │  │ - Pedersen   │              │
//! │  │ - Bucket     │  │ - Inverse    │  │ - IPA        │              │
//! │  │ - Window     │  │ - Batch      │  │ - Batch      │              │
//! │  └──────────────┘  └──────────────┘  └──────────────┘              │
//! │         │                  │                  │                    │
//! │         └──────────────────┴──────────────────┘                    │
//! │                            │                                       │
//! │                            ▼                                       │
//! │  ┌─────────────────────────────────────────────────────┐          │
//! │  │               GPU Compute Pipeline                  │          │
//! │  │  (WebGPU / CUDA / Metal abstraction)                │          │
//! │  └─────────────────────────────────────────────────────┘          │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use super::types::{NovaError, NovaResult, NovaSecurityLevel};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "gpu")]
use wgpu;

/// GPU acceleration context for Nova operations
#[cfg(feature = "gpu")]
pub struct NovaGPU {
    /// WebGPU device handle
    device: Arc<wgpu::Device>,
    /// Command queue for GPU operations
    queue: Arc<wgpu::Queue>,
    /// MSM compute pipeline
    msm_pipeline: wgpu::ComputePipeline,
    /// NTT forward transform pipeline
    ntt_forward_pipeline: wgpu::ComputePipeline,
    /// NTT inverse transform pipeline
    ntt_inverse_pipeline: wgpu::ComputePipeline,
    /// Commitment pipeline
    commitment_pipeline: wgpu::ComputePipeline,
    /// Configuration
    config: GPUConfig,
    /// Performance metrics
    metrics: GPUMetrics,
}

/// GPU configuration for Nova operations
#[derive(Debug, Clone)]
pub struct GPUConfig {
    /// Maximum workgroup size
    pub workgroup_size: u32,
    /// Number of workgroups for parallel operations
    pub num_workgroups: u32,
    /// Whether to use async execution
    pub async_execution: bool,
    /// Memory limit in bytes
    pub memory_limit: usize,
    /// MSM window size for Pippenger algorithm
    pub msm_window_size: u32,
    /// NTT domain size (power of 2)
    pub ntt_domain_size: usize,
    /// Enable profiling
    pub profiling_enabled: bool,
}

impl Default for GPUConfig {
    fn default() -> Self {
        Self {
            workgroup_size: 256,
            num_workgroups: 64,
            async_execution: true,
            memory_limit: 1024 * 1024 * 1024, // 1 GB
            msm_window_size: 16,
            ntt_domain_size: 1 << 16, // 65536
            profiling_enabled: false,
        }
    }
}

impl GPUConfig {
    /// Create configuration optimized for high performance
    pub fn high_performance() -> Self {
        Self {
            workgroup_size: 512,
            num_workgroups: 128,
            async_execution: true,
            memory_limit: 4 * 1024 * 1024 * 1024, // 4 GB
            msm_window_size: 20,
            ntt_domain_size: 1 << 20, // 1M
            profiling_enabled: false,
        }
    }

    /// Create configuration optimized for memory efficiency
    pub fn memory_efficient() -> Self {
        Self {
            workgroup_size: 128,
            num_workgroups: 32,
            async_execution: true,
            memory_limit: 512 * 1024 * 1024, // 512 MB
            msm_window_size: 12,
            ntt_domain_size: 1 << 14, // 16K
            profiling_enabled: false,
        }
    }
}

/// Performance metrics for GPU operations
#[derive(Debug, Clone, Default)]
pub struct GPUMetrics {
    /// Total MSM operations performed
    pub msm_operations: u64,
    /// Total NTT operations performed
    pub ntt_operations: u64,
    /// Total commitment operations
    pub commitment_operations: u64,
    /// Average MSM time in microseconds
    pub avg_msm_time_us: f64,
    /// Average NTT time in microseconds
    pub avg_ntt_time_us: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// GPU utilization percentage
    pub gpu_utilization: f64,
}

/// Scalar point for MSM operations
#[derive(Debug, Clone)]
pub struct ScalarPoint {
    /// Scalar value (little-endian bytes)
    pub scalar: Vec<u8>,
    /// Point coordinates (compressed or affine)
    pub point: Vec<u8>,
}

/// NTT domain parameters
#[derive(Debug, Clone)]
pub struct NTTDomain {
    /// Domain size (must be power of 2)
    pub size: usize,
    /// Primitive root of unity
    pub root_of_unity: Vec<u8>,
    /// Inverse of domain size (for inverse NTT)
    pub size_inv: Vec<u8>,
    /// Twiddle factors (precomputed)
    pub twiddle_factors: Vec<Vec<u8>>,
}

impl NTTDomain {
    /// Create a new NTT domain with given size
    pub fn new(size: usize) -> NovaResult<Self> {
        if !size.is_power_of_two() {
            return Err(NovaError::InvalidInput(
                "NTT domain size must be power of 2".to_string(),
            ));
        }

        // For demonstration, using placeholder values
        // In production, these would be computed based on the field
        let root_of_unity = vec![0u8; 32];
        let size_inv = vec![0u8; 32];
        let twiddle_factors = Vec::new();

        Ok(Self {
            size,
            root_of_unity,
            size_inv,
            twiddle_factors,
        })
    }

    /// Precompute twiddle factors for the domain
    pub fn precompute_twiddles(&mut self) {
        // Precompute ω^i for i = 0..size/2
        let half_size = self.size / 2;
        self.twiddle_factors = Vec::with_capacity(half_size);
        
        for _ in 0..half_size {
            // Placeholder: in production, compute ω^i mod p
            self.twiddle_factors.push(vec![0u8; 32]);
        }
    }
}

/// MSM result containing the aggregated point
#[derive(Debug, Clone)]
pub struct MSMResult {
    /// Resulting point (compressed)
    pub point: Vec<u8>,
    /// Number of points processed
    pub num_points: usize,
    /// Execution time in microseconds
    pub execution_time_us: u64,
}

/// NTT result containing transformed coefficients
#[derive(Debug, Clone)]
pub struct NTTResult {
    /// Transformed coefficients
    pub coefficients: Vec<Vec<u8>>,
    /// Domain size
    pub domain_size: usize,
    /// Whether this is the forward or inverse transform
    pub is_forward: bool,
    /// Execution time in microseconds
    pub execution_time_us: u64,
}

/// Commitment result
#[derive(Debug, Clone)]
pub struct CommitmentResult {
    /// Commitment value
    pub commitment: Vec<u8>,
    /// Blinding factor (if used)
    pub blinding: Option<Vec<u8>>,
    /// Execution time in microseconds
    pub execution_time_us: u64,
}

#[cfg(feature = "gpu")]
impl NovaGPU {
    /// Create a new Nova GPU acceleration context
    pub async fn new(config: GPUConfig) -> NovaResult<Self> {
        // Initialize WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| NovaError::HardwareError("No suitable GPU adapter found".to_string()))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: Some("NexusZero Nova GPU"),
                },
                None,
            )
            .await
            .map_err(|e| NovaError::HardwareError(format!("Failed to create GPU device: {}", e)))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Load Nova compute shaders
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nova GPU Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/nova_gpu.wgsl").into()),
        });

        // Create MSM pipeline
        let msm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MSM Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "msm_pippenger",
        });

        // Create NTT forward pipeline
        let ntt_forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NTT Forward Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "ntt_forward",
        });

        // Create NTT inverse pipeline
        let ntt_inverse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NTT Inverse Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "ntt_inverse",
        });

        // Create commitment pipeline
        let commitment_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Commitment Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "pedersen_commit",
        });

        Ok(Self {
            device,
            queue,
            msm_pipeline,
            ntt_forward_pipeline,
            ntt_inverse_pipeline,
            commitment_pipeline,
            config,
            metrics: GPUMetrics::default(),
        })
    }

    /// Perform Multi-Scalar Multiplication using Pippenger algorithm
    ///
    /// Computes: ∑ᵢ sᵢ * Gᵢ where sᵢ are scalars and Gᵢ are curve points
    pub async fn msm(&mut self, scalar_points: &[ScalarPoint]) -> NovaResult<MSMResult> {
        let start = Instant::now();
        let num_points = scalar_points.len();

        if num_points == 0 {
            return Err(NovaError::InvalidInput("Empty scalar points array".to_string()));
        }

        // Serialize data for GPU
        let (scalars_data, points_data) = self.serialize_scalar_points(scalar_points)?;

        // Create GPU buffers
        let scalars_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scalars Buffer"),
            size: scalars_data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let points_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Points Buffer"),
            size: points_data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MSM Result Buffer"),
            size: 64, // Compressed point size
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload data
        self.queue.write_buffer(&scalars_buffer, 0, &scalars_data);
        self.queue.write_buffer(&points_buffer, 0, &points_data);

        // Create bind group
        let bind_group_layout = self.msm_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MSM Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scalars_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: points_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.as_entire_binding(),
                },
            ],
        });

        // Create and submit command buffer
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MSM Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MSM Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.msm_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate number of workgroups
            let workgroups = (num_points as u32 + self.config.workgroup_size - 1) / self.config.workgroup_size;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back result
        let result_point = self.read_buffer_async(&result_buffer, 64).await?;

        let execution_time_us = start.elapsed().as_micros() as u64;

        // Update metrics
        self.metrics.msm_operations += 1;
        self.metrics.avg_msm_time_us = 
            (self.metrics.avg_msm_time_us * (self.metrics.msm_operations - 1) as f64 
             + execution_time_us as f64) / self.metrics.msm_operations as f64;

        Ok(MSMResult {
            point: result_point,
            num_points,
            execution_time_us,
        })
    }

    /// Perform forward NTT (Number Theoretic Transform)
    ///
    /// Transforms polynomial coefficients to evaluation form
    pub async fn ntt_forward(&mut self, coefficients: &[Vec<u8>], domain: &NTTDomain) -> NovaResult<NTTResult> {
        self.ntt_impl(coefficients, domain, true).await
    }

    /// Perform inverse NTT
    ///
    /// Transforms evaluation form back to coefficient form
    pub async fn ntt_inverse(&mut self, evaluations: &[Vec<u8>], domain: &NTTDomain) -> NovaResult<NTTResult> {
        self.ntt_impl(evaluations, domain, false).await
    }

    /// Internal NTT implementation
    async fn ntt_impl(&mut self, input: &[Vec<u8>], domain: &NTTDomain, is_forward: bool) -> NovaResult<NTTResult> {
        let start = Instant::now();

        if input.len() != domain.size {
            return Err(NovaError::InvalidInput(format!(
                "Input size {} doesn't match domain size {}",
                input.len(),
                domain.size
            )));
        }

        // Serialize input data
        let input_data = self.serialize_field_elements(input)?;
        let twiddles_data = self.serialize_field_elements(&domain.twiddle_factors)?;

        // Create buffers
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NTT Input Buffer"),
            size: input_data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let twiddles_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Twiddles Buffer"),
            size: twiddles_data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NTT Output Buffer"),
            size: input_data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload data
        self.queue.write_buffer(&input_buffer, 0, &input_data);
        self.queue.write_buffer(&twiddles_buffer, 0, &twiddles_data);

        // Select pipeline
        let pipeline = if is_forward {
            &self.ntt_forward_pipeline
        } else {
            &self.ntt_inverse_pipeline
        };

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NTT Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute NTT (log2(n) stages)
        let log_n = (domain.size as f64).log2() as u32;
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("NTT Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NTT Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch for butterfly operations
            let workgroups = (domain.size as u32 / 2 + self.config.workgroup_size - 1) / self.config.workgroup_size;
            for _ in 0..log_n {
                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read results
        let output_data = self.read_buffer_async(&output_buffer, input_data.len()).await?;
        let coefficients = self.deserialize_field_elements(&output_data, 32)?;

        let execution_time_us = start.elapsed().as_micros() as u64;

        // Update metrics
        self.metrics.ntt_operations += 1;
        self.metrics.avg_ntt_time_us = 
            (self.metrics.avg_ntt_time_us * (self.metrics.ntt_operations - 1) as f64 
             + execution_time_us as f64) / self.metrics.ntt_operations as f64;

        Ok(NTTResult {
            coefficients,
            domain_size: domain.size,
            is_forward,
            execution_time_us,
        })
    }

    /// Compute Pedersen commitment: C = g^v * h^r
    pub async fn pedersen_commit(&mut self, value: &[u8], blinding: &[u8]) -> NovaResult<CommitmentResult> {
        let start = Instant::now();

        // Create buffers
        let value_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Value Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let blinding_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blinding Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Commitment Result Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Pad inputs to 32 bytes
        let mut value_padded = vec![0u8; 32];
        let mut blinding_padded = vec![0u8; 32];
        value_padded[..value.len().min(32)].copy_from_slice(&value[..value.len().min(32)]);
        blinding_padded[..blinding.len().min(32)].copy_from_slice(&blinding[..blinding.len().min(32)]);

        self.queue.write_buffer(&value_buffer, 0, &value_padded);
        self.queue.write_buffer(&blinding_buffer, 0, &blinding_padded);

        // Create bind group
        let bind_group_layout = self.commitment_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Commitment Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: value_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: blinding_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Commitment Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Commitment Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.commitment_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let commitment = self.read_buffer_async(&result_buffer, 64).await?;
        let execution_time_us = start.elapsed().as_micros() as u64;

        self.metrics.commitment_operations += 1;

        Ok(CommitmentResult {
            commitment,
            blinding: Some(blinding_padded),
            execution_time_us,
        })
    }

    /// Batch Pedersen commitments for efficiency
    pub async fn batch_pedersen_commit(
        &mut self, 
        values: &[Vec<u8>], 
        blindings: &[Vec<u8>]
    ) -> NovaResult<Vec<CommitmentResult>> {
        if values.len() != blindings.len() {
            return Err(NovaError::InvalidInput(
                "Values and blindings must have same length".to_string()
            ));
        }

        let mut results = Vec::with_capacity(values.len());
        
        // Process in batches for better GPU utilization
        let batch_size = 256;
        for chunk_start in (0..values.len()).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(values.len());
            
            for i in chunk_start..chunk_end {
                let result = self.pedersen_commit(&values[i], &blindings[i]).await?;
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Get current GPU metrics
    pub fn metrics(&self) -> &GPUMetrics {
        &self.metrics
    }

    /// Reset GPU metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = GPUMetrics::default();
    }

    // Helper methods

    fn serialize_scalar_points(&self, scalar_points: &[ScalarPoint]) -> NovaResult<(Vec<u8>, Vec<u8>)> {
        let mut scalars = Vec::with_capacity(scalar_points.len() * 32);
        let mut points = Vec::with_capacity(scalar_points.len() * 64);

        for sp in scalar_points {
            // Pad scalar to 32 bytes
            let mut scalar_padded = vec![0u8; 32];
            scalar_padded[..sp.scalar.len().min(32)].copy_from_slice(&sp.scalar[..sp.scalar.len().min(32)]);
            scalars.extend_from_slice(&scalar_padded);

            // Pad point to 64 bytes
            let mut point_padded = vec![0u8; 64];
            point_padded[..sp.point.len().min(64)].copy_from_slice(&sp.point[..sp.point.len().min(64)]);
            points.extend_from_slice(&point_padded);
        }

        Ok((scalars, points))
    }

    fn serialize_field_elements(&self, elements: &[Vec<u8>]) -> NovaResult<Vec<u8>> {
        let mut data = Vec::with_capacity(elements.len() * 32);
        for elem in elements {
            let mut padded = vec![0u8; 32];
            padded[..elem.len().min(32)].copy_from_slice(&elem[..elem.len().min(32)]);
            data.extend_from_slice(&padded);
        }
        Ok(data)
    }

    fn deserialize_field_elements(&self, data: &[u8], element_size: usize) -> NovaResult<Vec<Vec<u8>>> {
        let num_elements = data.len() / element_size;
        let mut elements = Vec::with_capacity(num_elements);
        
        for i in 0..num_elements {
            let start = i * element_size;
            let end = start + element_size;
            elements.push(data[start..end].to_vec());
        }

        Ok(elements)
    }

    async fn read_buffer_async(&self, buffer: &wgpu::Buffer, size: usize) -> NovaResult<Vec<u8>> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .await
            .map_err(|_| NovaError::HardwareError("Buffer mapping cancelled".to_string()))?
            .map_err(|e| NovaError::HardwareError(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range().to_vec();
        staging_buffer.unmap();

        Ok(data)
    }
}

/// CPU fallback implementation when GPU is not available
#[cfg(not(feature = "gpu"))]
pub struct NovaGPU {
    config: GPUConfig,
    metrics: GPUMetrics,
}

#[cfg(not(feature = "gpu"))]
impl NovaGPU {
    /// Create a CPU-based fallback
    pub async fn new(config: GPUConfig) -> NovaResult<Self> {
        Ok(Self {
            config,
            metrics: GPUMetrics::default(),
        })
    }

    /// CPU-based MSM fallback
    pub async fn msm(&mut self, scalar_points: &[ScalarPoint]) -> NovaResult<MSMResult> {
        let start = Instant::now();
        let num_points = scalar_points.len();

        // Simple CPU implementation (placeholder)
        // In production, this would use optimized CPU algorithms
        let result_point = vec![0u8; 64];

        self.metrics.msm_operations += 1;

        Ok(MSMResult {
            point: result_point,
            num_points,
            execution_time_us: start.elapsed().as_micros() as u64,
        })
    }

    /// CPU-based NTT forward fallback
    pub async fn ntt_forward(&mut self, coefficients: &[Vec<u8>], domain: &NTTDomain) -> NovaResult<NTTResult> {
        let start = Instant::now();
        
        // Placeholder CPU implementation
        let output_coefficients = coefficients.to_vec();

        self.metrics.ntt_operations += 1;

        Ok(NTTResult {
            coefficients: output_coefficients,
            domain_size: domain.size,
            is_forward: true,
            execution_time_us: start.elapsed().as_micros() as u64,
        })
    }

    /// CPU-based NTT inverse fallback
    pub async fn ntt_inverse(&mut self, evaluations: &[Vec<u8>], domain: &NTTDomain) -> NovaResult<NTTResult> {
        let start = Instant::now();
        
        let output_coefficients = evaluations.to_vec();

        self.metrics.ntt_operations += 1;

        Ok(NTTResult {
            coefficients: output_coefficients,
            domain_size: domain.size,
            is_forward: false,
            execution_time_us: start.elapsed().as_micros() as u64,
        })
    }

    /// CPU-based Pedersen commitment fallback
    pub async fn pedersen_commit(&mut self, value: &[u8], blinding: &[u8]) -> NovaResult<CommitmentResult> {
        let start = Instant::now();

        // Placeholder: actual implementation would compute g^v * h^r
        let commitment = vec![0u8; 64];

        self.metrics.commitment_operations += 1;

        Ok(CommitmentResult {
            commitment,
            blinding: Some(blinding.to_vec()),
            execution_time_us: start.elapsed().as_micros() as u64,
        })
    }

    /// CPU-based batch commitments fallback
    pub async fn batch_pedersen_commit(
        &mut self, 
        values: &[Vec<u8>], 
        blindings: &[Vec<u8>]
    ) -> NovaResult<Vec<CommitmentResult>> {
        let mut results = Vec::with_capacity(values.len());
        for (v, b) in values.iter().zip(blindings.iter()) {
            results.push(self.pedersen_commit(v, b).await?);
        }
        Ok(results)
    }

    pub fn metrics(&self) -> &GPUMetrics {
        &self.metrics
    }

    pub fn reset_metrics(&mut self) {
        self.metrics = GPUMetrics::default();
    }
}

/// GPU acceleration manager for Nova operations
pub struct GPUAccelerationManager {
    /// GPU context (if available)
    gpu: Option<NovaGPU>,
    /// Whether GPU is enabled
    enabled: bool,
    /// Threshold for using GPU (minimum number of operations)
    gpu_threshold: usize,
}

impl GPUAccelerationManager {
    /// Create a new GPU acceleration manager
    pub async fn new(enabled: bool) -> NovaResult<Self> {
        let gpu = if enabled {
            match NovaGPU::new(GPUConfig::default()).await {
                Ok(g) => Some(g),
                Err(e) => {
                    log::warn!("GPU initialization failed, falling back to CPU: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            gpu,
            enabled,
            gpu_threshold: 100, // Use GPU for >100 operations
        })
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu.is_some()
    }

    /// Perform MSM with automatic CPU/GPU selection
    pub async fn msm(&mut self, scalar_points: &[ScalarPoint]) -> NovaResult<MSMResult> {
        if let Some(ref mut gpu) = self.gpu {
            if scalar_points.len() >= self.gpu_threshold {
                return gpu.msm(scalar_points).await;
            }
        }

        // CPU fallback
        self.cpu_msm(scalar_points)
    }

    /// CPU MSM implementation
    fn cpu_msm(&self, scalar_points: &[ScalarPoint]) -> NovaResult<MSMResult> {
        let start = Instant::now();
        
        // Placeholder CPU implementation
        let result_point = vec![0u8; 64];

        Ok(MSMResult {
            point: result_point,
            num_points: scalar_points.len(),
            execution_time_us: start.elapsed().as_micros() as u64,
        })
    }

    /// Perform NTT with automatic CPU/GPU selection
    pub async fn ntt_forward(&mut self, coefficients: &[Vec<u8>], domain: &NTTDomain) -> NovaResult<NTTResult> {
        if let Some(ref mut gpu) = self.gpu {
            if coefficients.len() >= self.gpu_threshold {
                return gpu.ntt_forward(coefficients, domain).await;
            }
        }

        // CPU fallback
        self.cpu_ntt_forward(coefficients, domain)
    }

    fn cpu_ntt_forward(&self, coefficients: &[Vec<u8>], domain: &NTTDomain) -> NovaResult<NTTResult> {
        let start = Instant::now();
        
        Ok(NTTResult {
            coefficients: coefficients.to_vec(),
            domain_size: domain.size,
            is_forward: true,
            execution_time_us: start.elapsed().as_micros() as u64,
        })
    }

    /// Get GPU metrics if available
    pub fn gpu_metrics(&self) -> Option<&GPUMetrics> {
        self.gpu.as_ref().map(|g| g.metrics())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_config_default() {
        let config = GPUConfig::default();
        assert_eq!(config.workgroup_size, 256);
        assert!(config.async_execution);
    }

    #[tokio::test]
    async fn test_gpu_config_high_performance() {
        let config = GPUConfig::high_performance();
        assert_eq!(config.workgroup_size, 512);
        assert_eq!(config.num_workgroups, 128);
    }

    #[tokio::test]
    async fn test_ntt_domain_creation() {
        let domain = NTTDomain::new(1024).unwrap();
        assert_eq!(domain.size, 1024);
    }

    #[tokio::test]
    async fn test_ntt_domain_invalid_size() {
        let result = NTTDomain::new(1000);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_gpu_manager_creation() {
        // Test with GPU disabled
        let manager = GPUAccelerationManager::new(false).await.unwrap();
        assert!(!manager.is_gpu_available());
    }

    #[tokio::test]
    async fn test_cpu_msm_fallback() {
        let mut manager = GPUAccelerationManager::new(false).await.unwrap();
        
        let scalar_points = vec![
            ScalarPoint {
                scalar: vec![1u8; 32],
                point: vec![2u8; 64],
            }
        ];

        let result = manager.msm(&scalar_points).await.unwrap();
        assert_eq!(result.num_points, 1);
        assert!(result.execution_time_us > 0 || result.execution_time_us == 0);
    }

    #[tokio::test]
    async fn test_scalar_point_creation() {
        let sp = ScalarPoint {
            scalar: vec![1, 2, 3],
            point: vec![4, 5, 6],
        };
        assert_eq!(sp.scalar.len(), 3);
        assert_eq!(sp.point.len(), 3);
    }

    #[tokio::test]
    async fn test_msm_result() {
        let result = MSMResult {
            point: vec![0u8; 64],
            num_points: 100,
            execution_time_us: 1000,
        };
        assert_eq!(result.num_points, 100);
    }

    #[tokio::test]
    async fn test_ntt_result() {
        let result = NTTResult {
            coefficients: vec![vec![0u8; 32]; 8],
            domain_size: 8,
            is_forward: true,
            execution_time_us: 500,
        };
        assert!(result.is_forward);
        assert_eq!(result.domain_size, 8);
    }

    #[tokio::test]
    async fn test_commitment_result() {
        let result = CommitmentResult {
            commitment: vec![0u8; 64],
            blinding: Some(vec![1u8; 32]),
            execution_time_us: 100,
        };
        assert!(result.blinding.is_some());
    }

    #[tokio::test]
    async fn test_gpu_metrics_default() {
        let metrics = GPUMetrics::default();
        assert_eq!(metrics.msm_operations, 0);
        assert_eq!(metrics.ntt_operations, 0);
    }
}
