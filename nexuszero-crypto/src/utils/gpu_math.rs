//! GPU-accelerated modular arithmetic operations
//!
//! This module provides hardware-accelerated implementations of modular arithmetic
//! operations using WebGPU, offering significant performance improvements for
//! large-scale cryptographic computations.

use crate::{CryptoError, CryptoResult};
use num_bigint::{BigUint, BigInt};
use std::sync::Arc;

/// GPU-accelerated modular arithmetic context
#[cfg(feature = "gpu")]
pub struct GPUModularMath {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Pipelines for different operations
    montgomery_mul_pipeline: wgpu::ComputePipeline,
    montgomery_exp_pipeline: wgpu::ComputePipeline,
    batch_mod_mul_pipeline: wgpu::ComputePipeline,
    big_int_mod_mul_pipeline: wgpu::ComputePipeline,
}

#[cfg(feature = "gpu")]
impl GPUModularMath {
    /// Create a new GPU modular math context
    pub async fn new() -> CryptoResult<Self> {
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
            .ok_or_else(|| CryptoError::HardwareError("No suitable GPU adapter found".to_string()))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: Some("NexusZero GPU Modular Math"),
                },
                None,
            )
            .await
            .map_err(|e| CryptoError::HardwareError(format!("Failed to create GPU device: {}", e)))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Load and create compute pipelines
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Modular Arithmetic Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../proof/shaders/modular_arithmetic.wgsl").into()),
        });

        let montgomery_mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Montgomery Multiplication Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "montgomery_mul",
        });

        let montgomery_exp_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Montgomery Exponentiation Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "montgomery_exp",
        });

        let batch_mod_mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Batch Modular Multiplication Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "batch_mod_mul",
        });

        let big_int_mod_mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Big Integer Modular Multiplication Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "big_int_mod_mul",
        });

        Ok(Self {
            device,
            queue,
            montgomery_mul_pipeline,
            montgomery_exp_pipeline,
            batch_mod_mul_pipeline,
            big_int_mod_mul_pipeline,
        })
    }

    /// GPU-accelerated Montgomery multiplication for multiple values
    pub async fn montgomery_mul_batch(
        &self,
        a_values: &[u32],
        b_values: &[u32],
        modulus: u32,
        montgomery_r: u32,
        montgomery_r_squared: u32,
        montgomery_r_inv: u32,
    ) -> CryptoResult<Vec<u32>> {
        if a_values.len() != b_values.len() {
            return Err(CryptoError::InvalidInput("Input arrays must have same length".to_string()));
        }

        let batch_size = a_values.len() as u64;

        // Create buffers
        let a_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("A Values Buffer"),
            size: (a_values.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("B Values Buffer"),
            size: (b_values.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let modulus_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modulus Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Montgomery Params Buffer"),
            size: (3 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Buffer"),
            size: (a_values.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload data
        self.queue.write_buffer(&a_buffer, 0, bytemuck::cast_slice(a_values));
        self.queue.write_buffer(&b_buffer, 0, bytemuck::cast_slice(b_values));
        self.queue.write_buffer(&modulus_buffer, 0, bytemuck::bytes_of(&modulus));
        self.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&[montgomery_r, montgomery_r_squared, montgomery_r_inv]));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Montgomery Mul Bind Group"),
            layout: &self.montgomery_mul_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &a_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &b_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &modulus_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &result_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Montgomery Mul Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Montgomery Mul Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.montgomery_mul_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((batch_size / 256 + 1) as u32, 1, 1);
        }

        // Download results
        let result_staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Staging Buffer"),
            size: (a_values.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&result_buffer, 0, &result_staging_buffer, 0, (a_values.len() * std::mem::size_of::<u32>()) as u64);

        self.queue.submit(Some(encoder.finish()));

        // Read results
        let result_slice = result_staging_buffer.slice(..);
        result_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let result_data = result_slice.get_mapped_range();
        let results: Vec<u32> = bytemuck::cast_slice(&result_data).to_vec();

        Ok(results)
    }

    /// GPU-accelerated modular exponentiation
    pub async fn modular_exponentiation(
        &self,
        base: u32,
        exponent: &BigUint,
        modulus: u32,
    ) -> CryptoResult<u32> {
        // Convert exponent to array of u32 (big-endian)
        let exp_bytes = exponent.to_bytes_be();
        let exp_words = exp_bytes.chunks(4)
            .map(|chunk| {
                let mut word = 0u32;
                for (i, &byte) in chunk.iter().enumerate() {
                    word |= (byte as u32) << (24 - i * 8);
                }
                word
            })
            .collect::<Vec<u32>>();

        // Create buffers
        let base_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Base Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let exp_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Exponent Buffer"),
            size: (exp_words.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mod_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modulus Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Exp Result Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload data
        self.queue.write_buffer(&base_buffer, 0, bytemuck::bytes_of(&base));
        self.queue.write_buffer(&exp_buffer, 0, bytemuck::cast_slice(&exp_words));
        self.queue.write_buffer(&mod_buffer, 0, bytemuck::bytes_of(&modulus));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Modular Exp Bind Group"),
            layout: &self.montgomery_exp_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &base_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &exp_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &mod_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &result_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Modular Exp Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Modular Exp Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.montgomery_exp_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        // Download result
        let result_staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Exp Result Staging Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&result_buffer, 0, &result_staging_buffer, 0, std::mem::size_of::<u32>() as u64);

        self.queue.submit(Some(encoder.finish()));

        // Read result
        let result_slice = result_staging_buffer.slice(..);
        result_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let result_data = result_slice.get_mapped_range();
        let result: u32 = *bytemuck::from_bytes(&result_data);

        Ok(result)
    }

    /// GPU-accelerated batch modular multiplication
    pub async fn batch_modular_multiplication(
        &self,
        a_values: &[u32],
        b_values: &[u32],
        moduli: &[u32],
    ) -> CryptoResult<Vec<u32>> {
        if a_values.len() != b_values.len() || a_values.len() != moduli.len() {
            return Err(CryptoError::InvalidInput("All input arrays must have same length".to_string()));
        }

        let batch_size = a_values.len() as u64;

        // Create buffers
        let a_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch A Buffer"),
            size: (a_values.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch B Buffer"),
            size: (b_values.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let moduli_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch Moduli Buffer"),
            size: (moduli.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch Result Buffer"),
            size: (a_values.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload data
        self.queue.write_buffer(&a_buffer, 0, bytemuck::cast_slice(a_values));
        self.queue.write_buffer(&b_buffer, 0, bytemuck::cast_slice(b_values));
        self.queue.write_buffer(&moduli_buffer, 0, bytemuck::cast_slice(moduli));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Batch Mod Mul Bind Group"),
            layout: &self.batch_mod_mul_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &a_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &b_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &moduli_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &result_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Batch Mod Mul Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batch Mod Mul Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.batch_mod_mul_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((batch_size / 256 + 1) as u32, 1, 1);
        }

        // Download results
        let result_staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch Result Staging Buffer"),
            size: (a_values.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&result_buffer, 0, &result_staging_buffer, 0, (a_values.len() * std::mem::size_of::<u32>()) as u64);

        self.queue.submit(Some(encoder.finish()));

        // Read results
        let result_slice = result_staging_buffer.slice(..);
        result_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let result_data = result_slice.get_mapped_range();
        let results: Vec<u32> = bytemuck::cast_slice(&result_data).to_vec();

        Ok(results)
    }
}

#[cfg(feature = "gpu")]
impl Default for GPUModularMath {
    fn default() -> Self {
        // This will panic if called without proper async initialization
        // Use GPUModularMath::new() instead
        panic!("GPUModularMath must be initialized asynchronously with new()");
    }
}

/// Fallback implementation when GPU is not available
#[cfg(not(feature = "gpu"))]
pub struct GPUModularMath;

#[cfg(not(feature = "gpu"))]
impl GPUModularMath {
    pub async fn new() -> CryptoResult<Self> {
        Err(CryptoError::HardwareError("GPU support not compiled in".to_string()))
    }

    pub async fn montgomery_mul_batch(
        &self,
        _a_values: &[u32],
        _b_values: &[u32],
        _modulus: u32,
        _montgomery_r: u32,
        _montgomery_r_squared: u32,
        _montgomery_r_inv: u32,
    ) -> CryptoResult<Vec<u32>> {
        Err(CryptoError::HardwareError("GPU support not available".to_string()))
    }

    pub async fn modular_exponentiation(
        &self,
        _base: u32,
        _exponent: &BigUint,
        _modulus: u32,
    ) -> CryptoResult<u32> {
        Err(CryptoError::HardwareError("GPU support not available".to_string()))
    }

    pub async fn batch_modular_multiplication(
        &self,
        _a_values: &[u32],
        _b_values: &[u32],
        _moduli: &[u32],
    ) -> CryptoResult<Vec<u32>> {
        Err(CryptoError::HardwareError("GPU support not available".to_string()))
    }
}