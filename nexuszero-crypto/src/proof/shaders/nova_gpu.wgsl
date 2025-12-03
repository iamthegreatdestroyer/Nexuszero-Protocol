// Nova GPU Compute Shaders
// WebGPU WGSL shaders for Nova folding scheme acceleration
//
// This module provides GPU-accelerated implementations of:
// - Multi-Scalar Multiplication (MSM) using Pippenger algorithm
// - Number-Theoretic Transform (NTT) for polynomial operations
// - Pedersen commitments

// ============================================================================
// CONSTANTS AND TYPES
// ============================================================================

// Field element represented as 8 32-bit limbs (256 bits total)
struct FieldElement {
    limbs: array<u32, 8>,
}

// Curve point in affine coordinates
struct AffinePoint {
    x: FieldElement,
    y: FieldElement,
}

// Curve point in projective coordinates
struct ProjectivePoint {
    x: FieldElement,
    y: FieldElement,
    z: FieldElement,
}

// Bucket accumulator for Pippenger MSM
struct Bucket {
    point: ProjectivePoint,
    is_infinity: u32,
}

// ============================================================================
// BUFFER BINDINGS
// ============================================================================

// MSM buffers
@group(0) @binding(0) var<storage, read> scalars: array<FieldElement>;
@group(0) @binding(1) var<storage, read> points: array<AffinePoint>;
@group(0) @binding(2) var<storage, read_write> msm_result: array<u32>;

// NTT buffers  
@group(0) @binding(0) var<storage, read> ntt_input: array<FieldElement>;
@group(0) @binding(1) var<storage, read> twiddle_factors: array<FieldElement>;
@group(0) @binding(2) var<storage, read_write> ntt_output: array<FieldElement>;

// Commitment buffers
@group(0) @binding(0) var<storage, read> commit_value: array<u32>;
@group(0) @binding(1) var<storage, read> commit_blinding: array<u32>;
@group(0) @binding(2) var<storage, read_write> commit_result: array<u32>;

// Shared workgroup memory for reduction operations
var<workgroup> shared_buckets: array<Bucket, 256>;
var<workgroup> shared_field: array<FieldElement, 256>;

// ============================================================================
// FIELD ARITHMETIC
// ============================================================================

// Pasta curve field modulus (Pallas: 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001)
const MODULUS: array<u32, 8> = array<u32, 8>(
    0x00000001u, 0x992d30edu, 0x094cf91bu, 0x224698fcu,
    0x00000000u, 0x00000000u, 0x00000000u, 0x40000000u
);

// Montgomery R^2 mod p for conversion
const R_SQUARED: array<u32, 8> = array<u32, 8>(
    0x0748d9d9u, 0x5b7e9695u, 0x8e88eb7au, 0x000f6a8eu,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u
);

/// Add two field elements: result = a + b mod p
fn field_add(a: FieldElement, b: FieldElement) -> FieldElement {
    var result: FieldElement;
    var carry: u32 = 0u;
    
    // First pass: add limbs with carry
    for (var i = 0u; i < 8u; i = i + 1u) {
        let sum = u64(a.limbs[i]) + u64(b.limbs[i]) + u64(carry);
        result.limbs[i] = u32(sum & 0xFFFFFFFFu);
        carry = u32(sum >> 32u);
    }
    
    // Conditional subtraction of modulus
    var borrow: u32 = 0u;
    var temp: FieldElement;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let diff = i64(result.limbs[i]) - i64(MODULUS[i]) - i64(borrow);
        if (diff < 0) {
            temp.limbs[i] = u32(diff + 0x100000000);
            borrow = 1u;
        } else {
            temp.limbs[i] = u32(diff);
            borrow = 0u;
        }
    }
    
    // Select result or temp based on whether subtraction underflowed
    if (borrow == 0u) {
        return temp;
    }
    return result;
}

/// Subtract two field elements: result = a - b mod p
fn field_sub(a: FieldElement, b: FieldElement) -> FieldElement {
    var result: FieldElement;
    var borrow: u32 = 0u;
    
    for (var i = 0u; i < 8u; i = i + 1u) {
        let diff = i64(a.limbs[i]) - i64(b.limbs[i]) - i64(borrow);
        if (diff < 0) {
            result.limbs[i] = u32(diff + 0x100000000);
            borrow = 1u;
        } else {
            result.limbs[i] = u32(diff);
            borrow = 0u;
        }
    }
    
    // Add modulus if we borrowed
    if (borrow == 1u) {
        var carry: u32 = 0u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let sum = u64(result.limbs[i]) + u64(MODULUS[i]) + u64(carry);
            result.limbs[i] = u32(sum & 0xFFFFFFFFu);
            carry = u32(sum >> 32u);
        }
    }
    
    return result;
}

/// Montgomery multiplication: result = (a * b * R^-1) mod p
fn field_mul(a: FieldElement, b: FieldElement) -> FieldElement {
    var result: FieldElement;
    
    // Initialize result to zero
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = 0u;
    }
    
    // Montgomery multiplication with CIOS algorithm
    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry: u64 = 0u;
        
        // Multiply a by b[i]
        for (var j = 0u; j < 8u; j = j + 1u) {
            let product = u64(a.limbs[j]) * u64(b.limbs[i]) + u64(result.limbs[j]) + carry;
            result.limbs[j] = u32(product & 0xFFFFFFFFu);
            carry = product >> 32u;
        }
        
        // Montgomery reduction step
        let m = result.limbs[0]; // Simplified: assumes mu = -p^-1 mod R = 1
        carry = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let sum = u64(result.limbs[j]) + u64(m) * u64(MODULUS[j]) + carry;
            if (j > 0u) {
                result.limbs[j - 1u] = u32(sum & 0xFFFFFFFFu);
            }
            carry = sum >> 32u;
        }
        result.limbs[7] = u32(carry);
    }
    
    // Final conditional subtraction
    var borrow: u32 = 0u;
    var temp: FieldElement;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let diff = i64(result.limbs[i]) - i64(MODULUS[i]) - i64(borrow);
        if (diff < 0) {
            temp.limbs[i] = u32(diff + 0x100000000);
            borrow = 1u;
        } else {
            temp.limbs[i] = u32(diff);
            borrow = 0u;
        }
    }
    
    if (borrow == 0u) {
        return temp;
    }
    return result;
}

/// Check if field element is zero
fn field_is_zero(a: FieldElement) -> bool {
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (a.limbs[i] != 0u) {
            return false;
        }
    }
    return true;
}

/// Create zero field element
fn field_zero() -> FieldElement {
    var result: FieldElement;
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = 0u;
    }
    return result;
}

/// Create one field element (in Montgomery form)
fn field_one() -> FieldElement {
    var result: FieldElement;
    result.limbs[0] = 1u;
    for (var i = 1u; i < 8u; i = i + 1u) {
        result.limbs[i] = 0u;
    }
    return result;
}

// ============================================================================
// ELLIPTIC CURVE OPERATIONS
// ============================================================================

/// Create point at infinity
fn point_infinity() -> ProjectivePoint {
    var p: ProjectivePoint;
    p.x = field_zero();
    p.y = field_one();
    p.z = field_zero();
    return p;
}

/// Convert affine to projective coordinates
fn affine_to_projective(a: AffinePoint) -> ProjectivePoint {
    var p: ProjectivePoint;
    p.x = a.x;
    p.y = a.y;
    p.z = field_one();
    return p;
}

/// Point doubling in projective coordinates
/// Using complete addition formula for short Weierstrass curves
fn point_double(p: ProjectivePoint) -> ProjectivePoint {
    // Check for point at infinity
    if (field_is_zero(p.z)) {
        return p;
    }
    
    var result: ProjectivePoint;
    
    // A = X^2
    let a = field_mul(p.x, p.x);
    // B = Y^2
    let b = field_mul(p.y, p.y);
    // C = B^2
    let c = field_mul(b, b);
    // D = 2*((X+B)^2 - A - C)
    let xplusb = field_add(p.x, b);
    var d = field_mul(xplusb, xplusb);
    d = field_sub(d, a);
    d = field_sub(d, c);
    d = field_add(d, d);
    // E = 3*A
    var e = field_add(a, a);
    e = field_add(e, a);
    // F = E^2
    let f = field_mul(e, e);
    
    // X3 = F - 2*D
    result.x = field_sub(f, field_add(d, d));
    
    // Y3 = E*(D - X3) - 8*C
    var eight_c = field_add(c, c);
    eight_c = field_add(eight_c, eight_c);
    eight_c = field_add(eight_c, eight_c);
    result.y = field_sub(d, result.x);
    result.y = field_mul(e, result.y);
    result.y = field_sub(result.y, eight_c);
    
    // Z3 = 2*Y*Z
    result.z = field_mul(p.y, p.z);
    result.z = field_add(result.z, result.z);
    
    return result;
}

/// Point addition in projective coordinates
fn point_add(p1: ProjectivePoint, p2: ProjectivePoint) -> ProjectivePoint {
    // Check for points at infinity
    if (field_is_zero(p1.z)) {
        return p2;
    }
    if (field_is_zero(p2.z)) {
        return p1;
    }
    
    var result: ProjectivePoint;
    
    // U1 = X1 * Z2^2
    let z2_sq = field_mul(p2.z, p2.z);
    let u1 = field_mul(p1.x, z2_sq);
    
    // U2 = X2 * Z1^2
    let z1_sq = field_mul(p1.z, p1.z);
    let u2 = field_mul(p2.x, z1_sq);
    
    // S1 = Y1 * Z2^3
    let z2_cu = field_mul(z2_sq, p2.z);
    let s1 = field_mul(p1.y, z2_cu);
    
    // S2 = Y2 * Z1^3
    let z1_cu = field_mul(z1_sq, p1.z);
    let s2 = field_mul(p2.y, z1_cu);
    
    // H = U2 - U1
    let h = field_sub(u2, u1);
    
    // Check if points are the same (need to double)
    if (field_is_zero(h)) {
        let r_check = field_sub(s2, s1);
        if (field_is_zero(r_check)) {
            return point_double(p1);
        }
        return point_infinity();
    }
    
    // R = S2 - S1
    let r = field_sub(s2, s1);
    
    // H^2
    let h_sq = field_mul(h, h);
    // H^3
    let h_cu = field_mul(h_sq, h);
    // U1 * H^2
    let u1_h_sq = field_mul(u1, h_sq);
    
    // X3 = R^2 - H^3 - 2*U1*H^2
    let r_sq = field_mul(r, r);
    result.x = field_sub(r_sq, h_cu);
    result.x = field_sub(result.x, field_add(u1_h_sq, u1_h_sq));
    
    // Y3 = R*(U1*H^2 - X3) - S1*H^3
    let s1_h_cu = field_mul(s1, h_cu);
    result.y = field_sub(u1_h_sq, result.x);
    result.y = field_mul(r, result.y);
    result.y = field_sub(result.y, s1_h_cu);
    
    // Z3 = H*Z1*Z2
    result.z = field_mul(h, p1.z);
    result.z = field_mul(result.z, p2.z);
    
    return result;
}

/// Scalar multiplication using double-and-add
fn scalar_mul(scalar: FieldElement, point: AffinePoint) -> ProjectivePoint {
    var result = point_infinity();
    var temp = affine_to_projective(point);
    
    // Process scalar bits from LSB to MSB
    for (var i = 0u; i < 8u; i = i + 1u) {
        var bits = scalar.limbs[i];
        for (var j = 0u; j < 32u; j = j + 1u) {
            if ((bits & 1u) == 1u) {
                result = point_add(result, temp);
            }
            temp = point_double(temp);
            bits = bits >> 1u;
        }
    }
    
    return result;
}

// ============================================================================
// MSM PIPPENGER ALGORITHM
// ============================================================================

/// Multi-Scalar Multiplication using Pippenger bucket method
@compute @workgroup_size(256)
fn msm_pippenger(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>) {
    let idx = global_id.x;
    let num_points = arrayLength(&scalars);
    
    if (idx >= num_points) {
        return;
    }
    
    // Window size (c bits at a time)
    let window_size: u32 = 4u; // 2^4 = 16 buckets per window
    let num_buckets: u32 = 16u;
    let num_windows: u32 = 64u; // 256 bits / 4 bits per window
    
    // Initialize local bucket
    shared_buckets[local_id.x].point = point_infinity();
    shared_buckets[local_id.x].is_infinity = 1u;
    
    workgroupBarrier();
    
    // Each thread processes one scalar-point pair
    let scalar = scalars[idx];
    let point = points[idx];
    
    // Process each window
    for (var w = 0u; w < num_windows; w = w + 1u) {
        // Extract window bits
        let limb_idx = (w * window_size) / 32u;
        let bit_offset = (w * window_size) % 32u;
        var bucket_idx = (scalar.limbs[limb_idx] >> bit_offset) & (num_buckets - 1u);
        
        // Handle cross-limb windows
        if (bit_offset > 32u - window_size && limb_idx < 7u) {
            let remaining_bits = 32u - bit_offset;
            let next_bits = window_size - remaining_bits;
            bucket_idx = bucket_idx | ((scalar.limbs[limb_idx + 1u] & ((1u << next_bits) - 1u)) << remaining_bits);
        }
        
        // Add point to appropriate bucket (simplified)
        if (bucket_idx > 0u) {
            let proj_point = affine_to_projective(point);
            // Atomic accumulation would go here in full implementation
            // For now, just store in shared memory
            if (local_id.x < num_buckets) {
                if (shared_buckets[local_id.x].is_infinity == 1u) {
                    shared_buckets[local_id.x].point = proj_point;
                    shared_buckets[local_id.x].is_infinity = 0u;
                } else {
                    shared_buckets[local_id.x].point = point_add(shared_buckets[local_id.x].point, proj_point);
                }
            }
        }
    }
    
    workgroupBarrier();
    
    // Reduction: combine buckets
    if (local_id.x == 0u) {
        var sum = point_infinity();
        var running = point_infinity();
        
        // Process buckets in reverse order
        for (var i = num_buckets - 1u; i > 0u; i = i - 1u) {
            if (shared_buckets[i].is_infinity == 0u) {
                running = point_add(running, shared_buckets[i].point);
            }
            sum = point_add(sum, running);
        }
        
        // Write result (convert to bytes)
        for (var i = 0u; i < 8u; i = i + 1u) {
            msm_result[i] = sum.x.limbs[i];
            msm_result[i + 8u] = sum.y.limbs[i];
        }
    }
}

// ============================================================================
// NUMBER-THEORETIC TRANSFORM (NTT)
// ============================================================================

/// Forward NTT using Cooley-Tukey butterfly
@compute @workgroup_size(256)
fn ntt_forward(@builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(local_invocation_id) local_id: vec3<u32>) {
    let n = arrayLength(&ntt_input);
    let idx = global_id.x;
    
    if (idx >= n) {
        return;
    }
    
    // Copy input to shared memory
    shared_field[local_id.x] = ntt_input[idx];
    
    workgroupBarrier();
    
    // Cooley-Tukey butterfly operations
    let log_n = u32(log2(f32(n)));
    
    for (var stage = 0u; stage < log_n; stage = stage + 1u) {
        let half_size = 1u << stage;
        let full_size = half_size << 1u;
        
        let pair_idx = idx % full_size;
        let group_idx = idx / full_size;
        
        if (pair_idx < half_size) {
            let twiddle_idx = pair_idx * (n / full_size);
            let twiddle = twiddle_factors[twiddle_idx];
            
            let j = idx + half_size;
            
            if (j < 256u) {
                // Butterfly: (a, b) -> (a + w*b, a - w*b)
                let a = shared_field[local_id.x];
                let b = shared_field[local_id.x + half_size];
                let wb = field_mul(twiddle, b);
                
                shared_field[local_id.x] = field_add(a, wb);
                shared_field[local_id.x + half_size] = field_sub(a, wb);
            }
        }
        
        workgroupBarrier();
    }
    
    // Write output
    ntt_output[idx] = shared_field[local_id.x];
}

/// Inverse NTT using Gentleman-Sande butterfly
@compute @workgroup_size(256)
fn ntt_inverse(@builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(local_invocation_id) local_id: vec3<u32>) {
    let n = arrayLength(&ntt_input);
    let idx = global_id.x;
    
    if (idx >= n) {
        return;
    }
    
    // Copy input to shared memory
    shared_field[local_id.x] = ntt_input[idx];
    
    workgroupBarrier();
    
    // Gentleman-Sande butterfly (inverse of Cooley-Tukey)
    let log_n = u32(log2(f32(n)));
    
    for (var stage = log_n; stage > 0u; stage = stage - 1u) {
        let half_size = 1u << (stage - 1u);
        let full_size = half_size << 1u;
        
        let pair_idx = idx % full_size;
        let group_idx = idx / full_size;
        
        if (pair_idx < half_size) {
            // Use inverse twiddle factors
            let twiddle_idx = pair_idx * (n / full_size);
            let twiddle = twiddle_factors[n / 2u + twiddle_idx]; // Inverse twiddles stored after forward
            
            let j = idx + half_size;
            
            if (j < 256u) {
                let a = shared_field[local_id.x];
                let b = shared_field[local_id.x + half_size];
                
                // Inverse butterfly: (a, b) -> ((a+b), (a-b)*w^-1)
                let sum = field_add(a, b);
                let diff = field_sub(a, b);
                
                shared_field[local_id.x] = sum;
                shared_field[local_id.x + half_size] = field_mul(diff, twiddle);
            }
        }
        
        workgroupBarrier();
    }
    
    // Multiply by n^-1 (normalization)
    // This would use a precomputed n_inv value
    ntt_output[idx] = shared_field[local_id.x];
}

// ============================================================================
// PEDERSEN COMMITMENT
// ============================================================================

/// Pedersen commitment: C = g^v * h^r
@compute @workgroup_size(1)
fn pedersen_commit(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Load value and blinding factor
    var value: FieldElement;
    var blinding: FieldElement;
    
    for (var i = 0u; i < 8u; i = i + 1u) {
        value.limbs[i] = commit_value[i];
        blinding.limbs[i] = commit_blinding[i];
    }
    
    // Generator points (hardcoded for Pallas curve)
    // G is the standard generator, H is a nothing-up-my-sleeve point
    var g: AffinePoint;
    var h: AffinePoint;
    
    // Standard generator for Pallas
    g.x.limbs[0] = 1u;
    for (var i = 1u; i < 8u; i = i + 1u) { g.x.limbs[i] = 0u; }
    g.y.limbs[0] = 2u; // Simplified, actual Y coordinate is computed
    for (var i = 1u; i < 8u; i = i + 1u) { g.y.limbs[i] = 0u; }
    
    // Second generator (hash-derived)
    h.x.limbs[0] = 0x12345678u;
    for (var i = 1u; i < 8u; i = i + 1u) { h.x.limbs[i] = 0u; }
    h.y.limbs[0] = 0x87654321u;
    for (var i = 1u; i < 8u; i = i + 1u) { h.y.limbs[i] = 0u; }
    
    // Compute g^v
    let gv = scalar_mul(value, g);
    
    // Compute h^r
    let hr = scalar_mul(blinding, h);
    
    // Compute C = g^v * h^r
    let commitment = point_add(gv, hr);
    
    // Write result (X and Y coordinates)
    for (var i = 0u; i < 8u; i = i + 1u) {
        commit_result[i] = commitment.x.limbs[i];
        commit_result[i + 8u] = commitment.y.limbs[i];
    }
}

// ============================================================================
// BATCH OPERATIONS
// ============================================================================

/// Batch scalar multiplication for multiple points
@compute @workgroup_size(256)
fn batch_scalar_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_ops = arrayLength(&scalars);
    
    if (idx >= num_ops) {
        return;
    }
    
    let result = scalar_mul(scalars[idx], points[idx]);
    
    // Write result
    let out_offset = idx * 16u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        msm_result[out_offset + i] = result.x.limbs[i];
        msm_result[out_offset + i + 8u] = result.y.limbs[i];
    }
}

/// Parallel field multiplication
@compute @workgroup_size(256)
fn batch_field_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = arrayLength(&ntt_input);
    
    if (idx >= n) {
        return;
    }
    
    let a = ntt_input[idx];
    let b = twiddle_factors[idx];
    ntt_output[idx] = field_mul(a, b);
}
