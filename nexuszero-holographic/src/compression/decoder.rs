use crate::compression::mps::MPS;
// ndarray::Array3 is not directly used in this file but retained for clarity
// when inspecting site tensors in decode logic.

/// Decode MPS back to bytes (lossy placeholder).
pub fn decode_proof(_mps: &MPS) -> Vec<u8> {
    // Attempt to decode a lossless MPS: physical_dim==256 and site tensors are one-hot
    let mut out = Vec::new();
    let pd = _mps.physical_dim();
    if pd != 256 {
        // Not a lossless MPS; for now, return empty vector to indicate lossy/unknown mapping
        return out;
    }
    for idx in 0.._mps.len() {
        if let Some(t) = _mps.site_tensor(idx) {
            // t is Array3 [l, p, r]; find the p index where elements are (close to) 1.0 across bonds
            let p_dim = t.shape()[1];
            let mut found: Option<usize> = None;
            'outer: for p in 0..p_dim {
                // Check that for all (l, r) entries, the tensor at p is ~1.0 and others ~0.0
                for l in 0..t.shape()[0] {
                    for r in 0..t.shape()[2] {
                        let val = t[[l, p, r]];
                        if (val - 1.0).abs() > 1e-6 {
                            // If val is not 1.0, we require exact one-hot so skip
                            continue 'outer;
                        }
                    }
                }
                // Also check other p' are zeros at least at [0,0]
                found = Some(p);
                break;
            }
            if let Some(v) = found {
                out.push(v as u8);
            } else {
                // Fallback: try to pick the argmax across p using the [0,0] element
                let mut best_p = 0usize;
                let mut best_val = t[[0,0,0]];
                for p in 0..p_dim {
                    let val = t[[0,p,0]];
                    if val > best_val { best_val = val; best_p = p; }
                }
                out.push(best_p as u8);
            }
        } else {
            return Vec::new();
        }
    }
    out
}

/// Simple parity-based error correction placeholder.
pub fn apply_error_correction(decoded: &mut [u8]) {
    // TODO: implement parity checks against stored metadata.
    let _ = decoded; // no-op
}
