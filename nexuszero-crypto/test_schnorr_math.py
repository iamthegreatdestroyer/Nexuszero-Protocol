#!/usr/bin/env python3
"""
Manual test to verify Schnorr protocol math
"""

# Simple values for testing
g = 2
x = 42  # secret
p = 255  # modulus (using vec![0xFF; 32] which is close to 2^256-1, but simplifying)

# Compute public value: h = g^x mod p
h = pow(g, x, p)
print(f"Generator g = {g}")
print(f"Secret x = {x}")
print(f"Modulus p = {p}")
print(f"Public h = g^x mod p = {h}")
print()

# Prover: Generate commitment
r = 17  # blinding factor
t = pow(g, r, p)
print(f"Blinding r = {r}")
print(f"Commitment t = g^r mod p = {t}")
print()

# Challenge (simplified)
c = 3
print(f"Challenge c = {c}")
print()

# Prover: Compute response
# s = r + c*x (we need to be careful about modulus here)
# In a proper Schnorr proof, this would be mod q where q is the order of g
# For simplicity, let's just use integer arithmetic
s = r + c * x
print(f"Response s = r + c*x = {r} + {c}*{x} = {s}")
print()

# Verifier: Check g^s = t * h^c (mod p)
left_side = pow(g, s, p)
right_side = (t * pow(h, c, p)) % p
print(f"Verification:")
print(f"  Left side: g^s mod p = {g}^{s} mod {p} = {left_side}")
print(f"  Right side: t * h^c mod p = {t} * {h}^{c} mod {p} = {right_side}")
print(f"  Match: {left_side == right_side}")
