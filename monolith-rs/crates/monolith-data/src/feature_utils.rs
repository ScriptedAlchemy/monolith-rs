//! Feature utility functions (Python parity for `monolith.native_training.data.feature_utils`).
//!
//! The upstream Python module wraps many TF custom ops. We port a small subset of
//! pure functions that are used in tests and are backend-independent.

/// Extract a feature id (FID) from an integer input and a slot.
///
/// Mirrors TF custom op `ExtractFid`:
/// - Computes `CityHash64(input_bytes)` where `input_bytes` are the in-memory 8 bytes
///   of the `int64` input (native-endian on the running machine; TF kernels run on
///   little-endian CPUs in practice).
/// - Masks hash to 49 bits and ORs with `slot << 48`.
///
/// Python parity test: `extract_fid(185, 4) == 1153447759131936`.
pub fn extract_fid(input: i64, slot: i64) -> i64 {
    let slot_bits = slot << 48;
    let bits_left: u64 = (1u64 << 49) - 1;
    let tmp: u64 = input as u64;
    // TF kernels run on little-endian CPUs in practice; use LE bytes for stable parity.
    let hash = cityhash::city_hash64(&tmp.to_le_bytes());
    ((hash & bits_left) as i64) | slot_bits
}

// =============================================================================
// CityHash64 (minimal port for ExtractFid parity)
// =============================================================================

mod cityhash {
    // Ported from https://github.com/google/cityhash (src/city.cc) with only the
    // helpers needed for `CityHash64`.

    const K0: u64 = 0xc3a5c85c97cb3127;
    const K1: u64 = 0xb492b66fbe98f273;
    const K2: u64 = 0x9ae16a3b2f90404f;

    #[inline]
    fn fetch64(s: &[u8]) -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&s[..8]);
        u64::from_le_bytes(b)
    }

    #[inline]
    fn fetch32(s: &[u8]) -> u32 {
        let mut b = [0u8; 4];
        b.copy_from_slice(&s[..4]);
        u32::from_le_bytes(b)
    }

    #[inline]
    fn rotate(val: u64, shift: u32) -> u64 {
        // Matches city.cc behavior: shift==0 => val, otherwise rotate.
        if shift == 0 {
            val
        } else {
            val.rotate_right(shift)
        }
    }

    #[inline]
    fn shift_mix(val: u64) -> u64 {
        val ^ (val >> 47)
    }

    #[inline]
    fn hash128to64(u: u64, v: u64) -> u64 {
        // city.cc: Hash128to64 from city.h
        const K_MUL: u64 = 0x9ddfea08eb382d69;
        let mut a = (u ^ v).wrapping_mul(K_MUL);
        a ^= a >> 47;
        let mut b = (v ^ a).wrapping_mul(K_MUL);
        b ^= b >> 47;
        b = b.wrapping_mul(K_MUL);
        b
    }

    #[inline]
    fn hash_len16(u: u64, v: u64) -> u64 {
        hash128to64(u, v)
    }

    #[inline]
    fn hash_len16_mul(u: u64, v: u64, mul: u64) -> u64 {
        // Murmur-inspired hashing (city.cc HashLen16(u, v, mul)).
        let mut a = (u ^ v).wrapping_mul(mul);
        a ^= a >> 47;
        let mut b = (v ^ a).wrapping_mul(mul);
        b ^= b >> 47;
        b = b.wrapping_mul(mul);
        b
    }

    fn hash_len0to16(s: &[u8]) -> u64 {
        let len = s.len();
        if len >= 8 {
            let mul = K2.wrapping_add((len as u64).wrapping_mul(2));
            let a = fetch64(s).wrapping_add(K2);
            let b = fetch64(&s[len - 8..]);
            let c = rotate(b, 37).wrapping_mul(mul).wrapping_add(a);
            let d = rotate(a, 25).wrapping_add(b).wrapping_mul(mul);
            return hash_len16_mul(c, d, mul);
        }
        if len >= 4 {
            let mul = K2.wrapping_add((len as u64).wrapping_mul(2));
            let a = fetch32(s) as u64;
            let b = fetch32(&s[len - 4..]) as u64;
            return hash_len16_mul((len as u64).wrapping_add(a << 3), b, mul);
        }
        if len > 0 {
            let a = s[0] as u64;
            let b = s[len >> 1] as u64;
            let c = s[len - 1] as u64;
            let y = a.wrapping_add(b << 8);
            let z = (len as u64).wrapping_add(c << 2);
            return shift_mix(y.wrapping_mul(K2) ^ z.wrapping_mul(K0)).wrapping_mul(K2);
        }
        K2
    }

    fn hash_len17to32(s: &[u8]) -> u64 {
        let len = s.len();
        let mul = K2.wrapping_add((len as u64).wrapping_mul(2));
        let a = fetch64(s).wrapping_mul(K1);
        let b = fetch64(&s[8..]);
        let c = fetch64(&s[len - 8..]).wrapping_mul(mul);
        let d = fetch64(&s[len - 16..]).wrapping_mul(K2);
        hash_len16_mul(
            rotate(a.wrapping_add(b), 43)
                .wrapping_add(rotate(c, 30))
                .wrapping_add(d),
            a.wrapping_add(rotate(b.wrapping_add(K2), 18)).wrapping_add(c),
            mul,
        )
    }

    fn hash_len33to64(s: &[u8]) -> u64 {
        let len = s.len();
        let mul = K2.wrapping_add((len as u64).wrapping_mul(2));
        let mut a = fetch64(s).wrapping_mul(K2);
        let b = fetch64(&s[8..]);
        let c = fetch64(&s[len - 24..]);
        let d = fetch64(&s[len - 32..]);
        let e = fetch64(&s[16..]).wrapping_mul(K2);
        let f = fetch64(&s[24..]).wrapping_mul(9);
        let g = fetch64(&s[len - 8..]);
        let h = fetch64(&s[len - 16..]).wrapping_mul(mul);
        let u = rotate(a.wrapping_add(g), 43)
            .wrapping_add(rotate(b, 30).wrapping_add(c).wrapping_mul(9));
        let v = a.wrapping_add(g) ^ d;
        let v = v.wrapping_add(f).wrapping_add(1);
        let w = (u.wrapping_add(v)).wrapping_mul(mul).swap_bytes().wrapping_add(h);
        let x = rotate(e.wrapping_add(f), 42).wrapping_add(c);
        let y = (v.wrapping_add(w)).wrapping_mul(mul).swap_bytes().wrapping_add(g).wrapping_mul(mul);
        let z = e.wrapping_add(f).wrapping_add(c);
        a = (x.wrapping_add(z)).wrapping_mul(mul).wrapping_add(y).swap_bytes().wrapping_add(b);
        let b2 = shift_mix(z.wrapping_add(a).wrapping_mul(mul).wrapping_add(d).wrapping_add(h))
            .wrapping_mul(mul);
        b2.wrapping_add(x)
    }

    fn weak_hash_len32_with_seeds(w: u64, x: u64, y: u64, z: u64, mut a: u64, mut b: u64) -> (u64, u64) {
        a = a.wrapping_add(w);
        b = rotate(b.wrapping_add(a).wrapping_add(z), 21);
        let c = a;
        a = a.wrapping_add(x);
        a = a.wrapping_add(y);
        b = b.wrapping_add(rotate(a, 44));
        (a.wrapping_add(z), b.wrapping_add(c))
    }

    fn weak_hash_len32_with_seeds_bytes(s: &[u8], a: u64, b: u64) -> (u64, u64) {
        weak_hash_len32_with_seeds(
            fetch64(s),
            fetch64(&s[8..]),
            fetch64(&s[16..]),
            fetch64(&s[24..]),
            a,
            b,
        )
    }

    /// CityHash64 for any length.
    pub fn city_hash64(s: &[u8]) -> u64 {
        let len = s.len();
        if len <= 32 {
            if len <= 16 {
                return hash_len0to16(s);
            }
            return hash_len17to32(s);
        }
        if len <= 64 {
            return hash_len33to64(s);
        }

        // For strings over 64 bytes we hash the end first, and then as we loop we
        // keep 56 bytes of state: v, w, x, y, and z.
        let mut x = fetch64(&s[len - 40..]);
        let mut y = fetch64(&s[len - 16..]).wrapping_add(fetch64(&s[len - 56..]));
        let mut z = hash_len16(fetch64(&s[len - 48..]).wrapping_add(len as u64), fetch64(&s[len - 24..]));
        let mut v = weak_hash_len32_with_seeds_bytes(&s[len - 64..], len as u64, z);
        let mut w = weak_hash_len32_with_seeds_bytes(&s[len - 32..], y.wrapping_add(K1), x);
        x = x.wrapping_mul(K1).wrapping_add(fetch64(s));

        // Decrease len to the nearest multiple of 64, and operate on 64-byte chunks.
        let mut pos = 0usize;
        let mut remaining = (len - 1) & !63usize;
        while remaining != 0 {
            x = rotate(
                x.wrapping_add(y)
                    .wrapping_add(v.0)
                    .wrapping_add(fetch64(&s[pos + 8..])),
                37,
            )
            .wrapping_mul(K1);
            y = rotate(
                y.wrapping_add(v.1)
                    .wrapping_add(fetch64(&s[pos + 48..])),
                42,
            )
            .wrapping_mul(K1);
            x ^= w.1;
            y = y.wrapping_add(v.0).wrapping_add(fetch64(&s[pos + 40..]));
            z = rotate(z.wrapping_add(w.0), 33).wrapping_mul(K1);
            v = weak_hash_len32_with_seeds_bytes(&s[pos..], v.1.wrapping_mul(K1), x.wrapping_add(w.0));
            w = weak_hash_len32_with_seeds_bytes(
                &s[pos + 32..],
                z.wrapping_add(w.1),
                y.wrapping_add(fetch64(&s[pos + 16..])),
            );
            std::mem::swap(&mut z, &mut x);
            pos += 64;
            remaining -= 64;
        }

        hash_len16(
            hash_len16(v.0, w.0)
                .wrapping_add(shift_mix(y).wrapping_mul(K1))
                .wrapping_add(z),
            hash_len16(v.1, w.1).wrapping_add(x),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_fid_matches_python_test_constant() {
        assert_eq!(extract_fid(185, 4), 1153447759131936);
    }
}
