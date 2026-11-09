use std::{
    arch::x86_64::{
        __m256i, __m512i, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_or_si256, _mm256_set_epi64x,
        _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_slli_epi64, _mm256_srli_epi64,
        _mm256_testz_si256, _mm256_xor_si256, _mm512_ternarylogic_epi32, _mm512_test_epi64_mask,
    },
    io::{Error, ErrorKind},
    sync::OnceLock,
};

use crate::transpose::{transpose_scalar, transpose_scalar_inline};

pub mod benchmark;
pub mod transpose;

//reduction function: (H+I)%MAX_SIZE
//H=Hash,I=Index in chain,MAX_SIZE=Max size of output in power of 2
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn des_reduction_inline_avx_512(h: &mut [__m512i; 64], i: u64) {
    unsafe { bitsliced_add_single_inline_avx_512(h, i) };
    bitsliced_modulo_power_of_two_inline_avx_512(h, 56).unwrap();
}

#[target_feature(enable = "avx2")]
pub unsafe fn des_reduction_inline_avx_2(h: &mut [__m256i; 64], i: u64) {
    unsafe { bitsliced_add_single_inline_avx_2(h, i) };
    bitsliced_modulo_power_of_two_inline_avx_2(h, 56).unwrap();
}

#[target_feature(enable = "avx2")]
pub unsafe fn des_reduction_grouped_avx_2<const N: usize>(h: &mut [__m256i; 64], i: &[u64; N]) {
    unsafe { bitsliced_add_grouped_avx_2::<N>(h, i) };
    bitsliced_modulo_power_of_two_inline_avx_2(h, 56).unwrap();
}

const M512_ONES: __m512i = unsafe { std::mem::transmute([!0u64; 8]) };
const M512_ZERO: __m512i = unsafe { std::mem::transmute([0u64; 8]) };

#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn bitsliced_add_single_inline_avx_512(a: &mut [__m512i; 64], b: u64) {
    let mut carry = M512_ZERO;
    let max_bit_pos = 64 - (b.leading_zeros() as usize);
    for i in (0..64).rev() {
        let bit_index = 63 - i;
        //break early to save cpu cycles
        if bit_index >= max_bit_pos {
            if _mm512_test_epi64_mask(carry, carry) == 0 {
                break;
            }
        }

        let current_bit = if ((b >> bit_index) & 1) == 1 {
            M512_ONES
        } else {
            M512_ZERO
        };

        let a_orig = a[i];

        a[i] = _mm512_ternarylogic_epi32(a_orig, current_bit, carry, 0x96);

        carry = _mm512_ternarylogic_epi32(a_orig, current_bit, carry, 0xE8);
    }
}

const M2_ZERO: __m256i = unsafe { std::mem::transmute([0u64; 4]) };

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bitsliced_add_single_inline_avx_2(a: &mut [__m256i; 64], b: u64) {
    let mut carry = M2_ZERO;

    // highest bit of b that's set, clamped to min 40
    let max_bit_pos = (64 - b.leading_zeros() as usize).min(40);
    // smallest i we need the full adder for
    let full_adder_floor = 64 - max_bit_pos;

    // full adder
    for i in (full_adder_floor..64).rev() {
        let bit_index = 63 - i;
        let bit = (b >> bit_index) & 1;
        let current_bit = _mm256_set1_epi64x((bit as i64).wrapping_neg());

        let a_orig = a[i];
        let xor_ab = _mm256_xor_si256(a_orig, current_bit);
        let and_ab = _mm256_and_si256(a_orig, current_bit);
        a[i] = _mm256_xor_si256(xor_ab, carry);
        carry = _mm256_or_si256(and_ab, _mm256_and_si256(carry, xor_ab));
    }

    // half adder (all subsequent bits in b are 0, only half adder is needed)
    for i in (8..full_adder_floor).rev() {
        if _mm256_testz_si256(carry, carry) != 0 {
            break;
        }
        let a_orig = a[i];
        a[i] = _mm256_xor_si256(a_orig, carry);
        carry = _mm256_and_si256(a_orig, carry);
    }
}

/// N = number of groups
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bitsliced_add_grouped_avx_2<const N: usize>(a: &mut [__m256i; 64], b: &[u64; N]) {
    const { assert!(N == 1 || N == 2 || N == 4) };

    // One 64-bit element per 64 columns; groups are replicated across their elements.
    let bv = match N {
        1 => _mm256_set1_epi64x(b[0] as i64),
        2 => _mm256_set_epi64x(b[1] as i64, b[1] as i64, b[0] as i64, b[0] as i64),
        _ => _mm256_set_epi64x(b[3] as i64, b[2] as i64, b[1] as i64, b[0] as i64),
    };

    // Width is set by the largest addend; smaller ones just contribute zero bits above their MSB.
    let b_or = b.iter().fold(0u64, |x, &y| x | y);
    let max_bit_pos = (64 - b_or.leading_zeros() as usize).min(40);
    let full_adder_floor = 64 - max_bit_pos;

    let zero = _mm256_setzero_si256();
    let mut carry = zero;
    let mut cur = bv;

    // full adder, bit 0 first (i = 63)
    for i in (full_adder_floor..64).rev() {
        // lowest bit of each element -> top bit -> all-ones/all-zeros mask
        let current_bit = _mm256_cmpgt_epi64(zero, _mm256_slli_epi64::<63>(cur));
        cur = _mm256_srli_epi64::<1>(cur);

        let a_orig = a[i];
        let xor_ab = _mm256_xor_si256(a_orig, current_bit);
        let and_ab = _mm256_and_si256(a_orig, current_bit);
        a[i] = _mm256_xor_si256(xor_ab, carry);
        carry = _mm256_or_si256(and_ab, _mm256_and_si256(carry, xor_ab));
    }

    // half adder with early exit
    for i in (8..full_adder_floor).rev() {
        if _mm256_testz_si256(carry, carry) != 0 {
            break;
        }
        let a_orig = a[i];
        a[i] = _mm256_xor_si256(a_orig, carry);
        carry = _mm256_and_si256(a_orig, carry);
    }
}

//this function only works when calculating the module with a number of the power of two
//currently only supports a single modulo operation for all integers
//example: if you want to calculate the modulo with 2^56, pass 56 to k
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub fn bitsliced_modulo_power_of_two_inline_avx_512(
    a: &mut [__m512i; 64],
    k: usize,
) -> Result<(), Error> {
    if k > 64 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "k must be <= 64 for bitsliced modulo",
        ));
    }
    let end: usize = 64 - k;
    for i in 0..end {
        a[i] = M512_ZERO
    }

    Ok(())
}
#[target_feature(enable = "avx2")]
pub fn bitsliced_modulo_power_of_two_inline_avx_2(
    a: &mut [__m256i; 64],
    k: usize,
) -> Result<(), Error> {
    if k > 64 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "k must be <= 64 for bitsliced modulo",
        ));
    }
    let end: usize = 64 - k;
    for i in 0..end {
        a[i] = M2_ZERO
    }

    Ok(())
}

static USE_GFNI: OnceLock<bool> = OnceLock::new();

//transpose 64x64 bit matrix
//use gfni if the cpu supports it, fallback to scalar if it doesn't
pub fn transpose_64x64(input: &[u64; 64]) -> [u64; 64] {
    if *USE_GFNI.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("gfni")
                && std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vbmi")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }) {
        unsafe { crate::transpose::transpose_gfni(input) }
    } else {
        transpose_scalar(input)
    }
}

//currently only supports scalar
pub fn transpose_64x64_inline(input: &mut [u64; 64]) {
    transpose_scalar_inline(input);
}

#[cfg(test)]
mod tests {
    use std::arch::x86_64::{
        _mm256_setzero_si256, _mm256_storeu_si256, _mm512_setzero_si512, _mm512_storeu_si512,
    };

    use super::*;

    #[test]
    fn test_add_single_inline_avx_512_works() {
        let mut a = [unsafe { _mm512_setzero_si512() }; 64];
        a[63] = unsafe { std::mem::transmute([!0u64; 8]) };
        unsafe { bitsliced_add_single_inline_avx_512(&mut a, 1) };
        let mut arr = [0u64; 8];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, a[63]) };
        assert_eq!(arr[0], 0);
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, a[62]) };
        assert_eq!(arr[0], 0xFFFFFFFFFFFFFFFF);
        for i in 0..62 {
            unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, a[i]) };
            assert_eq!(arr[0], 0);
        }
    }

    #[test]
    fn test_add_single_inline_avx_2_works() {
        let mut a = [unsafe { _mm256_setzero_si256() }; 64];
        a[63] = unsafe { std::mem::transmute([!0u64; 4]) };
        unsafe { bitsliced_add_single_inline_avx_2(&mut a, 1) };
        let mut arr = [0u64; 8];
        unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut _, a[63]) };
        assert_eq!(arr[0], 0);
        unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut _, a[62]) };
        assert_eq!(arr[0], 0xFFFFFFFFFFFFFFFF);
        for i in 0..62 {
            unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut _, a[i]) };
            assert_eq!(arr[0], 0);
        }
    }

    #[test]
    fn test_reduction_avx_2_works() {
        const MAX: u64 = (2 as u64).pow(56);
        let a = [[MAX - 1; 256]];

        let mut transposed = unsafe { transpose_avx_2::<1, 64>(&a) };
        unsafe { des_reduction_inline_avx_2(&mut transposed, 11) };
        let back = unsafe { transpose_back_avx_2::<1, 64>(&transposed) };

        let expected = (MAX - 1).wrapping_add(11) % MAX;
        for i in 0..256 {
            assert_eq!(back[0][i], expected);
        }
    }

    #[test]
    fn test_reduction_avx_2_high_works() {
        const MAX: u64 = (2 as u64).pow(56);
        let a = [[u64::MAX - 1; 256]];

        let mut transposed = unsafe { transpose_avx_2::<1, 64>(&a) };
        unsafe { des_reduction_inline_avx_2(&mut transposed, 11) };
        let back = unsafe { transpose_back_avx_2::<1, 64>(&transposed) };

        let expected = (u64::MAX - 1).wrapping_add(11) % MAX;
        for i in 0..256 {
            assert_eq!(back[0][i], expected);
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn transpose_avx_2<const T: usize, const L: usize>(
        blocks: &[[u64; 256]; T],
    ) -> [__m256i; L] {
        let mut outputs = [unsafe { core::mem::zeroed() }; L];

        for t in 0..T {
            let t0 = transpose_64x64(&blocks[t][0..64].try_into().unwrap());
            let t1 = transpose_64x64(&blocks[t][64..128].try_into().unwrap());
            let t2 = transpose_64x64(&blocks[t][128..192].try_into().unwrap());
            let t3 = transpose_64x64(&blocks[t][192..256].try_into().unwrap());

            let start = t * 64;
            let end = start + 64;
            for i in start..end {
                outputs[i] =
                    _mm256_set_epi64x(t0[i] as i64, t1[i] as i64, t2[i] as i64, t3[i] as i64);
            }
        }

        outputs
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn transpose_back_avx_2<const T: usize, const L: usize>(
        outputs_simd: &[__m256i; L],
    ) -> [[u64; 256]; T] {
        let mut blocks_transposed = [[0u64; 256]; T];

        for t in 0..T {
            for bit_idx in 0..64 {
                let reg = outputs_simd[bit_idx];

                let mut lanes = [0u64; 4];
                unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut _, reg) };

                // lanes[3-b] recovers t0..t3 given t0 is stored at the MSB
                blocks_transposed[t][bit_idx] = lanes[3];
                blocks_transposed[t][64 + bit_idx] = lanes[2];
                blocks_transposed[t][128 + bit_idx] = lanes[1];
                blocks_transposed[t][192 + bit_idx] = lanes[0];
            }

            for b in 0..4 {
                let block: &mut [u64; 64] = (&mut blocks_transposed[t][b * 64..b * 64 + 64])
                    .try_into()
                    .unwrap();
                transpose_64x64_inline(block);
            }
        }

        blocks_transposed
    }

    #[test]
    fn test_add_grouped_avx_2_works() {
        let mut a = [[0u64; 256]];
        for i in 0..128 {
            a[0][i] = 1;
        }
        let mut transposed = unsafe { transpose_avx_2::<1, 64>(&a) };
        unsafe { bitsliced_add_grouped_avx_2(&mut transposed, &[1u64; 2]) };
        let back = unsafe { transpose_back_avx_2::<1, 64>(&transposed) };
        for i in 0..128 {
            assert_eq!(back[0][i], 2);
        }
        for i in 128..256 {
            assert_eq!(back[0][i], 1);
        }
    }

    #[test]
    fn test_reduction_grouped_avx_2_works() {
        const MAX: u64 = (2 as u64).pow(56);
        let mut a = [[MAX - 1; 256]];

        let mut transposed = unsafe { transpose_avx_2::<1, 64>(&a) };
        unsafe { des_reduction_grouped_avx_2(&mut transposed, &[11u64, 22u64]) };
        let back = unsafe { transpose_back_avx_2::<1, 64>(&transposed) };

        let expected_lower_half = (MAX - 1).wrapping_add(11) % MAX;
        let expected_upper_half = (MAX - 1).wrapping_add(22) % MAX;
        for i in 0..128 {
            assert_eq!(back[0][i], expected_lower_half);
        }
        for i in 128..256 {
            assert_eq!(back[0][i], expected_upper_half);
        }
    }

    const M2_ONES: __m256i = unsafe { std::mem::transmute([!0u64; 4]) };

    #[test]
    fn test_modulo_inline_avx_2_works() {
        unsafe {
            let mut a = [M2_ONES; 64];
            let _ = bitsliced_modulo_power_of_two_inline_avx_2(&mut a, 56).unwrap();

            let zero_raw: [u8; 32] = std::mem::transmute(M2_ZERO);
            let ones_raw: [u8; 32] = std::mem::transmute(M2_ONES);

            for i in 0..8 {
                let actual: [u8; 32] = std::mem::transmute(a[i]);
                assert_eq!(actual, zero_raw, "Index {} should be ZERO", i);
            }
            for i in 8..64 {
                let actual: [u8; 32] = std::mem::transmute(a[i]);
                assert_eq!(actual, ones_raw, "Index {} should be ONES", i);
            }
        }
    }
}
