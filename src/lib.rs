use std::{
    arch::x86_64::{
        __m256i, __m512i, _mm256_and_si256, _mm256_or_si256, _mm256_ternarylogic_epi32,
        _mm256_test_epi64_mask, _mm256_testz_si256, _mm256_xor_si256, _mm512_and_si512,
        _mm512_or_si512, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_ternarylogic_epi32,
        _mm512_test_epi64_mask, _mm512_xor_si512,
    },
    io::{Error, ErrorKind},
    sync::OnceLock,
};

use wide::u64x8;

use crate::transpose::{transpose_scalar, transpose_scalar_inline};

pub mod benchmark;
pub mod transpose;

pub const ALL_ONES: u64x8 = u64x8::splat(0xFFFFFFFFFFFFFFFF);
pub const ZERO: u64x8 = u64x8::ZERO;

pub fn splat(n: u64) -> u64x8 {
    u64x8::splat(n)
}

//expects the input to be in bitsliced form e.g integers are columns, not rows
//last row is LSB
pub fn bitsliced_add(a: &[u64x8; 64], b: &[u64x8; 64]) -> [u64x8; 64] {
    let mut carry = u64x8::ZERO;
    let mut sum = [u64x8::ZERO; 64];
    for i in (0..64).rev() {
        let res = calc_sum_carry(a[i], b[i], carry);
        sum[i] = res.0;
        //only set carry if we haven't reached the end yet, we currently ignore overflows
        carry = res.1;
    }
    sum
}

pub fn bitsliced_add_single(a: &[u64x8; 64], b: u64) -> [u64x8; 64] {
    let mut carry = u64x8::ZERO;
    let mut sum = [u64x8::ZERO; 64];
    for i in (0..64).rev() {
        let shift_right = 63 - i;
        let current_bit = (b >> shift_right) & 1;
        let b_i = if current_bit == 1 { ALL_ONES } else { ZERO };
        let res = calc_sum_carry(a[i], b_i, carry);
        sum[i] = res.0;
        //only set carry if we haven't reached the end yet, we currently ignore overflows
        carry = res.1;
    }
    sum
}

pub fn bitsliced_add_inline(a: &mut [u64x8; 64], b: &[u64x8; 64]) {
    let mut carry = u64x8::ZERO;
    for i in (0..64).rev() {
        let res = calc_sum_carry(a[i], b[i], carry);
        a[i] = res.0;
        //only set carry if we haven't reached the end yet, we currently ignore overflows
        carry = res.1;
    }
}

pub fn bitsliced_add_single_inline(a: &mut [u64x8; 64], b: u64) {
    let mut carry = u64x8::ZERO;
    for i in (0..64).rev() {
        let shift_right = 63 - i;
        let current_bit = (b >> shift_right) & 1;
        let b_i = if current_bit == 1 { ALL_ONES } else { ZERO };
        let res = calc_sum_carry(a[i], b_i, carry);
        a[i] = res.0;
        //only set carry if we haven't reached the end yet, we currently ignore overflows
        carry = res.1;
    }
}

fn calc_sum_carry(a: u64x8, b: u64x8, carry: u64x8) -> (u64x8, u64x8) {
    let sum = a ^ b ^ carry;
    let next_carry = (a & b) | (carry & (a ^ b));
    (sum, next_carry)
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

const M2_ONES: __m256i = unsafe { std::mem::transmute([!0u64; 4]) };
const M2_ZERO: __m256i = unsafe { std::mem::transmute([0u64; 4]) };

#[target_feature(enable = "avx2")]
pub unsafe fn bitsliced_add_single_inline_avx_2(a: &mut [__m256i; 64], b: u64) {
    let mut carry = M2_ZERO;
    let max_bit_pos = 64 - (b.leading_zeros() as usize);
    for i in (0..64).rev() {
        let bit_index = 63 - i;
        //break early to save cpu cycles
        if bit_index >= max_bit_pos {
            if _mm256_testz_si256(carry, carry) != 0 {
                break;
            }
        }

        let current_bit = if ((b >> bit_index) & 1) == 1 {
            M2_ONES
        } else {
            M2_ZERO
        };

        let a_orig = a[i];

        let xor_ab = _mm256_xor_si256(a_orig, current_bit);
        let and_ab = _mm256_and_si256(a_orig, current_bit);
        a[i] = _mm256_xor_si256(xor_ab, carry);

        carry = _mm256_or_si256(and_ab, _mm256_and_si256(carry, xor_ab));
    }
}

//this function only works when calculating the module with a number of the power of two
//currently only supports a single modulo operation for all integers
//example: if you want to calculate the modulo with 2^56, pass 56 to k
pub fn bitsliced_modulo_power_of_two(a: &[u64x8; 64], k: usize) -> Result<[u64x8; 64], Error> {
    if k > 64 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "k must be <= 64 for bitsliced modulo",
        ));
    }
    let mut out = [u64x8::splat(0); 64];
    let start: usize = 64 - k;
    out[start..].copy_from_slice(&a[start..]);

    Ok(out)
}

pub fn bitsliced_modulo_power_of_two_inline(a: &mut [u64x8; 64], k: usize) -> Result<(), Error> {
    if k > 64 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "k must be <= 64 for bitsliced modulo",
        ));
    }
    let end: usize = 64 - k;
    for i in 0..end {
        a[i] = u64x8::splat(0);
    }

    Ok(())
}

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

//reduction function: (H+I)%MAX_SIZE
//H=Hash,I=Index in chain,MAX_SIZE=Max size of output in power of 2
pub fn des_reduction(h: &[u64x8; 64], i: u64) -> [u64x8; 64] {
    let mut sum = bitsliced_add_single(h, i);
    bitsliced_modulo_power_of_two_inline(&mut sum, 56).unwrap();
    sum
}

pub fn des_reduction_inline(h: &mut [u64x8; 64], i: u64) {
    bitsliced_add_single_inline(h, i);
    bitsliced_modulo_power_of_two_inline(h, 56).unwrap();
}

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
    use std::arch::x86_64::{_mm256_setzero_si256, _mm256_storeu_si256, _mm512_storeu_si512};

    use super::*;

    #[test]
    fn test_add_works() {
        let mut a = [ZERO; 64];
        a[63] = ALL_ONES;
        let mut b = [ZERO; 64];
        b[63] = ALL_ONES;
        let sum = bitsliced_add(&a, &b);
        assert_eq!(sum[63], ZERO);
        assert_eq!(sum[62], ALL_ONES);
        for i in 0..62 {
            assert_eq!(sum[i], ZERO);
        }
    }

    #[test]
    fn test_add_single_works() {
        let mut a = [ZERO; 64];
        a[63] = ALL_ONES;
        let sum = bitsliced_add_single(&a, 1);
        assert_eq!(sum[63], ZERO);
        assert_eq!(sum[62], ALL_ONES);
        for i in 0..62 {
            assert_eq!(sum[i], ZERO);
        }
    }

    #[test]
    fn test_add_inline_works() {
        let mut a = [ZERO; 64];
        a[63] = ALL_ONES;
        let mut b = [ZERO; 64];
        b[63] = ALL_ONES;
        bitsliced_add_inline(&mut a, &b);
        assert_eq!(a[63], ZERO);
        assert_eq!(a[62], ALL_ONES);
        for i in 0..62 {
            assert_eq!(a[i], ZERO);
        }
    }

    #[test]
    fn test_add_single_inline_works() {
        let mut a = [ZERO; 64];
        a[63] = ALL_ONES;
        bitsliced_add_single_inline(&mut a, 1);
        assert_eq!(a[63], ZERO);
        assert_eq!(a[62], ALL_ONES);
        for i in 0..62 {
            assert_eq!(a[i], ZERO);
        }
    }

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
    fn test_modulo_works() {
        let a = [ALL_ONES; 64];
        let res = bitsliced_modulo_power_of_two(&a, 56).unwrap();
        for i in 0..8 {
            assert_eq!(res[i], ZERO);
        }
        for i in 8..64 {
            assert_eq!(res[i], ALL_ONES);
        }
    }

    #[test]
    fn test_modulo_inline_works() {
        let mut a = [ALL_ONES; 64];
        let _ = bitsliced_modulo_power_of_two_inline(&mut a, 56).unwrap();
        for i in 0..8 {
            assert_eq!(a[i], ZERO);
        }
        for i in 8..64 {
            assert_eq!(a[i], ALL_ONES);
        }
    }

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
