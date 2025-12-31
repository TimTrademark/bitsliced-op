use std::{io::Error, io::ErrorKind};

use wide::u64x8;

pub mod benchmark;

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

#[cfg(test)]
mod tests {
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
}
