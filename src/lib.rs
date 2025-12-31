use std::{io::Error, io::ErrorKind};

use wide::u64x8;

pub mod benchmark;
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

pub fn bitsliced_add_inline(a: &mut [u64x8; 64], b: &[u64x8; 64]) {
    let mut carry = u64x8::ZERO;
    for i in (0..64).rev() {
        let res = calc_sum_carry(a[i], b[i], carry);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_works() {
        let all_ones = u64x8::splat(0xFFFFFFFFFFFFFFFF);
        let zero = u64x8::ZERO;
        let mut a = [zero; 64];
        a[63] = all_ones;
        let mut b = [zero; 64];
        b[63] = all_ones;
        let sum = bitsliced_add(&a, &b);
        assert_eq!(sum[63], zero);
        assert_eq!(sum[62], all_ones);
        for i in 0..62 {
            assert_eq!(sum[i], zero);
        }
    }

    #[test]
    fn test_add_inline_works() {
        let all_ones = u64x8::splat(0xFFFFFFFFFFFFFFFF);
        let zero = u64x8::ZERO;
        let mut a = [zero; 64];
        a[63] = all_ones;
        let mut b = [zero; 64];
        b[63] = all_ones;
        bitsliced_add_inline(&mut a, &b);
        assert_eq!(a[63], zero);
        assert_eq!(a[62], all_ones);
        for i in 0..62 {
            assert_eq!(a[i], zero);
        }
    }

    #[test]
    fn test_modulo_works() {
        let all_ones = u64x8::splat(0xFFFFFFFFFFFFFFFF);
        let zero = u64x8::ZERO;
        let a = [all_ones; 64];
        let res = bitsliced_modulo_power_of_two(&a, 56).unwrap();
        for i in 0..8 {
            assert_eq!(res[i], zero);
        }
        for i in 8..64 {
            assert_eq!(res[i], all_ones);
        }
    }

    #[test]
    fn test_modulo_inline_works() {
        let all_ones = u64x8::splat(0xFFFFFFFFFFFFFFFF);
        let zero = u64x8::ZERO;
        let mut a = [all_ones; 64];
        let _ = bitsliced_modulo_power_of_two_inline(&mut a, 56).unwrap();
        for i in 0..8 {
            assert_eq!(a[i], zero);
        }
        for i in 8..64 {
            assert_eq!(a[i], all_ones);
        }
    }
}
