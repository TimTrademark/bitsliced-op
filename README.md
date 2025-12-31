# Bitsliced-Op

A collection of bitsliced operations 🚀

## Background

This Rust crate includes [bitsliced](https://timtaubert.de/blog/2018/08/bitslicing-an-introduction/) operations such as addition. These operations are meant to be used in reduction functions and make generating rainbow tables more efficient when paired with bitsliced hashing (as results don't need to be transposed before passing it to a reduction function).

✨ Support for SIMD (AVX and other wide registers) is available and implemented through the use of the [wide](https://crates.io/crates/wide) crate. This allows up to 512 parallel operations depending on hardware.

## Usage

Bitsliced addition:

```rust
//construct bitsliced form of 1 in binary (integers are columns, so we have 256 columns with the last bit set to 1)
let all_ones = u64x8::splat(0xFFFFFFFFFFFFFFFF);
let zero = u64x8::ZERO;
let mut a = [zero; 64];
a[63] = all_ones;
let sum = bitsliced_add(&a, &b);
```

Or more efficient inline:

```rust
//construct bitsliced form of 1 in binary (integers are columns, so we have 256 columns with the last bit set to 1)
let all_ones = u64x8::splat(0xFFFFFFFFFFFFFFFF);
let zero = u64x8::ZERO;
let mut a = [zero; 64];
a[63] = all_ones;
bitsliced_add(&mut a, &b);
//result is stored in a
//this is more efficient and useful if you don't need to save the previous contents of 'a'
```
