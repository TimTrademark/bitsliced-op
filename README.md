# Bitsliced-Op

A collection of bitsliced operations 🚀

## Background

This Rust crate includes [bitsliced](https://timtaubert.de/blog/2018/08/bitslicing-an-introduction/) operations such as addition. These operations are mainly meant to be used in reduction functions (though they can be used for other purposes as well) and make generating rainbow tables more efficient when paired with bitsliced hashing (as results don't need to be transposed before passing it to a reduction function).

✨ Support for SIMD (AVX and other wide registers) is available and implemented for AVX2 and AVX512

## Usage

Bitsliced addition (AVX2 example):

```rust
//ZERO is an empty __m256i register (e.g all bits are 0)
let mut a = [ZERO; 64];
//set last bit to 1 for all columns
a[63] = ALL_ONES;
//add 1 to all columns
bitsliced_add_single_inline_avx_2(&a, 1);
```

Reduction function for DES (AVX2 example):

```rust
let mut H = [...]; // hashes in bitsliced format
let index: u64 = 0; //current index in rainbow chain, most likely just your loop counter

//reduction function does (HASH+INDEX)%MAX_SIZE for every column
let reduced = des_reduction_inline_avx_2(H, index);
```

Additionally, this crate also provides efficient functions to transpose (rows to columns and vice versa) a 64x64 bit matrix (which is needed to get data in bitsliced form):

```rust
let mut a = [ZERO; 64];
a[0] = ALL_ONES;

//first column of 'transposed' is all 1's now
let transposed = transpose_64x64(&a);
```
