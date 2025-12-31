# Bitsliced-Op

A collection of bitsliced operations 🚀

## Background

This Rust crate includes [bitsliced](https://timtaubert.de/blog/2018/08/bitslicing-an-introduction/) operations such as addition. These operations are mainly meant to be used in reduction functions (though they can be used for other purposes as well) and make generating rainbow tables more efficient when paired with bitsliced hashing (as results don't need to be transposed before passing it to a reduction function).

✨ Support for SIMD (AVX and other wide registers) is available and implemented through the use of the [wide](https://crates.io/crates/wide) crate. This allows up to 512 parallel operations depending on hardware.

## Usage

Bitsliced addition:

```rust
//construct bitsliced form of 1 in binary (integers are columns, so we have 256 columns with the last bit/row set to 1)
let mut a = [ZERO; 64];
a[63] = ALL_ONES;
let mut b = [ZERO; 64];
b[63] = ALL_ONES;
//add columns together (so column with index 0 of a will get added to column with index 0 of b)
let sum = bitsliced_add(&a, &b);
```

Or more efficient inline:

```rust
let mut a = [ZERO; 64];
a[63] = ALL_ONES;
let mut b = [ZERO; 64];
b[63] = ALL_ONES;
bitsliced_add_inline(&mut a, &b);
//result is stored in a
//this is more efficient and useful if you don't need to save the previous contents of 'a'
```

Add the same integer to all columns:

```rust
let mut a = [ZERO; 64];
a[63] = ALL_ONES;
//add 1 to all columns
bitsliced_add_single(&a, 1);
```

Reduction function for DES:

```rust
let mut H = [...]; // hashes in bitsliced format
let index = 0; //current index in rainbow chain, most likely just your loop counter

//reduction function does (HASH+INDEX)%MAX_SIZE for every column
let reduced = des_reduction(H, index);
```
