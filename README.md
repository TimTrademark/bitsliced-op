# Bitsliced-Op

A collection of bitsliced operations 🚀

## Background

This Rust crate includes [bitsliced](https://timtaubert.de/blog/2018/08/bitslicing-an-introduction/) operations such as addition. These operations are meant to be used in reduction functions and make generating rainbow tables more efficient when paired with bitsliced hashing (as results don't need to be transposed before passing it to a reduction function).

✨ Support for SIMD (AVX and other wide registers) is available and implemented through the use of the [wide](https://crates.io/crates/wide) crate. This allows up to 512 parallel operations depending on hardware.
