# JPEG-Compression-Pipeline
This Python script implements JPEG compression and decompression using discrete cosine transform (DCT) and Huffman coding. It takes an input image, converts it to YCbCr color space, performs DCT on 8x8 blocks, quantizes the coefficients, applies Huffman coding for both DC and AC coefficients, and writes the compressed data to files. The compressed data can then be decompressed to reconstruct the image.

## Code Overview

The script is organized into the following sections:

### DCT and Quantization

- The script defines functions for forward and inverse DCT.
- Quantization matrices for luminance (Y), and chrominance (Cr, Cb) are used to quantize the DCT coefficients.

### Zigzag Scan

- Functions `block_to_zigzag` and `zigzag_to_block` convert 8x8 blocks to and from zigzag order.

### Run-Length Encoding (RLE)

- The `rle` function performs Run-Length Encoding on the quantized DCT coefficients.

### Huffman Coding

- Huffman encoding is implemented using a custom `node` class for the Huffman tree.
- Functions `huffman_dc_coeff` and `huffman_ac_coeff` generate Huffman trees for DC and AC coefficients.
- The `bin_codes_to_rle` function decodes Huffman-coded binary strings into RLE format.

### Transform and Inverse Transform

- The script includes functions for the forward and inverse DCT transforms on image blocks.

### Color Space Conversion

- Functions `color_1_to_2` and `color_2_to_1` convert the image between RGB and YCbCr color spaces.

### Compression and Decompression

- The script reads an image, applies compression, writes the compressed data to files (`Y.txt`, `Cr.txt`, `Cb.txt`), and calculates the compression ratio.
- Decompression involves reading the compressed data, applying Huffman decoding, and reconstructing the image.

## Usage

1. Run the script on an input image (e.g., 'Lena.png').
2. The compressed data will be written to three separate text files (`Y.txt`, `Cr.txt`, `Cb.txt`).
3. To decompress, read the compressed data from these files and run the script.

## Results

The script outputs a compressed image ('out_image.jpg') and prints the compression ratio.

Feel free to use and modify the code for your own projects or contribute to its improvement!
