#!/usr/bin/env python3
"""
extract_mnist_image.py
Extracts a single MNIST image, saves it as JPEG, and generates Verilog/C header files.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MNIST_IMAGES_FILE = "data/train-images.idx3-ubyte"  # raw MNIST image file
OUTPUT_JPEG       = "data/mnist_digit.jpg"          # JPEG preview
OUTPUT_HEADER     = "data/mnist_image_data.h"       # C header for test_main.cpp
OUTPUT_MEMH       = "data/mnist_image_data.memh"    # Verilog hex memory file
DIGIT_INDEX       = 31                              # which image to extract (0..59999)

# ---------------------------------------------------------------------------
# Read MNIST IDX file format
# ---------------------------------------------------------------------------
def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2051:
            raise ValueError(f"Bad magic number: {magic}, expected 2051")
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows   = int.from_bytes(f.read(4), 'big')
        num_cols   = int.from_bytes(f.read(4), 'big')
        buffer = f.read(num_images * num_rows * num_cols)
        data = np.frombuffer(buffer, dtype=np.uint8)
        return data.reshape(num_images, num_rows, num_cols)

# ---------------------------------------------------------------------------
# Extract and save
# ---------------------------------------------------------------------------
images = read_mnist_images(MNIST_IMAGES_FILE)
print(f"Loaded {images.shape[0]} images, each {images.shape[1]}x{images.shape[2]}")

# Extract one image
img = images[DIGIT_INDEX]
print(f"Extracting image #{DIGIT_INDEX}, pixel range [{img.min()}, {img.max()}]")

# Save as JPEG (scale up for visibility, 8x magnification)
pil_img = Image.fromarray(img)
pil_img = pil_img.resize((28 * 8, 28 * 8), Image.NEAREST)
pil_img.save(OUTPUT_JPEG)
print(f"Saved JPEG preview as {OUTPUT_JPEG}")

# Show it (non-blocking if possible)
plt.imshow(img, cmap='gray')
plt.title(f"MNIST Image #{DIGIT_INDEX}")
plt.show(block=False)
plt.pause(1)

# ---------------------------------------------------------------------------
# Generate C header for test_main.cpp
# Each pixel maps to Q8.8 fixed-point: 0.0 = 0x0000, 1.0 = 0x0100
# ---------------------------------------------------------------------------
with open(OUTPUT_HEADER, 'w') as f:
    f.write("// Auto-generated MNIST image data (Q8.8 format)\n")
    f.write(f"// Image index: {DIGIT_INDEX}\n\n")
    f.write("#ifndef MNIST_IMAGE_DATA_H\n")
    f.write("#define MNIST_IMAGE_DATA_H\n\n")
    f.write(f"#define MNIST_IMAGE_INDEX {DIGIT_INDEX}\n")
    f.write("static const int16_t mnist_test_image[784] = {\n    ")
    
    for i, pixel in enumerate(img.flatten()):
        # Normalize 0-255 → 0.0-1.0, then Q8.8: multiply by 256
        q8_8 = int(round(pixel * 256.0 / 255.0))
        if q8_8 > 32767: q8_8 = 32767
        if q8_8 < -32768: q8_8 = -32768
        f.write(f"0x{q8_8 & 0xFFFF:04X}")
        if i < 783:
            f.write(", ")
        if (i + 1) % 10 == 0 and i != 783:
            f.write("\n    ")
    
    f.write("\n};\n\n")
    f.write("#endif // MNIST_IMAGE_DATA_H\n")

print(f"Generated C header: {OUTPUT_HEADER}")

# ---------------------------------------------------------------------------
# Generate Verilog .memh file
# Hex string representations for use with $readmemh()
# ---------------------------------------------------------------------------
with open(OUTPUT_MEMH, 'w') as f:
    # Add an optional comment at the top
    f.write(f"// MNIST Image #{DIGIT_INDEX} - Q8.8 fixed-point format\n")
    
    for pixel in img.flatten():
        # Normalize 0-255 → 0.0-1.0, then Q8.8: multiply by 256
        q8_8 = int(round(pixel * 256.0 / 255.0))
        if q8_8 > 32767: q8_8 = 32767
        if q8_8 < -32768: q8_8 = -32768
        
        # Write exactly 4 hex digits, one per line (no "0x" prefix for .memh)
        f.write(f"{q8_8 & 0xFFFF:04X}\n")

print(f"Generated Verilog memh: {OUTPUT_MEMH}")