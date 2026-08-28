import struct
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

def extract_mnist_samples(images_path: str, labels_path: str, output_dir: str, num_samples: int, base_index: int, custom_offsets: list):
    """
    Extracts a contiguous block of samples from MNIST IDX files starting from base_index.
    Generates unified .memh, and .coe files for DMA/BRAM loading.
    Renders the first 5 images and any valid user-specified custom offsets as JPEGs.
    """
    # 1. Check constraints
    if not (1 <= num_samples <= 10000):
        raise ValueError("Number of samples to extract must be between 1 and 10,000.")
    if base_index < 0:
        raise ValueError("Base index cannot be negative.")

    images_path = Path(images_path)
    labels_path = Path(labels_path)
    out_dir = Path(output_dir)

    # Completely remove the directory and its contents if it exists
    if out_dir.exists() and out_dir.is_dir():
        print(f"[*] Removing stale directory: {out_dir}")
        shutil.rmtree(out_dir)
        
    # Create a fresh directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse and Slice Labels
    with open(labels_path, 'rb') as lbl_file:
        magic, total_labels = struct.unpack(">II", lbl_file.read(8))
        assert magic == 2049, "Invalid label magic number!"
        if base_index + num_samples > total_labels:
            raise ValueError(f"Requested extraction exceeds total labels ({total_labels}).")
        
        # Seek directly to the starting base index (Header is 8 bytes long)
        lbl_file.seek(8 + base_index)
        labels = np.frombuffer(lbl_file.read(num_samples), dtype=np.uint8, count=num_samples)

    # 2. Parse and Slice Grayscale Images starting from base_index
    with open(images_path, 'rb') as img_file:
        magic, total_images, rows, cols = struct.unpack(">IIII", img_file.read(16))
        assert magic == 2051, "Invalid image magic number!"
        
        # Seek directly to the starting base index (Header is 16 bytes long)
        img_file.seek(16 + (base_index * rows * cols))
        bytes_to_read = num_samples * rows * cols
        images_raw = img_file.read(bytes_to_read)
        images = np.frombuffer(images_raw, dtype=np.uint8).reshape(num_samples, rows, cols)

    print(f"[*] Extracting {num_samples} samples starting from absolute index {base_index}...")

    # 3. Export Unified Hex Files (.memh)
    # Formats each 8-bit value sequentially (00 to FF) into a single master file.
    memh_img_file = out_dir / f"images_hex.memh"
    memh_lbl_file = out_dir / f"labels_hex.memh"
    
    hex_images = [f"{pixel:02X}" for pixel in images.ravel()]
    memh_img_file.write_text("\n".join(hex_images) + "\n", encoding="ascii")
    
    hex_labels = [f"{label:02X}" for label in labels.ravel()]
    memh_lbl_file.write_text("\n".join(hex_labels) + "\n", encoding="ascii")
    
    print(f"  [+] Unified image memh saved to:   {memh_img_file}")
    print(f"  [+] Unified label memh saved to:   {memh_lbl_file}")

    # 4. Export Unified COE File for Vivado BRAM (.coe)
    # Packs four 8-bit pixels/labels into 32-bit words, LSB first.
    coe_data_file = out_dir / f"images_labels_bram.coe"
    words_32bit = []
    
    # 4a. Pack Images
    flat_pixels = images.ravel()
    # 784 pixels per image is perfectly divisible by 4, so no padding is required
    for i in range(0, len(flat_pixels), 4):
        p0 = flat_pixels[i]
        p1 = flat_pixels[i+1]
        p2 = flat_pixels[i+2]
        p3 = flat_pixels[i+3]
        
        # Pack LSB first: p0 is at the lowest 8 bits, p3 is at the highest 8 bits
        word = (p3 << 24) | (p2 << 16) | (p1 << 8) | p0
        words_32bit.append(f"{word:08X}")

    # 4b. Pack Labels
    flat_labels = labels.ravel()
    remainder = len(flat_labels) % 4
    if remainder != 0:
        # Pad with zeros to ensure the final elements complete a 32-bit word
        padding = np.zeros(4 - remainder, dtype=np.uint8)
        flat_labels = np.concatenate((flat_labels, padding))

    for i in range(0, len(flat_labels), 4):
        l0 = flat_labels[i]
        l1 = flat_labels[i+1]
        l2 = flat_labels[i+2]
        l3 = flat_labels[i+3]
        
        # Pack LSB first
        word = (l3 << 24) | (l2 << 16) | (l1 << 8) | l0
        words_32bit.append(f"{word:08X}")

    # 4c. Write to file
    with open(coe_data_file, 'w', encoding="ascii") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        f.write(",\n".join(words_32bit))
        f.write(";\n") # Vivado requires the final entry to end with a semicolon
        
    print(f"  [+] Unified data COE saved to:     {coe_data_file}")

    print("\n[*] Generating visual JPEG renders...")
    
    # 5. Determine which offsets to render
    # Always include up to the first 5 images (offsets 0, 1, 2, 3, 4)
    offsets_to_render = set(range(min(num_samples, 5)))
    
    # Add any valid custom offsets provided by the user
    for offset in custom_offsets:
        if 0 <= offset < num_samples:
            offsets_to_render.add(offset)
        else:
            print(f"  [!] Warning: Custom offset {offset} is out of bounds and will be ignored.")

    # Sort the set so they render in numerical order
    offsets_to_render = sorted(list(offsets_to_render))

    # 6. Export Visual Assets (JPEGs)
    for offset in offsets_to_render:
        img_matrix = images[offset]
        label = labels[offset]
        absolute_index = base_index + offset
        
        base_name = f"offset_{offset}_absIdx_{absolute_index}_digit_{label}"

        pil_img = Image.fromarray(img_matrix, mode='L').convert('RGB')
        
        # Microscopic 28x28 canvases make text unreadable. We upscale to 280x280 
        # using NEAREST neighbor to preserve clean, unblurred pixel boundaries.
        pil_img_large = pil_img.resize((280, 280), resample=Image.NEAREST)
        
        # Initialize drawing canvas and burn the red label text onto it
        draw = ImageDraw.Draw(pil_img_large)
        try:
            # Modern Pillow versions allow direct size configurations on the default font
            draw.text((12, 12), f"Label: {label}", fill=(255, 0, 0), font_size=20)
        except TypeError:
            # Fallback wrapper for older Pillow environments
            draw.text((12, 12), f"Label: {label}", fill=(255, 0, 0))

        jpg_file = out_dir / f"{base_name}.jpg"
        pil_img_large.save(jpg_file)

        print(f"  [+] Generated JPEG: {jpg_file.name}")
        
    # Inform the user how many JPEGs were skipped
    total_skipped = num_samples - len(offsets_to_render)
    if total_skipped > 0:
        print(f"  [!] Skipped generating JPEGs for the remaining {total_skipped} images.")

    print(f"\n[SUCCESS] Check the '{output_dir}' directory for your assets.")


if __name__ == "__main__":
    # Setup the argument parser for command line execution
    parser = argparse.ArgumentParser(description="Extract a contiguous block of MNIST samples.")
    
    # Changed default from 5 to 100 as requested
    parser.add_argument("--num-sample", type=int, default=100, 
                        help="Number of samples to extract (1-10000). Default is 100.")
    parser.add_argument("--base-id", type=int, default=0, 
                        help="Starting absolute index in the dataset. Default is 0.")
    parser.add_argument("--offset", type=int, nargs="*", default=[], 
                        help="Additional custom offsets to render as JPEGs. E.g., --offsets 5 15 73")
    
    args = parser.parse_args()
    
    # Adjust paths if your files are named differently
    MNIST_IMAGES = "t10k-images.idx3-ubyte"
    MNIST_LABELS = "t10k-labels.idx1-ubyte" # Note: make sure this matches your dataset filename!
    OUTPUT_FOLDER = "extracted_samples"
    
    # Execute the extraction using the parsed command line arguments
    extract_mnist_samples(
        MNIST_IMAGES, 
        MNIST_LABELS, 
        OUTPUT_FOLDER, 
        num_samples=args.num_sample, 
        base_index=args.base_id, 
        custom_offsets=args.offset
    )