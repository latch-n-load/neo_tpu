import struct
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

NUM_SAMPLES = 5

def extract_mnist_samples(images_path: str, labels_path: str, output_dir: str, num_samples: int):
    """
    Extracts a variable number of samples (2-10) from MNIST IDX files.
    Generates unified .bin and .memh files for DMA loading, and individual 
    labeled JPEGs for visual verification.
    """
    # Defensive check for your requirements
    if not (2 <= num_samples <= 10):
        raise ValueError("This script is configured to extract between 2 and 10 images.")

    images_path = Path(images_path)
    labels_path = Path(labels_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse and Slice Labels
    with open(labels_path, 'rb') as lbl_file:
        magic, total_labels = struct.unpack(">II", lbl_file.read(8))
        assert magic == 2049, "Invalid label magic number!"
        # Read only the number of labels requested
        labels = np.frombuffer(lbl_file.read(num_samples), dtype=np.uint8, count=num_samples)

    # 2. Parse and Slice Grayscale Images
    with open(images_path, 'rb') as img_file:
        magic, total_images, rows, cols = struct.unpack(">IIII", img_file.read(16))
        assert magic == 2051, "Invalid image magic number!"
        
        # Read exactly the bytes needed for the requested number of images
        bytes_to_read = num_samples * rows * cols
        images_raw = img_file.read(bytes_to_read)
        images = np.frombuffer(images_raw, dtype=np.uint8).reshape(num_samples, rows, cols)

    print(f"[*] Processing {num_samples} samples...")

    # 3. Export Unified Binary Files (.bin)
    # Writes the entire multi-image/multi-label arrays to single files.
    bin_img_file = out_dir / f"all_{num_samples}_images.bin"
    bin_lbl_file = out_dir / f"all_{num_samples}_labels.bin"
    
    images.tofile(bin_img_file)
    labels.tofile(bin_lbl_file)
    print(f"  [+] Unified image binary saved to: {bin_img_file}")
    print(f"  [+] Unified label binary saved to: {bin_lbl_file}")

    # 4. Export Unified Hex Files (.memh)
    # Formats each 8-bit value sequentially (00 to FF) into a single master file.
    memh_img_file = out_dir / f"all_{num_samples}_images.memh"
    memh_lbl_file = out_dir / f"all_{num_samples}_labels.memh"
    
    hex_images = [f"{pixel:02X}" for pixel in images.ravel()]
    memh_img_file.write_text("\n".join(hex_images) + "\n", encoding="ascii")
    
    hex_labels = [f"{label:02X}" for label in labels.ravel()]
    memh_lbl_file.write_text("\n".join(hex_labels) + "\n", encoding="ascii")
    
    print(f"  [+] Unified image memh saved to:   {memh_img_file}")
    print(f"  [+] Unified label memh saved to:   {memh_lbl_file}")

    print("\n[*] Generating visual JPEG renders...")
    
    # 5. Export Individual Visual Assets (JPEGs)
    for i in range(num_samples):
        img_matrix = images[i]
        label = labels[i]
        
        base_name = f"sample_{i}_digit_{label}"

        # Convert 28x28 matrix to grayscale ('L') PIL image, then to RGB to handle red color text
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

    print(f"\n[SUCCESS] Check the '{output_dir}' directory for your assets.")


if __name__ == "__main__":
    # Adjust paths if your files are named differently
    MNIST_IMAGES = "t10k-images.idx3-ubyte"
    MNIST_LABELS = "t10k-labels.idx1-ubyte"
    OUTPUT_FOLDER = "extracted_samples"
    
    # Execute the extraction
    extract_mnist_samples(MNIST_IMAGES, MNIST_LABELS, OUTPUT_FOLDER, num_samples=NUM_SAMPLES)