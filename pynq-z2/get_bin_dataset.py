import struct
import numpy as np
from pathlib import Path

def convert_idx_to_grayscale_bin(images_path: str, labels_path: str, output_dir: str):
    """
    Parses standard MNIST IDX files and exports them as flat binary files.
    Retains the full 8-bit grayscale pixel values (0-255).
    """
    images_path = Path(images_path)
    labels_path = Path(labels_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read and Verify Labels
    with open(labels_path, 'rb') as lbl_file:
        magic, num_labels = struct.unpack(">II", lbl_file.read(8))
        assert magic == 2049, f"Invalid label magic number: {magic}"
        labels_array = np.frombuffer(lbl_file.read(), dtype=np.uint8)
        assert len(labels_array) == num_labels, "Label count mismatch!"

    # 2. Read and Verify Images
    with open(images_path, 'rb') as img_file:
        magic, num_images, rows, cols = struct.unpack(">IIII", img_file.read(16))
        assert magic == 2051, f"Invalid image magic number: {magic}"
        assert num_images == num_labels, f"CRITICAL ERROR: {num_images} images but {num_labels} labels!"
        
        # Read the raw 8-bit grayscale pixels
        images_array = np.frombuffer(img_file.read(), dtype=np.uint8)
        expected_pixels = num_images * rows * cols
        assert len(images_array) == expected_pixels, "Image data size mismatch!"

    # 3. Export to Flat Binary (.bin)
    img_out = out_dir / "t10k_images_grayscale.bin"
    lbl_out = out_dir / "t10k_labels_flat.bin"
    
    images_array.tofile(img_out)
    labels_array.tofile(lbl_out)

    print(f"[SUCCESS] Exported {num_images} aligned image-label pairs.")
    print(f"Images saved to: {img_out} ({images_array.nbytes} bytes)")
    print(f"Labels saved to: {lbl_out} ({labels_array.nbytes} bytes)")

if __name__ == "__main__":
    convert_idx_to_grayscale_bin(
        "t10k-images.idx3-ubyte", 
        "t10k-labels-idx1-ubyte", 
        "pynq_z2_dataset"
    )