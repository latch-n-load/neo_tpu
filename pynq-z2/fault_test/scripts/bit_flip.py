import time
import os
import re

# --- Configuration ---
GOLDEN_BITSTREAM = "/home/a_akif/tesi/neo_tpu_pynq/neo_tpu_pynq.runs/impl_1/neo_tpu_pynq_wrapper.bit"  # TODO: Point this to your actual .bit file
LL_FILE = "/home/a_akif/tesi/neo_tpu_pynq/neo_tpu_pynq.runs/impl_1/neo_tpu_pynq_wrapper.ll"            # TODO: Point this to your actual .ll file
CORRUPT_BITSTREAMS_DIR = "../corrupt_bit"

# Zynq-7000 / Artix-7 Specific Parameters
SYNC_WORD = b'\xAA\x99\x55\x66' # 0xAA995566
WORDS_PER_FRAME = 101
BYTES_PER_WORD = 4

# Target Node for Fault Injection
TARGET_NODE = "neo_tpu" # Set node search term
MAX_NODE_TARGS = 10000    # Maximum targets to corrupt for specified node

def find_sync_word(bit_data):
    """
    Check for Zynq-7000/Artix-7 bitstream. Searches for 0xAA995566 sync word.
    """
    sync_idx = bit_data.find(SYNC_WORD)
    if sync_idx == -1:
        print("[!] ERROR: Sync word 0xAA995566 not found! Cannot inject faults")
        return -1
    
    print(f"[+] Sync word found at byte index: {sync_idx} (0x{sync_idx:08X})")
    return sync_idx

def parse_ll_file(ll_filepath, target_keyword, max_node_targs):
    """
    Parse Vivado .ll file and extract absolute bit offsets for specified target node.
    Returns a list of dictionaries containing injection coordinates.
    """
    targets = []
    print(f"\n[*] Parsing .ll file for targets containing: '{target_keyword}'...")

    # Bit lines have the following form:
    # Bit <offset> <frame address> <frame offset> <information>
    # Example: Bit 23074723 0x00421c9f 1475 Block=SLICE_X90Y23 Latch=AQ Net=neo_tpu...
    ll_regex = re.compile(r"^Bit\s+(\d+)\s+(0x[0-9a-fA-F]+)\s+(\d+)\s+(.*)")
    
    try:
        with open(ll_filepath, 'r') as file:
            for line in file:
                if target_keyword in line:
                    match = ll_regex.match(line.strip())
                    if match:
                        targets.append({
                            'abs_bit_offset': int(match.group(1)),
                            'frame_addr': match.group(2),
                            'frame_offset': int(match.group(3)),
                            'info': match.group(4)
                        })
                        
                        if len(targets) >= max_node_targs:
                            break
                            
    except FileNotFoundError:
        print(f"[!] ERROR: Could not find .ll file at {ll_filepath}")
        return []

    print(f"[+] Found {len(targets)} injection targets matching '{target_keyword}'.")
    return targets

# def get_raw_data_start(bit_data):
#     """
#     Parse .bit file header to find exactly where the raw configuration data begins.
#     The .ll file offsets are relative to this point.
#     """
#     ptr = 0
    
#     # 1. Dummy string length (usually 0x0009)
#     length = int.from_bytes(bit_data[ptr:ptr+2], 'big')
#     ptr += 2 + length
    
#     # 2. Skip 2-byte header separator (0x00 0x01)
#     ptr += 2
    
#     # Field 'a': Design Name
#     if bit_data[ptr] == 0x61: 
#         ptr += 1
#     else: 
#         print(f"[!] Error: Expected 0x61 ('a'), found 0x{bit_data[ptr]:02X} at offset {ptr}")
#         return -1
#     length = int.from_bytes(bit_data[ptr:ptr+2], 'big')
#     ptr += 2 + length
#     # print(f"[DEBUG] Design Name Length: {length} bytes, skipped to offset {ptr}")
    
#     # Field 'b': Part Name
#     if bit_data[ptr] == 0x62: ptr += 1
#     length = int.from_bytes(bit_data[ptr:ptr+2], 'big')
#     ptr += 2 + length
    
#     # Field 'c': Date
#     if bit_data[ptr] == 0x63: ptr += 1
#     length = int.from_bytes(bit_data[ptr:ptr+2], 'big')
#     ptr += 2 + length
    
#     # Field 'd': Time
#     if bit_data[ptr] == 0x64: ptr += 1
#     length = int.from_bytes(bit_data[ptr:ptr+2], 'big')
#     ptr += 2 + length
    
#     # Field 'e': Raw Data Length (4 bytes)
#     if bit_data[ptr] == 0x65: ptr += 1
#     else: return -1
#     length = int.from_bytes(bit_data[ptr:ptr+4], 'big')
#     ptr += 4
    
#     return ptr, length

def analyze_ll_file(filepath):
    """
    Parses a Xilinx .ll file to extract the min/max ranges and the total 
    number of unique entries for absolute offsets, frame addresses, and frame offsets.
    """
    ll_regex = re.compile(r"^Bit\s+(\d+)\s+(0x[0-9a-fA-F]+)\s+(\d+)")
    
    # Using sets to automatically track unique entries
    abs_offsets = set()
    frame_addresses = set()
    frame_offsets = set()
    
    print(f"[*] Parsing {filepath}")
    
    try:
        with open(filepath, 'r') as file:
            for line in file:
                # Fast pre-filter
                if not line.startswith("Bit"):
                    continue
                
                match = ll_regex.match(line)
                if match:
                    abs_offsets.add(int(match.group(1)))
                    frame_addresses.add(match.group(2))
                    frame_offsets.add(int(match.group(3)))

        # Ensure we found data before calculating
        if not abs_offsets:
            print("[!] No valid bit lines found in the file.")
            return

        # Calculate metrics for Absolute Offsets
        min_abs = min(abs_offsets)
        max_abs = max(abs_offsets)
        
        # Calculate metrics for Frame Addresses (requires converting hex string to int for comparison)
        frame_addr_ints = {int(addr, 16) for addr in frame_addresses}
        min_frame_addr = (min(frame_addr_ints))
        max_frame_addr = (max(frame_addr_ints))
        
        # Calculate metrics for Frame Offsets
        min_frame_offset = min(frame_offsets)
        max_frame_offset = max(frame_offsets)

        # Display Results
        print("-" * 65)
        print(f"ABSOLUTE OFFSETS")
        print(f"  Min:  {min_abs} Max: {max_abs}")
        print(f"  Length: {len(abs_offsets)} unique entries")
        print(f"")
        print(f"FRAME ADDRESSES")
        print(f"  Min:  {min_frame_addr} Max: {max_frame_addr}")
        print(f"  Min:  0x{min_frame_addr:08X} Max: 0x{max_frame_addr:08X}")
        print(f"  Length: {len(frame_addresses)} unique entries")
        print(f"")
        print(f"FRAME OFFSETS")
        print(f"  Min:  {min_frame_offset} Max: {max_frame_offset}")
        print(f"  Length: {len(frame_offsets)} unique entries")
        print("-" * 65)

    except FileNotFoundError:
        print(f"[!] ERROR: Could not find file at {filepath}")

def get_fdri_data_start(bit_data):
    """
    Searches the bitstream for the standard Xilinx FDRI Write command sequence.
    The .ll file absolute offsets begin exactly after this command.
    """
    # 0x30004000 is the Type 1 Write to FDRI command.
    # It is immediately followed by a Type 2 command starting with 0x50 to 0x57 (depending on word count).
    pattern = re.compile(b'\x30\x00\x40\x00[\x50-\x57]')
    match = pattern.search(bit_data)
    
    if not match:
        print("[!] ERROR: Could not find FDRI write command sequence. Is this a valid bitstream?")
        return -1
        
    # The frame data payload starts exactly 8 bytes after the start of the match
    # (4 bytes for Type 1 command + 4 bytes for Type 2 command)
    fdri_data_start = match.start() + 8
    
    return fdri_data_start

def flip_bit_in_bytearray(data_array, abs_bit_offset, raw_data_start):
    """
    Calculates the exact byte and bit position incorporating the header offset.
    """
    # 1. Calculate Byte Index (Add the header offset!)
    byte_idx = raw_data_start + (abs_bit_offset // 8)
    
    # 2. Calculate Bit Index (Xilinx is MSB first)
    bit_in_byte = abs_bit_offset % 8
    shift_amount = 7 - bit_in_byte
    
    # Extract original byte
    original_byte = data_array[byte_idx]
    
    # 3. Flip the bit using XOR
    mask = 1 << shift_amount
    corrupted_byte = original_byte ^ mask
    
    # Apply to array
    data_array[byte_idx] = corrupted_byte
    
    # 4. Format binary strings for visual verification
    orig_bin = f"{original_byte:08b}"
    corr_bin = f"{corrupted_byte:08b}"
    
    return data_array, byte_idx, bit_in_byte, orig_bin, corr_bin

def generate_faulty_bitstreams():
    st_time = time.time()
    print("==================================================")
    print("      WHITE-BOX BITSTREAM CORRUPTION ENGINE       ")
    print("==================================================")
    
    # Create output directory
    os.makedirs(CORRUPT_BITSTREAMS_DIR, exist_ok=True)

    # 1. Load the Golden Bitstream
    try:
        with open(GOLDEN_BITSTREAM, 'rb') as f:
            golden_data = f.read()
            print(f"[*] Loaded Golden Bitstream: {len(golden_data)} bytes.")
    except FileNotFoundError:
        print(f"[!] ERROR: Golden bitstream not found at {GOLDEN_BITSTREAM}")
        return

    # Sanity Check
    if find_sync_word(golden_data) == -1: return

    # Find where the raw data starts
    fdri_data_start = get_fdri_data_start(golden_data)
    if fdri_data_start == -1:
        return
    
    print(f"[*] Raw frame payload begins at file offset: {fdri_data_start} (0x{fdri_data_start:08X})")

    # 2a. Analyze .ll to get ranges
    # analyze_ll_file(LL_FILE)

    # 2b. Parse the .ll file for targets
    # Examples: 'RAMB36' for memory, 'dma' for DMA engine, 'neo_tpu' for TPU logic.
    injection_targets = parse_ll_file(LL_FILE, target_keyword=TARGET_NODE, max_node_targs=MAX_NODE_TARGS)
    
    if not injection_targets:
        print("[!] No targets found. Exiting.")
        return

    # 3. Inject Faults and Verify
    print("\n[*] Commencing Targeted Bit Flips...\n")
    
    for i, target in enumerate(injection_targets):
        print(f"--- Injection #{i} ---")
        print(f"Target Node : {target['info']}")
        print(f"Frame Addr  : {target['frame_addr']} | Frame Offset: {target['frame_offset']}")
        print(f"Abs Bit Idx : {target['abs_bit_offset']}")
        
        # Make a fresh mutable copy of the golden bitstream
        faulty_data = bytearray(golden_data)
        
        # Fire the fault
        faulty_data, byte_idx, bit_in_byte, orig_bin, corr_bin = flip_bit_in_bytearray(
            faulty_data, target['abs_bit_offset'], fdri_data_start
        )
        
        # Verify the flip visually
        print(f"File Offset : Byte {byte_idx}, Bit {bit_in_byte}")
        print(f"Verification: Original Byte -> {orig_bin}")
        print(f"              Corrupt Byte  -> {corr_bin}")
        
        # Highlight which bit flipped
        pointer = " " * (17 + bit_in_byte) + "^"
        print(f"              {pointer} (Bit Flipped!)")
        
        # 4. Save the corrupted bitstream
        out_filepath = os.path.join(CORRUPT_BITSTREAMS_DIR, f"fi_{TARGET_NODE}_{i}.bit")
        with open(out_filepath, 'wb') as out_f:
            out_f.write(faulty_data)
        print(f"Saved to    : {out_filepath}\n")
        
    print(f"[*] Successfully generated {len(injection_targets)} corrupted bitstreams.")
    print("==================================================")
    end_time = time.time()
    print(f"Total Time: {end_time - st_time:.2f} seconds")

if __name__ == "__main__":
    generate_faulty_bitstreams()