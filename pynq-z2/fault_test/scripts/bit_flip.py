import time
import os
import shutil
import re
import random

# --- Configuration ---
GOLDEN_BITSTREAM = "/home/a_akif/tesi/neo_tpu_pynq2/neo_tpu_pynq2.runs/impl_1/neo_tpu_pynq_wrapper.bit"
LL_FILE = "/home/a_akif/tesi/neo_tpu_pynq2/neo_tpu_pynq2.runs/impl_1/neo_tpu_pynq_wrapper.ll"
EBD_FILE = "/home/a_akif/tesi/neo_tpu_pynq2/neo_tpu_pynq2.runs/impl_1/neo_tpu_pynq_wrapper.ebd" 
CORRUPT_BITSTREAMS_DIR = "../corrupt_bit"

# Zynq-7000 / Artix-7 Specific Parameters
SYNC_WORD = b'\xAA\x99\x55\x66' # 0xAA995566
# WORDS_PER_FRAME = 101
BYTES_PER_WORD = 4

# --- Campaign Mode Selection ---
# 1: Target specific Verilog nodes using the .ll file (Diagnostic)
# 2: Target random routing/LUTs using .ebd set subtraction (Statistical)
CAMPAIGN_PHASE = 1

# Phase 1 Config
LL_TARG_NODE = "neo_tpu" # Set node search term
MAX_PHASE1_TARGS = 200    # Maximum targets to corrupt for specified node
# Phase 2 Config
MAX_PHASE2_TARGS = 800    # Number of random routing/LUT bits to attack

def find_sync_word(bit_data):
    """
    Check for Zynq-7000/Artix-7 bitstream. Searches for 0xAA995566 sync word.
    """
    sync_idx = bit_data.find(SYNC_WORD)
    if sync_idx == -1:
        print("[!] ERROR: Sync word 0xAA995566 not found! Cannot inject faults")
        return -1
    
    print(f"    Sync word found at byte index: {sync_idx} (0x{sync_idx:08X})")
    return sync_idx

def get_ll_targs(ll_filepath, target_keyword, MAX_PHASE1_TARGS):
    """
    Parse Vivado .ll file and extract absolute bit offsets for specified target node.
    Returns a list of dictionaries containing injection coordinates.
    """
    targets = []
    print(f"[*] Parsing .ll file for targets containing: '{target_keyword}'...")

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
                        
                        if len(targets) >= MAX_PHASE1_TARGS:
                            break
                            
    except FileNotFoundError:
        print(f"[!] ERROR: Could not find .ll file at {ll_filepath}")
        return []

    print(f"    Found {len(targets)} injection targets matching '{target_keyword}'")
    return targets

def get_all_ll_bits(ll_filepath):
    """Extracts every absolute bit offset mapped in the .ll file to be used for set subtraction."""
    ll_bits = set()
    ll_regex = re.compile(r"^Bit\s+(\d+)\s+")
    try:
        with open(ll_filepath, 'r') as f:
            for line in f:
                match = ll_regex.match(line.strip())
                if match:
                    ll_bits.add(int(match.group(1)))
        print(f"[*] Total annotated state/memory bits extracted from .ll file: {len(ll_bits)} bits")
        return ll_bits
    except FileNotFoundError:
        print(f"[!] ERROR: Could not find .ll file at {ll_filepath}")
        return set()

def get_ll_stats(filepath):
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

def get_essential_bits(ebd_filepath):
    """Parse .ebd file. Get total bits, and set of all absolute bit offsets designated as essential logic/routing."""
    essential_bits = set()
    print(f"[*] Parsing .ebd file...")
    try:
        with open(ebd_filepath, 'r') as f:
            ebd_data = ""
            ebd_regex = re.compile(r"Bits:\s+(\d+)") # Capture total bits in .ebd
            for line in f:
                clean_line = line.strip()
                ebd_match = ebd_regex.match(clean_line)
                if ebd_match:
                    print(f"[*] Total bits (essential + non-essential) in .ebd file: {ebd_match.group(1)} bits")
                # Skip text header, grab the continuous string of 1s and 0s
                if clean_line.startswith('0') or clean_line.startswith('1'):
                    ebd_data += clean_line

        # Get absolute bit offsets of all 1s (essential bits)
        for idx, bit_char in enumerate(ebd_data):
            if bit_char == '1':
                essential_bits.add(idx)
                
        print(f"    Total essential bits in .ebd file: {len(essential_bits)} bits")
        return essential_bits
    except FileNotFoundError:
        print(f"[!] ERROR: Could not find .ebd file at {ebd_filepath}.")
        return set()

def get_fdri_data_start(bit_data):
    """
    Searches the bitstream for the standard Xilinx FDRI Write command sequence.
    The .ll file absolute offsets begin exactly after this command.
    """
    # Type 1 Write to FDRI command = 0x30004000
    # Type 2 Write to FDRI = Begins with 0x5 followed by 28 bits indicating bitstream WORD_COUNT
    pattern = re.compile(b'\x30\x00\x40\x00') # Binary re for 0x30004000
    match = pattern.search(bit_data)
    
    if not match:
        print("[!] ERROR: Could not find FDRI write command sequence.")
        return -1
        
    start_fdri_t2 = match.start() + 4
    start_fdri_bytes = bit_data[start_fdri_t2:start_fdri_t2 + 4]
    start_fdri_word = int.from_bytes(start_fdri_bytes, byteorder='big')

    if (start_fdri_word >> 28) != 0x5:
        print("[!] ERROR: Type 2 Write FDRI command does not begin with 0x5 header.")
        return -1
    
    fdri_data_start = match.start() + 8
    print(f"    Bitstream design payload begins at .bit offset: {fdri_data_start} (0x{fdri_data_start:08X})")
    word_count = start_fdri_word & 0x0FFFFFFF
    print(f"    Total bitstream payload in .bit: {word_count * BYTES_PER_WORD * 8} bits")

    
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
    print("           BITSTREAM CORRUPTION ENGINE            ")
    print("==================================================")
    
    # Create fresh output directory
    # shutil.rmtree(CORRUPT_BITSTREAMS_DIR, ignore_errors=True)
    os.makedirs(CORRUPT_BITSTREAMS_DIR, exist_ok=True)

    # 1a. Load Golden Bitstream
    try:
        with open(GOLDEN_BITSTREAM, 'rb') as f:
            golden_data = f.read()
            print(f"[*] Loaded Golden Bitstream: {len(golden_data)} bytes.")
    except FileNotFoundError:
        print(f"[!] ERROR: Golden bitstream not found at {GOLDEN_BITSTREAM}")
        return

    # Sanity Check
    if find_sync_word(golden_data) == -1: return

    # 1b. Find where the raw data starts
    fdri_data_start = get_fdri_data_start(golden_data)
    if fdri_data_start == -1:
        return

    # 2a. Analyze .ll to get ranges
    # get_ll_stats(LL_FILE)

    # PHASE 1: Flip bits targeting annotated nodes in .ll
    # Corrupt LL_TARG_NODE. Examples: 'RAMB36' for memory, 'dma' for DMA engine, 'neo_tpu' for TPU logic.
    # ---------------------------------------------------------------------------------------------------
    if CAMPAIGN_PHASE == 1:
        print(f"\n[PHASE 1] Executing Targeted Campaign for Node: '{LL_TARG_NODE}'")
        injection_targets = get_ll_targs(LL_FILE, target_keyword=LL_TARG_NODE, MAX_PHASE1_TARGS=MAX_PHASE1_TARGS)
    
        if not injection_targets:
            print("[!] No targets found. Exiting.")
            return

        # 3a. Inject Faults and Verify
        print("[*] Commencing Targeted Bit Flips...\n")
        
        for i, target in enumerate(injection_targets):
            print(f"--- Injection #{i} ---")
            print(f"Target Node : {target['info']}")
            print(f"Frame Addr  : {target['frame_addr']} | Frame Offset: {target['frame_offset']}")
            print(f"Abs Bit Idx : {target['abs_bit_offset']}")

            # 3b. Fire Faults
            faulty_data = bytearray(golden_data) # Make a copy of the golden bitstream
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
            
            # 3c. Save the corrupted bitstream
            out_filepath = os.path.join(CORRUPT_BITSTREAMS_DIR, f"seu_ph1_{target['abs_bit_offset']}_{LL_TARG_NODE}_{i}.bit")
            with open(out_filepath, 'wb') as out_f:
                out_f.write(faulty_data)
            print(f"Saved to    : {out_filepath}\n")
            
        print(f"[*] Successfully generated {len(injection_targets)} corrupted bitstreams.")

    # PHASE 2: ROUTING, LUT and DSP CAMPAIGN
    # ---------------------------------------------------------
    elif CAMPAIGN_PHASE == 2:
        print(f"\n[PHASE 2] Executing Routing, LUT and DSP Structural Fault Injection Campaign")
        ebd_bits = get_essential_bits(EBD_FILE) # Get bit offsets of essential logic from .ebd
        ll_bits = get_all_ll_bits(LL_FILE)      # Get bit offsets of ll_bits to avoid repeat fault injections on same bits
        
        if not ebd_bits or not ll_bits:
            print("[!] Missing necessary .ebd or .ll data. Exiting.")
            return

        # 4a. Perform set subtraction to isolate untested targets
        # TODO: Check this - ll contains state/memory type 1 (BRAM) and 0 (slice logic). 
        # Type 1 BRAM are not available in .ebd so pre-excluded. 
        # Subtraction only removes ALL Type 0 from ebd ~ 8200 bits
        untested_targs = list(ebd_bits - ll_bits)
        print(f"[*] Isolated un-tested bits (LUTs/Routing/DSPs) from .ebd for Phase 2 Injection: {len(untested_targs)} bits")
        
        # 4b. Select random sample from the massive list of untested targets
        sample_targs = min(MAX_PHASE2_TARGS, len(untested_targs))
        selected_targs = random.sample(untested_targs, sample_targs)
        print(f"[*] Commencing Bit Flips for {sample_targs} random targets from .ebd...\n")

        # 4c. Fire Faults and Save Corrupted Bitstreams
        for i, abs_offset in enumerate(selected_targs):
            faulty_data = bytearray(golden_data)
            faulty_data, byte_idx, bit_in_byte, orig_bin, corr_bin = flip_bit_in_bytearray(
                faulty_data, abs_offset, fdri_data_start
            )
            out_filepath = os.path.join(CORRUPT_BITSTREAMS_DIR, f"seu_ph2_{abs_offset}_{i}.bit")
            with open(out_filepath, 'wb') as out_f:
                out_f.write(faulty_data)
            print(f"  Saved : {out_filepath}")
            
        print(f"[*] Successfully generated {sample_targs} Phase 2 bitstreams.")

    else: print("[!] Invalid CAMPAIGN_PHASE selected.")
    end_time = time.time()
    print(f"Total Fault Injection Campaign Time: {end_time - st_time:.2f} seconds")

if __name__ == "__main__":
    generate_faulty_bitstreams()