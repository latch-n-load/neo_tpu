import os
import re

# --- Configuration ---
GOLDEN_BITSTREAM = "/home/a_akif/tesi/neo_tpu_pynq/neo_tpu_pynq.runs/impl_1/neo_tpu_pynq_wrapper.bit"  
LL_FILE = "/home/a_akif/tesi/neo_tpu_pynq/neo_tpu_pynq.runs/impl_1/neo_tpu_pynq_wrapper.ll"            
CORRUPT_BITSTREAMS_DIR = "../../corrupt_bit"

# Zynq-7000 / Artix-7 Specific Parameters
SYNC_WORD = b'\xAA\x99\x55\x66' 
WORDS_PER_FRAME = 101
BYTES_PER_WORD = 4

# Target Node for Fault Injection
TARGET_NODE = "ub_inst" 
MAX_NODE_TARGS = 500     # CHANGED: Increased to 500 for carpet bombing

def find_sync_word(bit_data):
    sync_idx = bit_data.find(SYNC_WORD)
    if sync_idx == -1:
        print("[!] ERROR: Sync word 0xAA995566 not found! Cannot inject faults")
        return -1
    print(f"[+] Sync word found at byte index: {sync_idx} (0x{sync_idx:08X})")
    return sync_idx

def parse_ll_file(ll_filepath, target_keyword, max_node_targs):
    targets = []
    print(f"\n[*] Parsing .ll file for targets containing: '{target_keyword}'...")
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

def analyze_ll_file(filepath):
    ll_regex = re.compile(r"^Bit\s+(\d+)\s+(0x[0-9a-fA-F]+)\s+(\d+)")
    abs_offsets = set()
    frame_addresses = set()
    frame_offsets = set()
    
    print(f"[*] Parsing {filepath}")
    try:
        with open(filepath, 'r') as file:
            for line in file:
                if not line.startswith("Bit"):
                    continue
                match = ll_regex.match(line)
                if match:
                    abs_offsets.add(int(match.group(1)))
                    frame_addresses.add(match.group(2))
                    frame_offsets.add(int(match.group(3)))

        if not abs_offsets: return

        min_abs, max_abs = min(abs_offsets), max(abs_offsets)
        frame_addr_ints = {int(addr, 16) for addr in frame_addresses}
        min_frame_addr, max_frame_addr = min(frame_addr_ints), max(frame_addr_ints)
        min_frame_offset, max_frame_offset = min(frame_offsets), max(frame_offsets)

        print("-" * 65)
        print(f"ABSOLUTE OFFSETS\n  Min:  {min_abs} Max: {max_abs}\n  Length: {len(abs_offsets)} unique entries\n")
        print(f"FRAME ADDRESSES\n  Min:  0x{min_frame_addr:08X} Max: 0x{max_frame_addr:08X}\n  Length: {len(frame_addresses)} unique entries\n")
        print(f"FRAME OFFSETS\n  Min:  {min_frame_offset} Max: {max_frame_offset}\n  Length: {len(frame_offsets)} unique entries")
        print("-" * 65)
    except FileNotFoundError:
        pass

def get_fdri_data_start(bit_data):
    pattern = re.compile(b'\x30\x00\x40\x00[\x50-\x57]')
    match = pattern.search(bit_data)
    if not match:
        print("[!] ERROR: Could not find FDRI write command sequence.")
        return -1
    return match.start() + 8

def flip_bit_in_bytearray(data_array, abs_bit_offset, raw_data_start):
    byte_idx = raw_data_start + (abs_bit_offset // 8)
    bit_in_byte = abs_bit_offset % 8
    shift_amount = 7 - bit_in_byte
    
    original_byte = data_array[byte_idx]
    corrupted_byte = original_byte ^ (1 << shift_amount)
    data_array[byte_idx] = corrupted_byte
    
    return data_array, byte_idx, bit_in_byte

def generate_faulty_bitstreams():
    print("==================================================")
    print("   WHITE-BOX CARPET BOMBING CORRUPTION ENGINE     ")
    print("==================================================")
    
    os.makedirs(CORRUPT_BITSTREAMS_DIR, exist_ok=True)

    try:
        with open(GOLDEN_BITSTREAM, 'rb') as f:
            golden_data = f.read()
            print(f"[*] Loaded Golden Bitstream: {len(golden_data)} bytes.")
    except FileNotFoundError:
        return

    if find_sync_word(golden_data) == -1: return

    fdri_data_start = get_fdri_data_start(golden_data)
    if fdri_data_start == -1: return
    
    print(f"[*] Raw frame payload begins at file offset: {fdri_data_start} (0x{fdri_data_start:08X})")

    # analyze_ll_file(LL_FILE)
    injection_targets = parse_ll_file(LL_FILE, target_keyword=TARGET_NODE, max_node_targs=MAX_NODE_TARGS)
    
    if not injection_targets: return

    print(f"\n[*] Commencing Carpet Bombing of {len(injection_targets)} bits...\n")
    
    # CHANGED: Create ONE mutable copy of the golden bitstream outside the loop
    faulty_data = bytearray(golden_data)
    
    # CHANGED: Apply all flips to the same bitstream
    for i, target in enumerate(injection_targets):
        faulty_data, byte_idx, bit_in_byte = flip_bit_in_bytearray(
            faulty_data, target['abs_bit_offset'], fdri_data_start
        )
        # Condensed print statement so the console isn't flooded
        print(f"  [Flip {i:03d}] File Byte: 0x{byte_idx:08x}, Bit: {bit_in_byte} | Node: {target['info'].split()[0]}")
        
    # CHANGED: Save the massively corrupted bitstream ONCE
    out_filepath = os.path.join(CORRUPT_BITSTREAMS_DIR, f"fi_CARPET_BOMB_{TARGET_NODE}.bit")
    with open(out_filepath, 'wb') as out_f:
        out_f.write(faulty_data)
        
    print(f"\n[*] SUCCESS: Saved massively corrupted bitstream to: {out_filepath}")
    print("==================================================")

if __name__ == "__main__":
    generate_faulty_bitstreams()