import subprocess
import time
import os
import re
import random

# ==============================================================================
# BITSTREAM CORRUPTION & METADATA PARSING UTILITIES
# ==============================================================================

SYNC_WORD = b'\xAA\x99\x55\x66'
BYTES_PER_WORD = 4

def find_sync_word(bit_data):
    """Check for Zynq-7000/Artix-7 bitstream. Searches for 0xAA995566 sync word."""
    sync_idx = bit_data.find(SYNC_WORD)
    if sync_idx == -1:
        print("[!] ERROR: Sync word 0xAA995566 not found! Cannot inject faults")
        return -1
                
    print(f"    Sync word found at byte index: {sync_idx} (0x{sync_idx:08X})")
    return sync_idx

def get_fdri_data_start(bit_data):
    """Searches the bitstream for the standard Xilinx FDRI Write command sequence."""
    # Type 1 Write to FDRI command = 0x30004000
    # Type 2 Write to FDRI = Begins with 0x5 followed by 28 bits indicating bitstream WORD_COUNT
    pattern = re.compile(b'\x30\x00\x40\x00')
    match = pattern.search(bit_data)
    
    if not match:
        print("[!] ERROR: Could not find FDRI write command sequence.")
        return -1, -1
        
    start_fdri_t2 = match.start() + 4
    start_fdri_word = int.from_bytes(bit_data[start_fdri_t2:start_fdri_t2 + 4], byteorder='big')

    if (start_fdri_word >> 28) != 0x5:
        print("[!] ERROR: Type 2 Write FDRI command does not begin with 0x5 MSB.")
        return -1, -1
    
    fdri_data_start = match.start() + 8
    print(f"    Bitstream design payload begins at .bit offset: {fdri_data_start} (0x{fdri_data_start:08X})")

    word_count = start_fdri_word & 0x0FFFFFFF
    total_payload_bits = word_count * BYTES_PER_WORD * 8
    print(f"    Total bitstream payload in .bit: {total_payload_bits} bits")

    return fdri_data_start, total_payload_bits

def disable_tail_crc(bit_data, fdri_data_start, total_payload_bits):
    """Disables CRC checks in the bitstream tail by replacing them with NOPs."""
    CRC_CMD = b'\x30\x00\x00\x01' # CRC Write CMD: 0x30000001
    NOP_CMD = b'\x20\x00\x00\x00' # NOP CMD: 0x20000000
    
    fdri_data_end = fdri_data_start + (total_payload_bits // 8)
    idx = fdri_data_end
    count = 0
    
    while True:
        idx = bit_data.find(CRC_CMD, idx)
        if idx == -1:
            break
        # Replace CRC Write and CRC value with two NOPs (8B replacement) 
        bit_data[idx : idx+8] = NOP_CMD + NOP_CMD
        count += 1
        idx += 8

    print(f"    Disabled {count} CRC checks in the bitstream tail.")
    return bit_data

def get_ll_targs(ll_filepath, max_targets, target_keyword=None):
    """
    Parses Vivado .ll file and extracts absolute bit offsets and information.
    - If target_keyword is specified, it filters nodes matching the keyword.
    - If target_keyword is None, it extracts globally from the entire .ll file.
    - Randomly samples down to max_targets if the resulting pool is too large.
    """
    targets = []
    target_desc = f"'{target_keyword}'" if target_keyword else "ALL nodes"
    print(f"[*] Parsing {os.path.basename(ll_filepath)} for targets matching: {target_desc}...")

    try:
        with open(ll_filepath, 'r') as file:
            for line in file:
                # 1. Fast pre-filter: Skip lines that don't match the keyword
                if target_keyword and target_keyword not in line:
                    continue

                # 2. Extract without Regex (Format: Bit    )
                if line.startswith("Bit "):
                    # Splits exactly 4 times, keeping the whole info string intact in parts[4]
                    parts = line.split(maxsplit=4) 
                    if len(parts) >= 5:
                        targets.append({
                            'abs_bit_offset': int(parts[1]),
                            'info': parts[4].strip()
                        })
    except FileNotFoundError:
        print(f"[!] ERROR: Could not find .ll file at {ll_filepath}")
        return []

    print(f"    Found {len(targets)} injection targets matching {target_desc}.")

    # Apply random sampling if pool exceeds requested max_targets
    if len(targets) > max_targets:
        print(f"    Randomly sampling {max_targets} targets from the pool...")
        return random.sample(targets, max_targets)

    return targets

# def get_all_ll_bits(ll_filepath):
#     """Extracts every absolute bit offset mapped in the .ll file."""
#     ll_bits = set()
#     ll_regex = re.compile(r"^Bit\s+(\d+)\s+")
#     try:
#         with open(ll_filepath, 'r') as f:
#             for line in f:
#                 match = ll_regex.match(line.strip())
#                 if match:
#                     ll_bits.add(int(match.group(1)))
#     except FileNotFoundError:
#         print(f"[!] ERROR: Could not find .ll file at {ll_filepath}")
#     return ll_bits

# TODO: Check diff with regex version. Remove if not needed.
def get_all_ll_bits(ll_filepath):
    """Extracts every absolute bit offset mapped in the .ll file for fast set subtraction."""
    ll_bits = set()
    print(f"[*] Extracting all active bit offsets from {os.path.basename(ll_filepath)}...")
    
    try:
        with open(ll_filepath, 'r') as f:
            for line in f:
                if line.startswith("Bit "):
                    # split(maxsplit=2) is the fastest way to isolate the second token
                    ll_bits.add(int(line.split(maxsplit=2)[1]))
    except FileNotFoundError:
        print(f"[!] ERROR: Could not find .ll file at {ll_filepath}")
        
    return ll_bits

def get_essential_bits(ebd_filepath):
    """Parses .ebd file and gets absolute bit offsets of all 1s (essential bits)."""
    essential_bits = set()

    print(f"[*] Parsing {os.path.basename(ebd_filepath)} for essential bits...")
    try:
        with open(ebd_filepath, 'r') as f:
            ebd_data = ""
            for line in f:
                clean_line = line.strip()
                if clean_line.startswith('0') or clean_line.startswith('1'):
                    ebd_data += clean_line
        for idx, bit_char in enumerate(ebd_data):
            if bit_char == '1':
                essential_bits.add(idx)
    except FileNotFoundError:
        print(f"[!] ERROR: Could not find .ebd file at {ebd_filepath}.")

    print(f"    Number of essential bits found in .ebd file: {len(essential_bits)} bits")
    return essential_bits

def generate_corrupt_bitstream(golden_data, abs_bit_offset, raw_data_start, out_filepath):
    """Calculates byte/bit position, flips the bit, and saves the file directly to disk."""
    byte_idx = raw_data_start + (abs_bit_offset // 8)
    bit_in_byte = abs_bit_offset % 8
    shift_amount = 7 - bit_in_byte
    
    # Copy buffer and flip bit
    faulty_data = bytearray(golden_data)
    faulty_data[byte_idx] ^= (1 << shift_amount)
    
    with open(out_filepath, 'wb') as out_f:
        out_f.write(faulty_data)

# ==============================================================================
# HARDWARE COMMUNICATION (XSCT)
# ==============================================================================

def start_xsct_session(tcl_script_path):
    """Launches XSCT as a persistent background process."""
    print(f"[*] Launching XSCT with script: {os.path.basename(tcl_script_path)}")
    xsct_proc = subprocess.Popen(
        ["xsct", tcl_script_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1
    )
    
    while True:
        line = xsct_proc.stdout.readline()
        if "READY" in line:
            print("    XSCT connected to hardware and ready!")
            return xsct_proc
        if line == "": # EOF means XSCT crashed
            print("[!] FATAL: XSCT failed to start or crashed.")
            print("\n[XSCT CRASH LOG]:\n", xsct_proc.stderr.read())
            return None

def program_fpga(xsct_proc, bitstream_path):
    """Sends bitstream path to persistent XSCT and waits for confirmation."""
    # Write .bit path to XSCT stdin
    xsct_proc.stdin.write(bitstream_path + "\n")
    xsct_proc.stdin.flush()

    # Wait for the success/error token from XSCT's standard output
    while True:
        line = xsct_proc.stdout.readline()
        clean_line = line.strip()
        
        if clean_line == "PROGRAM_DONE":
            return True
        elif "FPGA_PROGRAM_ERROR" in clean_line:
            return False
        elif line == "":
            print("\n[!] XSCT process terminated unexpectedly.")
            # Get any fatal crash logs from stderr
            error_output = xsct_proc.stderr.read()
            if error_output: 
                print(f"\n[XSCT CRASH LOG]:\n{error_output}")
            return False

# ==============================================================================
# UART COMMUNICATION & PARSING (DESIGN SPECIFIC)
# ==============================================================================

def _read_uart_base(ser, fv_regex, timeout_sec):
    """Base generic UART reader shared by both designs."""
    start_time = time.time()
    raw_buffer = b""
    
    while (time.time() - start_time) < timeout_sec:
        if ser.in_waiting > 0:
            raw_buffer += ser.read(ser.in_waiting)
            try:
                decoded = raw_buffer.decode('utf-8', errors='ignore')
                if fv_regex.search(decoded):
                    time.sleep(0.01)
                    raw_buffer += ser.read(ser.in_waiting)
                    break
            except Exception:
                pass
            # Break if UART outputs infinite garbage
            if len(raw_buffer) > 1000: break
        time.sleep(0.01)

    decoded_output = raw_buffer.decode('utf-8', errors='replace')
    # Create a legible version of raw binary using replacement character
    # legible_binary = raw_buffer.decode('utf-8', errors='replace').strip()

    # Check for Hardware Exceptions / Crashes
    hw_except_keywords = ("cpu", "neorv32", "fault", "bad")
    lower_out = decoded_output.lower()
    if any(word in lower_out for word in hw_except_keywords):
    # if "cpu" in lower_out or "neov32" in lower_out or "fault" in lower_out:
        first_error_line = "Unknown CPU Exception"
        for line in decoded_output.splitlines():
            clean_line = line.strip()
            if any(word in clean_line.lower() for word in hw_except_keywords):
            # if "cpu" in clean_line.lower() or "neorv32" in clean_line.lower() or "fault" in clean_line.lower():
                first_error_line = clean_line
                break
        return None, decoded_output, first_error_line, "Hardware Exception"
        
    if len(raw_buffer) == 0:
        return None, decoded_output, None, "No result obtained"

    return fv_regex.search(decoded_output), decoded_output, None, "Invalid UART Payload"

# --- DESIGN 1: NEO_CFS_TPU (32 Hex Characters) ---

def read_uart_hex(ser, fv_regex, timeout_sec, golden_fv=None):
    """UART logic specific to the 32-hex character CFS design."""
    match, decoded_output, err_line, fallback_state = _read_uart_base(ser, fv_regex, timeout_sec)
    
    if match:
        fv = match.group(1).lower()
        if golden_fv is None: return fv, "Golden FV Extracted"
        
        golden_cmp = golden_fv.lower()
        if fv == golden_cmp:
            return fv, "Matches Golden FV"
        elif fv == "0"*32 and golden_cmp != "0"*32:
            return fv, "False Perfect FV"
        else:
            return fv, "One or more faults"
            
    return (err_line if err_line else decoded_output), fallback_state

def parse_hex_fv(fv_hex, image_count=100):
    val = int(fv_hex, 16)
    clint_fault   = (val >> 127) & 1
    dma_lbl_fault = (val >> 126) & 1
    dma_img_fault = (val >> 125) & 1
    
    img_faults = val & ((1 << image_count) - 1)
    mismatches = bin(img_faults).count('1')
    accuracy_metric = (image_count - mismatches) / float(image_count)
    
    return accuracy_metric, clint_fault, dma_lbl_fault, dma_img_fault

def write_hex_log(filename, bitstream, fv, status, ll_info, parsed_data=None):
    with open(filename, 'w') as f:
        f.write(f"Corrupt_bit_file: {bitstream}\n")
        f.write(f"Fault vector from FPGA: {fv if fv else 'NONE'}\n")
        f.write(f"Test result: {status}\n")
        f.write(f"LL_Information: {ll_info}\n")

# --- DESIGN 2: NEO_EXTBUS_TPU (51 CSV Values) ---

def read_uart_csv(ser, fv_regex, timeout_sec, golden_fv=None):
    """Parse 51 CSV values generated by neo_extbus_tpu design and resturn fault category."""
    match, decoded_output, err_line, fallback_state = _read_uart_base(ser, fv_regex, timeout_sec)
    
    if match:
        fv = match.group(1)
        if golden_fv is None: return fv, "Golden FV Extracted"
        
        if fv == golden_fv:
            return fv, "Matches Golden FV"
        
        # False perfect not applicable since golden_fv is perfect -> 100% accuracy
        # elif fv.split(',')[0] == '100' and fv != golden_fv:
        #     return fv, "False Perfect FV"
        
        else:
            return fv, "One or more faults"
            
    return (err_line if err_line else decoded_output), fallback_state

def parse_csv_fv(fv_string, golden_fv, image_count=5):
    fv_vals = fv_string.split(',')
    gold_vals = golden_fv.split(',')
    
    acc_nom = int(fv_vals[0]) / 100.0
    
    b = [None] * image_count
    for i in range(image_count):
        b[i] = '1' if fv_vals[i*10+1:(i+1)*10+1] == gold_vals[i*10+1:(i+1)*10+1] else '0'

    img_results = "".join(b)
    acc_real = img_results.count('1') / float(image_count)
    
    return acc_nom, acc_real, img_results

def write_csv_log(filename, bitstream, fv, status, ll_info, parsed_data=None):
    with open(filename, 'w') as f:
        f.write(f"Corrupt_bit_file: {bitstream}\n")
        f.write(f"Fault vector from FPGA: {fv if fv else 'NONE'}\n")
        f.write(f"Test result: {status}\n")
        f.write(f"LL_Information: {ll_info}\n")
        
        if parsed_data and fv:
            acc_nom, acc_real, img_results = parsed_data
            f.write(f"Image_results: {img_results}\n")
            f.write(f"Accuracy_nominal: {acc_nom}\n")
            f.write(f"Accuracy_real: {acc_real}\n")
            

# ==============================================================================
# PIPELINE MANAGEMENT
# ==============================================================================

def cleanup_batch(directory_path):
    """Deletes all .bit and .log files in directory_path"""
    for filename in os.listdir(directory_path):
        if filename.endswith(".bit"):
            file_path = os.path.join(directory_path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[!] Warning: Could not delete {file_path}. Reason: {e}")
        elif filename.endswith(".log"):
            file_path = os.path.join(directory_path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[!] Warning: Could not delete {file_path}. Reason: {e}")