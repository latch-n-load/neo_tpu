import subprocess
import serial
import shutil
import time
import csv
import os
import re
import glob

# --- Configuration ---
GOLDEN_BITSTREAM = "/home/a_akif/tesi/neo_tpu_pynq2/neo_tpu_pynq2.runs/impl_1/neo_tpu_pynq_wrapper.bit"
UART_PORT = "/dev/ttyUSB0"      # TODO: Validate UART port using sudo dmesg -w | grep tty
BAUD_RATE = 921600
CORRUPT_BITSTREAMS_DIR = "../corrupt_bit"
TEST_RESULTS_CSV = "../fi_test_results.csv"
LOGS_DIR = "../logs"            # Directory to store individual run logs
TIMEOUT_SEC = 5                 # Wait for UART response
PROGRAM_FPGA_TCL = "program_fpga.tcl"
IMAGE_COUNT = 100

# Regex to capture exactly 32 hex characters
FV_REGEX = re.compile(r'([0-9a-fA-F]{32})') # Raw String re 32 chars, range 0-9, a-f, A-F

def start_xsct_session():
    """Launch XSCT as a persistent background process."""
    print("[*] Starting XSCT session...")
    xsct_proc = subprocess.Popen(
        ["xsct", PROGRAM_FPGA_TCL],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1 # Line buffered
    )
    
    # Wait for the TCL script to print "READY"
    while True:
        line = xsct_proc.stdout.readline()
        
        # --- DEBUG PRINT ---
        if line:
            print(f"[XSCT INIT] {line.strip()}")

        if "READY" in line:
            print("    XSCT connected to hardware and ready!")
            break
        if line == "": # EOF means XSCT crashed
            print("[!] FATAL: XSCT failed to start or crashed.")
            print("\n[XSCT CRASH LOG]:\n", xsct_proc.stderr.read())
            return None
            
    return xsct_proc

def program_fpga(xsct_proc, bitstream_path):
    """Send bitstream path to persistent XSCT and wait for confirmation."""
    # Write .bit path to XSCT stdin
    xsct_proc.stdin.write(bitstream_path + "\n")
    xsct_proc.stdin.flush()
    
    # Wait for the success/error token from XSCT's standard output
    while True:
        line = xsct_proc.stdout.readline()
        
        # --- DEBUG PRINT ---
        # This will print every single `puts` from your TCL script to your terminal
        if line:
            print(f"  [XSCT] {line.strip()}")
            
        clean_line = line.strip()
        
        if clean_line == "PROGRAM_DONE":
            return True
        elif "FPGA_PROGRAM_ERROR" in clean_line:
            return False
        elif line == "": # Process died
            print("\n[!] XSCT process terminated unexpectedly.")
            
            # Pull any fatal crash logs from stderr
            error_output = xsct_proc.stderr.read()
            if error_output:
                print(f"\n[XSCT CRASH LOG]:\n{error_output}")
                
            return False
        
def read_uart_for_fv(ser, golden_fv=None):
    """
    Read UART and categorize result into one of the following result_info categories:
    1. Matches Golden FV
    2. False Perfect FV
    3. One or more faults
    4. No result obtained
    5. Hardware Exception
    6. Invalid UART Payload (Truncated FV or Illegible binary) 
    Returns: (extracted_fv_string, result_info)
    """
    start_time = time.time()
    raw_buffer = b""
    
    while (time.time() - start_time) < TIMEOUT_SEC:
        if ser.in_waiting > 0:
            raw_buffer += ser.read(ser.in_waiting)
            # Attempt to decode and check for the 32-char hex string
            try:
                decoded = raw_buffer.decode('utf-8', errors='ignore')
                match = FV_REGEX.search(decoded)
                if match:
                    # Give it a tiny delay to finish printing the line, then break
                    time.sleep(0.01)
                    raw_buffer += ser.read(ser.in_waiting)
                    break
            except Exception:
                pass
            
            # Anti-Hang Protection: If buffer explodes past 5000 bytes, break early.
            if len(raw_buffer) > 5000:
                break
                
        time.sleep(0.01)

    decoded_output = raw_buffer.decode('utf-8', errors='ignore')
    # Create a legible version of raw binary using replacement character
    # legible_binary = raw_buffer.decode('utf-8', errors='replace').strip()

    # 1. Check for Legible Fault Vector
    match = FV_REGEX.search(decoded_output)
    if match:
        fv = match.group(1).lower() # Convert group 1 to lower case and assign as fv
        # print(f"[DEBUG] fv (match.group(1).lower()) = {fv}\n match.group(1) = {match.group(1)} ")
        # print(f"[DEBUG] golden_fv = {golden_fv} golden_fv.lower() = {golden_fv.lower()} ")
        # 1a. If fv == golden
        if golden_fv is None:
            # ser.reset_input_buffer()
            return fv, "Golden FV Extracted"
        elif fv == golden_fv.lower():
            # ser.reset_input_buffer()
            return fv, "Matches Golden FV"
        # 1b. if fv is false perfectly (all zeroes)
        elif fv == "00000000000000000000000000000000" and golden_fv != "00000000000000000000000000000000":
            # ser.reset_input_buffer()
            return fv, "False Perfect FV"
        # 1c. Mismatch between fv and golden_fv
        else:
            # ser.reset_input_buffer()
            return fv, "One or more faults"

    # 2. Check for No Results (Timeout)
    if len(raw_buffer) == 0:
        # ser.reset_input_buffer()
        return None, "No result obtained"

    # 3. Check for Hardware Exceptions / Crashes
    lower_out = decoded_output.lower()
    if "[cpu" in lower_out or "neov32" in lower_out or "access fault" in lower_out:
        # Extract the first meaningful line of the error
        first_error_line = "Unknown CPU Exception"
        for line in decoded_output.splitlines():
            clean_line = line.strip()
            # Grab the first line that looks like a CPU error log
            if "[cpu" in clean_line.lower() or "neorv32" in clean_line.lower() or "fault" in clean_line.lower():
                first_error_line = clean_line
                break
        # ser.reset_input_buffer()
        return first_error_line, "Hardware Exception"

    # 6. Invalid UART Payload (Truncated FV or Illegible binary) 
    # ser.reset_input_buffer()
    return decoded_output, "Invalid UART Payload"

def parse_fv(fv_hex):
    """
    Parses the 128-bit vector.
    Returns: (accuracy_metric, clint_fault, dma_lbl_fault, dma_img_fault)
    """
    val = int(fv_hex, 16)
    
    # Extract top MSBs for critical hardware failures
    clint_fault   = (val >> 127) & 1
    dma_lbl_fault = (val >> 126) & 1
    dma_img_fault = (val >> 125) & 1
    
    # Extract IMAGE_COUNT LSBs for image classification tracking
    img_faults = val & ((1 << IMAGE_COUNT) - 1)
    mismatches = bin(img_faults).count('1')
    
    # accuracy % = Image_count - number of 1s in fv (100LSB) / Image_count
    accuracy_metric =  (IMAGE_COUNT - mismatches) / IMAGE_COUNT
    
    return accuracy_metric, clint_fault, dma_lbl_fault, dma_img_fault

def generate_log_file(filename, bitstream, fv, status):
    """Generate an individual log file for each corrupt bitstream run."""
    with open(filename, 'w') as f:
        f.write(f"Corrupt_bit_file: {bitstream}\n")
        f.write(f"Fault vector from FPGA: {fv if fv else 'NONE'}\n")
        f.write(f"Test result: {status}\n")

def run_fault_campaign():
    st_time = time.time()
    print("==================================================")
    print("             FAULT SIMULATION CAMPAIGN            ")
    print("==================================================")

    # Refresh LOGS_DIR
    # shutil.rmtree(LOGS_DIR, ignore_errors=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # 1. Initialize UART and xsct
    # 1a. Open UART port
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=TIMEOUT_SEC) # Create serial object "ser"
        if ser.is_open:
            print (f"[*] Opened UART Serial Port")
            print (f"   Name: {ser.name}, Baudrate: {ser.baudrate}")
    except serial.SerialException as e:
        print(f"[!] Error opening UART: {e}")
        return

    # 1b. Launch XSCT session
    xsct_proc = start_xsct_session()
    if not xsct_proc:
        return

    # 2. Extract the Golden Fault Vector
    print("[*] Programming Golden Bitstream to obtain reference FV...")
    if not program_fpga(xsct_proc, GOLDEN_BITSTREAM):
        print("[!] Failed to program golden bitstream. Exiting.")
        return
        
    golden_fv, result_info = read_uart_for_fv(ser, golden_fv=None)
    
    if not golden_fv:
        print(f"[!] FATAL: Could not obtain legible Golden Fault Vector. Info: {result_info}")
        return
        
    print(f"    Golden Fault Vector obtained: {golden_fv}")

    # 3. Discover Corrupt Bitstreams
    corrupt_files = glob.glob(os.path.join(CORRUPT_BITSTREAMS_DIR, "*.bit"))
    total_tests = len(corrupt_files)
    if total_tests == 0:
        print("[!] No corrupt bitstreams found in directory. Exiting.")
        return
        
    print(f"[*] Found {total_tests} corrupt bitstreams. Starting campaign...")

    # 4. Open Summary CSV
    with open(TEST_RESULTS_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Test_ID', 
            'Corrupt_bitstream_filename', 
            'Fault_Vector', 
            'Accuracy', 
            'Result_Info', 
            'Test_Result',
            'CLINT_Fault',
            'DMA_Lbl_Fault',
            'DMA_Img_Fault'
        ])

        # 5. Main Testing Loop
        for i, bitstream_file in enumerate(corrupt_files):
            basename = os.path.basename(bitstream_file)
            print(f"\n--- Test {i+1}/{total_tests} : {basename} ---")
            
            # Program the board
            if not program_fpga(xsct_proc, bitstream_file):
                result_info = "XSCT Tool Error"
                fv = None
            else:
                # Read UART and assign fault category
                fv, result_info = read_uart_for_fv(ser, golden_fv)

            # Analyze valid fv
            acc, clint, dma_lbl, dma_img = ("N/A", "N/A", "N/A", "N/A")
            test_result = "Fail" # Default to Fail for crashes/garbage
            
            # Parse FV if it was categorized as a valid hex string match
            if fv and result_info in ["Matches Golden FV", "One or more faults"]:
                acc, clint, dma_lbl, dma_img = parse_fv(fv)
                # Define Pass as a match with Golden FV. Any data corruption or crash is a Fail.
                test_result = "Pass" if (result_info == "Matches Golden FV") else "Fail"
            # FV contains garbage data or partial text; leave hardware faults as "N/A"
            elif fv:
                test_result = "Fail"
            
            repr_fv = repr(fv) if fv else "NONE" # Representative symbols in case of raw binary fv
            
            # Write Summary CSV
            writer.writerow([
                i+1, 
                basename, 
                repr_fv if repr_fv else "NONE", 
                acc, 
                result_info, 
                test_result,
                clint, dma_lbl, dma_img
            ])
            
            # Write Individual Log
            log_filename = os.path.join(LOGS_DIR, f"log_{basename}.log")
            generate_log_file(log_filename, basename, repr_fv, test_result)
            
            print(f"  Result Info : {result_info}")
            print(f"  Test Result : {test_result}")
            if fv and acc != "N/A": print(f"  Accuracy    : {acc}")

    # Clean up
    ser.close()
    try:
        xsct_proc.stdin.write("EXIT\n")
        xsct_proc.stdin.flush()
        xsct_proc.wait(timeout=5)
    except:
        xsct_proc.kill()
        
    print("\n==================================================")
    print(f"Campaign Complete. Results saved to {TEST_RESULTS_CSV}")
    print(f"Individual logs saved in {LOGS_DIR}/")
    print(f"Total Time: {(time.time() - st_time)/60:.2f} minutes")

if __name__ == "__main__":
    run_fault_campaign()