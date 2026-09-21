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
UART_PORT = "/dev/ttyUSB5"      # TODO: Validate UART port using dmesg -w | grep tty
BAUD_RATE = 921600
CORRUPT_BITSTREAMS_DIR = "../corrupt_bit"
TEST_RESULTS_CSV = "../fault_test_results.csv"
LOGS_DIR = "../logs"            # Directory to store individual run logs
TIMEOUT_SEC = 30                # 30-second timeout
PROGRAM_FPGA_TCL = "program_fpga.tcl"

# Regex to capture exactly 32 hex characters
FV_REGEX = re.compile(r'([0-9a-fA-F]{32})') # Raw String re 32 chars, range 0-9, a-f, A-F
                                    # TODO if error: re.compile(r'^([0-9a-fA-F]{32})$')

def program_fpga(bitstream_path):
    """Flash the board via XSCT JTAG using TCL script."""
    try:
        subprocess.run(
            ["xsct", PROGRAM_FPGA_TCL, bitstream_path], 
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        print (f"   {os.path.basename(bitstream_path)} programmed succesfully to FPGA")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] XSCT Tool Error:\n{e.stderr.decode('utf-8')}")
        return False

def read_uart_for_fv(ser, golden_fv=None):
    """
    Read UART and categorize result into one of 6 defined states.
    Returns: (extracted_fv_string, result_info, raw_data_string)
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

    # Check for Legible Fault Vector
    match = FV_REGEX.search(decoded_output)
    if match:
        fv = match.group(1).lower()
        # print(f"[DEBUG] fv (match.group(1).lower()) = {fv}\n match.group(1) = {match.group(1)} ")
        # print(f"[DEBUG] golden_fv = {golden_fv} golden_fv.lower() = {golden_fv.lower()} ")
        if golden_fv and fv == golden_fv.lower():
            # ser.reset_input_buffer()
            return fv, "Match golden fv", decoded_output
        else:
            # ser.reset_input_buffer()
            return fv, "One or more faults", decoded_output

    # 6. Check for No Results (Timeout)
    if len(raw_buffer) == 0:
        return None, "No results obtained", decoded_output

    # Check for Binary Garbage
    # Heuristic: If more than 5% of characters are outside standard printable ASCII ranges
    non_ascii_count = sum(1 for b in raw_buffer if b > 127 or b < 8)
    if (non_ascii_count / len(raw_buffer)) > 0.05:
        # 4 & 5. Infinite vs Finite Garbage
        if len(raw_buffer) > 2000:
            # ser.reset_input_buffer()
            return None, "Infinite Garbage binary", decoded_output
        else:
            # ser.reset_input_buffer()
            return None, "Finite Garbage binary", decoded_output

    # 3. Missing UART Packets (Text exists, but no complete 32-char hex string)
    # ser.reset_input_buffer()
    return None, "Missing UART packets", decoded_output

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
    
    # Extract 100 LSBs for image classification tracking
    lsb_100 = val & ((1 << 100) - 1)
    mismatches = bin(lsb_100).count('1')
    
    # As requested: "accuracy % = number of 1s in fv (100LSB) / 100"
    accuracy_metric = mismatches / 100.0
    
    return accuracy_metric, clint_fault, dma_lbl_fault, dma_img_fault

def generate_log_file(filename, bitstream, fv, status):
    """Generates an individual log file for a specific corrupt bitstream run."""
    with open(filename, 'w') as f:
        f.write(f"Corrupt_bitstream_filename: {bitstream}\n")
        f.write(f"Fault vector obtained from FPGA: {fv if fv else 'NONE'}\n")
        f.write(f"Test result: {status}\n")

def run_fault_campaign():
    st_time = time.time()
    print("==================================================")
    print("             FAULT SIMULATION CAMPAIGN            ")
    print("==================================================")

    # Refersh LOGS_DIR
    shutil.rmtree(LOGS_DIR, ignore_errors=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # 1. Open the physical UART port
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=TIMEOUT_SEC) # Create serial object "ser"
        if ser.is_open:
            print (f"[*] Opened UART Serial Port")
            print (f"   Name: {ser.name}, Baudrate: {ser.baudrate}")
    except serial.SerialException as e:
        print(f"[!] Error opening UART: {e}")
        return

    # 2. Extract the Golden Fault Vector
    print("[*] Programming Golden Bitstream to obtain reference FV...")
    if not program_fpga(GOLDEN_BITSTREAM):
        print("[!] Failed to program golden bitstream. Exiting.")
        return
        
    golden_fv, result_info, _ = read_uart_for_fv(ser, golden_fv=None)
    
    if not golden_fv:
        print(f"[!] FATAL: Could not obtain legible Golden Fault Vector. Info: {result_info}")
        return
        
    print(f"[+] Golden Fault Vector obtained: {golden_fv}")

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
            'Info_Level_1', 
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
            if not program_fpga(bitstream_file):
                result_info = "XSCT Tool Error"
                fv = None
            else:
                # Read UART using the 6-state logic
                fv, result_info, raw_out = read_uart_for_fv(ser, golden_fv)

            # Analyze valid vectors
            acc, clint, dma_lbl, dma_img = ("N/A", "N/A", "N/A", "N/A")
            test_result = "Fail" # Default to Fail for crashes/garbage
            
            if fv:
                acc, clint, dma_lbl, dma_img = parse_fv(fv)
                # Define Pass as an exact match to Golden FV. Any data corruption or crash is a Fail.
                test_result = "Pass" if (result_info == "Match golden fv") else "Fail"
            
            # Write Summary CSV
            writer.writerow([
                i, 
                basename, 
                fv if fv else "NONE", 
                acc, 
                result_info, 
                test_result,
                clint, dma_lbl, dma_img
            ])
            
            # Write Individual Log
            log_filename = os.path.join(LOGS_DIR, f"log_{basename}.txt")
            generate_log_file(log_filename, basename, fv, test_result)
            
            print(f"  Info Level  : {result_info}")
            print(f"  Test Result : {test_result}")
            if fv: print(f"  Accuracy    : {acc}")

    ser.close()
    print("\n==================================================")
    print(f"Campaign Complete. Results saved to {TEST_RESULTS_CSV}")
    print(f"Individual logs saved in {LOGS_DIR}/")
    print(f"Total Time: {time.time() - st_time:.2f} seconds")

if __name__ == "__main__":
    run_fault_campaign()