# TODO CHECK IF PHASE 2 AND PHASE 3 RUN PROPERLY.

import serial
import time
import csv
import os
import re
import random
import utils # Import utility functions

# --- Configuration ---
GOLDEN_BITSTREAM = "/home/a_akif/tesi/neo_tpu_pynq2/neo_tpu_pynq2.runs/impl_1/neo_tpu_pynq_wrapper.bit"
LL_FILE = "/home/a_akif/tesi/neo_tpu_pynq2/neo_tpu_pynq2.runs/impl_1/neo_tpu_pynq_wrapper.ll"
EBD_FILE = "/home/a_akif/tesi/neo_tpu_pynq2/neo_tpu_pynq2.runs/impl_1/neo_tpu_pynq_wrapper.ebd" 

# TODO: Validate UART port using sudo dmesg -w | grep tty or ls -l /dev/serial/by-id/
UART_PORT = "/dev/ttyUSB10"
BAUD_RATE = 921600
TIMEOUT_SEC = 5 # Wait for UART response
PROGRAM_FPGA_TCL = "program_fpga.tcl"    

# --- Campaign Phase & Pipeline Configuration ---
CAMPAIGN_PHASE = 2
LL_TARG_NODE = "neorv32_cfs_inst" 
MAX_PHASE1_TARGS = 10000 # Maximum targets of LL_TARG_NODE extracted from .ll corrupted in .bit
MAX_PHASE2_TARGS = 10000 # Maximum targets of essential bits extracted from .ebd corrupted in .bit
MAX_PHASE3_TARGS = 10000 # Maximum targets of untested bits corrupted directly in .bit
BATCH_SIZE = 100 # Number of bitstreams to generate and test in one batch

DESIGN_NAME = "neo_cfs_tpu"
CORRUPT_BITSTREAMS_DIR = f"../corrupt_bit_ph{CAMPAIGN_PHASE}_{DESIGN_NAME}_{time.strftime('%y%m%d')}"
RESULTS_DIR = "../results/"
LOGS_DIR = f"../logs_ph{CAMPAIGN_PHASE}_{DESIGN_NAME}_{time.strftime('%y%m%d')}" # Directory to store individual run logs  

# REGEX for 32 HEX on UART_o, from neo_cfs_tpu
FV_REGEX = re.compile(r'([0-9a-fA-F]{32})')

def generate_campaign_targets(total_payload_bits):
    """Parses files ONCE and returns a unified list of (abs_bit_offset, ll_info) tuples."""
    targets = []
    
    if CAMPAIGN_PHASE == 1:
        print(f"\n[*] [PHASE 1] Extracting Target Node: '{LL_TARG_NODE}'")
        ph1_all_targs = utils.get_ll_targs(LL_FILE, MAX_PHASE1_TARGS, target_keyword=LL_TARG_NODE)
        if not ph1_all_targs: return []
        
        sample_size = min(MAX_PHASE1_TARGS, len(ph1_all_targs))
        sampled = random.sample(ph1_all_targs, sample_size) if len(ph1_all_targs) > sample_size else ph1_all_targs
        targets = [(t['abs_bit_offset'], t['info']) for t in sampled]

    elif CAMPAIGN_PHASE == 2:
        print(f"\n[*] [PHASE 2] Extracting Structural Bits from .ebd")
        ebd_bits = utils.get_essential_bits(EBD_FILE)
        if not ebd_bits: return []
        
        sample_size = min(MAX_PHASE2_TARGS, len(ebd_bits))
        sampled = random.sample(list(ebd_bits), sample_size)
        targets = [(offset, "N/A") for offset in sampled]

    elif CAMPAIGN_PHASE == 3:
        print(f"\n[*] [PHASE 3] Generating Random Payload Targets")
        ph3_exclude = [] # .ll and .ebd exclusions disabled
        sample_size = MAX_PHASE3_TARGS
        
        selected = set()
        while len(selected) < sample_size:
            candidate = random.randint(0, total_payload_bits - 1)
            if candidate not in ph3_exclude:
                selected.add(candidate)
        targets = [(offset, "N/A") for offset in selected]
        
    return targets

def run_pipelined_campaign():
    st_time = time.time()
    print("==================================================")
    print(f"        FAULT SIMULATION ({DESIGN_NAME.upper()}) ")
    print("==================================================")
    
    os.makedirs(CORRUPT_BITSTREAMS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    utils.cleanup_batch(CORRUPT_BITSTREAMS_DIR) # Clear out old runs
    utils.cleanup_batch(LOGS_DIR) # Clear out old runs

    # 1. Initialize Hardware
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=TIMEOUT_SEC)
        if ser.is_open:
            print (f"[*] Opened UART Serial Port")
            print (f"    Name: {ser.name}, Baudrate: {ser.baudrate}")
    except serial.SerialException as e:
        print(f"[!] Error opening UART: {e}")
        return
        
    xsct_proc = utils.start_xsct_session(PROGRAM_FPGA_TCL)
    if not xsct_proc: return

    # 2. Load & Prepare Golden Bitstream
    with open(GOLDEN_BITSTREAM, 'rb') as f:
        golden_data = bytearray(f.read())

    # Check if board is ZYNQ-7000
    if utils.find_sync_word(golden_data) == -1:return

    # Get FDRI start and total payload
    fdri_start, total_payload = utils.get_fdri_data_start(golden_data)
    if fdri_start == -1: return

    # Disable CRC (neo_cfs_tpu DOESN'T have it) - Just checking
    golden_data = utils.disable_tail_crc(golden_data, fdri_start, total_payload)

    # 3. Get Golden Fault Vector
    print("[*] Programming Golden Bitstream to obtain reference FV...")
    if not utils.program_fpga(xsct_proc, GOLDEN_BITSTREAM):
        print("[!] Failed to program golden bitstream. Exiting.")
        return
    golden_fv, result_info = utils.read_uart_hex(ser, FV_REGEX, TIMEOUT_SEC)
    if not golden_fv:
        print(f"[!] Failed to obtain legible Golden Fault Vector. Info: {result_info}")
        return
    print(f"    Golden Fault Vector obtained:\n    {golden_fv}")

    # 4. Generate Target Pool (Parse Files ONCE)
    campaign_targets = generate_campaign_targets(total_payload)
    total_tests = len(campaign_targets)
    if total_tests == 0:
        print("[!] No valid targets found for the campaign. Exiting.")
        return
    print(f"[*] Starting Pipelined Campaign: {total_tests} total tests in batches of {BATCH_SIZE}.")

    # 5. Open Summary CSV
    result_path = os.path.join(RESULTS_DIR, f"fi_results_ph{CAMPAIGN_PHASE}_{DESIGN_NAME}_{time.strftime('%y%m%d')}.csv"),
    with open(result_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Test_ID', 'Corrupt_bitstream_filename', 'Fault_Vector', 'Accuracy','Result_Info',
             'Test_Result', 'CLINT_Fault', 'DMA_Lbl_Fault', 'DMA_Img_Fault', 'LL_Info'
        ])

        # 6. Pipeline Main Loop
        for batch_start in range(0, total_tests, BATCH_SIZE):
            utils.cleanup_batch(CORRUPT_BITSTREAMS_DIR) # Delete previous batch files
            batch_targets = campaign_targets[batch_start : batch_start + BATCH_SIZE]
            batch_metadata = {}
            
            print(f"\n[*] Generating Batch {batch_start//BATCH_SIZE + 1}...")
            
            # --- GENERATION PHASE ---
            for i, (abs_offset, info) in enumerate(batch_targets):
                test_id = batch_start + i + 1
                basename = f"seu_ph{CAMPAIGN_PHASE}_{test_id}_{abs_offset}.bit"
                out_path = os.path.join(CORRUPT_BITSTREAMS_DIR, basename)
                
                utils.generate_corrupt_bitstream(golden_data, abs_offset, fdri_start, out_path)
                batch_metadata[basename] = info
            print(f"    Generated {BATCH_SIZE} corrupt bitstreams for Batch {batch_start//BATCH_SIZE + 1}")
                
            # --- SIMULATION PHASE ---
            for basename, ll_info in batch_metadata.items():
                test_idx = batch_start + list(batch_metadata.keys()).index(basename) + 1
                print(f"\n--- Test {test_idx}/{total_tests} : {basename} ---")
                
                bit_path = os.path.join(CORRUPT_BITSTREAMS_DIR, basename)
                
                if not utils.program_fpga(xsct_proc, bit_path):
                    result_info = "XSCT Tool Error"
                    fv = None
                else:
                    fv, result_info = utils.read_uart_hex(ser, FV_REGEX, TIMEOUT_SEC, golden_fv)

                acc, clint, dma_lbl, dma_img = ("N/A", "N/A", "N/A", "N/A")
                test_result = "Fail"
                
                if fv and result_info in ["Matches Golden FV", "One or more faults"]:
                    acc, clint, dma_lbl, dma_img = utils.parse_hex_fv(fv)
                    test_result = "Pass" if (result_info == "Matches Golden FV") else "Fail"
                elif fv:
                    test_result = "Fail"

                repr_fv = repr(fv) if fv else "NONE"
            
                # Write Outputs
                writer.writerow([
                    test_idx, basename, repr_fv, acc, result_info, test_result, 
                    clint, dma_lbl, dma_img, ll_info
                ])
                log_file = os.path.join(LOGS_DIR, f"log_{basename}.log")
                utils.write_hex_log(log_file, basename, repr_fv, test_result, ll_info)
                
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
    print(f"Campaign Complete. Results saved to {result_path}")
    print(f"Total Time: {(time.time() - st_time)/60:.2f} minutes")

if __name__ == "__main__":
    run_pipelined_campaign()