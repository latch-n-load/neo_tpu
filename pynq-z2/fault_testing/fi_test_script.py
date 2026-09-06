import subprocess
import serial
import time
import csv
import os

# --- Configuration ---
GOLDEN_BITSTREAM = "neo_tpu_golden.bit"
UART_PORT = "/dev/ttyUSB0"      # Update to your actual Ubuntu port
BAUD_RATE = 19200
TEST_ITERATIONS = 10000
EXPECTED_OUTPUT = "Perfect Classification! 0 errors encountered."

def run_fault_campaign():
    print("Starting Fault Injection Campaign...")
    
    # 1. Open the physical UART port
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=5)
    except Exception as e:
        print(f"Error opening UART: {e}")
        return

    # 2. Open a CSV file to store the statistics for all 10,000 runs
    with open('fault_campaign_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['Test_ID', 'Corrupted_File', 'Status', 'Raw_Output'])

        # 3. Main Testing Loop
        for i in range(TEST_ITERATIONS):
            corrupted_bit_name = f"corrupted_bitstreams/test_{i}.bit"
            
            # --- STEP A: Bitstream Corruption ---
            # Using subprocess to call the lab's "pixel" tool in the terminal.
            # (You will need to adjust the arguments to match how 'pixel' actually works)
            subprocess.run(["pixel", "--input", GOLDEN_BITSTREAM, "--output", corrupted_bit_name])
            
            # --- STEP B: Flash the Board ---
            # Call Vivado's XSCT tool. We pass it a small Tcl script that handles the JTAG connection.
            subprocess.run(["xsct", "program_fpga.tcl", corrupted_bit_name])
            
            # --- STEP C: Read UART Output ---
            # Clear out any junk data sitting in the buffer before we listen
            ser.reset_input_buffer()
            
            # Give the NEORV32 time to boot, run the inference, and print over UART
            # Adjust this sleep time based on your pipeline's actual latency
            time.sleep(3) 
            
            # Read everything the board sent us
            raw_output = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            
            # --- STEP D: Evaluate and Log ---
            if EXPECTED_OUTPUT in raw_output:
                status = "PASS"
            elif raw_output.strip() == "":
                status = "FATAL_CRASH (No Output)"
            else:
                status = "CORRUPTED_INFERENCE"
                
            writer.writerow([i, corrupted_bit_name, status, raw_output])
            print(f"Test {i}/{TEST_ITERATIONS} Completed | Status: {status}")

    ser.close()
    print("Campaign Complete. Results saved to fault_campaign_results.csv")

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs("corrupted_bitstreams", exist_ok=True)
    run_fault_campaign()