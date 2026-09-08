# This script receives bitstream as arguments from Python subprocess and programs
# PYNQ-Z2 board using XSCT

# Grab the bitstream name passed from the Python subprocess
set bitstream_file [lindex $argv 0]

# Connect to the physical JTAG hardware 
connect
# TODO check if name matches FPGA
targets -set -filter {name =~ "xc7z020*"}

# Attempt to program the FPGA and catch any hardware/file errors
if { [catch {fpga -file $bitstream_file} result] } {
    puts "FPGA_PROGRAM_ERROR: Failed to push bitstream. Reason: $result"
    disconnect
    exit 1
}

# Verify configuration status
# The -state flag returns 1 if configured, 0 if not configured.
set is_configured [fpga -state]
if { $is_configured == 0 } {
    puts "FPGA_PROGRAM_ERROR: Bitstream pushed, but FPGA DONE pin did not assert!"
    disconnect
    exit 1
}

# Exit successfully
disconnect
exit 0