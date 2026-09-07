# This script receives bitstream as arguments from Python subprocess and programs
# PYNQ-Z2 board using XSCT

# Grab the bitstream name passed from the Python subprocess
set bitstream_file [lindex $argv 0]

# Connect to the physical JTAG hardware 
connect
# TODO check if name matches FPGA
targets -set -filter {name =~ "xc7z020*"}
fpga -file $bitstream_file

# Disconnect gracefully
disconnect
exit