# This script receives bitstream over stdin from Python, stdout info, 
# and programs PYNQ-Z2 board using XSCT

connect
# TODO check if name matches FPGA
# Example
# 1  Xilinx HW-USB-II-G 0000185e495201
#      2  arm_dap (idcode: 4ba06477)
#      3  xczu9eg (idcode: 14730093)  
#   4  Digilent JTAG-SMT2 210279A42321
#      5  arm_dap (idcode: 4ba00477)
#      6  xc7z020 (idcode: 23727093)      <-- (My XC7Z020!)
targets -set -filter {name =~ "xc7z020"}

# Signal to Python that initialization is complete
puts "READY"
flush stdout

# Enter infinite loop
while {1} {
    # Wait for Python to send a bitstream path over stdin
    set bitstream_file [gets stdin]
    puts "Received .bit $bitstream_file"

    # Allow Python to cleanly close the session
    if {$bitstream_file == "EXIT"} {
        break
    }

    # Ignore empty lines
    if {$bitstream_file == ""} {
        continue
    }

    # Program the FPGA with error stdout
    if { [catch {fpga -file $bitstream_file} result] } {
        puts "FPGA_PROGRAM_ERROR: Failed to push bitstream. Reason: $result"
        flush stdout
        continue
    }

    # Get configuration status and stdout
    set is_configured [fpga -state]
    if { $is_configured != "FPGA is configured" } {
        puts "FPGA_PROGRAM_ERROR: Bitstream pushed, but FPGA DONE pin did not assert!"
        flush stdout
        continue
    }
    # Else stdout programming successful
    puts "PROGRAM_DONE"
    flush stdout
}

disconnect
exit 0