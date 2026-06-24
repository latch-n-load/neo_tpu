# ==============================================================================
# ModelSim / Questa Interactive Simulation Tcl Script
# ==============================================================================

# 1. Open the necessary GUI windows
view structure
view signals
view wave

# 2. Clear any existing signals from the wave window
delete wave *

# 3. Add Waveforms to the Window
# Customize these paths to match your exact testbench structure
# add wave -divider "System Clock & Reset"
# add wave -hex /neorv32_tb/clk_gen_i/clk
# add wave -hex /neorv32_tb/rst_gen_i/rstn

# add wave -divider "NEORV32 CPU Core"
# add wave -hex /neorv32_tb/neorv32_top_inst/neorv32_cpu_inst/pc_f

# add wave -divider "Custom Functions Subsystem (CFS)"
# add wave -hex /neorv32_tb/neorv32_top_inst/neorv32_cfs_inst/cfs_reg_wr_o
# add wave -hex /neorv32_tb/neorv32_top_inst/neorv32_cfs_inst/cfs_reg_addr_o
# add wave -hex /neorv32_tb/neorv32_top_inst/neorv32_cfs_inst/cfs_reg_data_o

# add wave -divider "Tiny-TPU Accelerator"
# Note: Adjust the internal path below to match where your Verilog TPU is instantiated inside CFS
# add wave -hex /neorv32_tb/neorv32_top_inst/neorv32_cfs_inst/YOUR_TPU_INST_NAME/*

add wave /neorv32_tb/neorv32_top/*
# 4. Run the Simulation
run 10ms

# 5. Control Wave Window Zoom 
# Options: 'wave zoom full' fits everything, or specify a precise time window range
wave zoom full
# Alternative example: wave zoom range 0ms 2ms

# 6. Add and Position a Simulation Cursor
# Create a named cursor, place it at a designated time, and focus the view on it
# wave cursor add -time 1ms -name "Inference_Start"
# wave cursor configure "Inference_Start" -color Cyan

# 7. Optional: Search for a specific signal value and jump there
# This searches forward from time 0 for the first time the CFS address matches your target
# set match_time [search_wave -start 0ns -pattern {/neorv32_tb/neorv32_top_inst/neorv32_cfs_inst/cfs_reg_addr_o == 32'h00000000}]
# if {$match_time ne ""} {
#     wave cursor add -time $match_time -name "First_CFS_Write"
#     wave cursor configure "First_CFS_Write" -color Yellow
# }