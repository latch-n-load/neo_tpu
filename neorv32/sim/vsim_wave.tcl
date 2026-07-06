# ==============================================================================
# ModelSim / Questa Interactive Simulation Tcl Script
# ==============================================================================

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

# 1. Log data to memory WITHOUT adding it to the visual wave window
# -r enables recursion.
log -r -depth 2 /neorv32_tb/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/*

# Hide Numeric_std warnings
set NumericStdNoWarnings 1

# 2. Run the Simulation
# Check if a custom simulation time was passed from the bash shell
if {[info exists env(SIM_TIME)]} {
    set run_time $env(SIM_TIME)
} else {
    set run_time "-all" ;
}

echo "Running simulation for: $run_time"
run $run_time

# 3. Open required GUI panes
view structure
view signals
view wave

# 4. Clear any existing signals from the wave window
delete wave *

config wave -signalnamewidth 1

# 5. Populate the wave window with the logged data
add wave -divider "NEORV32 CFS"
add wave /neorv32_tb/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/*

add wave -divider "TPU Classifier"
add wave /neorv32_tb/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/classifier_inst/*

# 6. Adjust zoom so you can see the results immediately
wave zoom full

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