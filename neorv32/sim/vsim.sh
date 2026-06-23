#!/usr/bin/env bash

set -e

# Jump to the directory where this script lives
CUR_DIR=$(dirname "$0")
BUILD_DIR="${CUR_DIR}/build_vsim"
rm -rf "${BUILD_DIR}"

# Define vsim toolchain executables (assumes they are in your PATH)
VLIB="${VLIB:-vlib}"
VMAP="${VMAP:-vmap}"
VLOG="${VLOG:-vlog}"
VCOM="${VCOM:-vcom}"
VSIM="${VSIM:-vsim}"

# Path to the NEORV32 file list
FILE_LIST="../../rtl/file_list_soc.f"

# 1. Create Build Directory & Library
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
echo "Setting up vsim library..."
$VLIB neorv32
$VMAP neorv32 neorv32

# 2. Compile ALL TPU Verilog Files
echo "Compiling Verilog TPU files..."
$VLOG -work neorv32 ../../rtl/tpu_integration/*.v 2>/dev/null || true

# 3. Parse .f file and Compile VHDL Files in Strict Order
echo "Parsing $FILE_LIST and compiling VHDL files..."

if [ ! -f "$FILE_LIST" ]; then
  echo "ERROR: File list not found at $FILE_LIST"
  exit 1
fi

# Read the file line by line
while IFS= read -r line || [ -n "$line" ]; do
  # Clean the line: remove Windows carriage returns
  clean_line=$(echo "$line" | tr -d '\r')

  # Skip empty lines and comments (lines starting with # or //)
  if [[ -z "$clean_line" ]] || [[ "$clean_line" == \#* ]] || [[ "$clean_line" == //* ]]; then
    continue
  fi

  # Replace the placeholder text with the actual relative path to the rtl folder
  target_file="${clean_line/NEORV32_RTL_PATH_PLACEHOLDER/..\/..\/rtl}"

  # Check if the file exists before compiling
  if [ -f "$target_file" ]; then
    $VCOM -work neorv32 -2008 "$target_file"
  else
    echo "WARNING: File $target_file not found. Skipping."
  fi

done < "$FILE_LIST"

# 4A. Compile all simulation helper files EXCEPT the main testbench
echo "Compiling simulation helper files..."
find ../../sim -type f -name '*.vhd' ! -name 'neorv32_tb.vhd' -exec $VCOM -work neorv32 -2008 {} +

# 4B. Compile the main testbench last
echo "Compiling main testbench..."
$VCOM -work neorv32 -2008 ../../sim/neorv32_tb.vhd

# 5. Prepare and Run Simulation
if [ -z "$1" ]
  then
    VSIM_RUN_ARGS="-do \"run 10ms; quit\""
  else
    VSIM_RUN_ARGS=$@
fi

echo "Vsim simulation run parameters: $VSIM_RUN_ARGS";
runcmd="$VSIM -c -work neorv32 neorv32_tb $VSIM_RUN_ARGS"

if [ -n "$VSIM_NOLOG" ]; then
  eval "$runcmd"
else
  eval "$runcmd" 2>&1 | tee vsim.log
fi