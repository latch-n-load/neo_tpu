#!/usr/bin/env bash
# Color Definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m' # Bold yellow stands out bright
NC='\033[0m'        # No Color (Resets the terminal back to normal)

set -e

# Setup build directory
CUR_DIR=$(dirname "$0")
BUILD_DIR="${CUR_DIR}/vsim_build"
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
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}" || exit 1
echo -e "${YELLOW}Working directory: $(pwd)${NC}"

echo -e "${YELLOW}[INFO] Setting up vsim libraries...${NC}"
$VLIB neorv32
$VMAP neorv32 neorv32
$VLIB tiny_tpu
$VMAP tiny_tpu tiny_tpu

# 2. Compile ALL TPU Verilog Files
echo -e "${YELLOW}[INFO] Compiling Verilog TPU files...${NC}"
$VLOG -work tiny_tpu  "../../../tiny-tpu/mnist_demo/rtl/*.v"

# Currently Unused: In case some glue logic is needed
# echo -e "${YELLOW}[INFO] Compiling NEO-TPU integration modules...${NC}"
# $VLOG -work tiny_tpu "../../../rtl/tpu_integration/*.v"

# 3. Parse .f file and Compile VHDL Files in Strict Order
echo -e "${YELLOW}[INFO] Parsing $FILE_LIST and compiling NEORV32 VHDL files...${NC}"

if [ ! -f "$FILE_LIST" ]; then
  echo -e "${RED}[ERROR] File list not found at $FILE_LIST${NC}"
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
    echo -e "${ORANGE}[WARNING] File $target_file not found. Skipping.${NC}"
  fi

done < "$FILE_LIST"

# 4A. Compile all simulation helper files EXCEPT the main testbench
echo -e "${YELLOW}[INFO] Compiling simulation helper files...${NC}"
find ../../sim -type f -name '*.vhd' ! -name 'neorv32_tb.vhd' -exec $VCOM -work neorv32 -2008 {} +

# 4B. Compile the main testbench last
echo -e "${YELLOW}[INFO] Compiling main testbench...${NC}"
$VCOM -work neorv32 -2008 ../../sim/neorv32_tb.vhd

# Check user argument
if [ -n "$1" ]; then
  export SIM_TIME="$1"
else 
  export SIM_TIME="all"
fi

# Check for Waveforms (Argument 2)
if [ "$2" == "wave" ]; then
  echo -e "${YELLOW}[INFO] Running vsim simulation in Wave Mode...${NC}";
  runcmd="$VSIM -voptargs="+acc" -t ns -L neorv32 -L tiny_tpu neorv32.neorv32_tb -do ../vsim_wave.tcl"
else
  echo -e "${YELLOW}[INFO] Running vsim simulation in Console Mode...${NC}";
  runcmd="$VSIM -voptargs="+acc" -c -t ns -L neorv32 -L tiny_tpu neorv32.neorv32_tb -do \"run \$SIM_TIME; quit -f\""
fi

eval "$runcmd" 2>&1 | tee vsim.log
