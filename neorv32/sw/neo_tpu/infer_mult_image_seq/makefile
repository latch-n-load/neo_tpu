# Application makefile.
# Use this makefile to configure all relevant CPU / compiler options.

# Use this statement in shell to make firmware:
# make clean_all all

# Use # Use this statement in run vsim in console mode:
# make vsim

# Override the default CPU ISA
MARCH = rv32i_zicsr_zifencei

# Override the default RISC-V GCC prefix
#RISCV_PREFIX ?= riscv-none-elf-

# Override default optimization goal
EFFORT = -Os

# Add extended debug symbols
USER_FLAGS += -ggdb -gdwarf-3

# Adjust processor IMEM size
USER_FLAGS += -Wl,--defsym,__neorv32_rom_size=16k

# Adjust processor DMEM size
USER_FLAGS += -Wl,--defsym,__neorv32_ram_size=8k

# Adjust maximum heap size
#USER_FLAGS += -Wl,--defsym,__neorv32_heap_size=1k

# Additional sources
#APP_SRC += $(wildcard ./*.c)
#APP_INC += -I .

# Ale
# USER_FLAGS += -DUART0_SIM_MODE

# GHDL not used, simulation is in modelsim
# GHDL_RUN_FLAGS += --wave=neo_tpu.ghw 

# Set path to NEORV32 root directory
NEORV32_HOME ?= ../../..

# Include the main NEORV32 makefile
include $(NEORV32_HOME)/sw/common/common.mk

# =========================================================
# Target for launching vsim

# Colors for log
LOG_YELLOW = \033[1;33m
# LOG_RED    = \033[1;31m
LOG_GREEN  = \033[1;32m
LOG_NC  = \033[0m

# Default simulation values
SIM_TIME ?= -all
WAVE_EN  ?= 0

.PHONY: vsim
vsim:
	@printf "=================================================\n"
	@printf "$(LOG_GREEN) Launching vsim simulation $(LOG_NC)\n"
	@printf "=================================================\n"
	@printf "$(LOG_YELLOW) Simulation Time : $(SIM_TIME)$(LOG_NC)\n"
	@printf "$(LOG_YELLOW) Wave Generation : $(WAVE_EN)$(LOG_NC)\n"
	@printf "=================================================\n"
	@SIM_TIME=$(SIM_TIME) WAVE_EN=$(WAVE_EN) $(NEORV32_HOME)/sim/vsim.sh

# /Ale