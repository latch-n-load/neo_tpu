# ==============================================================================
# FILE: Makefile
# Verilator build with Verible linting and auto-correction
# Requires: verilator, verible-verilog-lint, verible-verilog-format
# ==============================================================================

# Directory Structure
SOURCE_DIR = src
TEST_DIR = test
OUTPUT_DIR = obj_dir
WAVE_DIR = waveforms
DATA_DIR = data

# Output Executable Settings
EXEC_NAME = Vmnist_tpu_tiled_classifier_tb
EXECUTABLE = $(OUTPUT_DIR)/$(EXEC_NAME)

# Tools
VERILATOR = verilator
VERIBLE = verible-verilog-lint
VERIBLE_FORMAT = verible-verilog-format

# Verilator flags (Includes all warning suppressions)
VERILATOR_FLAGS = -Wall \
                  -Wno-WIDTH -Wno-LITENDIAN -Wno-UNOPTFLAT \
                  -Wno-COMBDLY -Wno-INITIALDLY \
                  -Wno-UNUSEDSIGNAL -Wno-UNDRIVEN \
                  -Wno-PINCONNECTEMPTY -Wno-PINMISSING \
                  -Wno-DECLFILENAME -Wno-UNUSEDPARAM -Wno-BLKSEQ \
                  --trace --cc --exe --build -j 0 \
                  -CFLAGS "-O3 -march=native -I$(PWD)/$(DATA_DIR)" \
                  --top-module mnist_tpu_tiled_classifier_tb


# Verible format rules for tabular alignment
VERIBLE_FORMAT_FLAGS = \
    --module_net_variable_alignment=align \
    --assignment_statement_alignment=align \
    --port_declarations_alignment=align \
    --formal_parameters_alignment=align \
    --class_member_variable_alignment=align \
    --struct_union_members_alignment=align \
    --case_items_alignment=align \
    --named_port_alignment=align \
    --named_parameter_alignment=align

# Verible lint rules
VERIBLE_RULES = \
    --rules=always-comb \
    --rules=always-ff-non-blocking \
    --rules=case-missing-default \
    --rules=explicit-parameter-storage-type \
    --rules=forbid-consecutive-null-statements \
    --rules=forbid-defparam \
    --rules=generate-label-prefix \
    --rules=interface-name-style \
    --rules=module-begin-block \
    --rules=module-filename \
    --rules=no-trailing-spaces \
    --rules=no-tabs \
    --rules=packed-dimensions-range-ordering \
    --rules=parameter-name-style \
    --rules=positive-meaning-parameter-name \
    --rules=signal-name-style \
    --rules=unpacked-dimensions-range-ordering \
    --rules=void-cast \
    --rules=always-comb-blocking

# Disable some strict rules
VERIBLE_WAIVE = \
    --rules=-line-length \
    --rules=-macro-name-style \
    --rules=-signal-name-style

# Source files
VERILOG_SOURCES = $(SOURCE_DIR)/pe.v \
					$(SOURCE_DIR)/leaky_relu_child.v \
					$(SOURCE_DIR)/leaky_relu_parent.v \
					$(SOURCE_DIR)/leaky_relu_derivative_child.v \
					$(SOURCE_DIR)/leaky_relu_derivative_parent.v \
					$(SOURCE_DIR)/systolic.v \
					$(SOURCE_DIR)/bias_child.v \
					$(SOURCE_DIR)/bias_parent.v \
					$(SOURCE_DIR)/fixedpoint_simple.v \
					$(SOURCE_DIR)/control_unit.v \
					$(SOURCE_DIR)/unified_buffer.v \
					$(SOURCE_DIR)/vpu.v \
					$(SOURCE_DIR)/loss_parent.v \
					$(SOURCE_DIR)/loss_child.v \
					$(SOURCE_DIR)/tpu_mnist.v \
					$(SOURCE_DIR)/gradient_descent.v \
					$(SOURCE_DIR)/mnist_tpu_tiled_classifier.v

# Links to files
VERILOG_TB = $(TEST_DIR)/mnist_tpu_tiled_classifier_tb.v
CPP_TB = $(TEST_DIR)/test_main.cpp
ALL_VERILOG = $(VERILOG_SOURCES) $(VERILOG_TB)
WAVE_FILE = $(WAVE_DIR)/mnist_tpu_classifier.vcd
MNIST_HEADER = $(DATA_DIR)/mnist_image_data.h

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Log levels: 0 = Silent (Default), 1 = High (Verbose), 2 = Medium, 3 = Low
LOG_LEVEL ?= 0

.PHONY: all clean run wave lint format fix help distclean

# ==============================================================================
# Default target
# ==============================================================================
all: lint format fix build

# ==============================================================================
# Help
# ==============================================================================
help:
	@echo "$(BLUE)MNIST TPU Tiled Classifier - Build System$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@echo "  $(YELLOW)make help$(NC)     - Show this help message"
	@echo "  $(YELLOW)make all$(NC)      - Lint, format, fix, and build (default)"
	@echo "  $(YELLOW)make lint$(NC)     - Run Verible lint check only"
	@echo "  $(YELLOW)make format$(NC)   - Auto-format with Verible"
	@echo "  $(YELLOW)make fix$(NC)      - Apply Verible auto-fixes"
	@echo "  $(YELLOW)make build$(NC)    - Build with Verilator (no linting)"
	@echo "  $(YELLOW)make run$(NC)      - Run simulation (no trace)"
	@echo "  $(YELLOW)make wave$(NC)     - Run simulation with waveform trace"
	@echo "  $(YELLOW)make view$(NC)     - Open waveforms in GTKWave"
	@echo "  $(YELLOW)make clean$(NC)    - Clean all build artifacts"
	@echo ""

# ==============================================================================
# Verible lint check
# ==============================================================================
lint:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Verible Lint Check$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@$(VERIBLE) $(VERIBLE_RULES) $(VERIBLE_WAIVE) $(ALL_VERILOG) 2>&1 || true
	@echo ""
	@echo "$(GREEN)Verible lint check complete$(NC)"

# ==============================================================================
# Strict lint (fails on warnings)
# ==============================================================================
lint-strict:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Verible Strict Lint Check$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@$(VERIBLE) $(VERIBLE_RULES) $(VERIBLE_WAIVE) --fatal_errors $(ALL_VERILOG)
	@echo "$(GREEN)Verible strict lint passed$(NC)"

# ==============================================================================
# Auto-format with Verible
# ==============================================================================
format:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Verible Auto-Format$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@for f in $(ALL_VERILOG); do \
		echo "$(YELLOW)Formatting $$f...$(NC)"; \
		$(VERIBLE_FORMAT) $(VERIBLE_FORMAT_FLAGS) --inplace $$f; \
	done
	@echo ""
	@echo "$(GREEN)Formatting complete$(NC)"

# ==============================================================================
# Apply Verible auto-fixes
# ==============================================================================
fix:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Verible Auto-Fix$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(YELLOW)Scanning for fixable issues...$(NC)"
	@# Capture linter output. We use 2> to grab the patch suggestions. 
	@$(VERIBLE) $(VERIBLE_RULES) $(VERIBLE_WAIVE) --patch_context_lines=3 $(ALL_VERILOG) 2> verible_fixes.patch || true
	@if [ -s verible_fixes.patch ]; then \
		echo "$(YELLOW)Found fixes. Applying...$(NC)"; \
		yes | verible-patch-tool apply-pick verible_fixes.patch; \
		if [ $$? -eq 0 ]; then \
			echo "$(GREEN)Fixes applied successfully.$(NC)"; \
		else \
			echo "$(RED)Error: verible-patch-tool failed.$(NC)"; \
			rm -f verible_fixes.patch; exit 1; \
		fi; \
	else \
		echo "$(GREEN)No auto-fixable issues found.$(NC)"; \
	fi
	@rm -f verible_fixes.patch
	@echo ""

# ==============================================================================
# Fix common Verilator compatibility issues
# ==============================================================================
fix-verilator:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Verilator Compatibility Fixes$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@for f in $(ALL_VERILOG); do \
		echo "$(YELLOW)Fixing $$f for Verilator...$(NC)"; \
		\
		# Remove synthesis translate_off/on comments; \
		sed -i 's/\/\/\s*synthesis\s*translate_off/\/\/ synthesis translate_off removed/g' $$f; \
		sed -i 's/\/\/\s*synthesis\s*translate_on/\/\/ synthesis translate_on removed/g' $$f; \
		\
		# Ensure default_nettype none at top; \
		if ! grep -q '`default_nettype none' $$f; then \
			sed -i '1i `default_nettype none' $$f; \
		fi; \
		\
		# Add default_nettype wire at end if not present; \
		if ! grep -q '`default_nettype wire' $$f; then \
			echo '' >> $$f; \
			echo '`default_nettype wire' >> $$f; \
		fi; \
	done
	@echo ""
	@echo "$(GREEN)Verilator compatibility fixes applied$(NC)"

# ==============================================================================
# Verilator lint only
# ==============================================================================
verilator-lint:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Verilator Lint Check$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@$(VERILATOR) --lint-only -Wall $(ALL_VERILOG)
	@echo "$(GREEN)Verilator lint passed$(NC)"

# ==============================================================================
# Build with Verilator
# ==============================================================================
build: $(EXECUTABLE)

$(EXECUTABLE): $(ALL_VERILOG) $(CPP_TB) $(MNIST_HEADER)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Verilator Build$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	$(VERILATOR) $(VERILATOR_FLAGS) \
		$(ALL_VERILOG) $(CPP_TB) \
		-o $(EXEC_NAME)
	@echo ""
	@echo "$(GREEN)Build successful: $(EXECUTABLE)$(NC)"

# ==============================================================================
# Run simulation without trace
# ==============================================================================
run: $(EXECUTABLE)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Running Simulation$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	./$(EXECUTABLE) --log-level $(LOG_LEVEL)

# ==============================================================================
# Run simulation with waveform trace
# ==============================================================================
wave: $(EXECUTABLE)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Running Simulation with Trace$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@mkdir -p $(WAVE_DIR)
	./$(EXECUTABLE) --trace --wave-file $(WAVE_FILE) --log-level $(LOG_LEVEL)
	@echo "$(GREEN)Waveform successfully written to $(WAVE_FILE)$(NC)"

# ==============================================================================
# View waveforms
# ==============================================================================
view:
	@echo "$(BLUE)Opening waveform in GTKWave...$(NC)"
	@if [ -f $(WAVE_FILE) ]; then \
		if [ -f $(WAVE_DIR)/mnist_tpu_classifier.gtkw ]; then \
			gtkwave $(WAVE_FILE) $(WAVE_DIR)/mnist_tpu_classifier.gtkw & \
		else \
			echo "$(YELLOW)No .gtkw save file found. Opening default layout...$(NC)"; \
			gtkwave $(WAVE_FILE) & \
		fi \
	else \
		echo "$(RED)No waveform file found. Run 'make wave' first.$(NC)"; \
	fi

# ==============================================================================
# Full pipeline: format -> lint -> fix -> fix-verilator -> build
# ==============================================================================
pipeline:
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) fix
	@$(MAKE) fix-verilator
	@$(MAKE) build
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)  Full Pipeline Complete!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(YELLOW)Run 'make run' or 'make wave' to simulate$(NC)"

# ==============================================================================
# Check tool availability
# ==============================================================================
check-tools:
	@echo "$(BLUE)Checking tool availability...$(NC)"
	@which $(VERILATOR) > /dev/null 2>&1 && \
		echo "$(GREEN)✓ verilator$(NC)" || \
		echo "$(RED)✗ verilator not found$(NC)"
	@which $(VERIBLE) > /dev/null 2>&1 && \
		echo "$(GREEN)✓ verible-verilog-lint$(NC)" || \
		echo "$(RED)✗ verible-verilog-lint not found$(NC)"
	@which $(VERIBLE_FORMAT) > /dev/null 2>&1 && \
		echo "$(GREEN)✓ verible-verilog-format$(NC)" || \
		echo "$(RED)✗ verible-verilog-format not found$(NC)"
	@which gtkwave > /dev/null 2>&1 && \
		echo "$(GREEN)✓ gtkwave$(NC)" || \
		echo "$(YELLOW)⚠ gtkwave not found (optional)$(NC)"

# ==============================================================================
# Clean
# ==============================================================================
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf $(OUTPUT_DIR)
	rm -rf $(WAVE_DIR)/*.vcd
	rm -f verible_fixes.txt
	rm -f *.bak
	@echo "$(GREEN)Clean complete$(NC)"

# ==============================================================================
# Deep clean (including Verilator cache)
# ==============================================================================
distclean: clean
	@echo "$(YELLOW)Deep cleaning...$(NC)"
	rm -rf obj_dir_*
	rm -f .verilator_cache
	rm -rf $(WAVE_DIR)/*.vcd
	@echo "$(GREEN)Deep clean complete$(NC)"