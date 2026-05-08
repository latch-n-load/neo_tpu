# ==============================================================================
# FILE: Makefile
# Verilator build with Verible linting and auto-correction
# Requires: verilator, verible-verilog-lint, verible-verilog-format
# ==============================================================================

# Directory Structure
SOURCE_DIR = src
TEST_DIR = test
OUTPUT_DIR = obj_dir

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
                  -CFLAGS "-O3 -march=native" \
                  --top-module mnist_tpu_tiled_classifier_tb

# Verible lint rules (Defined individually to prevent parsing errors)
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

# Waive certain rules for legacy or test files
VERIBLE_WAIVE = --rules=-line-length

# File Discoveries
SRCS = $(wildcard $(SOURCE_DIR)/*.v)
TEST_SRCS = $(wildcard $(TEST_DIR)/*.v) $(wildcard $(TEST_DIR)/*.cpp)
ALL_VERILOG = $(wildcard $(SOURCE_DIR)/*.v) $(wildcard $(TEST_DIR)/*.v)

# Color Codes for Pretty Output
BLUE = \033[1;34m
GREEN = \033[1;32m
YELLOW = \033[1;33m
RED = \033[1;31m
NC = \033[0m # No Color

# ==============================================================================
# Default target - Runs full styling, patching, compilation, and simulation
# ==============================================================================
all: lint format fix fix-verilator build wave

# ==============================================================================
# Step 1: Linting
# ==============================================================================
lint:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Verible Lint Check$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@$(VERIBLE) $(VERIBLE_RULES) $(VERIBLE_WAIVE) $(ALL_VERILOG) 2>&1 || true
	@echo ""
	@echo "$(GREEN)Verible lint check complete$(NC)"

# ==============================================================================
# Step 2: Code Formatting
# ==============================================================================
format:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Verible Code Formatting$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@for f in $(ALL_VERILOG); do \
		echo "$(YELLOW)Formatting $$f...$(NC)"; \
		$(VERIBLE_FORMAT) --inplace $$f; \
	done
	@echo ""
	@echo "$(GREEN)Formatting complete$(NC)"

# ==============================================================================
# Step 3: Apply Verible Auto-Fixes
# ==============================================================================
fix:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Verible Auto-Fix$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(YELLOW)Scanning for fixable issues...$(NC)"
	@$(VERIBLE) $(VERIBLE_RULES) $(VERIBLE_WAIVE) --patch_context_lines=3 $(ALL_VERILOG) 2> verible_fixes.patch || true
	@if [ -s verible_fixes.patch ]; then \
		echo "$(YELLOW)Found fixes. Applying automatically...$(NC)"; \
		yes | verible-patch-tool apply-pick verible_fixes.patch; \
		if [ $$? -eq 0 ]; then \
			echo "$(GREEN)Fixes applied successfully.$(NC)"; \
		else \
			echo "$(RED)Error: verible-patch-tool failed during apply-pick.$(NC)"; \
			rm -f verible_fixes.patch; exit 1; \
		fi; \
	else \
		echo "$(GREEN)No auto-fixable issues found.$(NC)"; \
	fi
	@rm -f verible_fixes.patch
	@echo ""

# ==============================================================================
# Step 4: Fix common Verilator compatibility issues
# ==============================================================================
fix-verilator:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Verilator Compatibility Fixes$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@for f in $(ALL_VERILOG); do \
		echo "$(YELLOW)Fixing $$f for Verilator...$(NC)"; \
		sed -i 's/\/\/\s*synthesis\s*translate_off/\/\/ synthesis translate_off removed/g' $$f; \
		sed -i 's/\/\/\s*synthesis\s*translate_on/\/\/ synthesis translate_on removed/g' $$f; \
		if ! grep -q '`default_nettype none' $$f; then \
			sed -i '1i `default_nettype none' $$f; \
		fi; \
		if ! grep -q '`default_nettype wire' $$f; then \
			echo '' >> $$f; \
			echo '`default_nettype wire' >> $$f; \
		fi; \
	done
	@echo ""
	@echo "$(GREEN)Verilator compatibility fixes applied$(NC)"

# ==============================================================================
# Step 5: Build Verilator Model
# ==============================================================================
build:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Building Verilator Model$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	$(VERILATOR) $(VERILATOR_FLAGS) $(SRCS) $(TEST_SRCS)
	@echo ""
	@echo "$(GREEN)Build complete: $(EXECUTABLE)$(NC)"

# ==============================================================================
# Step 6: Run simulation and generate waveform file
# ==============================================================================
wave:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Running Simulation & Generating Wave$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@if [ -f $(EXECUTABLE) ]; then \
		$(EXECUTABLE) +trace; \
		echo "$(GREEN)Simulation complete. Waveform generated as mnist_tpu_classifier.vcd$(NC)"; \
	else \
		echo "$(RED)Error: Executable not found. Please run 'make build' first.$(NC)"; \
		exit 1; \
	fi
	@echo ""

# ==============================================================================
# Optional step: View waves in GTKWave
# ==============================================================================
view:
	@echo "$(BLUE)Opening waveform in GTKWave...$(NC)"
	@if [ -f mnist_tpu_classifier.vcd ]; then \
		gtkwave mnist_tpu_classifier.vcd & \
	else \
		echo "$(RED)Error: mnist_tpu_classifier.vcd not found. Run 'make wave' first.$(NC)"; \
	fi

# ==============================================================================
# Pipeline targets for segmented runs
# ==============================================================================
pipeline: lint format fix fix-verilator build

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
	rm -f mnist_tpu_classifier.vcd
	rm -f verible_fixes.patch
	rm -f *.bak
	@echo "$(GREEN)Clean complete$(NC)"

# ==============================================================================
# Deep clean (including Verilator cache)
# ==============================================================================
distclean: clean
	@echo "$(YELLOW)Performing deep clean...$(NC)"
	rm -rf $(OUTPUT_DIR)
	@echo "$(GREEN)Deep clean complete$(NC)"

.PHONY: all lint format fix fix-verilator build wave view clean distclean check-tools pipeline