# ==============================================================================
# FILE: Makefile
# Verilator build for MNIST TPU Tiled Classifier testbench
# ==============================================================================

# Directory Structure
SOURCE_DIR = src
TEST_DIR = test
SIM_BUILD_DIR = sim

# Verilator settings
VERILATOR = verilator
VERILATOR_FLAGS = -Wall -Wno-fatal --cc --exe --build -j 0 \
                  -CFLAGS "-O3 -march=native" \
                  --top-module mnist_tpu_tiled_classifier_tb

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

# Testbench files
VERILOG_TB = $(TEST_DIR)/mnist_tpu_tiled_classifier_tb.v
CPP_TB = $(TEST_DIR)/test_main.cpp

# Output
EXECUTABLE = Vmnist_tpu_tiled_classifier_tb

.PHONY: all clean run wave

all: $(EXECUTABLE)

# Build the simulation executable
$(EXECUTABLE): $(VERILOG_SOURCES) $(VERILOG_TB) $(CPP_TB)
	$(VERILATOR) $(VERILATOR_FLAGS) \
		$(VERILOG_SOURCES) $(VERILOG_TB) $(CPP_TB) \
		-o $(EXECUTABLE)

# Run simulation
run: $(EXECUTABLE)
	./$(SIM_BUILD_DIR)/$(EXECUTABLE) --trace

# View waveforms with GTKWave
wave:
	gtkwave mnist_tpu_classifier.vcd &

# Clean build artifacts
clean:
	rm -rf $(SIM_BUILD_DIR)
	rm -f mnist_tpu_classifier.vcd