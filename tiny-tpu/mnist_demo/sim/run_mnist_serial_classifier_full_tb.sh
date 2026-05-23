#!/usr/bin/env bash
# ABOUTME: Builds and runs the full-size MNIST serial-classifier regression in ModelSim.
# ABOUTME: Native Linux Version (Removed WSL and .exe dependencies)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TPU_DIR="$PROJECT_DIR/rtl"

# If vlib/vsim are already in your PATH, you can leave this blank or export it externally.
# Otherwise, change this to your actual ModelSim/Questa bin path:
MODELSIM_DIR="${MODELSIM_DIR:-}"

# Helper to prefix commands if MODELSIM_DIR is provided
CMD_PREFIX=""
if [ -n "$MODELSIM_DIR" ]; then
    CMD_PREFIX="${MODELSIM_DIR}/"
fi

WORK_DIR="$PROJECT_DIR/artifacts/sim/modelsim_mnist_serial_classifier_full"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

cd "$WORK_DIR"

# Create and map library using standard Linux commands
${CMD_PREFIX}vlib "$WORK_DIR/work"
${CMD_PREFIX}vmap work "$WORK_DIR/work"

# Compile Verilog files
${CMD_PREFIX}vlog \
  -work work \
  "$TPU_DIR/bias_child.v" \
  "$TPU_DIR/bias_parent.v" \
  "$TPU_DIR/fixedpoint_simple.v" \
  "$TPU_DIR/gradient_descent.v" \
  "$TPU_DIR/leaky_relu_child.v" \
  "$TPU_DIR/leaky_relu_derivative_child.v" \
  "$TPU_DIR/leaky_relu_derivative_parent.v" \
  "$TPU_DIR/leaky_relu_parent.v" \
  "$TPU_DIR/loss_child.v" \
  "$TPU_DIR/loss_parent.v" \
  "$TPU_DIR/pe.v" \
  "$TPU_DIR/systolic.v" \
  "$TPU_DIR/unified_buffer.v" \
  "$TPU_DIR/vpu.v" \
  "$PROJECT_DIR/rtl/uart_rx.v" \
  "$PROJECT_DIR/rtl/uart_frame_receiver.v" \
  "$PROJECT_DIR/rtl/mnist_frame_buffer.v" \
  "$PROJECT_DIR/rtl/mnist_uart_ingress.v" \
  "$PROJECT_DIR/rtl/tpu_mnist.v" \
  "$PROJECT_DIR/rtl/mnist_classifier_core.v" \
  "$PROJECT_DIR/rtl/mnist_serial_classifier.v" \
  "$SCRIPT_DIR/tb_mnist_serial_classifier_full.v"

# Run simulation in CLI mode
${CMD_PREFIX}vsim \
  -c \
  -do "run -all; quit -f" \
  work.tb_mnist_serial_classifier_full