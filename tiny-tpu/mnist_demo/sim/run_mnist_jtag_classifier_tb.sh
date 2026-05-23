#!/usr/bin/env bash
# ABOUTME: Compiles and runs the full JTAG MMIO plus classifier regression in ModelSim.
# ABOUTME: Checks that host-style image writes produce the expected tracked-sample prediction.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TPU_DIR="$PROJECT_DIR/rtl"
# MODELSIM_DIR="${MODELSIM_DIR:-/mnt/c/intelFPGA/18.1/modelsim_ase/win32aloem}"
WORK_DIR="$PROJECT_DIR/artifacts/sim/modelsim_mnist_jtag_classifier"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

cd "$WORK_DIR"
vlib "$WORK_DIR/work"
vmap work "$WORK_DIR/work"

vlog -work work \
 "$TPU_DIR/fixedpoint_simple.v" \
 "$TPU_DIR/unified_buffer.v" \
 "$TPU_DIR/systolic.v" \
 "$TPU_DIR/vpu.v" \
 "$TPU_DIR/bias_child.v" \
 "$TPU_DIR/bias_parent.v" \
 "$TPU_DIR/leaky_relu_child.v" \
 "$TPU_DIR/leaky_relu_parent.v" \
 "$TPU_DIR/leaky_relu_derivative_child.v" \
 "$TPU_DIR/leaky_relu_derivative_parent.v" \
 "$TPU_DIR/loss_child.v" \
 "$TPU_DIR/loss_parent.v" \
 "$TPU_DIR/gradient_descent.v" \
 "$TPU_DIR/pe.v" \
 "$TPU_DIR/tpu_mnist.v" \
 "$TPU_DIR/mnist_classifier_core.v" \
 "$TPU_DIR/mnist_jtag_mmio.v" \
 "$SCRIPT_DIR/tb_mnist_jtag_classifier.v"

vsim \
  -c \
  -do "run -all; quit -f" \
  work.tb_mnist_jtag_classifier
