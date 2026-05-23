# RTL

This folder contains the Verilog sources used by the verilator for testing MNIST.

Includes MNIST specific `mnist_tpu_tiled_classifier.v`.

The following are different versions of files between `ale\src` and the main `tiny-tpu/src`.

| ale/src | tiny-tpu/src |
|-----------|----------------|
| `tpu_mnist.v` | `tpu.sv` |
| `fixedpoint_simple.v` | `fixedpoint.sv` |
