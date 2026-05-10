// ==============================================================================
// MODULE: mnist_tpu_tiled_classifier
//
// SUMMARY:
// This module implements a highly optimized, hardware-accelerated Multilayer 
// Perceptron (MLP) for MNIST digit classification utilizing a custom "Tiny-TPU" 
// core. Unlike earlier "chunked" iterations where the host FSM performed the 
// neural network accumulation and activation logic manually, this "tiled" 
// architecture is designed to fully exploit the TPU's internal Vector Processing 
// Unit (VPU) and a massive Unified Buffer (UB) of 4096 elements. 
//
// This enables true System-on-Chip (SoC) acceleration. The FSM acts as a pure 
// orchestrator for a two-layer network (784 inputs -> 64 hidden -> 10 outputs). 
// It loads entire tiles of input activations, weight matrices, and bias vectors 
// into the UB sequentially. It then issues precise control signals (`ub_rd_start_in`, 
// `ub_ptr_select`) to stream weights into the systolic array's shadow buffers, 
// swap them into active registers (`sys_switch_in`), and stream the inputs and 
// biases through the array. 
//
// Crucially, the FSM dynamically configures the VPU data pathway: it activates 
// both hardware bias addition and Leaky ReLU modules (`4'b1100`) for the hidden 
// layer, and only the bias module (`4'b1000`) for the output layer. The fully 
// activated partial sums are captured directly from the VPU, completely 
// eliminating host-side accumulation, before an Argmax state predicts the digit.
// ==============================================================================

`timescale 1ns / 1ps
`default_nettype none

module mnist_tpu_tiled_classifier #(
    parameter integer PIXELS               = 784,
    parameter integer PIXEL_ADDR_WIDTH     = 10,
    parameter integer HIDDEN_NEURONS       = 64,
    parameter integer HIDDEN_ADDR_WIDTH    = 6,
    parameter integer OUTPUT_NEURONS       = 10,
    parameter integer OUTPUT_ADDR_WIDTH    = 4,
    parameter integer TILE_WIDTH           = 2,
    // Massive Unified Buffer to hold full inputs, weights, and biases simultaneously
    parameter integer UNIFIED_BUFFER_WIDTH = 4096,

    // Ale
    // parameter integer PRELOAD_MODEL = 0,
    parameter W1_INIT_FILE = "/home/ale/tesi/tesi_git/tiny-tpu/ale_mnist/model/reference/w1_tiled_q8_8.memh",
    parameter B1_INIT_FILE = "/home/ale/tesi/tesi_git/tiny-tpu/ale_mnist/model/reference/b1_q8_8.memh",
    parameter W2_INIT_FILE = "/home/ale/tesi/tesi_git/tiny-tpu/ale_mnist/model/reference/w2_tiled_q8_8.memh",
    parameter B2_INIT_FILE = "/home/ale/tesi/tesi_git/tiny-tpu/ale_mnist/model/reference/b2_q8_8.memh"
    // \Ale
) (
    input wire clk,
    input wire rst,  // Active-high reset
    input wire start,
    input wire [15:0] pixel_data_in,  // 16b Q8.8 pixel data input
    output wire [PIXEL_ADDR_WIDTH - 1:0] pixel_addr_out, // Address for fetching pixel_data_in into UB
    output reg busy,
    output reg done,
    output reg [3:0] prediction_out  // Final predicted digit (0-9)
);

  // Calculate how many TILE_WIDTH x TILE_WIDTH matrix blocks are required
  localparam integer HIDDEN_TILES = (HIDDEN_NEURONS + TILE_WIDTH - 1) / TILE_WIDTH;
  localparam integer OUTPUT_TILES = (OUTPUT_NEURONS + TILE_WIDTH - 1) / TILE_WIDTH;

  // FSM State Definitions
  localparam [4:0] STATE_IDLE = 5'd0;  // 0 - Waiting for start signal
  localparam [4:0] STATE_RESET_ASSERT = 5'd1; // 1 - Assert reset to clear TPU state before starting new inference
  localparam [4:0] STATE_RESET_RELEASE = 5'd2; // 2 - Release reset and initialize indices for new inference sequence
  localparam [4:0] STATE_LOAD_INPUT = 5'd3;  // 3 - Load X or H1 into UB
  localparam [4:0] STATE_LOAD_WEIGHT = 5'd4;  // 4 - Load W1 or W2 into UB
  localparam [4:0] STATE_LOAD_BIAS = 5'd5;  // 5 - Load B1 or B2 into UB
  localparam [4:0] STATE_START_WEIGHT = 5'd6;  // 6 - Stream Weights -> Systolic Array
  localparam [4:0] STATE_START_WEIGHT_GAP = 5'd7; // 7 - Gap state to ensure weights are latched before streaming inputs
  localparam [4:0] STATE_START_INPUT = 5'd8;  // 8 - Stream Inputs -> Systolic Array
  localparam [4:0] STATE_SWITCH_WEIGHTS = 5'd9;  // 9 - Latch weights in Array
  localparam [4:0] STATE_START_BIAS = 5'd10;  // 10 - Stream Biases -> VPU Hardware
  localparam [4:0] STATE_WAIT_OUTPUT = 5'd11;  // 11 - Poll VPU for final valid outputs
  localparam [4:0] STATE_NEXT_TILE = 5'd12; // 12 - Move to next tile of weights/activations or switch layers
  localparam [4:0] STATE_ARGMAX = 5'd13; // 13 - Compute Argmax on final logits to determine predicted digit
  localparam [4:0] STATE_DONE = 5'd14;  // 14 - Inference complete, wait for next start signal

  reg [4:0] state;
  reg current_layer;  // 0 = Layer 1 (Input->Hidden), 1 = Layer 2 (Hidden->Output)

  // Tracks which tile of hidden neurons is being processed 
  reg [HIDDEN_ADDR_WIDTH - 1:0] hidden_tile_index; // = HIDDEN_NEURONS/TILE_WIDTH = 64/2=32, range(0,31).

  // Tracks which tile of output neurons is being processed
  reg [OUTPUT_ADDR_WIDTH - 1:0] output_tile_index; // = OUTPUT_NEURONS/TILE_WIDTH = 10/2=5, range(0,4).
                                                   // Note: For last tile, 1 output neuron remains, which is handled by active_tile_outputs and dynamic loading logic in FSM.
  reg [PIXEL_ADDR_WIDTH - 1:0] l1_input_index;  // Tracks pixel input loading for Layer 1
  reg [HIDDEN_ADDR_WIDTH - 1:0] l2_input_index;  // Tracks hidden activation loading for Layer 2
  reg [15:0] weight_load_index; // Counts the specific weight being loaded within the current tile boundary
  reg [15:0] bias_load_index;
  reg [15:0] active_tile_outputs;

  // Status Registers for VPU output validation
  reg output_seen_0;
  reg output_seen_1;

  // Dynamic routing signal for VPU, used with vpu_data_pathway
  reg [3:0] active_vpu_mode;

  reg tpu_rst;

  // Host-to-UB Write Interface
  reg [15:0] ub_wr_host_data_in_0;
  reg [15:0] ub_wr_host_data_in_1;
  reg ub_wr_host_valid_in_0;
  reg ub_wr_host_valid_in_1;

  // UB Read Control Interface (Drives data out of UB into TPU components)
  reg ub_rd_start_in;
  reg ub_rd_transpose;
  reg [8:0] ub_ptr_select; // TODO: Check how -> Critical routing: 0=Left Array, 1=Top Array, 2=VPU Bias
  reg [15:0] ub_rd_addr_in;
  reg [15:0] ub_rd_row_size;
  reg [15:0] ub_rd_col_size;

  reg sys_switch_in;  // Signal to latch shadow weights to active weights

  // Internal wires connecting TPU modules
  wire [15:0] sys_data_out_21;
  wire [15:0] sys_data_out_22;
  wire sys_valid_out_21;
  wire sys_valid_out_22;

  // VPU Outputs (Fully activated, biased, ready for memory)
  wire [15:0] vpu_data_out_1;
  wire [15:0] vpu_data_out_2;
  wire vpu_valid_out_1;
  wire vpu_valid_out_2;

  // TODO: NOT USED. May be useful for debug
  wire [15:0] ub_rd_input_data_out_0;
  wire [15:0] ub_rd_input_data_out_1;
  wire ub_rd_input_valid_out_0;
  wire ub_rd_input_valid_out_1;
  wire [15:0] ub_rd_weight_data_out_0;
  wire [15:0] ub_rd_weight_data_out_1;
  wire ub_rd_weight_valid_out_0;
  wire ub_rd_weight_valid_out_1;
  wire [15:0] ub_rd_bias_data_out_0;
  wire [15:0] ub_rd_bias_data_out_1;
  wire [15:0] ub_rd_Y_data_out_0;
  wire [15:0] ub_rd_Y_data_out_1;
  wire [15:0] ub_rd_H_data_out_0;
  wire [15:0] ub_rd_H_data_out_1;
  wire [15:0] ub_rd_col_size_out;
  wire ub_rd_col_size_valid_out;

  // Internal buffer arrays for checking TPU outputs and performing Argmax at the end
  reg signed [15:0] hidden_buffer[0:HIDDEN_NEURONS - 1];
  reg signed [15:0] logits_buffer[0:OUTPUT_NEURONS - 1];

  // TODO: Weight and Bias Memories: Loaded from off-chip memory or stored in on-chip ROM
  // Currently no ports and FSM states in mnist_tpu_tiled_classifier to load from host.
  reg signed [15:0] w1_mem[0:(PIXELS * HIDDEN_NEURONS) - 1];
  reg signed [15:0] b1_mem[0:HIDDEN_NEURONS - 1];
  reg signed [15:0] w2_mem[0:(HIDDEN_NEURONS * OUTPUT_NEURONS) - 1];
  reg signed [15:0] b2_mem[0:OUTPUT_NEURONS - 1];

  // TODO: Check if T -> Variables for Argmax computation at the end
  integer clear_index;
  integer compare_index;
  integer best_index;
  reg signed [15:0] best_value;

  // pixel_addr_out = l1_input_index (if STATE_LOAD_INPUT and current_layer =0), else 00..0
  assign pixel_addr_out = (state == STATE_LOAD_INPUT && !current_layer) ? l1_input_index : {PIXEL_ADDR_WIDTH{1'b0}};

  // Ale : Preload weights and biases into internal memories from files.
  initial begin
    $readmemh(B1_INIT_FILE, b1_mem);
    $readmemh(B2_INIT_FILE, b2_mem);
    $readmemh(W1_INIT_FILE, w1_mem);
    $readmemh(W2_INIT_FILE, w2_mem);
  end
  // \Ale

  tpu_mnist #(
      .SYSTOLIC_ARRAY_WIDTH(2),
      .UNIFIED_BUFFER_WIDTH(UNIFIED_BUFFER_WIDTH)
  ) tpu_inst (
      .clk                  (clk),
      .rst                  (tpu_rst),
      .ub_wr_host_data_in_0 (ub_wr_host_data_in_0),
      .ub_wr_host_data_in_1 (ub_wr_host_data_in_1),
      .ub_wr_host_valid_in_0(ub_wr_host_valid_in_0),
      .ub_wr_host_valid_in_1(ub_wr_host_valid_in_1),
      .ub_rd_start_in       (ub_rd_start_in),
      .ub_rd_transpose      (ub_rd_transpose),
      .ub_ptr_select        (ub_ptr_select),
      .ub_rd_addr_in        (ub_rd_addr_in),
      .ub_rd_row_size       (ub_rd_row_size),
      .ub_rd_col_size       (ub_rd_col_size),
      .learning_rate_in     (16'h0001),

      // Dynamically fed from FSM to turn VPU features on/off
      .vpu_data_pathway(active_vpu_mode),

      .sys_switch_in              (sys_switch_in),
      .vpu_leak_factor_in         (16'h0003),                  // Standard leaky ReLU constant
      .inv_batch_size_times_two_in(16'h0000),
      .sys_data_out_21            (sys_data_out_21),
      .sys_data_out_22            (sys_data_out_22),
      .sys_valid_out_21           (sys_valid_out_21),
      .sys_valid_out_22           (sys_valid_out_22),
      .vpu_data_out_1             (vpu_data_out_1),
      .vpu_data_out_2             (vpu_data_out_2),
      .vpu_valid_out_1            (vpu_valid_out_1),
      .vpu_valid_out_2            (vpu_valid_out_2),
      .ub_rd_input_data_out_0     (ub_rd_input_data_out_0),
      .ub_rd_input_data_out_1     (ub_rd_input_data_out_1),
      .ub_rd_input_valid_out_0    (ub_rd_input_valid_out_0),
      .ub_rd_input_valid_out_1    (ub_rd_input_valid_out_1),
      .ub_rd_weight_data_out_0    (ub_rd_weight_data_out_0),
      .ub_rd_weight_data_out_1    (ub_rd_weight_data_out_1),
      .ub_rd_weight_valid_out_0   (ub_rd_weight_valid_out_0),
      .ub_rd_weight_valid_out_1   (ub_rd_weight_valid_out_1),
      .ub_rd_bias_data_out_0      (ub_rd_bias_data_out_0),
      .ub_rd_bias_data_out_1      (ub_rd_bias_data_out_1),
      .ub_rd_Y_data_out_0         (ub_rd_Y_data_out_0),
      .ub_rd_Y_data_out_1         (ub_rd_Y_data_out_1),
      .ub_rd_H_data_out_0         (ub_rd_H_data_out_0),
      .ub_rd_H_data_out_1         (ub_rd_H_data_out_1),
      .ub_rd_col_size_out         (ub_rd_col_size_out),
      .ub_rd_col_size_valid_out   (ub_rd_col_size_valid_out)
  );

  // Helper function to determine how many outputs are active in the current tile, based on remaining neurons
  function [15:0] clipped_tile_outputs;
    input integer remaining;
    begin
      if (remaining >= TILE_WIDTH) begin
        clipped_tile_outputs = TILE_WIDTH;
      end else begin
        clipped_tile_outputs = remaining[15:0];
      end
    end
  endfunction

  // ==============================================================================
  // CORE ORCHESTRATION FSM
  // ==============================================================================
  always @(posedge clk or posedge rst) begin
    if (rst) begin
      // Reset all internal state and outputs
      state                 <= STATE_IDLE;
      current_layer         <= 1'b0;
      hidden_tile_index     <= {HIDDEN_ADDR_WIDTH{1'b0}};
      output_tile_index     <= {OUTPUT_ADDR_WIDTH{1'b0}};
      l1_input_index        <= {PIXEL_ADDR_WIDTH{1'b0}};
      l2_input_index        <= {HIDDEN_ADDR_WIDTH{1'b0}};
      weight_load_index     <= 16'd0;
      bias_load_index       <= 16'd0;
      active_tile_outputs   <= 16'd0;
      output_seen_0         <= 1'b0;
      output_seen_1         <= 1'b0;
      active_vpu_mode       <= 4'b0000;
      tpu_rst               <= 1'b1;  // Reset TPU (active high)
      busy                  <= 1'b0;
      done                  <= 1'b0;
      prediction_out        <= 4'd0;

      ub_wr_host_data_in_0  <= 16'h0000;
      ub_wr_host_data_in_1  <= 16'h0000;
      ub_wr_host_valid_in_0 <= 1'b0;
      ub_wr_host_valid_in_1 <= 1'b0;
      ub_rd_start_in        <= 1'b0;
      ub_rd_transpose       <= 1'b0;
      ub_ptr_select         <= 9'd0;
      ub_rd_addr_in         <= 16'd0;
      ub_rd_row_size        <= 16'd0;
      ub_rd_col_size        <= 16'd0;
      sys_switch_in         <= 1'b0;

      // Clear intermediate buffers
      for (clear_index = 0; clear_index < HIDDEN_NEURONS; clear_index = clear_index + 1) begin
        hidden_buffer[clear_index] <= 16'h0000;
      end
      for (clear_index = 0; clear_index < OUTPUT_NEURONS; clear_index = clear_index + 1) begin
        logits_buffer[clear_index] <= 16'h0000;
      end
    end else begin
      // Default 1-cycle pulses for TPU control
      done                  <= 1'b0;
      ub_wr_host_data_in_0  <= 16'h0000;
      ub_wr_host_data_in_1  <= 16'h0000;
      ub_wr_host_valid_in_0 <= 1'b0;
      ub_wr_host_valid_in_1 <= 1'b0;
      ub_rd_start_in        <= 1'b0;
      ub_rd_transpose       <= 1'b0;
      ub_ptr_select         <= 9'd0;
      ub_rd_addr_in         <= 16'd0;
      ub_rd_row_size        <= 16'd0;
      ub_rd_col_size        <= 16'd0;
      sys_switch_in         <= 1'b0;

      case (state)
        STATE_IDLE: begin
          tpu_rst         <= 1'b0;
          active_vpu_mode <= 4'b0000;  // Ensure VPU is passthrough in idle
          busy            <= 1'b0;
          if (start) begin
            busy <= 1'b1;
            current_layer <= 1'b0;
            hidden_tile_index <= {HIDDEN_ADDR_WIDTH{1'b0}};
            output_tile_index <= {OUTPUT_ADDR_WIDTH{1'b0}};
            l1_input_index <= {PIXEL_ADDR_WIDTH{1'b0}};
            l2_input_index <= {HIDDEN_ADDR_WIDTH{1'b0}};
            weight_load_index <= 16'd0;
            bias_load_index <= 16'd0;
            active_tile_outputs <= clipped_tile_outputs(
                HIDDEN_NEURONS
            );  // Calculate # of outputs for the first tile
                // For HIDDEN_NEURONS = 64, returns 2 for TILE_WIDTH=2
            output_seen_0 <= 1'b0;
            output_seen_1 <= 1'b0;
            tpu_rst <= 1'b1;  // Reset TPU to clear any internal state before starting
            // Clear previous inference results
            for (clear_index = 0; clear_index < HIDDEN_NEURONS; clear_index = clear_index + 1) begin
              hidden_buffer[clear_index] <= 16'h0000;
            end
            for (clear_index = 0; clear_index < OUTPUT_NEURONS; clear_index = clear_index + 1) begin
              logits_buffer[clear_index] <= 16'h0000;
            end
            state <= STATE_RESET_ASSERT;
          end
        end

        STATE_RESET_ASSERT: begin
          tpu_rst         <= 1'b1;
          active_vpu_mode <= 4'b0000;
          state           <= STATE_RESET_RELEASE;
        end

        STATE_RESET_RELEASE: begin
          tpu_rst           <= 1'b0;
          l1_input_index    <= {PIXEL_ADDR_WIDTH{1'b0}};
          l2_input_index    <= {HIDDEN_ADDR_WIDTH{1'b0}};
          weight_load_index <= 16'd0;
          bias_load_index   <= 16'd0;
          output_seen_0     <= 1'b0;
          output_seen_1     <= 1'b0;
          state             <= STATE_LOAD_INPUT;
        end

        // --- UNIFIED BUFFER LOADING SEQUENCE ---

        // Write Input Activations (Layer 0) and hidden activations (Layer 1) into UB
        STATE_LOAD_INPUT: begin
          if (!current_layer) begin  // If current layer = 0
            if (l1_input_index < PIXELS) begin  // If current no of pixel < total pixels
              ub_wr_host_data_in_0 <= pixel_data_in; // Input to TPU (UB) from host, pixel data Q8.8
              ub_wr_host_valid_in_0 <= 1'b1;
              l1_input_index <= l1_input_index + {{(PIXEL_ADDR_WIDTH - 1){1'b0}}, 1'b1}; // Increments pixel_addr_out++ for next pixel
            end else begin
              weight_load_index <= 16'd0;  // If pixel loading complete, initialize weight index
              state             <= STATE_LOAD_WEIGHT;  // and move to weight loading state
            end
          end else begin
            if (l2_input_index < HIDDEN_NEURONS) begin // If current layer = 1, check if current hidden activation index < total hidden neurons
              if (l2_input_index + 1 < HIDDEN_NEURONS) begin // Check if 2 hidden activations can be loaded into the 2-port UB
                ub_wr_host_data_in_1 <= hidden_buffer[l2_input_index];
                ub_wr_host_data_in_0 <= hidden_buffer[l2_input_index+1];
                ub_wr_host_valid_in_1 <= 1'b1;
                ub_wr_host_valid_in_0 <= 1'b1;
                l2_input_index <= l2_input_index + {{(HIDDEN_ADDR_WIDTH - 2){1'b0}}, 2'd2}; // Increment hidden activation index by 2
                // if only 1 hidden activation remains, load into port 0 and increment l2_input_index by 1
              end else begin
                ub_wr_host_data_in_0 <= hidden_buffer[l2_input_index];
                ub_wr_host_valid_in_0 <= 1'b1;
                l2_input_index <= l2_input_index + {{(HIDDEN_ADDR_WIDTH - 1){1'b0}}, 1'b1}; // Increment hidden activation index by 1
              end
            end else begin
              weight_load_index <= 16'd0; // If hidden activation loading complete, initialize weight index
              state <= STATE_LOAD_WEIGHT;  // and move to weight loading state
            end
          end
        end

        // Write Weights into UB (Offsets past the Inputs)
        STATE_LOAD_WEIGHT: begin
          // If weight_load_index < {[Total Pixels (for Layer 0) or Total Hidden Neurons (for Layer 1)] * (2 normally, or 1 for edge case)}
          // For !current_layer -> (weight_load_index < (784= * 2)) i.e. load 1568 weights for Layer 0
          if (weight_load_index < ((!current_layer ? PIXELS : HIDDEN_NEURONS) * active_tile_outputs)) begin
            if (weight_load_index + 1 < ((!current_layer ? PIXELS : HIDDEN_NEURONS) * active_tile_outputs)) begin // Check if 2 weights can be loaded into the 2-port UB
              // If Layer 0, load 2 weights from w1_mem, till weight_load_index 
              if (!current_layer) begin
                ub_wr_host_data_in_1 <= w1_mem[(hidden_tile_index * PIXELS * TILE_WIDTH) + weight_load_index];
                ub_wr_host_data_in_0 <= w1_mem[(hidden_tile_index * PIXELS * TILE_WIDTH) + weight_load_index + 1];
                // If Layer 1, load 2 weights from w2_mem
              end else begin
                ub_wr_host_data_in_1 <= w2_mem[(output_tile_index * HIDDEN_NEURONS * TILE_WIDTH) + weight_load_index];
                ub_wr_host_data_in_0 <= w2_mem[(output_tile_index * HIDDEN_NEURONS * TILE_WIDTH) + weight_load_index + 1];
              end
              ub_wr_host_valid_in_1 <= 1'b1;
              ub_wr_host_valid_in_0 <= 1'b1;
              weight_load_index     <= weight_load_index + 16'd2;
            end else begin
              if (!current_layer) begin
                ub_wr_host_data_in_0 <= w1_mem[(hidden_tile_index * PIXELS * TILE_WIDTH) + weight_load_index];
              end else begin
                ub_wr_host_data_in_0 <= w2_mem[(output_tile_index * HIDDEN_NEURONS * TILE_WIDTH) + weight_load_index];
              end
              ub_wr_host_valid_in_0 <= 1'b1;
              weight_load_index     <= weight_load_index + 16'd1;
            end
          end else begin
            bias_load_index <= 16'd0;
            state           <= STATE_LOAD_BIAS;
          end
        end

        // Write Biases into UB (Offsets past Inputs and Weights)
        STATE_LOAD_BIAS: begin
          if (bias_load_index < active_tile_outputs) begin
            if (bias_load_index + 1 < active_tile_outputs) begin
              if (!current_layer) begin
                ub_wr_host_data_in_1 <= b1_mem[(hidden_tile_index*TILE_WIDTH)+bias_load_index];
                ub_wr_host_data_in_0 <= b1_mem[(hidden_tile_index*TILE_WIDTH)+bias_load_index+1];
              end else begin
                ub_wr_host_data_in_1 <= b2_mem[(output_tile_index*TILE_WIDTH)+bias_load_index];
                ub_wr_host_data_in_0 <= b2_mem[(output_tile_index*TILE_WIDTH)+bias_load_index+1];
              end
              ub_wr_host_valid_in_1 <= 1'b1;
              ub_wr_host_valid_in_0 <= 1'b1;
              bias_load_index       <= bias_load_index + 16'd2;
            end else begin
              if (!current_layer) begin
                ub_wr_host_data_in_0 <= b1_mem[(hidden_tile_index*TILE_WIDTH)+bias_load_index];
              end else begin
                ub_wr_host_data_in_0 <= b2_mem[(output_tile_index*TILE_WIDTH)+bias_load_index];
              end
              ub_wr_host_valid_in_0 <= 1'b1;
              bias_load_index       <= bias_load_index + 16'd1;
            end
          end else begin
            state <= STATE_START_WEIGHT;
          end
        end

        // --- TPU READ COMMAND SEQUENCE ---

        // Command UB to stream Weights -> Top of Systolic Array
        STATE_START_WEIGHT: begin  // 0x06
          ub_rd_start_in <= 1'b1;
          ub_ptr_select <= 9'd1;  // Routing code: 1 = Read weights into Top of Array
          ub_rd_addr_in <= (!current_layer ? PIXELS : HIDDEN_NEURONS);  // Offset memory read to weights section of UB (past the inputs)
          ub_rd_row_size <= (!current_layer ? PIXELS : HIDDEN_NEURONS);
          ub_rd_col_size <= active_tile_outputs;
          ub_rd_transpose <= 1'b1;
          state <= STATE_START_WEIGHT_GAP;
        end

        STATE_START_WEIGHT_GAP: begin  // 0x07
          state <= STATE_START_INPUT;
        end

        // Command UB to stream Inputs -> Left Side of Systolic Array
        STATE_START_INPUT: begin  // 0x08
          // DYNAMIC HARDWARE ACTIVATION:
          // Layer 1 (current_layer=0): Set 4'b1100 to enable VPU Bias (bit 3) and Leaky ReLU (bit 2).
          // Layer 2 (current_layer=1): Set 4'b1000 to enable only VPU Bias (bit 3) for Logits.
          active_vpu_mode <= current_layer ? 4'b1000 : 4'b1100;

          ub_rd_start_in  <= 1'b1;
          ub_ptr_select   <= 9'd0;  // Routing code: 0 = Left side of Array
          ub_rd_addr_in   <= 16'd0;  // Inputs are at the start of the UB memory space, so no offset
          ub_rd_row_size  <= 16'd1;
          ub_rd_col_size  <= (!current_layer ? PIXELS : HIDDEN_NEURONS);
          ub_rd_transpose <= 1'b0;
          state           <= STATE_SWITCH_WEIGHTS;
        end

        STATE_SWITCH_WEIGHTS: begin
          // Latch weights from shadow registers into active processing registers
          sys_switch_in <= 1'b1;
          state         <= STATE_START_BIAS;
        end

        // Command UB to stream Biases -> VPU Hardware Module
        STATE_START_BIAS: begin
          ub_rd_start_in <= 1'b1;
          ub_ptr_select <= 9'd2;  // Routing code: 2 = VPU Bias Interface
          ub_rd_addr_in <= (!current_layer ? (PIXELS + (PIXELS * active_tile_outputs)) : (HIDDEN_NEURONS + (HIDDEN_NEURONS * active_tile_outputs)));
          ub_rd_row_size <= 16'd1;
          ub_rd_col_size <= active_tile_outputs;
          ub_rd_transpose <= 1'b0;
          output_seen_0 <= 1'b0;
          output_seen_1 <= 1'b0;
          state <= STATE_WAIT_OUTPUT;
        end

        // --- VPU HARVESTING ---

        // Wait for the VPU to finish its pipeline.
        // Because VPU hardware bias and ReLU are enabled, the returned data 
        // is completely finished and can be dumped directly into memory without math.
        STATE_WAIT_OUTPUT: begin
          if (vpu_valid_out_1) begin
            output_seen_0 <= 1'b1;
            if (!current_layer) begin
              hidden_buffer[hidden_tile_index*TILE_WIDTH] <= vpu_data_out_1;
            end else begin
              logits_buffer[output_tile_index*TILE_WIDTH] <= vpu_data_out_1;
            end
          end

          if (active_tile_outputs > 1 && vpu_valid_out_2) begin
            output_seen_1 <= 1'b1;
            if (!current_layer) begin
              hidden_buffer[(hidden_tile_index*TILE_WIDTH)+1] <= vpu_data_out_2;
            end else begin
              logits_buffer[(output_tile_index*TILE_WIDTH)+1] <= vpu_data_out_2;
            end
          end

          // Once all expected outputs arrive, shut off VPU and move to next tile
          if (!vpu_valid_out_1 && !vpu_valid_out_2 &&
                        output_seen_0 &&
                        ((active_tile_outputs == 1) || output_seen_1)) begin
            active_vpu_mode <= 4'b0000;
            state           <= STATE_NEXT_TILE;
          end
        end

        STATE_NEXT_TILE: begin
          if (!current_layer) begin
            if (hidden_tile_index + 1 < HIDDEN_TILES) begin
              hidden_tile_index <= hidden_tile_index + {{(HIDDEN_ADDR_WIDTH - 1) {1'b0}}, 1'b1};
              active_tile_outputs <= clipped_tile_outputs(
                  HIDDEN_NEURONS - ((hidden_tile_index + 1) * TILE_WIDTH)
              );
              tpu_rst <= 1'b1;
              state <= STATE_RESET_ASSERT;
            end else begin
              // Switch to Output Layer (Logits)
              current_layer       <= 1'b1;
              output_tile_index   <= {OUTPUT_ADDR_WIDTH{1'b0}};
              active_tile_outputs <= clipped_tile_outputs(OUTPUT_NEURONS);
              tpu_rst             <= 1'b1;
              state               <= STATE_RESET_ASSERT;
            end
          end else begin
            if (output_tile_index + 1 < OUTPUT_TILES) begin
              output_tile_index <= output_tile_index + {{(OUTPUT_ADDR_WIDTH - 1) {1'b0}}, 1'b1};
              active_tile_outputs <= clipped_tile_outputs(
                  OUTPUT_NEURONS - ((output_tile_index + 1) * TILE_WIDTH)
              );
              tpu_rst <= 1'b1;
              state <= STATE_RESET_ASSERT;
            end else begin
              state <= STATE_ARGMAX;
            end
          end
        end

        // Determine the highest Logit score for prediction
        STATE_ARGMAX: begin
          best_index = 0;
          best_value = logits_buffer[0];
          for (
              compare_index = 1; compare_index < OUTPUT_NEURONS; compare_index = compare_index + 1
          ) begin
            if ($signed(logits_buffer[compare_index]) > $signed(best_value)) begin
              best_index = compare_index;
              best_value = logits_buffer[compare_index];
            end
          end
          prediction_out <= best_index[3:0];
          busy           <= 1'b0;
          done           <= 1'b1;
          state          <= STATE_DONE;
        end

        STATE_DONE: begin
          state <= STATE_IDLE;
        end

        default: begin
          state <= STATE_IDLE;
        end
      endcase
    end
  end
endmodule

`default_nettype wire
