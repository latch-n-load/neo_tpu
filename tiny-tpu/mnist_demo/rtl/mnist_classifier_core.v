// ABOUTME: Runs a two-layer inference by chunking input features in pairs through the Tiny-TPU core.
// ABOUTME: Accumulates raw partial sums externally, then applies bias and ReLU to match the trained model.
//
// SUMMARY:
// This module, `mnist_classifier_core`, acts as the master orchestrator for a custom, 
// hardware-accelerated Tiny-TPU (Tensor Processing Unit) designed to run a quantized 
// Multilayer Perceptron (MLP) for MNIST digit recognition. Because the TPU systolic 
// array has a limited physical size (e.g., 2x2), this controller breaks down large matrix 
// multiplications (784 inputs x 64 hidden neurons) into small, manageable "chunks." 
// 
// It manages a complex finite state machine (FSM) that shuttles data between memory and 
// the TPU. The FSM loads input pixels and model weights into the TPU's internal Unified 
// Buffer (UB), triggers the systolic array to multiply them, and retrieves the raw partial 
// sums. Because the TPU is configured purely for multiplication in this design, this core 
// handles the external accumulation of those chunks, applies the bias vectors (`b1`, `b2`), 
// and performs the non-linear ReLU activation for the hidden layer. 
//
// Finally, it sequences the transition from Layer 1 (pixel to hidden) to Layer 2 
// (hidden to output/logits). Once Layer 2 is complete, it runs an Argmax search over 
// the resulting 10 logits to determine and output the final predicted digit (0-9).

`timescale 1ns/1ps
`default_nettype none

// A simple dual-port synchronous ROM used to store weights.
// Used primarily for simulation when PRELOAD_MODEL is set to 1.
module mnist_sync_rom_2r #(
    parameter integer DEPTH = 1,
    parameter INIT_FILE = ""
) (
    input wire clk,
    input wire [15:0] addr_a, // Address port A
    input wire [15:0] addr_b, // Address port B
    output reg [15:0] q_a,    // Data out port A
    output reg [15:0] q_b     // Data out port B
);
    // Intel FPGA attribute to map this to M10K block RAM blocks
    (* ramstyle = "M10K" *) reg [15:0] mem [0:DEPTH - 1];
    
    // Synthesizable ONLY for FPGA block RAM initialization
    initial begin
        $readmemh(INIT_FILE, mem);
    end

    // Synchronous reads: addresses clocked in, data available next cycle
    always @(posedge clk) begin
        q_a <= mem[addr_a];
        q_b <= mem[addr_b];
    end
endmodule


// The brain of the operation. Drives the TPU by loading data into the Unified Buffer, 
// toggling systolic array control signals, and accumulating the outputs.
module mnist_classifier_core #(
    // Network architecture parameters
    parameter integer PIXELS = 784,
    parameter integer PIXEL_ADDR_WIDTH = 10,
    parameter integer HIDDEN_NEURONS = 64,
    parameter integer HIDDEN_ADDR_WIDTH = 6,
    parameter integer OUTPUT_NEURONS = 10,
    parameter integer OUTPUT_ADDR_WIDTH = 4,
    
    // Hardware constraints
    parameter integer TILE_WIDTH = 2,           // Size of the systolic array (e.g., 2x2)
    parameter integer UNIFIED_BUFFER_WIDTH = 128,
    
    // Simulation / Preload flags
    parameter integer PRELOAD_MODEL = 0,
    parameter W1_INIT_FILE = "data/model/reference/w1_tiled_q8_8.memh",
    parameter B1_INIT_FILE = "data/model/reference/b1_q8_8.memh",
    parameter W2_INIT_FILE = "data/model/reference/w2_tiled_q8_8.memh",
    parameter B2_INIT_FILE = "data/model/reference/b2_q8_8.memh"
) (
    input wire clk,
    input wire rst,                        // Active-high reset to initialize the controller and TPU
    input wire start,                      // Pulsed to begin inference
    input wire [15:0] pixel_data_in,       // 16-bit pixel data coming from UART ingress
    output wire [PIXEL_ADDR_WIDTH - 1:0] pixel_addr_out, // Requesting specific pixel index
    output reg busy,                       // High while inference is running
    output reg done,                       // Pulsed high when prediction is ready
    output reg [3:0] prediction_out        // Final predicted digit (0-9)
);
    
    // Calculate total number of tiles needed based on layer sizes and TPU width
    localparam integer HIDDEN_TILES = (HIDDEN_NEURONS + TILE_WIDTH - 1) / TILE_WIDTH;
    localparam integer OUTPUT_TILES = (OUTPUT_NEURONS + TILE_WIDTH - 1) / TILE_WIDTH;
    
    // FSM States for orchestrating the TPU operations and external math
    localparam [4:0] STATE_IDLE = 5'd0;                 // Waiting for 'start' signal
    localparam [4:0] STATE_TILE_PREP = 5'd1;            // Setup loop counters for new tile/chunk
    localparam [4:0] STATE_RESET_ASSERT = 5'd2;         // Pulse TPU reset high
    localparam [4:0] STATE_RESET_RELEASE = 5'd3;        // Pulse TPU reset low
    localparam [4:0] STATE_LOAD_INPUT = 5'd4;           // Pushing data (pixels/hidden) to UB
    localparam [4:0] STATE_LOAD_WEIGHT = 5'd5;          // Generating addresses to fetch weights
    localparam [4:0] STATE_START_WEIGHT = 5'd6;         // Trigger weight read from UB to Systolic Array
    localparam [4:0] STATE_START_WEIGHT_GAP = 5'd7;     // Timing gap for UB read latency
    localparam [4:0] STATE_START_INPUT = 5'd8;          // Trigger input read from UB to Systolic Array
    localparam [4:0] STATE_SWITCH_WEIGHTS = 5'd9;       // Swap shadow buffer into active buffer in TPU
    localparam [4:0] STATE_WAIT_OUTPUT = 5'd10;         // Poll TPU for VPU valid signal indicating done
    localparam [4:0] STATE_NEXT_CHUNK = 5'd11;          // Iterate chunk loop or finish tile
    localparam [4:0] STATE_FINALIZE_TILE = 5'd12;       // Add bias and apply ReLU to chunked sums
    localparam [4:0] STATE_NEXT_TILE = 5'd13;           // Move to next tile, or switch layers
    localparam [4:0] STATE_ARGMAX = 5'd14;              // Find max probability logit (setup)
    localparam [4:0] STATE_DONE = 5'd15;                // Complete inference, assert 'done'
    localparam [4:0] STATE_LOAD_WEIGHT_WAIT = 5'd16;    // Wait 1 cycle for Block RAM read latency
    localparam [4:0] STATE_LOAD_WEIGHT_COMMIT = 5'd17;  // Write fetched Block RAM data into UB
    localparam [4:0] STATE_ARGMAX_COMPARE = 5'd18;      // Loop comparing logits to find maximum
    localparam [4:0] STATE_ARGMAX_LATCH = 5'd19;        // Store final max index to output

    // --- Controller Registers ---
    reg [4:0] state;
    reg current_layer; // 0 = Layer 1 (Input->Hidden), 1 = Layer 2 (Hidden->Output)
    
    // Tracking current position in the matrix multiplications
    reg [HIDDEN_ADDR_WIDTH - 1:0] hidden_tile_index;
    reg [OUTPUT_ADDR_WIDTH - 1:0] output_tile_index;
    reg [15:0] chunk_index; // How many 2x2 pieces of the current tile we've processed
    
    // UB loading indices
    reg [15:0] input_load_index;
    reg [15:0] weight_load_index;
    
    // Active widths for boundary conditions (e.g. if neurons isn't multiple of TILE_WIDTH)
    reg [15:0] active_tile_outputs; 
    reg [15:0] active_input_words;
    
    // Tracking TPU output signals
    reg output_seen_0;
    reg output_seen_1;
    reg tpu_rst;

    // Holding partial sums returned from TPU
    reg signed [15:0] partial_out_0;
    reg signed [15:0] partial_out_1;
    
    // Accumulators for chunked processing (summing up 2x2 chunks into full neuron values)
    reg signed [15:0] accum_0;
    reg signed [15:0] accum_1;

    // Registers to drive Unified Buffer (UB) writes
    reg [15:0] ub_wr_host_data_in_0;
    reg [15:0] ub_wr_host_data_in_1;
    reg ub_wr_host_valid_in_0;
    reg ub_wr_host_valid_in_1;

    // Registers to control TPU read/execute sequences
    reg ub_rd_start_in;
    reg ub_rd_transpose;
    reg [8:0] ub_ptr_select;
    reg [15:0] ub_rd_addr_in;
    reg [15:0] ub_rd_row_size;
    reg [15:0] ub_rd_col_size;
    reg sys_switch_in;
    
    // Wires returning from TPU (Raw systolic array outputs, mostly unused here as VPU handles it)
    wire [15:0] sys_data_out_21;
    wire [15:0] sys_data_out_22;
    wire sys_valid_out_21;
    wire sys_valid_out_22;
    
    // VPU (Vector Processing Unit) Data Pathway. 
    // In this specific implementation, VPU is set to passthrough (4'b0000), 
    // so these lines carry the raw systolic partial sums.
    wire [15:0] vpu_data_out_1;
    wire [15:0] vpu_data_out_2;
    wire vpu_valid_out_1;
    wire vpu_valid_out_2;
    
    // UB Read output wires (largely unused by host, consumed internally by TPU)
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
    
    // Intermediate memory arrays managed by this controller to hold activations
    reg signed [15:0] hidden_buffer [0:HIDDEN_NEURONS - 1]; // Output of Layer 1
    reg signed [15:0] logits_buffer [0:OUTPUT_NEURONS - 1]; // Output of Layer 2

    // Memory arrays for Biases (Pre-loaded from files)
    reg signed [15:0] b1_mem [0:HIDDEN_NEURONS - 1];
    reg signed [15:0] b2_mem [0:OUTPUT_NEURONS - 1];

    // Read addresses to pull from Weight Block RAMs
    reg [15:0] weight_read_addr_a;
    reg [15:0] weight_read_addr_b;
    reg weight_read_dual; // Flag to indicate if we are reading 2 weights this cycle

    // Data returning from Weight Block RAMs
    wire [15:0] w1_read_data_a;
    wire [15:0] w1_read_data_b;
    wire [15:0] w2_read_data_a;
    wire [15:0] w2_read_data_b;
    
    // Multiplexed data depending on which layer is active
    wire [15:0] active_weight_read_data_a;
    wire [15:0] active_weight_read_data_b;

    // Utility registers for Argmax
    integer clear_index;
    reg [OUTPUT_ADDR_WIDTH - 1:0] argmax_index;
    reg [3:0] best_index_reg;
    reg signed [15:0] best_value;
    
    // Initialization block for Bias memories (used in simulation/FPGA synthesis)
    initial begin
        if (PRELOAD_MODEL) begin
            $readmemh(B1_INIT_FILE, b1_mem);
            $readmemh(B2_INIT_FILE, b2_mem);
        end
    end

    // Route weight read data based on current layer status
    assign active_weight_read_data_a = current_layer ? w2_read_data_a : w1_read_data_a;
    assign active_weight_read_data_b = current_layer ? w2_read_data_b : w1_read_data_b;
    
    // Instantiate ROMs or RAMs for model weights based on PRELOAD_MODEL parameter
    generate
        if (PRELOAD_MODEL) begin : model_preload
            // Layer 1 weights (Pixels -> Hidden)
            mnist_sync_rom_2r #(
                .DEPTH(PIXELS * HIDDEN_NEURONS),
                .INIT_FILE(W1_INIT_FILE)
            ) w1_rom (
                .clk(clk),
                .addr_a(weight_read_addr_a),
                .addr_b(weight_read_addr_b),
                .q_a(w1_read_data_a),
                .q_b(w1_read_data_b)
            );
            // Layer 2 weights (Hidden -> Outputs)
            mnist_sync_rom_2r #(
                .DEPTH(HIDDEN_NEURONS * OUTPUT_NEURONS),
                .INIT_FILE(W2_INIT_FILE)
            ) w2_rom (
                .clk(clk),
                .addr_a(weight_read_addr_a),
                .addr_b(weight_read_addr_b),
                .q_a(w2_read_data_a),
                .q_b(w2_read_data_b)
            );
        end else begin : model_runtime
            // Placeholder for runtime loadable memory (if PRELOAD_MODEL == 0)
            (* ramstyle = "M10K" *) reg signed [15:0] w1_mem [0:(PIXELS * HIDDEN_NEURONS) - 1];
            (* ramstyle = "M10K" *) reg signed [15:0] w2_mem [0:(HIDDEN_NEURONS * OUTPUT_NEURONS) - 1];
            
            reg [15:0] w1_read_data_a_reg;
            reg [15:0] w1_read_data_b_reg;
            reg [15:0] w2_read_data_a_reg;
            reg [15:0] w2_read_data_b_reg;

            assign w1_read_data_a = w1_read_data_a_reg;
            assign w1_read_data_b = w1_read_data_b_reg;
            assign w2_read_data_a = w2_read_data_a_reg;
            assign w2_read_data_b = w2_read_data_b_reg;

            always @(posedge clk) begin
                w1_read_data_a_reg <= w1_mem[weight_read_addr_a];
                w1_read_data_b_reg <= w1_mem[weight_read_addr_b];
                w2_read_data_a_reg <= w2_mem[weight_read_addr_a];
                w2_read_data_b_reg <= w2_mem[weight_read_addr_b];
            end
        end
    endgenerate

    // Request pixel data from ingress module if we are currently loading inputs for Layer 1.
    // If not Layer 1, output address 0.
    assign pixel_addr_out = (state == STATE_LOAD_INPUT && !current_layer)
        ? ((chunk_index << 1) + input_load_index)
        : {PIXEL_ADDR_WIDTH{1'b0}};
        
    // Instantiate the actual TPU core module
    tpu_mnist #(
        .SYSTOLIC_ARRAY_WIDTH(2),
        .UNIFIED_BUFFER_WIDTH(UNIFIED_BUFFER_WIDTH)
    ) tpu_inst (
        .clk(clk),
        .rst(tpu_rst),

        // Unified Buffer Host Write Interface
        .ub_wr_host_data_in_0(ub_wr_host_data_in_0), // i_data from host to TPU (UB) write port 0
        .ub_wr_host_data_in_1(ub_wr_host_data_in_1), // i_data from host to TPU (UB) write port 1 (Use for loading weights)
        .ub_wr_host_valid_in_0(ub_wr_host_valid_in_0), // i_valid signal for host to TPU (UB) write port 0
        .ub_wr_host_valid_in_1(ub_wr_host_valid_in_1), // i_valid signal for host to TPU (UB) write port 1
        
        // Unified Buffer Read Control Interface
        .ub_rd_start_in(ub_rd_start_in), // i_Signal to start UB read transaction
        .ub_rd_transpose(ub_rd_transpose), // i_Signal to indicate if UB read should transpose the data
        .ub_ptr_select(ub_ptr_select), // i_Signal TODO
        .ub_rd_addr_in(ub_rd_addr_in), // i_Address for UB read transactions
        .ub_rd_row_size(ub_rd_row_size), // i_Number of rows to read for this transaction (for internal address generation)
        .ub_rd_col_size(ub_rd_col_size), // i_Number of columns to read for this transaction (for internal address generation)
        
        .learning_rate_in(16'h0001), // i_Learning rate for weight updates (NOT used in inference)
        
        // VPU Configuration:
        // 4'b0000 = Passthrough mode. The TPU will NOT apply biases or activations internally.
        // It simply multiplies and returns the raw sums to be handled by this controller.
        .vpu_data_pathway(4'b0000), 
        .sys_switch_in(sys_switch_in),
        .vpu_leak_factor_in(16'h0000),
        .inv_batch_size_times_two_in(16'h0000),
        
        // Systolic Output Wires (Unused due to VPU usage)
        .sys_data_out_21(sys_data_out_21),
        .sys_data_out_22(sys_data_out_22),
        .sys_valid_out_21(sys_valid_out_21),
        .sys_valid_out_22(sys_valid_out_22),
        
        // VPU and TPU outputs (which contain raw systolic sums because vpu_data_pathway = 0000)
        .vpu_data_out_1(vpu_data_out_1), 
        .vpu_data_out_2(vpu_data_out_2), 
        .vpu_valid_out_1(vpu_valid_out_1),
        .vpu_valid_out_2(vpu_valid_out_2),
        
        // -------------------------------------------------------------
        // TODO: NOT USED 
        // UB Read Output Wires (Internal routing checks)
        // -------------------------------------------------------------
        .ub_rd_input_data_out_0(ub_rd_input_data_out_0), // o_data input activations to host from TPU (UB) read port 0
        .ub_rd_input_data_out_1(ub_rd_input_data_out_1), // o_data input activations to host from TPU (UB) read port 1
        .ub_rd_input_valid_out_0(ub_rd_input_valid_out_0), // o_valid to host from TPU (UB) for read port 0
        .ub_rd_input_valid_out_1(ub_rd_input_valid_out_1), // o_valid to host from TPU (UB) for read port 1
        .ub_rd_weight_data_out_0(ub_rd_weight_data_out_0), // o_weight to host from TPU (UB) read port 0
        .ub_rd_weight_data_out_1(ub_rd_weight_data_out_1), // o_weight to host from TPU (UB) read port 1
        .ub_rd_weight_valid_out_0(ub_rd_weight_valid_out_0), // o_valid to host from TPU (UB) for weight read port 0
        .ub_rd_weight_valid_out_1(ub_rd_weight_valid_out_1), // o_valid to host from TPU (UB) for weight read port 1
        .ub_rd_bias_data_out_0(ub_rd_bias_data_out_0), // o_bias to host from TPU (UB) read port 0
        .ub_rd_bias_data_out_1(ub_rd_bias_data_out_1), // o_bias to host from TPU (UB) read port 1
        .ub_rd_Y_data_out_0(ub_rd_Y_data_out_0), // o_Y (output activations) to host from TPU (UB) read port 0
        .ub_rd_Y_data_out_1(ub_rd_Y_data_out_1), // o_Y (output activations) to host from TPU (UB) read port 1
        .ub_rd_H_data_out_0(ub_rd_H_data_out_0), // Not used (Maybe used for hidden layer activations)
        .ub_rd_H_data_out_1(ub_rd_H_data_out_1), // Not used (Maybe used for hidden layer activations)
        .ub_rd_col_size_out(ub_rd_col_size_out), 
        .ub_rd_col_size_valid_out(ub_rd_col_size_valid_out)
    );

    // Helper Function: Ensure we don't ask the TPU for more outputs than exist in a tile
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

    // Helper Function: Ensure we don't feed the TPU more inputs than TILE_WIDTH (2)
    function [15:0] clipped_input_words;
        input integer remaining;
        begin
            if (remaining >= 2) begin
                clipped_input_words = 16'd2;
            end else begin
                clipped_input_words = remaining[15:0];
            end
        end
    endfunction

    // 16-bit saturating addition to prevent mathematical overflow during external accumulation
    // If numbers get too large, clamp them to the max/min 16-bit integer values
    function signed [15:0] sat_add16;
        input signed [15:0] left;
        input signed [15:0] right;
        reg signed [16:0] sum;
        begin
            sum = left + right;
            if (sum > 17'sd32767) begin
                sat_add16 = 16'sh7fff;
            end else if (sum < -17'sd32768) begin
                sat_add16 = -16'sd32768;
            end else begin
                sat_add16 = sum[15:0];
            end
        end
    endfunction

    // External ReLU (Rectified Linear Unit) activation function
    // Turns negative numbers to 0, leaves positive numbers alone.
    // Checks the sign bit ([15]) to determine if negative.
    function signed [15:0] relu16;
        input signed [15:0] value;
        begin
            if (value[15]) begin
                relu16 = 16'sh0000;
            end else begin
                relu16 = value;
            end
        end
    endfunction

    // =========================================================================
    // THE CORE FINITE STATE MACHINE (FSM)
    // =========================================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            // Reset all internal state and outputs
            state <= STATE_IDLE;
            current_layer <= 1'b0;
            hidden_tile_index <= {HIDDEN_ADDR_WIDTH{1'b0}}; // Set index to 6'b0 (from tb) for first hidden tile
            output_tile_index <= {OUTPUT_ADDR_WIDTH{1'b0}}; // Set index to 4'b0 (from tb) for first output tile
            chunk_index <= 16'd0;
            input_load_index <= 16'd0;
            weight_load_index <= 16'd0;
            active_tile_outputs <= 16'd0;
            active_input_words <= 16'd0;
            output_seen_0 <= 1'b0;
            output_seen_1 <= 1'b0;
            tpu_rst <= 1'b1; // Reset TPU (active high)
            partial_out_0 <= 16'h0000;
            partial_out_1 <= 16'h0000;
            accum_0 <= 16'h0000;
            accum_1 <= 16'h0000;
            busy <= 1'b0;
            done <= 1'b0;
            prediction_out <= 4'd0;
            argmax_index <= {OUTPUT_ADDR_WIDTH{1'b0}};
            best_index_reg <= 4'd0;
            best_value <= 16'h0000;
            weight_read_addr_a <= 16'd0;
            weight_read_addr_b <= 16'd0;
            weight_read_dual <= 1'b0;
            
            // Clear TPU interface signals
            ub_wr_host_data_in_0 <= 16'h0000;
            ub_wr_host_data_in_1 <= 16'h0000;
            ub_wr_host_valid_in_0 <= 1'b0;
            ub_wr_host_valid_in_1 <= 1'b0;
            ub_rd_start_in <= 1'b0;
            ub_rd_transpose <= 1'b0;
            ub_ptr_select <= 9'd0;
            ub_rd_addr_in <= 16'd0;
            ub_rd_row_size <= 16'd0;
            ub_rd_col_size <= 16'd0;
            sys_switch_in <= 1'b0;
            
            // Clear intermediate buffers
            for (clear_index = 0; clear_index < HIDDEN_NEURONS; clear_index = clear_index + 1) begin
                hidden_buffer[clear_index] <= 16'h0000;
            end
            for (clear_index = 0; clear_index < OUTPUT_NEURONS; clear_index = clear_index + 1) begin
                logits_buffer[clear_index] <= 16'h0000;
            end
        end else begin
            // Default 1-cycle pulses: De-assert write/read valid flags immediately
            done <= 1'b0;
            ub_wr_host_data_in_0 <= 16'h0000;
            ub_wr_host_data_in_1 <= 16'h0000;
            ub_wr_host_valid_in_0 <= 1'b0;
            ub_wr_host_valid_in_1 <= 1'b0;
            ub_rd_start_in <= 1'b0;
            ub_rd_transpose <= 1'b0;
            ub_ptr_select <= 9'd0;
            ub_rd_addr_in <= 16'd0;
            ub_rd_row_size <= 16'd0;
            ub_rd_col_size <= 16'd0;
            sys_switch_in <= 1'b0;
            
            case (state)
                // -------------------------------------------------------------
                // IDLE: Wait for start signal from upper level
                // -------------------------------------------------------------
                // On 'start', initialize indices and control signals for first tile and move to TILE_PREP
                STATE_IDLE: begin
                    tpu_rst <= 1'b0; // TPU ready
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        current_layer <= 1'b0;
                        hidden_tile_index <= {HIDDEN_ADDR_WIDTH{1'b0}};
                        output_tile_index <= {OUTPUT_ADDR_WIDTH{1'b0}};
                        chunk_index <= 16'd0;
                        input_load_index <= 16'd0;
                        weight_load_index <= 16'd0;
                        active_tile_outputs <= clipped_tile_outputs(HIDDEN_NEURONS); // Calculate outputs for this tile (2 or less if edge case)
                        active_input_words <= clipped_input_words(PIXELS); // Calculate input words for this tile (2 or less if edge case)
                        output_seen_0 <= 1'b0;
                        output_seen_1 <= 1'b0;
                        partial_out_0 <= 16'h0000;
                        partial_out_1 <= 16'h0000;
                        accum_0 <= 16'h0000;
                        accum_1 <= 16'h0000;
                        argmax_index <= {OUTPUT_ADDR_WIDTH{1'b0}};
                        best_index_reg <= 4'd0;
                        best_value <= 16'h0000;
                        weight_read_addr_a <= 16'd0;
                        weight_read_addr_b <= 16'd0;
                        weight_read_dual <= 1'b0;
                        tpu_rst <= 1'b1; // Reset TPU to clear any internal state before starting
                        
                        // Clear previous inference results
                        for (clear_index = 0; clear_index < HIDDEN_NEURONS; clear_index = clear_index + 1) begin
                            hidden_buffer[clear_index] <= 16'h0000;
                        end
                        for (clear_index = 0; clear_index < OUTPUT_NEURONS; clear_index = clear_index + 1) begin
                            logits_buffer[clear_index] <= 16'h0000;
                        end
                        state <= STATE_TILE_PREP;
                    end
                end

                // -------------------------------------------------------------
                // PREP: Initialize variables for the next tile processing
                // -------------------------------------------------------------
                STATE_TILE_PREP: begin
                    input_load_index <= 16'd0;
                    weight_load_index <= 16'd0;
                    output_seen_0 <= 1'b0;
                    output_seen_1 <= 1'b0;
                    partial_out_0 <= 16'h0000;
                    partial_out_1 <= 16'h0000;
                    state <= STATE_RESET_ASSERT;
                end

                // Toggle TPU Reset to clear internal accumulators/buffers
                STATE_RESET_ASSERT: begin
                    tpu_rst <= 1'b1;
                    state <= STATE_RESET_RELEASE;
                end

                STATE_RESET_RELEASE: begin
                    tpu_rst <= 1'b0;
                    state <= STATE_LOAD_INPUT;
                end

                // -------------------------------------------------------------
                // DATA PUSH: Write pixel values or hidden activations into UB
                // -------------------------------------------------------------
                // Loads 16b in the UB either pixel in or hidden_buffer value depending on layer
                STATE_LOAD_INPUT: begin
                    if (input_load_index < active_input_words) begin // 0 < 2, rest to be seen
                        if (!current_layer) begin
                            // If Layer 1: Get raw pixels (16b) from host to TPU (UB) (using pixel_addr_out to fetch)
                            ub_wr_host_data_in_0 <= pixel_data_in;
                        end else begin
                            // Else - Layer 2: Get calculated Layer 1 hidden value (16b) - hidden_buffer[chunk_index * 2 + input_load_index]
                            ub_wr_host_data_in_0 <= hidden_buffer[(chunk_index << 1) + input_load_index];
                        end
                        // Assert write valid for port 0
                        ub_wr_host_valid_in_0 <= 1'b1;
                        input_load_index <= input_load_index + 16'd1; // Increment id to load next hidden value in next cycle
                    end else begin // if input_load_index >= active_input_words, we are done loading inputs for this tile
                        weight_load_index <= 16'd0; // Reset weight load index for next state
                        state <= STATE_LOAD_WEIGHT;
                    end
                end

                // -------------------------------------------------------------
                // WEIGHT PUSH: Address logic to fetch weights from ROM/RAM
                // -------------------------------------------------------------
                STATE_LOAD_WEIGHT: begin
                    // Check if we still have weights to load for this 2x2 chunk
                    if (weight_load_index < (active_input_words * active_tile_outputs)) begin
                        // If we need at least 2 weights, use dual-port read optimization
                        if (weight_load_index + 1 < (active_input_words * active_tile_outputs)) begin
                            if (!current_layer) begin // Layer 1 calculation
                                // Full 2x2 chunk logic
                                if (active_input_words == 16'd2 && active_tile_outputs == 16'd2) begin
                                    if (weight_load_index == 16'd0) begin
                                        // Fetch top row of weights for this tile
                                        weight_read_addr_a <= (hidden_tile_index * PIXELS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH);
                                        weight_read_addr_b <= (hidden_tile_index * PIXELS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH) + 16'd2;
                                    end else begin
                                        // Fetch bottom row
                                        weight_read_addr_a <= (hidden_tile_index * PIXELS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH) + 16'd1;
                                        weight_read_addr_b <= (hidden_tile_index * PIXELS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH) + 16'd3;
                                    end
                                end else begin // Partial edge-case chunk
                                    weight_read_addr_a <= (hidden_tile_index * PIXELS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH) + weight_load_index;
                                    weight_read_addr_b <= (hidden_tile_index * PIXELS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH) + weight_load_index + 16'd1;
                                end
                            end else begin // Layer 2 calculation
                                if (active_input_words == 16'd2 && active_tile_outputs == 16'd2) begin
                                    if (weight_load_index == 16'd0) begin
                                        weight_read_addr_a <= (output_tile_index * HIDDEN_NEURONS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH);
                                        weight_read_addr_b <= (output_tile_index * HIDDEN_NEURONS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH) + 16'd2;
                                    end else begin
                                        weight_read_addr_a <= (output_tile_index * HIDDEN_NEURONS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH) + 16'd1;
                                        weight_read_addr_b <= (output_tile_index * HIDDEN_NEURONS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH) + 16'd3;
                                    end
                                end else begin
                                    weight_read_addr_a <= (output_tile_index * HIDDEN_NEURONS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH) + weight_load_index;
                                    weight_read_addr_b <= (output_tile_index * HIDDEN_NEURONS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH) + weight_load_index + 16'd1;
                                end
                            end
                            weight_read_dual <= 1'b1;
                        end else begin 
                            // Only 1 weight left to load (boundary conditions)
                            if (!current_layer) begin
                                weight_read_addr_a <= (hidden_tile_index * PIXELS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH) + weight_load_index;
                            end else begin
                                weight_read_addr_a <= (output_tile_index * HIDDEN_NEURONS * TILE_WIDTH) + (chunk_index * 2 * TILE_WIDTH) + weight_load_index;
                            end
                            weight_read_addr_b <= 16'd0;
                            weight_read_dual <= 1'b0;
                        end
                        state <= STATE_LOAD_WEIGHT_WAIT;
                    end else begin
                        // All weights loaded into UB, begin TPU trigger sequence
                        state <= STATE_START_WEIGHT;
                    end
                end

                // Wait for synchronous memory read delay
                STATE_LOAD_WEIGHT_WAIT: begin
                    state <= STATE_LOAD_WEIGHT_COMMIT;
                end

                // Write fetched weights to TPU Unified Buffer
                STATE_LOAD_WEIGHT_COMMIT: begin
                    ub_wr_host_data_in_1 <= active_weight_read_data_a;
                    ub_wr_host_valid_in_1 <= weight_read_dual;
                    
                    if (weight_read_dual) begin
                        ub_wr_host_data_in_0 <= active_weight_read_data_b;
                        ub_wr_host_valid_in_0 <= 1'b1;
                        weight_load_index <= weight_load_index + 16'd2;
                    end else begin
                        ub_wr_host_data_in_0 <= active_weight_read_data_a;
                        ub_wr_host_valid_in_0 <= 1'b1;
                        weight_load_index <= weight_load_index + 16'd1;
                    end
                    state <= STATE_LOAD_WEIGHT;
                end

                // -------------------------------------------------------------
                // TPU TRIGGERS: Start pushing data from UB to Systolic Array
                // -------------------------------------------------------------
                
                // Trigger Systolic Array weight read from UB
                STATE_START_WEIGHT: begin
                    ub_rd_start_in <= 1'b1;
                    ub_ptr_select <= 9'd1; // Address the weight section (top of array)
                    ub_rd_addr_in <= active_input_words; // Offset past input data
                    ub_rd_row_size <= active_input_words;
                    ub_rd_col_size <= active_tile_outputs;
                    ub_rd_transpose <= 1'b1; // Weights are transposed entering array
                    state <= STATE_START_WEIGHT_GAP;
                end

                // Timing gap required by TPU architecture between weight/input loads
                STATE_START_WEIGHT_GAP: begin
                    state <= STATE_START_INPUT;
                end

                // Trigger Systolic Array input read from UB
                STATE_START_INPUT: begin
                    ub_rd_start_in <= 1'b1;
                    ub_ptr_select <= 9'd0; // Address input section (left of array)
                    ub_rd_addr_in <= 16'd0;
                    ub_rd_row_size <= 16'd1;
                    ub_rd_col_size <= active_input_words;
                    ub_rd_transpose <= 1'b0;
                    state <= STATE_SWITCH_WEIGHTS;
                end

                STATE_SWITCH_WEIGHTS: begin
                    sys_switch_in <= 1'b1; // Latch weights into array shadow buffers
                    
                    // Prep variables for receiving data
                    output_seen_0 <= 1'b0; 
                    output_seen_1 <= 1'b0;
                    partial_out_0 <= 16'h0000;
                    partial_out_1 <= 16'h0000;
                    state <= STATE_WAIT_OUTPUT;
                end

                // -------------------------------------------------------------
                // TPU POLL: Wait for results and accumulate
                // -------------------------------------------------------------
                // Poll TPU for VPU output valid signals
                STATE_WAIT_OUTPUT: begin
                    if (vpu_valid_out_1) begin
                        output_seen_0 <= 1'b1;
                        partial_out_0 <= vpu_data_out_1; // Capture column 0 result
                    end

                    if (active_tile_outputs > 1 && vpu_valid_out_2) begin
                        output_seen_1 <= 1'b1;
                        partial_out_1 <= vpu_data_out_2; // Capture column 1 result
                    end

                    // Check if we have received all expected outputs for this chunk
                    if (!vpu_valid_out_1 && !vpu_valid_out_2 &&
                        output_seen_0 &&
                        ((active_tile_outputs == 1) || output_seen_1)) begin
                        
                        // Because TPU is in passthrough mode, we must accumulate the chunks
                        // manually using safe saturating addition
                        accum_0 <= sat_add16(accum_0, partial_out_0);
                        if (active_tile_outputs > 1) begin
                            accum_1 <= sat_add16(accum_1, partial_out_1);
                        end
                        state <= STATE_NEXT_CHUNK;
                    end
                end

                // -------------------------------------------------------------
                // CHUNK ITERATION: Move to next 2x2 chunk of the current row
                // -------------------------------------------------------------
                STATE_NEXT_CHUNK: begin
                    if (!current_layer) begin
                        // Are there more pixels to process for this hidden neuron tile?
                        if (((chunk_index + 1) << 1) < PIXELS) begin
                            chunk_index <= chunk_index + 16'd1;
                            active_input_words <= clipped_input_words(PIXELS - ((chunk_index + 1) * 2));
                            state <= STATE_TILE_PREP; // Loop back and load next chunk
                        end else begin
                            state <= STATE_FINALIZE_TILE; // Done summing chunks
                        end
                    end else begin
                        // Are there more hidden neurons to process for this output tile?
                        if (((chunk_index + 1) << 1) < HIDDEN_NEURONS) begin
                            chunk_index <= chunk_index + 16'd1;
                            active_input_words <= clipped_input_words(HIDDEN_NEURONS - ((chunk_index + 1) * 2));
                            state <= STATE_TILE_PREP; // Loop back
                        end else begin
                            state <= STATE_FINALIZE_TILE;
                        end
                    end
                end

                // -------------------------------------------------------------
                // POST-PROCESSING: Externally apply Bias and Activation (ReLU)
                // -------------------------------------------------------------
                STATE_FINALIZE_TILE: begin
                    if (!current_layer) begin
                        // Add bias for Neuron 0, apply ReLU, store in hidden buffer
                        hidden_buffer[hidden_tile_index * TILE_WIDTH] <= relu16(sat_add16(accum_0, b1_mem[hidden_tile_index * TILE_WIDTH]));
                        if (active_tile_outputs > 1) begin
                            // Add bias for Neuron 1, apply ReLU, store
                            hidden_buffer[(hidden_tile_index * TILE_WIDTH) + 1] <= relu16(sat_add16(accum_1, b1_mem[(hidden_tile_index * TILE_WIDTH) + 1]));
                        end
                    end else begin
                        // For Layer 2, Add Bias, but DO NOT apply ReLU to Logits
                        logits_buffer[output_tile_index * TILE_WIDTH] <= sat_add16(accum_0, b2_mem[output_tile_index * TILE_WIDTH]);
                        if (active_tile_outputs > 1) begin
                            logits_buffer[(output_tile_index * TILE_WIDTH) + 1] <= sat_add16(accum_1, b2_mem[(output_tile_index * TILE_WIDTH) + 1]);
                        end
                    end
                    state <= STATE_NEXT_TILE;
                end

                // -------------------------------------------------------------
                // TILE/LAYER ITERATION
                // -------------------------------------------------------------
                STATE_NEXT_TILE: begin
                    if (!current_layer) begin
                        if (hidden_tile_index + 1 < HIDDEN_TILES) begin
                            // Move to next hidden neuron tile (e.g. Neurons 2,3)
                            hidden_tile_index <= hidden_tile_index + {{(HIDDEN_ADDR_WIDTH - 1){1'b0}}, 1'b1};
                            chunk_index <= 16'd0;
                            input_load_index <= 16'd0;
                            weight_load_index <= 16'd0;
                            active_tile_outputs <= clipped_tile_outputs(HIDDEN_NEURONS - ((hidden_tile_index + 1) * TILE_WIDTH));
                            active_input_words <= clipped_input_words(PIXELS);
                            accum_0 <= 16'h0000;
                            accum_1 <= 16'h0000;
                            state <= STATE_TILE_PREP;
                        end else begin
                            // Transition from Layer 1 (Hidden) to Layer 2 (Output)
                            current_layer <= 1'b1;
                            output_tile_index <= {OUTPUT_ADDR_WIDTH{1'b0}};
                            chunk_index <= 16'd0;
                            input_load_index <= 16'd0;
                            weight_load_index <= 16'd0;
                            active_tile_outputs <= clipped_tile_outputs(OUTPUT_NEURONS);
                            active_input_words <= clipped_input_words(HIDDEN_NEURONS);
                            accum_0 <= 16'h0000;
                            accum_1 <= 16'h0000;
                            state <= STATE_TILE_PREP;
                        end
                    end else begin
                        if (output_tile_index + 1 < OUTPUT_TILES) begin
                            // Move to next output logit tile
                            output_tile_index <= output_tile_index + {{(OUTPUT_ADDR_WIDTH - 1){1'b0}}, 1'b1};
                            chunk_index <= 16'd0;
                            input_load_index <= 16'd0;
                            weight_load_index <= 16'd0;
                            active_tile_outputs <= clipped_tile_outputs(OUTPUT_NEURONS - ((output_tile_index + 1) * TILE_WIDTH));
                            active_input_words <= clipped_input_words(HIDDEN_NEURONS);
                            accum_0 <= 16'h0000;
                            accum_1 <= 16'h0000;
                            state <= STATE_TILE_PREP;
                        end else begin
                            // Layer 2 complete. Begin final classification.
                            state <= STATE_ARGMAX;
                        end
                    end
                end

                // -------------------------------------------------------------
                // PREDICTION: Find highest probability logit (Argmax)
                // -------------------------------------------------------------
                STATE_ARGMAX: begin
                    best_index_reg <= 4'd0; // Default guess is 0
                    best_value <= logits_buffer[0]; // Score for digit 0
                    
                    if (OUTPUT_NEURONS > 1) begin
                        // Start comparing at index 1
                        argmax_index <= {{(OUTPUT_ADDR_WIDTH - 1){1'b0}}, 1'b1};
                        state <= STATE_ARGMAX_COMPARE;
                    end else begin
                        // Edge case fallback
                        prediction_out <= 4'd0;
                        busy <= 1'b0;
                        done <= 1'b1;
                        state <= STATE_DONE;
                    end
                end

                STATE_ARGMAX_COMPARE: begin
                    // Compare current highest score to the next logit
                    // Use $signed to ensure negative logits are evaluated correctly
                    if ($signed(logits_buffer[argmax_index]) > $signed(best_value)) begin
                        best_index_reg <= argmax_index; // Update new winner
                        best_value <= logits_buffer[argmax_index];
                    end

                    // Iterate until all 10 logits are checked
                    if (argmax_index + {{(OUTPUT_ADDR_WIDTH - 1){1'b0}}, 1'b1} < OUTPUT_NEURONS) begin
                        argmax_index <= argmax_index + {{(OUTPUT_ADDR_WIDTH - 1){1'b0}}, 1'b1};
                    end else begin
                        state <= STATE_ARGMAX_LATCH;
                    end
                end

                STATE_ARGMAX_LATCH: begin
                    // Output the final predicted digit (0-9)
                    prediction_out <= best_index_reg;
                    busy <= 1'b0;
                    done <= 1'b1; // Pulse done signal
                    state <= STATE_DONE;
                end

                STATE_DONE: begin
                    // Return to start
                    state <= STATE_IDLE;
                end

                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end
endmodule