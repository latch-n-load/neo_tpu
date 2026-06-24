// ABOUTME: Interface between NEORV32 CFS and TPU top (classifer_core).
// ABOUTME: Presents one-bit image pixels to the classifier core as Q8.8 values and latches inference results.

`timescale 1ns/1ps
`default_nettype none

/* * MODULE: cfs_tpu_mnist_interface
 * PURPOSE: Acts as a bridge between the host processor (via CFS-MM bus) and the Tiny-TPU hardware.
 * It provides memory-mapped registers for control (start/clear), status (busy/done), and results (prediction).
 * It also includes a dedicated memory space to buffer the 784-pixel input image.
 */
module cfs_tpu_mnist_interface #(
    parameter integer PIXELS = 784,                           // Total pixels in a 28x28 MNIST image
    parameter integer PIXEL_ADDR_WIDTH = 10,                  // Bits needed to address 784 pixels (2^10 = 1024)
    parameter [31:0] IMAGE_BASE_ADDR = 32'h00000100           // Base byte address where the image buffer starts
) (
    // System Clock and Reset
    input wire clk,
    input wire rst,

    // CFS side interface
    input wire [31:0] cfs_addr_i,                            // 32-bit byte address from the bus master
    input wire cfs_read_en_i,                                      // Read enable signal
    input wire cfs_write_en_i,                                     // Write enable signal
    input wire [31:0] cfs_write_data_i,                          // 32-bit data to be written to a register
    input wire [3:0] cfs_byte_en_i,                          // Indicates which bytes of the 32-bit word are valid
    output reg [31:0] cfs_read_data_o,                           // 32-bit data read back to the bus master
    output reg cfs_read_data_valid_o,                             // Validates the read data
    output wire cfs_wait_req_o,                              // Used to stall the bus (hardcoded to 0 here for no stalls)
    output reg [3:0] cfs_prediction_o,                   // Safely stored prediction result to be read by host


    // TPU Interface (Signals connecting directly to Tiny-TPU hardware)
    input wire tpu_busy_i,                                       // High when TPU is computing
    input wire tpu_done_i,                                       // 1-cycle pulse from TPU when inference finishes
    input wire [3:0] tpu_prediction_i,                           // 4-bit prediction (0-9) from the TPU
    input wire [PIXEL_ADDR_WIDTH - 1:0] tpu_pixel_addr_i,        // TPU requesting a specific pixel index
    output reg [15:0] tpu_pixel_data_o,                         // Pixel data sent to TPU (formatted as 16-bit Q8.8)
    output reg tpu_start_o,                               // 1-cycle pulse to wake up the TPU
    output reg tpu_frame_ldd_o,                              // Flag indicating a full image is in the buffer
    output reg tpu_done_o,                               // Persistent flag indicating inference is complete
    output reg tpu_write_while_busy_o                          // Error flag if host writes while TPU is running
);

    // Register Map Word Addresses 
    // Note: CFS-MM uses byte addressing, but registers are 32-bit (4 bytes) wide.
    // By ignoring the lowest 2 bits of the address, we convert byte addresses to word addresses.
    localparam [31:0] CTRL_WORD_ADDR = 32'h00000000;          // Byte Addr: 0x00
    localparam [31:0] STATUS_WORD_ADDR = 32'h00000001;        // Byte Addr: 0x04
    localparam [31:0] RESULT_WORD_ADDR = 32'h00000002;        // Byte Addr: 0x08
    localparam [31:0] VERSION_WORD_ADDR = 32'h00000003;       // Byte Addr: 0x0C

    // Image Buffer Word Addresses
    localparam [31:0] IMAGE_BASE_WORD_ADDR = IMAGE_BASE_ADDR[31:2]; // Convert 0x100 to word address (0x40)
    localparam [31:0] IMAGE_LAST_WORD_ADDR = IMAGE_BASE_WORD_ADDR + PIXELS - 1; 

    // Magic Number for Device Identification ("MNIS" in ASCII)
    localparam [31:0] VERSION_VALUE = 32'h4D4E4953;

    // Internal Storage
    reg [PIXELS - 1:0] frame_bits;                            // 784-bit wide register storing the binary image
    reg [31:0] read_data_next;                                // Combinational signal holding the next read value
    wire [31:0] word_addr;                                    // The translated word address
    integer clear_index;                                      // Loop variable for clearing the image buffer
    integer pixel_index;                                      // Variable to calculate pixel offset during writes

    /* * FUNCTION: pixel_bit_from_write
     * PURPOSE: Extracts a single bit from the 32-bit CFS write data based on which byte lane is enabled.
     * Since the image is binary, we only care about the lowest bit of the valid byte.
     */
    function pixel_bit_from_write;
        input [31:0] data;
        input [3:0] byteenable;
        begin
            pixel_bit_from_write = 1'b0;
            if (byteenable[0]) begin
                pixel_bit_from_write = data[0];
            end else if (byteenable[1]) begin
                pixel_bit_from_write = data[8];
            end else if (byteenable[2]) begin
                pixel_bit_from_write = data[16];
            end else if (byteenable[3]) begin
                pixel_bit_from_write = data[24];
            end
        end
    endfunction

    // Convert incoming byte address to word address by dropping the lower 2 bits (divide by 4)
    assign word_addr = cfs_addr_i[31:2];

    // We never stall the CFS bus; all our reads/writes complete in 1 cycle
    assign cfs_wait_req_o = 1'b0;

    /*
     * COMBINATIONAL READ LOGIC
     * Multiplexes the correct register data onto the read bus based on the requested address.
     */
    always @(*) begin
        read_data_next = 32'h00000000; // Default to 0
        if (word_addr == CTRL_WORD_ADDR) begin
            // Control register is write-only, reads return 0
            read_data_next[0] = 1'b0;
            read_data_next[1] = 1'b0;
            read_data_next[2] = 1'b0;
            read_data_next[3] = 1'b0;
        end else if (word_addr == STATUS_WORD_ADDR) begin
            // Pack status flags into a single 32-bit word
            read_data_next[0] = tpu_busy_i;
            read_data_next[1] = tpu_done_o;
            read_data_next[2] = tpu_frame_ldd_o;
            read_data_next[3] = tpu_write_while_busy_o;
        end else if (word_addr == RESULT_WORD_ADDR) begin
            // Return the latched 4-bit prediction
            read_data_next[3:0] = cfs_prediction_o;
        end else if (word_addr == VERSION_WORD_ADDR) begin
            // Return the device ID magic number
            read_data_next = VERSION_VALUE;
        end else if (word_addr >= IMAGE_BASE_WORD_ADDR && word_addr <= IMAGE_LAST_WORD_ADDR) begin
            // Allow the host to read back specific pixels from the buffer for verification
            read_data_next[0] = frame_bits[word_addr - IMAGE_BASE_WORD_ADDR];
        end
    end

    /*
     * TPU DATA FORMATTING (Q8.8 Fixed-Point Conversion)
     * The TPU hardware asks for a pixel using `tpu_pixel_addr_i`.
     * This block fetches the 1-bit pixel from `frame_bits` and casts it to a 16-bit Q8.8 value.
     * Binary '1' (White) becomes 0x0100 (1.0 in Q8.8).
     * Binary '0' (Black) becomes 0x0000 (0.0 in Q8.8).
     */
    always @(*) begin
        tpu_pixel_data_o = 16'h0000;
        if (tpu_pixel_addr_i < PIXELS && frame_bits[tpu_pixel_addr_i]) begin
            tpu_pixel_data_o = 16'h0100;
        end
    end

    /*
     * MAIN SEQUENTIAL LOGIC
     * Handles Reset, CFS-MM Writes, CFS-MM Reads, and asynchronous TPU signals.
     */
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            // Hard reset: clear all buffers, flags, and outputs
            frame_bits <= {PIXELS{1'b0}};
            tpu_start_o <= 1'b0;
            tpu_frame_ldd_o <= 1'b0;
            tpu_done_o <= 1'b0;
            tpu_write_while_busy_o <= 1'b0;
            cfs_prediction_o <= 4'd0;
            cfs_read_data_o <= 32'h00000000;
            cfs_read_data_valid_o <= 1'b0;
        end else begin
            // Default pulse behavior: start signal is only high for 1 clock cycle
            tpu_start_o <= 1'b0;
            
            // Latch read requests to the output bus
            cfs_read_data_valid_o <= cfs_read_en_i;
            if (cfs_read_en_i) begin
                cfs_read_data_o <= read_data_next;
            end

            // Capture the 1-cycle 'done' pulse from the TPU and make it sticky/persistent 
            // so the host CPU can read it via polling
            if (tpu_done_i) begin
                tpu_done_o <= 1'b1;
                cfs_prediction_o <= tpu_prediction_i;
            end

            // Handle CFS-MM Writes from the Host Processor
            if (cfs_write_en_i) begin
                if (word_addr == CTRL_WORD_ADDR) begin
                    
                    // Bit 0: START command. Only fire if image is loaded and TPU is idle.
                    if (cfs_byte_en_i[0] && cfs_write_data_i[0] && tpu_frame_ldd_o && !tpu_busy_i) begin
                        tpu_start_o <= 1'b1;
                        tpu_done_o <= 1'b0; // Auto-clear old results on new run
                    end
                    
                    // Bit 1: SOFT RESET / CLEAR command. Wipes image buffer and resets all flags.
                    if (cfs_byte_en_i[0] && cfs_write_data_i[1]) begin
                        for (clear_index = 0; clear_index < PIXELS; clear_index = clear_index + 1) begin
                            frame_bits[clear_index] <= 1'b0;
                        end
                        tpu_frame_ldd_o <= 1'b0;
                        tpu_done_o <= 1'b0;
                        cfs_prediction_o <= 4'd0;
                        tpu_write_while_busy_o <= 1'b0;
                    end
                    
                    // Bit 2: Clear 'Done' flag manually
                    if (cfs_byte_en_i[0] && cfs_write_data_i[2]) begin
                        tpu_done_o <= 1'b0;
                    end
                    
                    // Bit 3: Clear 'Write While Busy' error flag manually
                    if (cfs_byte_en_i[0] && cfs_write_data_i[3]) begin
                        tpu_write_while_busy_o <= 1'b0;
                    end

                end else if (word_addr >= IMAGE_BASE_WORD_ADDR && word_addr <= IMAGE_LAST_WORD_ADDR) begin
                    // Writing pixels into the image buffer
                    pixel_index = word_addr - IMAGE_BASE_WORD_ADDR;
                    
                    if (tpu_busy_i) begin
                        // ERROR: Host tried to change image data while TPU is currently calculating!
                        tpu_write_while_busy_o <= 1'b1;
                    end else begin
                        // Valid pixel write: update the bit array and flag the image as loaded
                        frame_bits[pixel_index] <= pixel_bit_from_write(cfs_write_data_i, cfs_byte_en_i);
                        tpu_frame_ldd_o <= 1'b1;
                        tpu_done_o <= 1'b0; // Invalidate any old prediction since data changed
                    end
                end
            end
        end
    end
endmodule