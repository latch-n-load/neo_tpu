// ABOUTME: Interface between NEORV32 CFS and TPU top (classifer_core).
// ABOUTME: Presents one-bit image pixels to the classifier core as Q8.8 values and latches inference results.

`timescale 1ns/1ps
`default_nettype none

/* * MODULE: cfs_tpu_mnist_interface
 * PURPOSE: Acts as a bridge between the host processor CFS and the Tiny-TPU hardware.
 * It provides memory-mapped registers for control (start/clear), status (busy/done), and results (prediction).
 * It also includes a dedicated memory space to buffer the 784-pixel input image.
 */
module cfs_tpu_mnist_interface #(
    parameter integer PIXELS = 784,                           // Total pixels in a 28x28 MNIST image
    parameter integer PIXEL_ADDR_WIDTH = 10,                  // Bits needed to address 784 pixels (2^10 = 1024)
    parameter [31:0] IMAGE_BASE_ADDR = 32'h00000100           // Base byte address for image data @ 0x...100
) (
    // System Clock and Reset
    input wire clk,
    input wire rst,

    // CFS side interface
    input wire [31:0] cfs_addr_i,                           // 32-bit byte address from the NEORV32 / CFS
    input wire cfs_read_en_i,                               // From CFS, read enable for interface registers
    input wire cfs_write_en_i,                              // From CFS, write enable for interface registers
    input wire [31:0] cfs_write_data_i,                     // 32-bit image to be written to frame_bits register
    input wire [3:0] cfs_byte_en_i,                         // Which byte(s) of cfs_write_data_i are valid
    output reg [31:0] cfs_read_data_o,                      // 32-bit data read out from interface registers to CFS
    output reg cfs_read_data_valid_o,                       // Indicates if value on cfs_read_data_o is valid
    output wire cfs_wait_req_o,                             // May be used to stall CFS rd/wr (set to 0, not used otherwise)
    output reg [3:0] cfs_prediction_o,                      // Dedicated out port to CFS for 0-9 prediction from TPU
    output reg tpu_done_o,                                  // Flag to CFS indicating if TPU inference is done
    output reg tpu_frame_ldd_o,                             // Flag to CFS indicating that frame is loaded in TPU
    output reg tpu_write_while_busy_o                       // Error flag to CFS if it writes while TPU is busy


    // TPU side Interface 
    input wire tpu_busy_i,                                       // High when TPU is computing
    input wire tpu_done_i,                                       // 1-cycle pulse from TPU when inference finishes
    input wire [3:0] tpu_prediction_i,                           // 4-bit prediction (0-9) from TPU to interface
    input wire [PIXEL_ADDR_WIDTH - 1:0] tpu_pixel_addr_i,        // TPU_read_addr_i for reading pixels in frame_bits
    output reg [15:0] tpu_pixel_data_o,                          // Pixel data_o to TPU (formatted as 16-bit Q8.8)
    output reg tpu_start_o,                                      // 1-cycle pulse to start TPU inference
);

    // Register Map Word Addresses 
    // Note: CFS uses byte addressing, but registers are 32-bit (4 bytes) wide.
    // By ignoring the lowest 2 bits of the address, we convert byte addresses to word addresses.
    localparam [31:0] CTRL_WORD_ADDR = 32'h00000000;          // Byte Addr: 0x00
    localparam [31:0] STATUS_WORD_ADDR = 32'h00000001;        // Byte Addr: 0x04
    localparam [31:0] RESULT_WORD_ADDR = 32'h00000002;        // Byte Addr: 0x08
    localparam [31:0] VERSION_WORD_ADDR = 32'h00000003;       // Byte Addr: 0x0C

    // Image Buffer Word Addresses
    localparam [31:0] IMAGE_BASE_WORD_ADDR = IMAGE_BASE_ADDR[31:2]; // Convert 0x100 to word address (0x40)
    localparam [31:0] IMAGE_LAST_WORD_ADDR = IMAGE_BASE_WORD_ADDR + PIXELS - 1; // 0x...34F

    // Device Identification Number ("MNIS" in ASCII)
    localparam [31:0] VERSION_VALUE = 32'h4D4E4953;

    // Internal Storage
    reg [PIXELS - 1:0] frame_bits;                            // 784-bit register for 28x28x1 1b image
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

    // We never stall the CFS bus; all read/write complete in 1 cycle
    assign cfs_wait_req_o = 1'b0;

    /*
     * COMBINATIONAL READ LOGIC
     * Multiplexes the correct register data onto the read bus based on the requested address.
     */
    always @(*) begin
        read_data_next = 32'h00000000; // Default to 0
        if (word_addr == CTRL_WORD_ADDR) begin
            // Control register is write-only, reads 0
            read_data_next = 32'h00000000;
        end else if (word_addr == STATUS_WORD_ADDR) begin
            // Sets read_data_next as b'...{tpu_write_while_busy_o, tpu_frame_ldd_o, tpu_done_o, tpu_busy_i}
            read_data_next[0] = tpu_busy_i;
            read_data_next[1] = tpu_done_o;
            read_data_next[2] = tpu_frame_ldd_o;
            read_data_next[3] = tpu_write_while_busy_o;
        end else if (word_addr == RESULT_WORD_ADDR) begin
            // Return the latched 4-bit prediction on 4 LSBs
            read_data_next[3:0] = cfs_prediction_o;
        end else if (word_addr == VERSION_WORD_ADDR) begin
            // Return the device ID number on 32b
            read_data_next = VERSION_VALUE;
        end else if (word_addr >= IMAGE_BASE_WORD_ADDR && word_addr <= IMAGE_LAST_WORD_ADDR) begin
            // read_data_next 0th bit set to bit from pixel frame
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
     * Handles Reset, CFS Writes/Reads, and asynchronous TPU signals.
     */
    always @(posedge clk or posedge rst) begin
        if (rst) begin // Active high reset
            // Hard reset: clear all buffers, flags, and outputs
            frame_bits <= {PIXELS{1'b0}};       // Set pixel frame to all 0s
            tpu_start_o <= 1'b0;                // Clear start pulse
            tpu_frame_ldd_o <= 1'b0;            
            tpu_done_o <= 1'b0;
            tpu_write_while_busy_o <= 1'b0;
            cfs_prediction_o <= 4'd0;
            cfs_read_data_o <= 32'h00000000;
            cfs_read_data_valid_o <= 1'b0;
        end else begin
            // Default pulse behavior: start signal is only high for 1 clock cycle
            tpu_start_o <= 1'b0;
            
            // Send read_data_next from interface registers to CFS readout
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

            // Handle CFS Writes from the Host Processor
            if (cfs_write_en_i) begin
                // If CFS writes to CTRL register
                if (word_addr == CTRL_WORD_ADDR) begin
                    
                    // cfs_write_data_i BIT 0: START command. Only fire if image is loaded and TPU is idle.
                    if (cfs_byte_en_i[0] && cfs_write_data_i[0] && tpu_frame_ldd_o && !tpu_busy_i) begin
                        tpu_start_o <= 1'b1;    // Send start signal to TPU
                        tpu_done_o <= 1'b0;     // Clear TPU done flag seen by CFS
                    end
                    
                    // cfs_write_data_i BIT 1: CLEAR command. Wipes image buffer and resets all flags.
                    if (cfs_byte_en_i[0] && cfs_write_data_i[1]) begin
                        for (clear_index = 0; clear_index < PIXELS; clear_index = clear_index + 1) begin
                            frame_bits[clear_index] <= 1'b0;
                        end
                        tpu_frame_ldd_o <= 1'b0;
                        tpu_done_o <= 1'b0;
                        cfs_prediction_o <= 4'd0;
                        tpu_write_while_busy_o <= 1'b0;
                    end
                    
                    // cfs_write_data_i BIT 2: Clear 'Done' flag manually
                    if (cfs_byte_en_i[0] && cfs_write_data_i[2]) begin
                        tpu_done_o <= 1'b0;
                    end
                    
                    // cfs_write_data_i BIT 3: Clear 'Write While Busy' error flag manually
                    if (cfs_byte_en_i[0] && cfs_write_data_i[3]) begin
                        tpu_write_while_busy_o <= 1'b0;
                    end

                // If CFS writes to image buffer
                end else if (word_addr >= IMAGE_BASE_WORD_ADDR && word_addr <= IMAGE_LAST_WORD_ADDR) begin
                    /*  Initialize frame_bits' pixel index
                        pixel_index is handled by CFS, i.e. CFS writes to TPU pixel-1b by pixel-1b on each write
                        So 784 WRITES for 784 PIXELS
                    */  
                    pixel_index = word_addr - IMAGE_BASE_WORD_ADDR;
                    
                    // If write happens when tpu_busy_i
                    if (tpu_busy_i) begin
                        // ERROR: Host tried to change image data while TPU is currently calculating!
                        tpu_write_while_busy_o <= 1'b1;

                    // If TPU is idle
                    end else begin
                        // Valid pixel write: update the bit array and flag the image as loaded
                        frame_bits[pixel_index] <= pixel_bit_from_write(cfs_write_data_i, cfs_byte_en_i);
                        tpu_frame_ldd_o <= 1'b1;
                        tpu_done_o <= 1'b0; // De-assert done flag
                    end
                end
            end
        end
    end
endmodule