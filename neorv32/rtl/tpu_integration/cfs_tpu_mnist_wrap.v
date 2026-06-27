// ABOUTME: Instantiates the cfs_tpu_mnist_interface and mnist_classifier_core 

`timescale 1ns/1ps
`default_nettype none

module cfs_tpu_mnist_wrap #(
    parameter integer PIXELS = 784;
    parameter integer PIXEL_ADDR_WIDTH = 10;
    parameter integer CLOCK_HZ = 50000000;
    parameter integer MAX_WAIT_CYCLES = 2000000;
    parameter [31:0] IMAGE_BASE = 32'h00000100;
    parameter W1_INIT_FILE = "../../../data/model/reference/w1_tiled_q8_8.memh";
    parameter B1_INIT_FILE = "../../../data/model/reference/b1_q8_8.memh";
    parameter W2_INIT_FILE = "../../../data/model/reference/w2_tiled_q8_8.memh";
    parameter B2_INIT_FILE = "../../../data/model/reference/b2_q8_8.memh";
) ( 
    input wire clk,
    input wire rst,
    input wire [31:0] cfs_addr_i;

);
    
    reg cfs_read_en_i;
    reg cfs_write_en_i;
    reg [31:0] cfs_write_data_i;
    reg [3:0] cfs_byte_en_i;
    wire [31:0] cfs_read_data_o;
    wire cfs_read_data_valid_o;
    wire cfs_wait_req_o;

    wire tpu_busy;
    wire tpu_done;
    wire [3:0] tpu_prediction;
    wire [PIXEL_ADDR_WIDTH - 1:0] tpu_pixel_addr;
    wire [15:0] tpu_pixel_data;
    wire tpu_start;
    wire tpu_frame_ldd;
    wire tpu_done;
    wire tpu_write_while_busy;
    wire [3:0] tpu_prediction;

    reg [7:0] sample_bytes [0:(PIXELS / 8) - 1];
    integer pixel_index;
    integer wait_cycles;
    reg [31:0] status_value;
    reg [31:0] result_value;
    integer label_file;
    integer label_scan;
    integer expected_label_int;
    reg [3:0] expected_label;

    cfs_tpu_mnist_interface #(
        .PIXELS(PIXELS),
        .PIXEL_ADDR_WIDTH(PIXEL_ADDR_WIDTH),
        .IMAGE_BASE_ADDR(IMAGE_BASE)
    ) cfs_tpu_mnist_interface_inst (
        .clk(clk),
        .rst(rst),
        .cfs_addr_i(cfs_addr_i),
        .cfs_read_en_i(cfs_read_en_i),
        .cfs_write_en_i(cfs_write_en_i),
        .cfs_write_data_i(cfs_write_data_i),
        .cfs_byte_en_i(cfs_byte_en_i),
        .cfs_read_data_o(cfs_read_data_o),
        .cfs_read_data_valid_o(cfs_read_data_valid_o),
        .cfs_wait_req_o(cfs_wait_req_o),
        .tpu_busy_i(tpu_busy),
        .tpu_done_i(tpu_done),
        .tpu_prediction_i(tpu_prediction),
        .tpu_pixel_addr_i(tpu_pixel_addr),
        .tpu_pixel_data_o(tpu_pixel_data),
        .tpu_start_o(tpu_start),
        .tpu_frame_ldd_o(tpu_frame_ldd),
        .tpu_done_o(tpu_done),
        .tpu_write_while_busy_o(tpu_write_while_busy),
        .cfs_prediction_o(tpu_prediction)
    );

    mnist_classifier_core #(
        .PIXELS(PIXELS),
        .PIXEL_ADDR_WIDTH(PIXEL_ADDR_WIDTH),
        .HIDDEN_NEURONS(64),
        .HIDDEN_ADDR_WIDTH(6),
        .OUTPUT_NEURONS(10),
        .OUTPUT_ADDR_WIDTH(4),
        .TILE_WIDTH(2),
        .UNIFIED_BUFFER_WIDTH(128),
        .PRELOAD_MODEL(1),
        .W1_INIT_FILE(W1_INIT_FILE),
        .B1_INIT_FILE(B1_INIT_FILE),
        .W2_INIT_FILE(W2_INIT_FILE),
        .B2_INIT_FILE(B2_INIT_FILE)
    ) classifier_inst (
        .clk(clk),
        .rst(rst),
        .start(tpu_start),
        .pixel_data_in(tpu_pixel_data),
        .pixel_addr_out(tpu_pixel_addr),
        .busy(tpu_busy),
        .done(tpu_done),
        .prediction_out(tpu_prediction)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task write32;
        input [31:0] addr;
        input [31:0] value;
        begin
            @(posedge clk);
            cfs_addr_i <= addr;
            cfs_write_data_i <= value;
            cfs_byte_en_i <= 4'h1;
            cfs_write_en_i <= 1'b1;
            cfs_read_en_i <= 1'b0;
            @(posedge clk);
            cfs_write_en_i <= 1'b0;
            cfs_addr_i <= 32'h0;
            cfs_write_data_i <= 32'h0;
        end
    endtask

    task read32;
        input [31:0] addr;
        output [31:0] value;
        begin
            @(posedge clk);
            cfs_addr_i <= addr;
            cfs_read_en_i <= 1'b1;
            cfs_write_en_i <= 1'b0;
            @(posedge clk);
            cfs_read_en_i <= 1'b0;
            cfs_addr_i <= 32'h0;
            wait (cfs_read_data_valid_o);
            value = cfs_read_data_o;
        end
    endtask

    initial begin
        rst = 1'b1;
        cfs_addr_i = 32'h0;
        cfs_read_en_i = 1'b0;
        cfs_write_en_i = 1'b0;
        cfs_write_data_i = 32'h0;
        cfs_byte_en_i = 4'h1;
        expected_label = 4'd0;
        expected_label_int = 0;

        $readmemh("../../../data/model/reference/sample_image_0.memh", sample_bytes);
        label_file = $fopen("../../../data/model/reference/sample_expected_prediction_0.txt", "r");
        if (label_file == 0) begin
            $display("FAIL: could not open expected prediction file");
            $finish(1);
        end
        label_scan = $fscanf(label_file, "%d", expected_label_int);
        $fclose(label_file);
        if (label_scan != 1 || expected_label_int < 0 || expected_label_int > 15) begin
            $display("FAIL: malformed expected prediction value %0d", expected_label_int);
            $finish(1);
        end
        expected_label = expected_label_int[3:0];

        repeat (5) @(posedge clk);
        rst = 1'b0;
        repeat (5) @(posedge clk);

        for (pixel_index = 0; pixel_index < PIXELS; pixel_index = pixel_index + 1) begin
            write32(
                IMAGE_BASE + (pixel_index * 4),
                sample_bytes[pixel_index / 8][pixel_index % 8]
            );
        end

        if (!tpu_frame_ldd) begin
            $display("FAIL: tpu_frame_ldd should be high after image writes");
            $finish(1);
        end

        write32(32'h00000000, 32'h00000001);

        wait_cycles = 0;
        begin : wait_for_done
            while (wait_cycles < MAX_WAIT_CYCLES) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
                if (tpu_done) begin
                    disable wait_for_done;
                end
            end
        end

        if (!tpu_done) begin
            $display(
                "FAIL: timed out waiting for done after %0d cycles (state=%0d layer=%0d chunk=%0d)",
                wait_cycles,
                classifier_inst.state,
                classifier_inst.current_layer,
                classifier_inst.chunk_index
            );
            $finish(1);
        end

        read32(32'h00000004, status_value);
        read32(32'h00000008, result_value);

        if (!(status_value & 32'h2)) begin
            $display("FAIL: done bit not set in status register: %h", status_value);
            $finish(1);
        end
        if (tpu_write_while_busy) begin
            $display("FAIL: tpu_write_while_busy should remain low in nominal flow");
            $finish(1);
        end
        if ((result_value[3:0] != expected_label) || (tpu_prediction != expected_label)) begin
            $display(
                "FAIL: prediction mismatch expected=%0d result_reg=%0d latched=%0d classifier=%0d",
                expected_label,
                result_value[3:0],
                tpu_prediction,
                tpu_prediction
            );
            $finish(1);
        end

        $display(
            "PASS: jtag mmio + classifier predicted expected digit %0d (cycles=%0d),
            total time = %0d",
            expected_label,
            wait_cycles,
            $time/10
        );
        $finish(0);
    end
endmodule
