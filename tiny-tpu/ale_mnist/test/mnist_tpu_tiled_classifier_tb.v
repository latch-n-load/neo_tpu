// ==============================================================================
// FILE: mnist_tpu_tiled_classifier_tb.v
// ==============================================================================

`timescale 1ns / 1ps
`default_nettype none

module mnist_tpu_tiled_classifier_tb #(
    parameter integer PIXELS = 784,
    parameter integer PIXEL_ADDR_WIDTH = 10,
    parameter integer HIDDEN_NEURONS = 64,
    parameter integer HIDDEN_ADDR_WIDTH = 6,
    parameter integer OUTPUT_NEURONS = 10,
    parameter integer OUTPUT_ADDR_WIDTH = 4,
    parameter integer TILE_WIDTH = 2,
    parameter integer UNIFIED_BUFFER_WIDTH = 4096
) (
    input wire clk,
    input wire rst,
    input wire start,
    input wire [15:0] pixel_data_in,
    output wire [PIXEL_ADDR_WIDTH-1:0] pixel_addr_out,
    output wire busy,
    output wire done,
    output wire [3:0] prediction_out,
    // Debug outputs
    output wire [4:0] debug_state,
    output wire debug_current_layer,
    output wire [HIDDEN_ADDR_WIDTH-1:0] debug_hidden_tile,
    output wire [OUTPUT_ADDR_WIDTH-1:0] debug_output_tile,
    output wire [15:0] debug_vpu_out_1,
    output wire [15:0] debug_vpu_out_2,
    output wire debug_vpu_valid_1,
    output wire debug_vpu_valid_2,
    output wire debug_sys_switch,
    output wire debug_tpu_rst
);

  mnist_tpu_tiled_classifier #(
      .PIXELS(PIXELS),
      .PIXEL_ADDR_WIDTH(PIXEL_ADDR_WIDTH),
      .HIDDEN_NEURONS(HIDDEN_NEURONS),
      .HIDDEN_ADDR_WIDTH(HIDDEN_ADDR_WIDTH),
      .OUTPUT_NEURONS(OUTPUT_NEURONS),
      .OUTPUT_ADDR_WIDTH(OUTPUT_ADDR_WIDTH),
      .TILE_WIDTH(TILE_WIDTH),
      .UNIFIED_BUFFER_WIDTH(UNIFIED_BUFFER_WIDTH)
      //.PRELOAD_MODEL(0)
  ) dut (
      .clk(clk),
      .rst(rst),
      .start(start),
      .pixel_data_in(pixel_data_in),
      .pixel_addr_out(pixel_addr_out),
      .busy(busy),
      .done(done),
      .prediction_out(prediction_out)
  );

  // Debug signal assignments
  assign debug_state = dut.state;
  assign debug_current_layer = dut.current_layer;
  assign debug_hidden_tile = dut.hidden_tile_index;
  assign debug_output_tile = dut.output_tile_index;
  assign debug_vpu_out_1 = dut.vpu_data_out_1;
  assign debug_vpu_out_2 = dut.vpu_data_out_2;
  assign debug_vpu_valid_1 = dut.vpu_valid_out_1;
  assign debug_vpu_valid_2 = dut.vpu_valid_out_2;
  assign debug_sys_switch = dut.sys_switch_in;
  assign debug_tpu_rst = dut.tpu_rst;

endmodule

`default_nettype wire
