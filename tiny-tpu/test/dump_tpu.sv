module dump();
initial begin
  $dumpfile("waveforms/tpu.vcd");
  $dumpvars(0, tpu); 
  // DEBUG: Explicitly dump the unpacked arrays
    for (integer i = 0; i < 2; i++) begin
        $dumpvars(0, tpu.ub_wr_host_data_in[i]);
        $dumpvars(0, tpu.ub_wr_host_valid_in[i]);
    end
end
endmodule