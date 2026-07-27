module uart_top (
    input wire clk,      // 125 MHz clock on pin H16
    output reg tx_pin    // PMODA Pin 1
);

    // 125 MHz / 115200 baud = ~1085 clock cycles per bit
    parameter CLKS_PER_BIT = 1085;
    
    // Character 'A' (ASCII 0x41 = 8'b01000001)
    // Transmission order: Start bit (0), LSB to MSB (10000010), Stop bit (1)
    wire [9:0] tx_frame = 10'b1_01000001_0;

    integer clk_count = 0;
    integer bit_index = 0;
    reg [2:0] state = 0;

    initial begin
        tx_pin = 1'b1; // Idle HIGH
    end

    always @(posedge clk) begin
        if (clk_count < CLKS_PER_BIT - 1) begin
            clk_count <= clk_count + 1;
        end else begin
            clk_count <= 0;
            
            // Output current bit
            tx_pin <= tx_frame[bit_index];

            if (bit_index < 9) begin
                bit_index <= bit_index + 1;
            end else begin
                bit_index <= 0; // Infinite loop restart
            end
        end
    end
endmodule