module blink_led(
    input wire clk,
    output reg led
);
    reg [26:0] counter = 0;
    always @(posedge clk) begin
        counter <= counter + 1;
        if (counter == 125_000_000) begin
            led <= ~led;
            counter <= 0;
        end
    end
endmodule