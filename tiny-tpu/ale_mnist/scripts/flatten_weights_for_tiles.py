def flatten_weights_for_tiles_debug(
    matrix: list[list[str]],
    tile_width: int,
) -> list[str]:
    """
    Rearranges a standard 2D weight matrix into a highly specialized 1D array
    designed specifically for a hardware systolic array with a limited tile size.
    
    Instead of standard row-major (C) or column-major (Fortran) flattening, this 
    groups weights by 'Output Tiles', allowing the FPGA to hold target output neurons
    stationary while streaming all input pixels through them.
    """
    if not matrix:
        return []
    if tile_width <= 0:
        raise ValueError("tile_width must be positive")

    input_size = len(matrix) # Length of outer list (if 784)
    output_size = len(matrix[0]) # Lenght of inner list (then 64)
    
    print(f"\n[*] Initialization: Matrix Dimensions = {input_size}x{output_size}. Tile Width = {tile_width}\n")

    for row in matrix: # Extracts the outer list in matrix [outer_list[inner_list[int]]]
        if len(row) != output_size:
            raise ValueError("matrix rows must have equal length")

    flattened: list[str] = []
    # Step by the hardware's maximum tile width (e.g., 2 for a 2x2 TPU)
    for tile_start in range(0, output_size, tile_width): # range(start, stop, step) -> tile_start = 0, 2, 4, 6 ... output_size-tile_width
        tile_stop = min(tile_start + tile_width, output_size) # tile_stop = (not 0), 2, 4, 6, 8 ... output_size 
        
        print(f"\n=== New Tile Bounds: Matrix Column {tile_start} to {tile_stop - 1} ===\n")
        
        # For this specific tile of outputs, stream through every single input pixel
        for input_index in range(input_size): # input_index = 0, 1, 2 ... input_size-1 (784x64-1 or 64x784-1)
            grabbed_values = []
            
            for output_index in range(tile_start, tile_stop): # output_index = 0, 1 (for tile_start=0), 2, 3 (for tile_start=2) ... up to output_size
                val = matrix[input_index][output_index]
                flattened.append(val) # append in flattened in0out0, in0out1, in1out0, in1out1, in2out0, in2out1 ... inNoutM-1
                grabbed_values.append(val)
                print(f"flattened[{len(flattened) - 1}]\t <= matrix[{input_index}][{output_index}] ({val})")            
    return flattened


# ==========================================
# Test Setup: 98 Inputs x 8 Outputs
# ==========================================

# 1. Generate a dummy matrix where each element knows its original row and column
# We use strings so we can visually trace exactly where each element ends up.
# Example: Row 5, Column 2 will be the string "In05_Out2"
rows = 49
cols = 4
dummy_matrix = [[f"W{row:02d}_W{col}" for col in range(cols)] for row in range(rows)]
# print("=== Input Matrix ({rows}x{cols}) ===")
for row in dummy_matrix:
    print(row)

# 2. Run the debug function with a tile width of 2
result_array = flatten_weights_for_tiles_debug(dummy_matrix, tile_width=2)

# 3. Verify the final array size
print(f"\n[*] Flattening Complete.")
print(f"[*] Original Matrix Elements: {rows} * {cols} = {rows * cols}")
print(f"[*] Final 1D Array Size: {len(result_array)}")