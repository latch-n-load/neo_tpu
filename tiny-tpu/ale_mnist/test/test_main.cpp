// ==============================================================================
// FILE: test_main.cpp
// Verilator C++ Testbench for MNIST TPU Tiled Classifier
// ==============================================================================

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vmnist_tpu_tiled_classifier_tb.h"
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>

// Simulation parameters
static const int CLK_PERIOD_PS = 10000;  // 10ns = 10000ps (100 MHz)
static const int PIXELS = 784;
static const int HIDDEN_NEURONS = 64;
static const int OUTPUT_NEURONS = 10;
static const uint64_t MAX_SIM_TIME = 50000000000; // 50ms max (5M cycles)

// FSM State names
static const char* state_names[] = {
    "IDLE",
    "RESET_ASSERT",
    "RESET_RELEASE",
    "LOAD_INPUT",
    "LOAD_WEIGHT",
    "LOAD_BIAS",
    "START_WEIGHT",
    "START_WEIGHT_GAP",
    "START_INPUT",
    "SWITCH_WEIGHTS",
    "START_BIAS",
    "WAIT_OUTPUT",
    "NEXT_TILE",
    "ARGMAX",
    "DONE"
};

// Globals
static Vmnist_tpu_tiled_classifier_tb* top;
static VerilatedVcdC* tfp;
static uint64_t sim_time = 0;
static uint64_t cycle_count = 0;

// Test image (28x28 vertical line pattern)
static int16_t test_image[PIXELS];

// ==============================================================================
// Generate a simple test image
// ==============================================================================
static void generate_test_image() {
    // Initialize to zero (background)
    for (int i = 0; i < PIXELS; i++) {
        test_image[i] = 0;
    }
    
    // Draw vertical line in columns 13-14 resembling digit "1"
    for (int row = 4; row < 24; row++) {
        test_image[row * 28 + 13] = 0x8000; // Q8.8: 128.0
        test_image[row * 28 + 14] = 0xFF00; // Q8.8: 255.0
    }
    
    std::cout << "[TB] Generated test image (vertical line pattern)" << std::endl;
}

// ==============================================================================
// Clock toggle
// ==============================================================================
static void toggle_clock() {
    top->clk = !top->clk;
    top->eval();
    
    if (tfp) {
        tfp->dump(sim_time);
    }
    
    sim_time += CLK_PERIOD_PS / 2;
    
    if (top->clk) {
        cycle_count++;
    }
}

// ==============================================================================
// Apply reset
// ==============================================================================
static void apply_reset(int cycles) {
    top->rst = 1;
    top->start = 0;
    top->pixel_data_in = 0;
    
    for (int i = 0; i < cycles; i++) {
        toggle_clock();
        toggle_clock();
    }
    
    top->rst = 0;
    std::cout << "[TB] Reset released after " << cycles << " cycles" << std::endl;
}

// ==============================================================================
// Get state name string
// ==============================================================================
static const char* get_state_name(uint8_t state) {
    if (state < 15) {
        return state_names[state];
    }
    return "UNKNOWN";
}

// ==============================================================================
// State transition logger
// ==============================================================================
static uint8_t prev_state = 0;
static uint8_t prev_layer = 0;
static uint8_t prev_hidden_tile = 0;
static uint8_t prev_output_tile = 0;
static bool prev_busy = false;
static bool prev_done = false;

static void log_state_changes() {
    uint8_t state = top->debug_state;
    uint8_t layer = top->debug_current_layer;
    uint8_t hidden_tile = top->debug_hidden_tile;
    uint8_t output_tile = top->debug_output_tile;
    
    // State changes
    if (state != prev_state) {
        std::cout << "[STATE] Cycle " << std::setw(8) << cycle_count 
                  << ": " << std::setw(15) << get_state_name(prev_state)
                  << " -> " << std::setw(15) << get_state_name(state)
                  << " | Layer=" << (layer ? "L2" : "L1")
                  << " | HTile=" << (int)hidden_tile 
                  << " OTile=" << (int)output_tile
                  << std::endl;
        prev_state = state;
    }
    
    // Layer changes
    if (layer != prev_layer) {
        std::cout << "[LAYER] Cycle " << std::setw(8) << cycle_count 
                  << ": Switching to " << (layer ? "Layer 2 (Hidden->Output)" : "Layer 1 (Input->Hidden)")
                  << std::endl;
        prev_layer = layer;
    }
    
    // Tile changes within same layer
    if (layer == 0 && hidden_tile != prev_hidden_tile) {
        std::cout << "[TILE]  Cycle " << std::setw(8) << cycle_count 
                  << ": Layer 1 Tile " << (int)hidden_tile << " / 32"
                  << std::endl;
        prev_hidden_tile = hidden_tile;
    }
    if (layer == 1 && output_tile != prev_output_tile) {
        std::cout << "[TILE]  Cycle " << std::setw(8) << cycle_count 
                  << ": Layer 2 Tile " << (int)output_tile << " / 5"
                  << std::endl;
        prev_output_tile = output_tile;
    }
    
    // Busy/done transitions
    if (top->busy && !prev_busy) {
        std::cout << "[DUT]   Cycle " << std::setw(8) << cycle_count 
                  << ": DUT busy asserted" << std::endl;
    }
    if (top->done && !prev_done) {
        std::cout << "[DUT]   Cycle " << std::setw(8) << cycle_count 
                  << ": DUT done asserted" << std::endl;
    }
    prev_busy = top->busy;
    prev_done = top->done;
}

// ==============================================================================
// VPU output logger
// ==============================================================================
static bool prev_vpu_valid_1 = false;
static bool prev_vpu_valid_2 = false;

static void log_vpu_outputs() {
    if (top->debug_vpu_valid_1 && !prev_vpu_valid_1) {
        std::cout << "[VPU]   Cycle " << std::setw(8) << cycle_count 
                  << ": Output1 = 0x" << std::hex << std::setw(4) << std::setfill('0')
                  << (uint16_t)top->debug_vpu_out_1 << std::dec << std::setfill(' ')
                  << " (" << (int16_t)top->debug_vpu_out_1 << ")"
                  << std::endl;
    }
    if (top->debug_vpu_valid_2 && !prev_vpu_valid_2) {
        std::cout << "[VPU]   Cycle " << std::setw(8) << cycle_count 
                  << ": Output2 = 0x" << std::hex << std::setw(4) << std::setfill('0')
                  << (uint16_t)top->debug_vpu_out_2 << std::dec << std::setfill(' ')
                  << " (" << (int16_t)top->debug_vpu_out_2 << ")"
                  << std::endl;
    }
    prev_vpu_valid_1 = top->debug_vpu_valid_1;
    prev_vpu_valid_2 = top->debug_vpu_valid_2;
}

// ==============================================================================
// Signal edge logger
// ==============================================================================
static bool prev_sys_switch = false;
static bool prev_tpu_rst = false;

static void log_control_signals() {
    if (top->debug_sys_switch && !prev_sys_switch) {
        std::cout << "[CTRL]  Cycle " << std::setw(8) << cycle_count 
                  << ": sys_switch pulsed (weights latched)" << std::endl;
    }
    if (top->debug_tpu_rst && !prev_tpu_rst) {
        std::cout << "[CTRL]  Cycle " << std::setw(8) << cycle_count 
                  << ": TPU reset asserted" << std::endl;
    }
    prev_sys_switch = top->debug_sys_switch;
    prev_tpu_rst = top->debug_tpu_rst;
}

// ==============================================================================
// Main
// ==============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    // Create DUT instance
    top = new Vmnist_tpu_tiled_classifier_tb;
    
    // Enable waveform tracing if --trace argument provided
    bool trace_enabled = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trace") == 0) {
            trace_enabled = true;
        }
    }
    
    if (trace_enabled) {
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC;
        top->trace(tfp, 99);
        tfp->open("mnist_tpu_classifier.vcd");
        std::cout << "[TB] Waveform tracing enabled -> mnist_tpu_classifier.vcd" << std::endl;
    }
    
    // Generate test image
    generate_test_image();
    
    // Initialize inputs
    top->clk = 0;
    top->rst = 1;
    top->start = 0;
    top->pixel_data_in = 0;
    
    // Evaluate initial state
    top->eval();
    
    std::cout << "============================================" << std::endl;
    std::cout << "MNIST TPU Tiled Classifier - Verilator Testbench" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Apply reset
    apply_reset(10);
    
    // Pulse start signal
    std::cout << "[TB] Asserting start pulse..." << std::endl;
    toggle_clock();
    top->start = 1;
    toggle_clock();
    toggle_clock();
    top->start = 0;
    toggle_clock();
    std::cout << "[TB] Start pulse complete" << std::endl;
    
    bool all_pixels_sent = false;
    
    // Main simulation loop
    while (sim_time < MAX_SIM_TIME) {
        toggle_clock();
        
        // Log state and signals on posedge
        if (top->clk) {
            log_state_changes();
            log_vpu_outputs();
            log_control_signals();
            
            // Feed pixels when DUT is loading inputs for Layer 1
            if (top->debug_state == 3 && top->debug_current_layer == 0) { // STATE_LOAD_INPUT
                uint16_t addr = top->pixel_addr_out;
                if (addr < PIXELS) {
                    top->pixel_data_in = test_image[addr];
                    if (addr == 0 || addr < 10 || addr % 100 == 0) {
                        std::cout << "[PIX]   Cycle " << std::setw(8) << cycle_count 
                                  << ": Pixel[" << std::setw(3) << addr 
                                  << "] = 0x" << std::hex << std::setw(4) << std::setfill('0')
                                  << (uint16_t)test_image[addr] << std::dec << std::setfill(' ')
                                  << std::endl;
                    }
                    if (addr == PIXELS - 1) {
                        all_pixels_sent = true;
                    }
                }
            }
        }
        
        // Check for completion
        if (top->done) {
            std::cout << std::endl;
            std::cout << "============================================" << std::endl;
            std::cout << "[RESULT] Inference Complete!" << std::endl;
            std::cout << "[RESULT] Predicted Digit: " << (int)top->prediction_out << std::endl;
            std::cout << "[RESULT] Total Cycles: " << cycle_count << std::endl;
            std::cout << "[RESULT] Simulation Time: " << sim_time / 1000 << " ns" << std::endl;
            std::cout << "============================================" << std::endl;
            break;
        }
        
        // Timeout check
        if (sim_time >= MAX_SIM_TIME) {
            std::cout << std::endl;
            std::cout << "============================================" << std::endl;
            std::cout << "[ERROR] Timeout! Inference took too long." << std::endl;
            std::cout << "[ERROR] Current state: " << get_state_name(top->debug_state) << std::endl;
            std::cout << "[ERROR] Cycles: " << cycle_count << std::endl;
            std::cout << "[ERROR] Layer: " << (top->debug_current_layer ? "L2" : "L1") << std::endl;
            std::cout << "[ERROR] HiddenTile: " << (int)top->debug_hidden_tile << std::endl;
            std::cout << "[ERROR] OutputTile: " << (int)top->debug_output_tile << std::endl;
            std::cout << "============================================" << std::endl;
            break;
        }
    }
    
    // Add extra cycles for observation
    for (int i = 0; i < 10; i++) {
        toggle_clock();
        toggle_clock();
    }
    
    // Cleanup
    if (tfp) {
        tfp->close();
        delete tfp;
    }
    delete top;
    
    std::cout << "[TB] Simulation completed." << std::endl;
    return 0;
}