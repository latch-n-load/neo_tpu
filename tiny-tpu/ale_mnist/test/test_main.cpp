// ==============================================================================
// FILE: test_main.cpp
// Verilator C++ Testbench for MNIST TPU Tiled Classifier (Runtime VPU Check)
// ==============================================================================

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vmnist_tpu_tiled_classifier_tb.h"
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>

// Simulation parameters
static const int CLK_PERIOD_PS = 10000;  
static const int PIXELS = 784;
static const int HIDDEN_NEURONS = 64;
static const int OUTPUT_NEURONS = 10;
static const uint64_t MAX_SIM_TIME = 100000000;

// FSM State names
static const char* state_names[] = {
    "IDLE", "RESET_ASSERT", "RESET_RELEASE", "LOAD_INPUT", "LOAD_WEIGHT", 
    "LOAD_BIAS", "START_WEIGHT", "START_WEIGHT_GAP", "START_INPUT", 
    "SWITCH_WEIGHTS", "START_BIAS", "WAIT_OUTPUT", "NEXT_TILE", "ARGMAX", "DONE"
};

// Globals
static Vmnist_tpu_tiled_classifier_tb* top;
static VerilatedVcdC* tfp;
static uint64_t sim_time = 0;
static uint64_t cycle_count = 0;

static int16_t test_image[PIXELS];

// Expected Math Globals
std::vector<int16_t> w1_mem;
std::vector<int16_t> b1_mem;
std::vector<int16_t> w2_mem;
std::vector<int16_t> b2_mem;
std::vector<int16_t> expected_hidden(HIDDEN_NEURONS, 0);
std::vector<int16_t> expected_logits(OUTPUT_NEURONS, 0);

// ==============================================================================
// Q8.8 Math Utilities
// ==============================================================================
static int16_t sat_add(int16_t a, int16_t b) {
    int32_t sum = (int32_t)a + (int32_t)b;
    if (sum > 32767) return 32767;
    if (sum < -32768) return -32768;
    return (int16_t)sum;
}

static int16_t q8_8_mul(int16_t a, int16_t b) {
    int32_t prod = (int32_t)a * (int32_t)b;
    int32_t shifted = prod >> 8;
    return (int16_t)(shifted & 0xFFFF);
}

// ==============================================================================
// File Loading
// ==============================================================================
static void load_memh(const std::string& filepath, std::vector<int16_t>& mem) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open " << filepath << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '/') continue;
        uint32_t val;
        std::stringstream ss;
        ss << std::hex << line;
        ss >> val;
        mem.push_back((int16_t)val);
    }
    std::cout << "[TB] Loaded " << mem.size() << " elements from " << filepath << std::endl;
}

// ==============================================================================
// Compute Golden Reference
// ==============================================================================
static void compute_expected_outputs() {
    std::cout << "[TB] Computing C++ Expected Outputs..." << std::endl;
    
    // Layer 1
    for (int h = 0; h < HIDDEN_NEURONS; h++) {
        int h_tile = h / 2;
        int h_rem = h % 2;
        int16_t acc = 0;
        for (int p = 0; p < PIXELS; p++) {
            int w_idx = (h_tile * PIXELS * 2) + (p * 2) + h_rem;
            acc = sat_add(acc, q8_8_mul(test_image[p], w1_mem[w_idx]));
            std::cout << "[TB] Current Product: test_image[" << p << "] (" << test_image[p] << ") * w1_mem[" << w_idx << "] = (" << w1_mem[w_idx] << ") = " << q8_8_mul(test_image[p], w1_mem[w_idx]) << std::endl;
            std::cout << "[TB] Accumulator after adding product: " << acc << std::endl;
        }
        acc = sat_add(acc, b1_mem[h]);
        std::cout << "[TB] Accumulator + bias b1_mem[" << h << "] = " << b1_mem[h] << " = " << acc << std::endl;
        if (acc < 0) acc = 0; // ReLU
        expected_hidden[h] = acc;
    }

    // Layer 2
    for (int o = 0; o < OUTPUT_NEURONS; o++) {
        int o_tile = o / 2;
        int o_rem = o % 2;
        int16_t acc = 0;
        for (int h = 0; h < HIDDEN_NEURONS; h++) {
            int w_idx = (o_tile * HIDDEN_NEURONS * 2) + (h * 2) + o_rem;
            acc = sat_add(acc, q8_8_mul(expected_hidden[h], w2_mem[w_idx]));
        }
        acc = sat_add(acc, b2_mem[o]);
        expected_logits[o] = acc;
    }
}

// ==============================================================================
// Generate a simple test image
// ==============================================================================
static void generate_test_image() {
    for (int i = 0; i < PIXELS; i++) {
        test_image[i] = 0;
    }
    for (int row = 4; row < 24; row++) {
        test_image[row * 28 + 13] = 0x8000; 
        test_image[row * 28 + 14] = 0xFF00; 
    }
    std::cout << "[TB] Generated test image (vertical line pattern)" << std::endl;
}

// ==============================================================================
// Clock & Reset
// ==============================================================================
static void toggle_clock() {
    top->clk = !top->clk;
    top->eval();
    if (tfp) tfp->dump(sim_time);
    sim_time += CLK_PERIOD_PS / 2;
    if (top->clk) cycle_count++;
}

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
// Main
// ==============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    top = new Vmnist_tpu_tiled_classifier_tb;
    
    bool trace_enabled = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trace") == 0) trace_enabled = true;
    }
    
    if (trace_enabled) {
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC;
        top->trace(tfp, 99);
        tfp->open("mnist_tpu_classifier.vcd");
        std::cout << "[TB] Waveform tracing enabled -> mnist_tpu_classifier.vcd" << std::endl;
    }
    
    // 1. Generate Image & Load Weights
    generate_test_image();
    load_memh("/home/ale/tesi/tesi_git/tiny-tpu/ale_mnist/model/reference/w1_tiled_q8_8.memh", w1_mem);
    load_memh("/home/ale/tesi/tesi_git/tiny-tpu/ale_mnist/model/reference/b1_q8_8.memh", b1_mem);
    load_memh("/home/ale/tesi/tesi_git/tiny-tpu/ale_mnist/model/reference/w2_tiled_q8_8.memh", w2_mem);
    load_memh("/home/ale/tesi/tesi_git/tiny-tpu/ale_mnist/model/reference/b2_q8_8.memh", b2_mem);
    
    // 2. Compute C++ Golden Reference
    compute_expected_outputs();

    top->clk = 0;
    top->rst = 1;
    top->start = 0;
    top->pixel_data_in = 0;
    top->eval();
    
    apply_reset(10); 
    
    std::cout << "[TB] Asserting start pulse..." << std::endl;
    toggle_clock(); 
    top->start = 1; 
    toggle_clock(); 
    toggle_clock(); 
    top->start = 0; 
    toggle_clock(); 
    
    // Runtime Checking Variables
    int hidden_checked_count = 0;
    int logits_checked_count = 0;
    int hidden_errors = 0;
    int logit_errors = 0;
    bool inference_complete = false;

    while (sim_time < MAX_SIM_TIME) {
        toggle_clock();
        
        if (top->clk) {
            // Feed pixels when DUT is loading inputs for Layer 1
            if (top->debug_state == 3 && top->debug_current_layer == 0) { 
                uint16_t addr = top->pixel_addr_out;
                if (addr < PIXELS) top->pixel_data_in = test_image[addr];
            }

            // ------------------------------------------------------------------
            // 3. RUNTIME VPU STREAM CHECKING (Bypass DONE signal)
            // ------------------------------------------------------------------
            uint8_t current_layer = top->debug_current_layer;

            // Check Channel 1
            if (top->debug_vpu_valid_1) {
                int16_t hw_val = (int16_t)top->debug_vpu_out_1;
                if (current_layer == 0) {
                    if (hidden_checked_count < HIDDEN_NEURONS) {
                        int16_t exp_val = expected_hidden[hidden_checked_count];
                        if (hw_val != exp_val) {
                            if (hidden_errors < 10) std::cout << "[FAIL] L1 Hidden[" << hidden_checked_count << "] HW: " << hw_val << " | EXP: " << exp_val << " @ cycle " << cycle_count << std::endl;
                            hidden_errors++;
                        }
                        hidden_checked_count++;
                    }
                } else {
                    if (logits_checked_count < OUTPUT_NEURONS) {
                        int16_t exp_val = expected_logits[logits_checked_count];
                        if (hw_val != exp_val) {
                            if (logit_errors < 10) std::cout << "[FAIL] L2 Logit[" << logits_checked_count << "] HW: " << hw_val << " | EXP: " << exp_val << " @ cycle " << cycle_count << std::endl;
                            logit_errors++;
                        }
                        logits_checked_count++;
                    }
                }
            }

            // Check Channel 2
            if (top->debug_vpu_valid_2) {
                int16_t hw_val = (int16_t)top->debug_vpu_out_2;
                if (current_layer == 0) {
                    if (hidden_checked_count < HIDDEN_NEURONS) {
                        int16_t exp_val = expected_hidden[hidden_checked_count];
                        if (hw_val != exp_val) {
                            if (hidden_errors < 10) std::cout << "[FAIL] L1 Hidden[" << hidden_checked_count << "] HW: " << hw_val << " | EXP: " << exp_val << " @ cycle " << cycle_count << std::endl;
                            hidden_errors++;
                        }
                        hidden_checked_count++;
                    }
                } else {
                    if (logits_checked_count < OUTPUT_NEURONS) {
                        int16_t exp_val = expected_logits[logits_checked_count];
                        if (hw_val != exp_val) {
                            if (logit_errors < 10) std::cout << "[FAIL] L2 Logit[" << logits_checked_count << "] HW: " << hw_val << " | EXP: " << exp_val << " @ cycle " << cycle_count << std::endl;
                            logit_errors++;
                        }
                        logits_checked_count++;
                    }
                }
            }

            // Check if we have collected all expected data
            if (hidden_checked_count >= HIDDEN_NEURONS && logits_checked_count >= OUTPUT_NEURONS) {
                inference_complete = true;
                break; // Exit the while loop manually!
            }
        }
    }
    
    std::cout << "\n============================================" << std::endl;
    if (inference_complete) {
        std::cout << "[RESULT] All expected VPU outputs harvested successfully!" << std::endl;
    } else {
        std::cout << "[RESULT] TIMEOUT! Failed to harvest all VPU outputs." << std::endl;
        std::cout << "         Harvested " << hidden_checked_count << "/" << HIDDEN_NEURONS << " Hidden Neurons" << std::endl;
        std::cout << "         Harvested " << logits_checked_count << "/" << OUTPUT_NEURONS << " Logits" << std::endl;
    }
    
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Hidden Errors: " << hidden_errors << " / " << hidden_checked_count << std::endl;
    std::cout << "Logit Errors:  " << logit_errors << " / " << logits_checked_count << std::endl;
    
    // Predict manually from our collected logits if we got them all
    if (logits_checked_count == OUTPUT_NEURONS) {
        int best_idx = 0;
        int16_t best_val = expected_logits[0];
        for(int i = 1; i < OUTPUT_NEURONS; i++){
            if(expected_logits[i] > best_val){
                best_val = expected_logits[i];
                best_idx = i;
            }
        }
        std::cout << "Expected Digit (C++ Argmax): " << best_idx << std::endl;
    }
    
    std::cout << "Total Cycles: " << cycle_count << std::endl;
    std::cout << "============================================\n" << std::endl;
    
    // Add extra cycles for observation in the waveform
    for (int i = 0; i < 20; i++) {
        toggle_clock();
        toggle_clock();
    }
    
    if (tfp) { tfp->close(); delete tfp; }
    delete top;
    
    return 0;
}