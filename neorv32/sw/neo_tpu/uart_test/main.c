#include <neorv32.h>
#include <stdint.h>

// A safe magic word address in DMEM (1KB in, away from stack and .data)
#define MAGIC_MEM_ADDR ((volatile uint32_t *) (0x80000000 + 1024))
#define MAGIC_WORD     0xDEADBEEF

// Bulletproof Trap Handler
void __attribute__((__interrupt__, aligned(4))) debug_trap_handler(void) {
  uint32_t mcause = neorv32_cpu_csr_read(CSR_MCAUSE);
  uint32_t mepc   = neorv32_cpu_csr_read(CSR_MEPC);
  uint32_t mtval  = neorv32_cpu_csr_read(CSR_MTVAL);
  
  neorv32_uart0_printf("\n\n*** CPU TRAP/EXCEPTION DETECTED ***\n");
  neorv32_uart0_printf("MCAUSE : 0x%x\n", mcause);
  neorv32_uart0_printf("MEPC   : 0x%x\n", mepc);
  neorv32_uart0_printf("MTVAL  : 0x%x\n", mtval);
  neorv32_uart0_printf("System Halted.\n");
  
  while(1) { neorv32_cpu_sleep(); } 
}

int main(void) {
  // 1. Immediately read MCAUSE. If we arrived here via a crash, the error code is still here!
  uint32_t initial_mcause = neorv32_cpu_csr_read(CSR_MCAUSE);
  
  neorv32_rte_setup();
  neorv32_uart0_setup(19200, 0); 
  
  // 2. Override the default RTE trap handler with our custom diagnostic one
  neorv32_cpu_csr_write(CSR_MTVEC, (uint32_t)(&debug_trap_handler));

  neorv32_uart0_printf("\n\n<<< TIMEOUT DIAGNOSTIC BOOT >>>\n");

  // 3. Reset Cause Analysis
  if (initial_mcause != 0) {
    neorv32_uart0_printf("[RESET CAUSE] Software Trap! The CPU crashed and jumped to 0x0.\n");
    neorv32_uart0_printf("[RESET CAUSE] Previous MCAUSE: 0x%x\n", initial_mcause);
  } else if (*MAGIC_MEM_ADDR == MAGIC_WORD) {
    neorv32_uart0_printf("[RESET CAUSE] Warm Boot. (Hardware Soft-Reset, RAM survived)\n");
  } else {
    neorv32_uart0_printf("[RESET CAUSE] Cold Boot. (Power Cycle or Hard Reset wiping RAM)\n");
    *MAGIC_MEM_ADDR = MAGIC_WORD; // Set it for the next reset
  }

  uint32_t sys_freq = NEORV32_SYSINFO->CLK;
  
  // Verify CLINT is available before using it
  if (!neorv32_clint_available()) {
    neorv32_uart0_printf("[ERROR] CLINT is not available!\n");
    return 1;
  }

  // Verify if WDT is available
  if (!neorv32_wdt_available()) {
    neorv32_uart0_printf("[INFO] WDT is not available!\n");
    return 1;
  }

  while (1) {
    neorv32_uart0_printf("\n--------------------------------\n");
    neorv32_uart0_printf(" 1. Test DMA Transfer\n");
    neorv32_uart0_printf(" 2. Run TPU Inference\n");
    neorv32_uart0_printf(" Waiting for input (10 seconds timeout)");

    // 4. Setup CLINT hardware timeout
    uint64_t start_time = neorv32_clint_time_get();
    uint64_t timeout_time = start_time + (uint64_t)(10ULL * sys_freq);
    char selected_option = '\0';
    uint32_t last_print_time = (uint32_t)start_time;

    // 5. Hardware Polling Loop
    while (1) {
      if (neorv32_uart0_available() && neorv32_uart0_char_received()) {
        selected_option = (char)neorv32_uart0_char_received_get();
        break; 
      }
      
      uint64_t current_time = neorv32_clint_time_get();
      
      if (current_time >= timeout_time) {
        break; 
      }
      
      // Print a dot every 1 second exactly. 
      // If it crashes inside this loop, we will see exactly how many seconds it survived.
      if (((uint32_t)current_time - last_print_time) >= sys_freq) {
          neorv32_uart0_printf(".");
          last_print_time = (uint32_t)current_time;
      }
    }

    // Handle Output
    if (selected_option != '\0') {
      neorv32_uart0_printf("\n-> You selected: %c\n", selected_option);
      uint64_t pause = neorv32_clint_time_get() + (uint64_t)(2ULL * sys_freq);
      while(neorv32_clint_time_get() < pause);
    } else {
      neorv32_uart0_printf("\n-> Timeout reached.\n");
      uint64_t pause = neorv32_clint_time_get() + (uint64_t)(5ULL * sys_freq);
      while(neorv32_clint_time_get() < pause);
    }
  }

  return 0;
}