#include <neorv32.h>
#include <stdint.h>

// Force 4-byte alignment required by RISC-V for the trap vector
void __attribute__((__interrupt__, aligned(4))) debug_trap_handler(void) {
  uint32_t mcause = neorv32_cpu_csr_read(CSR_MCAUSE); 
  uint32_t mepc   = neorv32_cpu_csr_read(CSR_MEPC);   
  uint32_t mtval  = neorv32_cpu_csr_read(CSR_MTVAL);  
  
  neorv32_uart0_printf("\n\n*** CPU EXCEPTION DETECTED ***\n");
  neorv32_uart0_printf("MCAUSE : 0x%x\n", mcause);
  neorv32_uart0_printf("MEPC   : 0x%x\n", mepc);
  neorv32_uart0_printf("MTVAL  : 0x%x\n", mtval);
  neorv32_uart0_printf("System Halted.\n");
  
  while(1) { neorv32_cpu_sleep(); } 
}

int main(void) {
  // Disable the Watchdog Timer immediately so it doesn't hard-reset the CPU
  if (NEORV32_SYSINFO->SOC & (1 << SYSINFO_SOC_IO_WDT)) {
    neorv32_wdt_disable();
  }

  neorv32_uart0_setup(19200, 0); 

  // Wire the aligned CPU Trap Vector
  neorv32_cpu_csr_write(CSR_MTVEC, (uint32_t)(&debug_trap_handler));

  uint32_t sys_freq = NEORV32_SYSINFO->CLK;

  neorv32_uart0_printf("\n<<< NEORV32 CLINT Diagnostic Interface >>>\n");

  if (neorv32_clint_available() == 0) {
    neorv32_uart0_printf("[ERROR] SYSINFO reports CLINT is NOT synthesized!\n");
    return 1;
  }
  neorv32_uart0_printf("[OK] SYSINFO reports CLINT is present.\n");

  neorv32_uart0_printf("Attempting to read CLINT MTIME... \n");
  uint64_t current_time = neorv32_clint_time_get(); 
  neorv32_uart0_printf("Success! MTIME = %u\n", (uint32_t)current_time);

  while (1) {
    neorv32_uart0_printf("\n--------------------------------\n");
    neorv32_uart0_printf("Please select an option:\n");
    neorv32_uart0_printf(" 1. Test DMA Transfer\n");
    neorv32_uart0_printf(" 2. Run TPU Inference\n");
    neorv32_uart0_printf(" Waiting for input (10 seconds timeout)...\n");

    uint64_t timeout_time = neorv32_clint_time_get() + (uint64_t)(10ULL * sys_freq);
    char selected_option = '\0';

    while (1) {
      if (neorv32_uart0_available() && neorv32_uart0_char_received()) {
        selected_option = (char)neorv32_uart0_char_received_get();
        break; 
      }
      if (neorv32_clint_time_get() >= timeout_time) {
        break; 
      }
    }

    if (selected_option != '\0') {
      neorv32_uart0_printf("\n-> You selected: %c\n", selected_option);
    } else {
      neorv32_uart0_printf("\n-> Timeout. Pausing 5s...\n");
      uint64_t pause_time = neorv32_clint_time_get() + (uint64_t)(5ULL * sys_freq);
      while (neorv32_clint_time_get() < pause_time);
    }
  }

  return 0;
}