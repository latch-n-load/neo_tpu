/**
 * @file neorv32_cfs.c
 * @brief Custom Functions Subsystem (CFS) hardware driver source.
 */

// Includes the header file that defines hardware addresses and constants
#include "neorv32_cfs.h"

volatile uint32_t neorv32_cfs_irq_status = 0u;
volatile uint32_t neorv32_cfs_irq_prediction = 0u;
volatile uint32_t neorv32_cfs_irq_pending = 0u;

// DEBUG: Lookup for print debug matching the enum order
 const char* cfs_reg_names[] = {
  "CFS_REG_CONTROL",
  "CFS_REG_STATUS",
  "CFS_REG_RESULT",
  "CFS_REG_VERSION"
};

// Ale: Possibly redundant, could use NEORV32_CFS_BASE directly
// "static inline" asks the compiler to embed this function directly where it's called to save time.
// "volatile uint32_t *" means it returns a pointer to an exact 32-bit memory location that the hardware can change.
static inline volatile uint32_t *neorv32_cfs_word_base(void) {
  // Type Casting: Takes the raw number NEORV32_CFS_BASE and forces the compiler to treat it as a hardware pointer.
  return (volatile uint32_t *)(uintptr_t)NEORV32_CFS_BASE;
}

// Ale: Pre-existing NEORV32 function
// Function returning a standard integer to act as a boolean (1 for true, 0 for false)
int neorv32_cfs_available(void) {
  // Bitwise Operations: 
  // 1. "1u << SYSINFO_SOC_IO_CFS" shifts a '1' left to the exact bit representing the CFS.
  // 2. "&" (AND) masks out all other bits in the SOC register, returning non-zero only if the CFS bit is 1.
  return (int)(NEORV32_SYSINFO->SOC & (1u << SYSINFO_SOC_IO_CFS));
}

// Fixed-Width Integers: Takes an exact 32-bit register offset and a 32-bit value to write.
void neorv32_cfs_write_reg(uint32_t reg, uint32_t value) {
  // Pointer Initialization: Calls our inline function to get the base address of the hardware.
  volatile uint32_t *base = neorv32_cfs_word_base();
  // neorv32_uart0_printf("[DEBUG neorv32_cfs.c] CFS write addr: %s, value: %x\n", cfs_reg_names[reg], value);

  // Array Indexing / Pointer Arithmetic: Jumps 'reg' spaces forward from 'base' and writes the data to silicon.
  base[reg] = value;
}

// Fixed-Width Integers: Returns an exact 32-bit number read from the hardware.
uint32_t neorv32_cfs_read_reg(uint32_t reg) {
  // Pointer Initialization: Gets the base address again.
  volatile uint32_t *base = neorv32_cfs_word_base();
  // Array Indexing: Reads the data directly from the hardware memory address.
  // neorv32_uart0_printf("[DEBUG neorv32_cfs.c] CFS read addr: %s, value: %x\n", cfs_reg_names[reg], base[reg]);
  return base[reg];

}

// The following functions are wrapper functions. They use our write function 
// to send specific command bits to the hardware's CONTROL register.
void neorv32_cfs_clear_frame(void) {
  neorv32_cfs_write_reg(CFS_REG_CONTROL, CFS_CTRL_CLEAR_FRAME_BIT);
}

void neorv32_cfs_clear_done(void) {
  neorv32_cfs_write_reg(CFS_REG_CONTROL, CFS_CTRL_CLEAR_DONE_BIT);
}

void neorv32_cfs_clear_error(void) {
  neorv32_cfs_write_reg(CFS_REG_CONTROL, CFS_CTRL_CLEAR_ERR_BIT);
}

void neorv32_cfs_clear_irq(void) {
  neorv32_cfs_write_reg(CFS_REG_CONTROL, CFS_CTRL_CLEAR_IRQ_BIT);
}

void neorv32_cfs_irq_enable(void) {
  neorv32_cfs_irq_status = 0u;
  neorv32_cfs_irq_prediction = 0u;
  neorv32_cfs_irq_pending = 0u;
  neorv32_rte_handler_install(CFS_TRAP_CODE, neorv32_cfs_irq_handler);
  neorv32_cpu_csr_set(CSR_MSTATUS, (1u << CSR_MSTATUS_MIE));
  neorv32_cpu_csr_set(CSR_MIE, (1u << CFS_FIRQ_ENABLE));
}

void neorv32_cfs_irq_disable(void) {
  neorv32_cpu_csr_clr(CSR_MIE, (1u << CFS_FIRQ_ENABLE));
  // neorv32_uart0_printf("[DEBUG neorv32_cfs.c] CFS interrupt disabled.\n");
}

void neorv32_cfs_irq_handler(void) {
  neorv32_cfs_irq_status = neorv32_cfs_read_reg(CFS_REG_STATUS);
  neorv32_cfs_irq_prediction = neorv32_cfs_read_reg(CFS_REG_RESULT);
  neorv32_cfs_irq_pending = 1u;
  neorv32_cfs_clear_irq();
}

void neorv32_cfs_start_inference(void) {
  neorv32_cfs_write_reg(CFS_REG_CONTROL, CFS_CTRL_START_BIT);
  // neorv32_uart0_printf("[DEBUG neorv32_cfs.c] Start sent to TPU.\n");
}

// Pointers: Takes a read-only ("const") pointer to an array of 8-bit bytes (pixels), and the total count.
void neorv32_cfs_load_image(const uint8_t *pixel_array, uint32_t pixel_cnt) {
  volatile uint32_t *image = neorv32_cfs_word_base() + CFS_IMAGE_WORD_BASE;
  
  // Calculate bytes, 32-bit words, and left-over
  uint32_t byte_cnt = (pixel_cnt + 7u) / 8u; // 784 pixels = 98 bytes
  uint32_t word_cnt = byte_cnt / 4u;         // 96 bytes = 24 full 32-bit words
  uint32_t leftover_bytes = byte_cnt % 4u;     // 98 % 4 = 2 leftover bytes (16 pixels)

  const uint8_t *src = pixel_array; // Initialize byte pointer to base addr of pixel_array

  // Process 32-bit words (4 bytes in parallel)
  for (uint32_t w = 0; w < word_cnt; ++w) {
    // Packed = B3 (MSB), B2, B1, B0 (LSB)
    uint32_t packed = ((uint32_t)src[0])        | // Write current byte
                      ((uint32_t)src[1] << 8)   | // Write current+1th byte shifted by 8
                      ((uint32_t)src[2] << 16)  | // Write current+2th byte shifted by 16
                      ((uint32_t)src[3] << 24);   // Write current+3th byte shifted by 24
    
    image[w] = packed;
    src += 4; // Inc pointer by word size
  }

  // For leftover_bytes -- 2 for MNIST
  if (leftover_bytes > 0) {
    uint32_t packed = 0u;
    // Store 1 by 1
    for (uint32_t i = 0; i < leftover_bytes; ++i) {
      // src = 4*(24-1) = 95th byte, get 96th and 97th byte
      packed |= ((uint32_t)src[i] << (i * 8));
    }
    image[word_cnt] = packed; // Write as the last partial word
  }
}

// Pass-by-Reference: Takes a pointer ("*prediction") to a variable created elsewhere so it can write the answer there.
uint32_t neorv32_cfs_busy_wait_result(uint32_t *prediction) {
  // Busy-Waiting & Bitwise ops: 
  // Reads the hardware STATUS register, uses AND ("&") to isolate the DONE_BIT. 
  // It stays trapped in this empty loop as long as the result is 0 (hardware is not done yet).
  while ((neorv32_cfs_read_reg(CFS_REG_STATUS) & CFS_STATUS_DONE_BIT) == 0u) {
    /* busy wait until the classifier completes */
  }
  // neorv32_uart0_printf("[DEBUG neorv32_cfs.c] Busy-wait done. Read @ %s, value: %x\n", cfs_reg_names[CFS_REG_STATUS], neorv32_cfs_read_reg(CFS_REG_STATUS));  

  // Pointer Dereferencing: Follows the pointer to the original variable and overwrites it with the hardware's result.
  *prediction = neorv32_cfs_read_reg(CFS_REG_RESULT);
  // neorv32_uart0_printf("[DEBUG neorv32_cfs.c] CFS prediction read @ %s, value: %x\n", cfs_reg_names[CFS_REG_RESULT], *prediction);

  // Returns the final state of the hardware STATUS register to the main program.
  return neorv32_cfs_read_reg(CFS_REG_STATUS);
}

uint32_t neorv32_cfs_wait_for_result_irq(uint32_t *prediction) {
  neorv32_cfs_irq_pending = 0u;

  while (neorv32_cfs_irq_pending == 0u) {
    neorv32_cpu_sleep();
  }

  *prediction = neorv32_cfs_irq_prediction;
  return neorv32_cfs_irq_status;
}