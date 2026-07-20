/**********************************************************************//**
 * @file neo_tpu/main.c
 * @brief Simple CFS-driven MNIST inference demo for NEORV32.
 * @details Contains a ready image as binary array of 784 pixels (1 pixel = 1 bit)
 * stored as constants in IMEM of NEORV32 (No DMA). Pixels are loaded by NEORV32 to
 * TinyTPU, starts inference and awaits result interrupt using WFI.
 **************************************************************************/

#include <neorv32.h>
#include "neorv32_cfs.h"

#define BAUD_RATE 19200u
#define PIXEL_COUNT 784u
#define PIXEL_ENTRY_COUNT 98u

static uint8_t has_zicntr = 0;
static uint32_t sys_freq = 0;

/**
 * @brief Macro to wrap and time a block of code.
 * Bypasses reading CSR_MCYCLE if the hardware counters are missing.
 */
#define GET_TIME_US(step_name_str, code_block) \
  do { \
    uint32_t _start = 0, _end = 0; \
    if (has_zicntr) { _start = neorv32_cpu_csr_read(CSR_MCYCLE); } \
    \
    code_block \
    \
    if (has_zicntr) { \
      _end = neorv32_cpu_csr_read(CSR_MCYCLE); \
      uint32_t _us = (uint32_t)((((uint64_t)(_end - _start)) * 1000000ULL) / sys_freq); \
      neorv32_uart0_printf("%s time: %u us\n", step_name_str, _us); \
    } \
  } while(0)


int main(void) {
  uint8_t pixel_array[PIXEL_ENTRY_COUNT] = {0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	
  0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0xE0,	0x1F,	0x00,	0xE0,	0xFF,	0x01,	0x80,	0xFF,	0x0F,	0x00,
  0xFE,	0x3F,	0x00,	0xF8,	0xF3,	0x00,	0x80,	0x87,	0x03,	0x00,	0x38,	0x38,	0x00,	0x80,	0xCF,	0x01,	0x00,	0xF0,
  0x0F,	0x00,	0x00,	0xFE,	0x00,	0x00,	0xC0,	0x07,	0x00,	0x00,	0xFC,	0x00,	0x00,	0xC0,	0x0F,	0x00,	0x00,	0xDE,	
  0x01,	0x00,	0xF0,	0x1C,	0x00,	0x00,	0xCF,	0x01,	0x00,	0x70,	0x1C,	0x00,	0x00,	0xFF,	0x01,	0x00,	0xF0,	0x0F,	
  0x00,	0x00,	0x7E,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00,	0x00};

  uint32_t version_value;
  uint32_t status_value;
  uint32_t prediction;

  /* Initialize CFS interrupt, exception handling and UART. */
  neorv32_rte_setup();
  neorv32_uart0_setup(BAUD_RATE, 0);
  neorv32_cfs_irq_enable();
  
  sys_freq = NEORV32_SYSINFO->CLK;

  /* Safely check if the Zicntr extension (base counters) is implemented in hardware */
  has_zicntr = (neorv32_cpu_csr_read(CSR_MXISA) & (1 << CSR_MXISA_ZICNTR)) != 0;
    // NEORV32_SYSINFO->CPU_FEATURES & (1 << NEORV32_SYSINFO_CPU_FEATURES_ZICNTR)) != 0;

  /* Verify that the CFS peripheral is present. */
  if (neorv32_cfs_available() == 0) {
    neorv32_uart0_printf("[ERROR] No CFS synthesized!\n");
    return 1;
  }

  neorv32_uart0_printf("\n<<< NEORV32 TinyTPU Singular Image inference without Preprocessing >>>\n");
  
  if (!has_zicntr) {
    neorv32_uart0_printf("[WARNING] Zicntr hardware counters DISABLED. Timing skipped.\n");
  }

  /* 1. Read and check the version register. */
  GET_TIME_US("Version read", {
  version_value = neorv32_cfs_read_reg(CFS_REG_VERSION);
  });
  neorv32_uart0_printf("Version register: 0x%x\n", version_value);
  

  if (version_value != CFS_VERSION_VALUE) {
    neorv32_uart0_printf("[ERROR] Unexpected version value.\n");
    return 2;
  }

  /* 2. Clear the frame and verify the status bits. */
  neorv32_uart0_printf("Clearing image frame and flags.\n");
  neorv32_cfs_clear_frame();
  status_value = neorv32_cfs_read_reg(CFS_REG_STATUS);
  neorv32_uart0_printf("Status after clear: 0x%x\n", status_value);

  if ((status_value & CFS_STATUS_DONE_BIT) != 0u) {
    neorv32_uart0_printf("[WARNING] Done flag was already set.\n");
  }

  /* 3. Load the 784 pixels into the CFS image buffer. */
  neorv32_uart0_printf("Loading %u pixels into the image buffer.\n", PIXEL_COUNT);
  
  MEASURE_TIME_US("Pixel load", {
  neorv32_cfs_load_image(pixel_array, PIXEL_COUNT);
  });

  status_value = neorv32_cfs_read_reg(CFS_REG_STATUS);
  neorv32_uart0_printf("Status after load: 0x%x\n", status_value);

  if ((status_value & CFS_STATUS_FRAME_LOADED_BIT) == 0u) {
    neorv32_uart0_printf("[ERROR] Image was not marked as loaded.\n");
    return 3;
  }

  /* 4. Start inference and wait for completion via interrupt. */
  MEASURE_TIME_US("Inference", {
  neorv32_cfs_start_inference();
  status_value = neorv32_cfs_wait_for_result_irq(&prediction);
  });

  neorv32_uart0_printf("Inference complete. Status=0x%x Prediction=%u Time-taken=%uus\n",
                       status_value, prediction, inf_t_us);

  neorv32_cfs_irq_disable();

  // Raise GPIO 0 for sim_terminate in neorv32_tb.vhd
  neorv32_gpio_pin_set(0, 1);

  return 0;
}