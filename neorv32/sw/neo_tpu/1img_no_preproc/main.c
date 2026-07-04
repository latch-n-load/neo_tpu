/**********************************************************************//**
 * @file neo_tpu_1img_nopreproc/main.c
 * @brief Simple CFS-driven MNIST inference demo for NEORV32.
 **************************************************************************/

#include <neorv32.h>
#include "neorv32_cfs.h"

#define BAUD_RATE 19200u
#define PIXEL_COUNT 784u
#define PIXEL_ENTRY_COUNT 98u

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

  /* Initialize exception handling and UART. */
  neorv32_rte_setup();
  neorv32_uart0_setup(BAUD_RATE, 0);

  /* Verify that the CFS peripheral is present. */
  if (neorv32_cfs_available() == 0) {
    neorv32_uart0_printf("Error! No CFS synthesized!\n");
    return 1;
  }

  neorv32_uart0_printf("\n<<< NEORV32 TinyTPU Singular Image inference without Preprocessing >>>\n");

  /* 1. Read and check the version register. */
  version_value = neorv32_cfs_read_reg(CFS_REG_VERSION);
  neorv32_uart0_printf("Version register: 0x%x\n", version_value);

  if (version_value != CFS_VERSION_VALUE) {
    neorv32_uart0_printf("Error! Unexpected version value.\n");
    return 2;
  }

  /* 2. Clear the frame and verify the status bits. */
  neorv32_uart0_printf("Clearing image frame and flags.\n");
  neorv32_cfs_clear_frame();
  status_value = neorv32_cfs_read_reg(CFS_REG_STATUS);
  neorv32_uart0_printf("Status after clear: 0x%x\n", status_value);

  if ((status_value & CFS_STATUS_DONE_BIT) != 0u) {
    neorv32_uart0_printf("Warning: done flag was already set.\n");
  }

  /* 3. Load the 784 pixels into the CFS image buffer. */
  neorv32_uart0_printf("Loading %u pixels into the image buffer.\n", PIXEL_COUNT);
  neorv32_cfs_load_image(pixel_array, PIXEL_COUNT);

  status_value = neorv32_cfs_read_reg(CFS_REG_STATUS);
  neorv32_uart0_printf("Status after load: 0x%x\n", status_value);

  if ((status_value & CFS_STATUS_FRAME_LOADED_BIT) == 0u) {
    neorv32_uart0_printf("Error! Image was not marked as loaded.\n");
    return 3;
  }

  /* 4. Start inference and wait for completion. */
  neorv32_uart0_printf("Starting inference.\n");
  neorv32_cfs_start_inference();
  status_value = neorv32_cfs_wait_for_result(&prediction);

  neorv32_uart0_printf("Inference complete. Status=0x%x Prediction=%u\n",
                       status_value, prediction);

  return 0;
}

