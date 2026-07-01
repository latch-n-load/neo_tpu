/**********************************************************************//**
 * @file neo_tpu_1img_nopreproc/main.c
 * @author Syed Akif Ali
 * @brief Simple program for inference of single image data file on TinyTPU, without
 * preprocessing.
 **************************************************************************/

#include <neorv32.h>


/**********************************************************************//**
 * @name Configuration
 **************************************************************************/
#define BAUD_RATE 19200

/**********************************************************************//**
 * Main function
 *
 * @note This program requires the CFS and UART0.
 *
 * @return 0 if execution was successful
 **************************************************************************/
int main() {
  // capture all exceptions and give debug info via UART0
  // this is not required, but keeps us safe
  neorv32_rte_setup();

  // setup UART at default baud rate, no interrupts
  neorv32_uart0_setup(BAUD_RATE, 0);  


  // check if CFS is implemented at all
  if (neorv32_cfs_available() == 0) {
    neorv32_uart0_printf("Error! No CFS synthesized!\n");
    return 1;
  }

  neorv32_uart0_printf("\n\n<<< NEORV32 1 Image Data File Inference on TinyTPU Program >>>\n\n");

  neorv32_uart0_printf("NOTE: The singular test image is a availabe as a ___ file,\n"
                       "      ready for inference, no image preprocessing is performed by NEORV32.\n\n");

  neorv32_uart0_printf(" CFS memory-mapped registers for communication with TPU classifer:\n"
                       " * NEORV32_CFS->REG[0] (r/w): Control\n"
                       " * NEORV32_CFS->REG[1] (r/w): Status\n"
                       " * NEORV32_CFS->REG[2] (r): Result\n"
                       " * NEORV32_CFS->REG[3] (r): TPU Version\n");
  
  

  

  neorv32_uart0_printf("\nCFS demo program completed.\n");

  return 0;
}
