/**********************************************************************//**
 * @file neo_tpu/main.c
 * @brief DMA-driven multi-image MNIST inference demo for NEORV32.
 * @details Images and labels are read from external memory by DMA, unpacked,
 * thresholded to 1-bit pixels, and passed to the CFS TPU. Low performance due to
 * sequential algorithm - NEORV32 sleeps while DMA and CFS are working.
 **************************************************************************/

#include <neorv32.h>
#include "neorv32_cfs.h"
#include "neorv32_dma.h"

// #define EN_TIME  // Macro for obtaining timing results
#define BAUD_RATE 19200u
#define PIXEL_COUNT 784u
#define PIXEL_ENTRY_COUNT CFS_IMAGE_BYTE_COUNT
#define PIXEL_WORD_COUNT ((PIXEL_COUNT + 3u) / 4u) // Round up to closest word
#define IMAGE_COUNT 100u
#define EXT_MEM_BASE 0xC0000000u
#define IMAGE_STRIDE_WORDS PIXEL_WORD_COUNT
#define LABEL_WORD_COUNT ((IMAGE_COUNT + 3u) / 4u)
#define LABEL_BASE_ADDR (EXT_MEM_BASE + (IMAGE_COUNT * IMAGE_STRIDE_WORDS * 4u))
#define PIXEL_THRESHOLD 127u

// static uint8_t has_zicntr = 0;
static uint32_t sys_freq = 0;
volatile uint32_t dma_irq_pending = 0u;

void dma_firq_handler(void) {
  neorv32_dma_irq_ack();
  dma_irq_pending = 1u;
}

static void dma_wait_for_done(void) {
  dma_irq_pending = 0u;
  while (dma_irq_pending == 0u) {
    /* busy wait */
    // neorv32_cpu_sleep();
  }
}

static void dma_start_transfer(uint32_t src_addr, uint32_t *dst_words, uint32_t word_count) {
  uint32_t config = DMA_SRC_INC_WORD | DMA_DST_INC_WORD | word_count;
  dma_irq_pending = 0u;
  // neorv32_uart0_printf("[DEBUG] Starting DMA Transfer with DMA_SRC_INC_WORD and DMA_DST_INC_WORD\n");
  neorv32_dma_program(src_addr, (uint32_t)dst_words, config);
  neorv32_dma_start();
}

static void unpack_pixels_to_bits(const uint32_t *src_words, uint8_t *dst_bits) {
  uint32_t pixel_idx = 0u;

  for (uint32_t i = 0; i < PIXEL_ENTRY_COUNT; ++i) {
    dst_bits[i] = 0u;
  }

  for (uint32_t w = 0; w < PIXEL_WORD_COUNT; ++w) {
    uint32_t word = src_words[w];
    for (uint32_t p = 0; p < 4u && pixel_idx < PIXEL_COUNT; ++p) {
      uint8_t gray = (uint8_t)(word >> (8u * p));
      uint8_t bit = (gray > PIXEL_THRESHOLD) ? 1u : 0u;
      uint32_t byte_idx = pixel_idx / 8u;
      uint32_t bit_idx = pixel_idx & 7u;

      if (bit) {
        dst_bits[byte_idx] |= (uint8_t)(1u << bit_idx);
      }
      pixel_idx++;
    }
  }
}

static void unpack_labels(const uint32_t *src_words, uint8_t *dst_labels) {
  for (uint32_t i = 0; i < IMAGE_COUNT; ++i) {
    uint32_t word_idx = i / 4u;
    uint32_t byte_idx = i % 4u;
    dst_labels[i] = (uint8_t)((src_words[word_idx] >> (8u * byte_idx)) & 0xffu);
    // neorv32_uart0_printf("[DEBUG] true_labels[%u] = %u\n", i, dst_labels[i]);
  }
}

#ifdef EN_TIME
#define GET_TIME_US(step_name_str, code_block) \
  do { \
    uint32_t _start = 0, _end = 0; \
    if (has_zicntr) { _start = neorv32_cpu_csr_read(CSR_MCYCLE); } \
    code_block \
    if (has_zicntr) { \
      _end = neorv32_cpu_csr_read(CSR_MCYCLE); \
      uint32_t _cnt = _end - _start; \
      uint32_t _ns = (uint32_t)((((uint64_t)(_end - _start)) * 1000000000ULL) / sys_freq); \
      neorv32_uart0_printf("[TIMER] %s | MCYCLE_cnt: %u, time: %u ns\n", step_name_str, _cnt, _ns); \
    } \
  } while(0)
#else
#define GET_TIME_US(step_name_str, code_block) do { code_block } while(0)
#endif

int main(void) {
  uint8_t pixel_bits[PIXEL_ENTRY_COUNT] = {0};
  uint32_t pixel_dma_buf_0[PIXEL_WORD_COUNT];
  uint32_t pixel_dma_buf_1[PIXEL_WORD_COUNT];
  uint32_t label_dma_buf[LABEL_WORD_COUNT] = {0};
  uint8_t true_labels[IMAGE_COUNT] = {0};
  uint8_t predictions[IMAGE_COUNT] = {0};
  uint32_t version_value = 0u;
  uint32_t status_value = 0u;
  uint32_t prediction = 0u;
  uint16_t err_cnt = 0u;

  // Timing Accumulators (Using 64-bit to prevent overflow)
  uint64_t t_tick_start = 0;
  uint64_t t_total_labels = 0;
  uint64_t t_total_images = 0;
  uint64_t t_total_tpu = 0;
  uint64_t t_pipeline_start = 0;
  uint64_t t_pipeline_end = 0;

  neorv32_rte_setup();
  neorv32_uart0_setup(BAUD_RATE, 0);
  neorv32_rte_handler_install(DMA_TRAP_CODE, dma_firq_handler);
  neorv32_cfs_irq_enable();
  neorv32_gpio_pin_set(0, 0);
  neorv32_dma_enable();
  neorv32_cpu_csr_set(CSR_MIE, (1u << CFS_FIRQ_ENABLE) | (1u << DMA_FIRQ_ENABLE));
  neorv32_cpu_csr_set(CSR_MSTATUS, (1u << CSR_MSTATUS_MIE));

  sys_freq = NEORV32_SYSINFO->CLK;
  // has_zicntr = (neorv32_cpu_csr_read(CSR_MXISA) & (1u << CSR_MXISA_ZICNTR)) != 0u;

  if (neorv32_cfs_available() == 0) {
    neorv32_uart0_printf("[ERROR] No CFS synthesized!\n");
    return 1;
  }
  if (!neorv32_clint_available()) {
    neorv32_uart0_printf("[ERROR] No CLINT synthesized!\n");
    return 1;
  }

  neorv32_uart0_printf("\n<<< NEORV32 TinyTPU DMA-driven multi-image inference >>>\n\n");
  // if (!has_zicntr) {
  //   neorv32_uart0_printf("[WARNING] Zicntr hardware counters DISABLED. Timing skipped.\n");
  // }

  // GET_TIME_US("Version read", {
    version_value = neorv32_cfs_read_reg(CFS_REG_VERSION);
  // });
  // neorv32_uart0_printf("Version register: 0x%x\n", version_value);

  if (version_value != CFS_VERSION_VALUE) {
    neorv32_uart0_printf("[ERROR] Unexpected version value.\n");
    return 2;
  }

  // PIPELINE START
  // =========================================================================
  t_pipeline_start = neorv32_clint_time_get();
  neorv32_uart0_printf("Loading labels from external memory.\n");
  // neorv32_uart0_printf("[DEBUG] LABEL_BASE_ADDR 0x%x, label_dma_buf 0x%x, LABEL_WORD_COUNT %u.\n", 
    // LABEL_BASE_ADDR, (uint32_t)&label_dma_buf, LABEL_WORD_COUNT);
  t_tick_start = neorv32_clint_time_get();
  dma_start_transfer(LABEL_BASE_ADDR, label_dma_buf, LABEL_WORD_COUNT);
  dma_wait_for_done();
  t_total_labels += (neorv32_clint_time_get() - t_tick_start);


  // neorv32_uart0_printf("[DEBUG] LABELS:\n label_dma_buf[0] 0x%x, \n label_dma_buf[1] 0x%x\n",
  //   label_dma_buf[0], label_dma_buf[1] );
  // neorv32_uart0_printf("[DEBUG] Unpack LABELS: label_dma_buf 0x%x, true_labels 0x%x\n",
  //   (uint32_t)&label_dma_buf, (uint32_t)&true_labels);
  unpack_labels(label_dma_buf, true_labels);

  neorv32_uart0_printf("Clearing frame and flags before inference.\n");
  neorv32_cfs_clear_frame();
  status_value = neorv32_cfs_read_reg(CFS_REG_STATUS);
  neorv32_uart0_printf("Status after clear: 0x%x\n", status_value);

  for (uint32_t img_idx = 0u; img_idx < IMAGE_COUNT; ++img_idx) {
    uint32_t *active_buffer = (img_idx & 1u) ? pixel_dma_buf_1 : pixel_dma_buf_0;
    uint32_t image_src_addr = EXT_MEM_BASE + (img_idx * IMAGE_STRIDE_WORDS * 4u);

    neorv32_uart0_printf("Image %u: Loading %u Pixels via DMA from EXT_MEM[0x%x].\n", img_idx, PIXEL_COUNT, image_src_addr);
    // neorv32_uart0_printf("[DEBUG] Image %u: DMA from MEM[0x%x] %u, PIXEL_WORD_COUNT %u\n", img_idx, image_src_addr, image_src_addr, PIXEL_WORD_COUNT);
    t_tick_start = neorv32_clint_time_get();
    dma_start_transfer(image_src_addr, active_buffer, PIXEL_WORD_COUNT);
    dma_wait_for_done();
    t_total_images += (neorv32_clint_time_get() - t_tick_start);


    unpack_pixels_to_bits(active_buffer, pixel_bits);
    neorv32_uart0_printf("Image %u: unpacked and thresholded.\n", img_idx);

    // if ((img_idx + 1u) < IMAGE_COUNT) {
    //   next_src_addr = EXT_MEM_BASE + ((img_idx + 1u) * IMAGE_STRIDE_WORDS * 4u);
    //   // next_src_addr = EXT_MEM_BASE + ((img_idx + 1u) * IMAGE_STRIDE_WORDS);
    //   next_buffer = preload_buffer;
  
    //   neorv32_uart0_printf("Image %u: preloading next image into ping-pong buffer.\n", img_idx + 1u);
    //   neorv32_uart0_printf("DMA from [0x%x] %u, PIXEL_WORD_COUNT %u\n", next_src_addr, next_src_addr, PIXEL_WORD_COUNT);
    //   dma_start_transfer(next_src_addr, next_buffer, PIXEL_WORD_COUNT);
    // }

    neorv32_uart0_printf("Image %u: starting inference.\n", img_idx);
    neorv32_cfs_clear_frame();
    neorv32_cfs_load_image(pixel_bits, PIXEL_COUNT);
    t_tick_start = neorv32_clint_time_get(); // obtain inference start time

    neorv32_cfs_start_inference();

    // GET_TIME_US("Inference", {
      status_value = neorv32_cfs_wait_for_result_irq(&prediction);
      t_total_tpu += (neorv32_clint_time_get() - t_tick_start);

    // });

    predictions[img_idx] = (uint8_t)prediction;


    // if ((img_idx + 1u) < IMAGE_COUNT) {
    //   dma_wait_for_done();
    // }

    // neorv32_uart0_printf("[DEBUG] Comparing Prediction with true_lables @ [0x%x]\n", (uint32_t)&true_labels[img_idx]);
    neorv32_uart0_printf("Image %u: Inference Complete.\n", img_idx); 
    neorv32_uart0_printf("Image %u: Prediction=%u, Label=%u, Status=0x%x\n",
                         img_idx, prediction, true_labels[img_idx], status_value);
    if (prediction != true_labels[img_idx]) {
      neorv32_uart0_printf("[ERROR] Mismatch for image %u.\n\n", img_idx);
      err_cnt++;
    }
    else neorv32_uart0_printf("[SUCCESS] Prediction matches true label.\n\n");
  }

  // =========================================================================
  // PIPELINE END
  // =========================================================================
  t_pipeline_end = neorv32_clint_time_get();

  /* -------------------------------------------------------------
   * Timing Calculations (Converted from CLINT Ticks to Microseconds)
   * ------------------------------------------------------------- */
  uint32_t time_labels_us   = (uint32_t)((t_total_labels * 1000000ULL) / sys_freq);
  uint32_t time_images_us   = (uint32_t)((t_total_images * 1000000ULL) / sys_freq);
  uint32_t time_tpu_us      = (uint32_t)((t_total_tpu * 1000000ULL) / sys_freq);
  uint32_t time_pipeline_us = (uint32_t)(((t_pipeline_end - t_pipeline_start) * 1000000ULL) / sys_freq);

  // Calculate averages per image
  uint32_t avg_img_dma_us = time_images_us / IMAGE_COUNT;
  uint32_t avg_tpu_us     = time_tpu_us / IMAGE_COUNT;

  /* -------------------------------------------------------------
   * Report 1: Evaluation Metrics
   * ------------------------------------------------------------- */
  uint32_t crct_cnt = IMAGE_COUNT - err_cnt;

  // Fixed-point calculation (scaling by 10000 gives 2 decimal places: 9540 = 95.40%)
  uint32_t acc_scaled = (crct_cnt * 10000u) / IMAGE_COUNT;
  uint32_t acc_int = acc_scaled / 100u;
  uint32_t acc_frac = acc_scaled % 100u;

  uint32_t err_scaled = (err_cnt * 10000u) / IMAGE_COUNT;
  uint32_t err_int = err_scaled / 100u;
  uint32_t err_frac = err_scaled % 100u;

  neorv32_uart0_printf("\n=======================================================\n");
  neorv32_uart0_printf("             MNIST TPU EVALUATION REPORT               \n");
  neorv32_uart0_printf("=======================================================\n");
  neorv32_uart0_printf("Total Images Evaluated  : %u\n", IMAGE_COUNT);
  neorv32_uart0_printf("Correct Classifications : %u\n", crct_cnt);
  neorv32_uart0_printf("Misclassifications      : %u\n", err_cnt);
  neorv32_uart0_printf("Top-1 Accuracy          : %u.%u%u%%\n", acc_int, acc_frac / 10u, acc_frac % 10u);
  neorv32_uart0_printf("Error Rate (1 - Acc)    : %u.%u%u%%\n", err_int, err_frac / 10u, err_frac % 10u);
  neorv32_uart0_printf("-------------------------------------------------------\n");

  if (err_cnt > 0u) {
    neorv32_uart0_printf("Misclassified Samples Details:\n");
    neorv32_uart0_printf("Image Index | Predicted | True Label\n");
    neorv32_uart0_printf("------------+-----------+-----------\n");
    for (uint32_t i = 0u; i < IMAGE_COUNT; ++i) {
      if (predictions[i] != true_labels[i]) {
        neorv32_uart0_printf("     %u     |     %u     |     %u\n", i, predictions[i], true_labels[i]);
      }
    }
  } else {
    neorv32_uart0_printf("Perfect Classification! 0 errors encountered.\n");
  }
  neorv32_uart0_printf("=======================================================\n\n");

  neorv32_uart0_printf("\n=======================================================\n");
  neorv32_uart0_printf("                     TIMING REPORT                       \n");
  neorv32_uart0_printf("=========================================================\n");
  neorv32_uart0_printf("System Clock Frequency  : %u KHz\n", sys_freq / 1000u);
  neorv32_uart0_printf("-------------------------------------------------------\n");
  neorv32_uart0_printf("Label Loading (DMA)     : %u us (Total for %u labels)\n", time_labels_us, IMAGE_COUNT);
  neorv32_uart0_printf("Image Loading (DMA)     : %u us (Total for %u images)\n", time_images_us, IMAGE_COUNT);
  neorv32_uart0_printf("TPU Operation Time      : %u us (Total for %u images)\n", time_tpu_us, IMAGE_COUNT);
  neorv32_uart0_printf("Total Pipeline Latency  : %u us\n", time_pipeline_us);
  neorv32_uart0_printf("-------------------------------------------------------\n");
  neorv32_uart0_printf("Avg Image Fetch Latency : %u us / image\n", avg_img_dma_us);
  neorv32_uart0_printf("Avg TPU Inference Time  : %u us / image\n", avg_tpu_us);
  neorv32_uart0_printf("=======================================================\n\n");

  neorv32_cfs_irq_disable();
  neorv32_gpio_pin_set(0, 1);
  return 0;
}