/**********************************************************************//**
 * @file neo_tpu/main.c
 * @brief Hashed, DMA-driven multi-image MNIST inference demo for NEORV32.
 * @details Images and labels are read from external memory by DMA, unpacked,
 * thresholded to 1-bit pixels, and passed to the CFS TPU. Low performance due to
 * sequential algorithm - NEORV32 sleeps while DMA and CFS are working.
 **************************************************************************/

#include <neorv32.h>
#include "neorv32_cfs.h"
#include "neorv32_dma.h"

/* -------------------------------------------------------------
 * Logging Configuration
 * ------------------------------------------------------------- */
#define LOG_LEVEL_HASH 0 // Final Test Hash Only
#define LOG_LEVEL_LOW  1 // Prologue, Fatal Errors, and Final Reports 
#define LOG_LEVEL_MID  2 // Initialization steps and periodic progress updates
#define LOG_LEVEL_HIGH 3 // Verbose: DEBUG and detailed info

// Global runtime log level variable
uint8_t log_lvl = LOG_LEVEL_HASH; 

#define LOG_HASH(...)  neorv32_uart0_printf(__VA_ARGS__)

#define LOG_LOW(...)  do { \
    if (log_lvl >= LOG_LEVEL_LOW) neorv32_uart0_printf(__VA_ARGS__); \
} while(0)

#define LOG_MID(...)  do { \
    if (log_lvl >= LOG_LEVEL_MID) neorv32_uart0_printf(__VA_ARGS__); \
} while(0)

#define LOG_HIGH(...) do { \
    if (log_lvl >= LOG_LEVEL_HIGH) neorv32_uart0_printf(__VA_ARGS__); \
} while(0)


/* -------------------------------------------------------------
 * Application Configuration
 * ------------------------------------------------------------- */
#define BAUD_RATE 921600u
#define PIXEL_COUNT 784u
#define PIXEL_ENTRY_COUNT CFS_IMAGE_BYTE_COUNT
#define PIXEL_WORD_COUNT ((PIXEL_COUNT + 3u) / 4u) 
#define IMAGE_COUNT 100u
#define EXT_MEM_BASE 0xC0000000u
#define IMAGE_STRIDE_WORDS PIXEL_WORD_COUNT
#define LABEL_WORD_COUNT ((IMAGE_COUNT + 3u) / 4u)
#define LABEL_BASE_ADDR (EXT_MEM_BASE + (IMAGE_COUNT * IMAGE_STRIDE_WORDS * 4u))
#define PIXEL_THRESHOLD 127u

extern uint64_t neorv32_cfs_get_total_ticks(void);

static uint32_t sys_freq = 0;
volatile uint32_t dma_irq_pending = 0u;
const char* log_lvl_nomi[4] = {"LOG_LEVEL_HASH", "LOG_LEVEL_LOW", "LOG_LEVEL_MID", "LOG_LEVEL_HIGH"};

// DMA Timing Accumulators
volatile uint64_t dma_start_tick = 0;
volatile uint64_t dma_total_label_ticks = 0;
volatile uint64_t dma_total_image_ticks = 0;
volatile uint8_t  dma_is_label = 0;
volatile uint32_t img_idx = 0u;

void dma_firq_handler(void) {
  uint64_t end_tick = neorv32_clint_time_get();
  neorv32_dma_irq_ack();
  dma_irq_pending = 1u;
  
  if (dma_is_label) dma_total_label_ticks += (end_tick - dma_start_tick);
  else {
    dma_total_image_ticks += (end_tick - dma_start_tick);
    LOG_HIGH("[DEBUG] Image %u: DMA IRQ received.\n", img_idx);
  }
}

static void dma_wait_for_done(void) {
  while (dma_irq_pending == 0u) {
    neorv32_cpu_sleep(); // wfi
  }
}

static void dma_start_transfer(uint32_t src_addr, uint32_t *dst_words, uint32_t word_count, uint8_t is_label) {
  uint32_t config = DMA_SRC_INC_WORD | DMA_DST_INC_WORD | word_count;
  dma_irq_pending = 0u;
  dma_is_label = is_label;
  neorv32_dma_program(src_addr, (uint32_t)dst_words, config);
  
  dma_start_tick = neorv32_clint_time_get();
  neorv32_dma_start();
}

static void unpack_pixels_to_bits(const uint32_t *src_words, uint8_t *dst_bits) {
  uint32_t pixel_idx = 0u;
  for (uint32_t i = 0; i < PIXEL_ENTRY_COUNT; ++i) dst_bits[i] = 0u;

  for (uint32_t w = 0; w < PIXEL_WORD_COUNT; ++w) {
    uint32_t word = src_words[w];
    for (uint32_t p = 0; p < 4u && pixel_idx < PIXEL_COUNT; ++p) {
      uint8_t gray = (uint8_t)(word >> (8u * p));
      uint8_t bit = (gray > PIXEL_THRESHOLD) ? 1u : 0u;
      
      uint32_t byte_idx = pixel_idx >> 3;
      uint32_t bit_idx = pixel_idx & 7u;
      dst_bits[byte_idx] |= (uint8_t)(bit << bit_idx); 
      
      pixel_idx++;
    }
  }
}

static void unpack_labels(const uint32_t *src_words, uint8_t *dst_labels) {
  for (uint32_t i = 0; i < IMAGE_COUNT; ++i) {
    uint32_t word_idx = i / 4u;
    uint32_t byte_idx = i % 4u;
    dst_labels[i] = (uint8_t)((src_words[word_idx] >> (8u * byte_idx)) & 0xffu);
    // LOG_HIGH("[DEBUG] true_labels[%u] = %u\n", i, dst_labels[i]);
  }
}

uint32_t upd_hash(uint32_t cur_hash, uint32_t new_data) {
    // Unrolled FNV-1a hash using bitwise shifts
    cur_hash ^= (new_data & 0xFF);
    cur_hash *= 0x01000193;
    cur_hash ^= ((new_data >> 8) & 0xFF);
    cur_hash *= 0x01000193;    
    cur_hash ^= ((new_data >> 16) & 0xFF);
    cur_hash *= 0x01000193;   
    cur_hash ^= ((new_data >> 24) & 0xFF);
    cur_hash *= 0x01000193;
    
    return cur_hash;
}

int main(void) { 
  uint8_t pixel_bits[PIXEL_ENTRY_COUNT] = {0};
  uint32_t pixel_dma_buf_0[PIXEL_WORD_COUNT]; // Int. Memory buffer for current image fetched via DMA
  uint32_t pixel_dma_buf_1[PIXEL_WORD_COUNT]; // Int. Memory buffer for next image fetched via DMA
  uint32_t label_dma_buf[LABEL_WORD_COUNT] = {0}; // Int. Memory buffer for labels fetched via DMA
  uint8_t true_labels[IMAGE_COUNT] = {0};
  uint8_t predictions[IMAGE_COUNT] = {0};
  uint32_t test_hash = 0x811c9dc5; // FNV offset basis
  
  uint32_t version_value = 0u, status_value = 0u, prediction = 0u;
  uint16_t err_cnt = 0u;

  // Pipeline Timing Accumulators
  uint64_t t_pipeline_start = 0;
  uint64_t t_pipeline_end = 0;
  uint64_t t_total_unpack_ticks = 0;

  neorv32_rte_setup();
  neorv32_uart0_setup(BAUD_RATE, 0);
  neorv32_rte_handler_install(DMA_TRAP_CODE, dma_firq_handler);
  neorv32_cfs_irq_enable();
  neorv32_gpio_pin_set(0, 0);
  neorv32_dma_enable();
  neorv32_cpu_csr_set(CSR_MIE, (1u << CFS_FIRQ_ENABLE) | (1u << DMA_FIRQ_ENABLE));
  neorv32_cpu_csr_set(CSR_MSTATUS, (1u << CSR_MSTATUS_MIE));

  sys_freq = NEORV32_SYSINFO->CLK;

  if (neorv32_cfs_available() == 0) {
    LOG_LOW("[ERROR] No CFS synthesized!\n");
    return 1;
  }
  if (!neorv32_clint_available()) {
    LOG_LOW("[ERROR] No CLINT synthesized!\n");
    return 1;
  }

  /* -------------------------------------------------------------
   * Interactive Boot and Log Level Selection
   * ------------------------------------------------------------- */
  // neorv32_uart0_printf("\n===========================================================\n");
  // neorv32_uart0_printf(" NEORV32 TinyTPU Pipelined Multi-Image Classification\n");
  // neorv32_uart0_printf("===========================================================\n\n");
  
  // neorv32_uart0_printf("Select Log Level:\n");
  // neorv32_uart0_printf("0 : LOG_LEVEL_HASH HASH only\n");
  // neorv32_uart0_printf("1 : LOG_LEVEL_LOW Reports \n");
  // neorv32_uart0_printf("2 : LOG_LEVEL_MID Progress & Init Info\n");
  // neorv32_uart0_printf("3 : LOG_LEVEL_HIGH Verbose Debug\n");
  // neorv32_uart0_printf("Enter choice (0-3): ");

  // // Poll UART until valid input is received
  // while (1) {
  //   if (neorv32_uart0_available()) {
  //     char log_in = neorv32_uart0_getc();
  //     if (log_in >= '0' && log_in <= '3') {
  //       log_lvl = (uint8_t)(log_in - '0');
  //       neorv32_uart0_printf("%c\n\n", log_in); // Echo character to terminal
  //       break;
  //     } else {
  //       neorv32_uart0_printf("\nInvalid input. Enter 0, 1, 2, or 3: ");
  //     }
  //   }
  // }

  LOG_LOW("Logging at: %s\n\n", log_lvl_nomi[log_lvl]);

  version_value = neorv32_cfs_read_reg(CFS_REG_VERSION);
  LOG_MID("Version register: 0x%x\n", version_value);
  if (version_value != CFS_VERSION_VALUE) {
    LOG_LOW("[ERROR] Unexpected version value.\n");
    return 2;
  }

  /* -------------------------------------------------------------
   * Prologue: Pre-fetch Labels and Image 0
   * ------------------------------------------------------------- */
  t_pipeline_start = neorv32_clint_time_get();
  
  LOG_MID("Loading labels...\n");
    LOG_HIGH("[DEBUG] LABEL_BASE_ADDR 0x%x, label_dma_buf 0x%x, LABEL_WORD_COUNT %u.\n", 
    LABEL_BASE_ADDR, (uint32_t)&label_dma_buf, LABEL_WORD_COUNT);
  // DMA labels with busy wait, sequential.
  dma_start_transfer(LABEL_BASE_ADDR, label_dma_buf, LABEL_WORD_COUNT, 1);
  dma_wait_for_done();
  // #if LOG_LEVEL == LOG_LEVEL_HIGH
  //   LOG_HIGH("[DEBUG] NEORV32 Internal Memory Buffer for Labels fetched via DMA\n");
  //   for (uint32_t lbl_wrd_i = 0; lbl_wrd_i < LABEL_WORD_COUNT; lbl_wrd_i++) {
  //       LOG_HIGH("[DEBUG] label_dma_buf[%u] 0x%x\n", lbl_wrd_i, label_dma_buf[lbl_wrd_i]);
  //   }
  //   LOG_HIGH("[DEBUG] Staring labels unpacking from src-label_dma_buf 0x%x to dest-true_labels 0x%x\n",
  //     (uint32_t)&label_dma_buf, (uint32_t)&true_labels);
  // #endif

  unpack_labels(label_dma_buf, true_labels);

  LOG_MID("Pre-fetching Image 0...\n");
  LOG_HIGH("[DEBUG] Image %u: Starting DMA Transfer, src_addr 0x%x, dst_words 0x%x, word_count %u, is_label %u\n", 
   img_idx, EXT_MEM_BASE, (uint32_t)&pixel_dma_buf_0, PIXEL_WORD_COUNT, 0);
  dma_start_transfer(EXT_MEM_BASE, pixel_dma_buf_0, PIXEL_WORD_COUNT, 0);
  dma_wait_for_done();

  neorv32_cfs_clear_frame();
  status_value = neorv32_cfs_read_reg(CFS_REG_STATUS);
  LOG_MID("Status after clear: 0x%x\n\n", status_value);

  /* -------------------------------------------------------------
   * The Ping-Pong Pipeline
   * ------------------------------------------------------------- */
  for (img_idx = 0u; img_idx < IMAGE_COUNT; ++img_idx) {
    uint32_t *active_buffer = (img_idx & 1u) ? pixel_dma_buf_1 : pixel_dma_buf_0;
    uint32_t *fetch_buffer  = (img_idx & 1u) ? pixel_dma_buf_0 : pixel_dma_buf_1;
    
    // 1. Kick off DMA fetch for the NEXT image in the background
    if (img_idx + 1 < IMAGE_COUNT) {
        uint32_t next_idx = img_idx + 1;
        uint32_t next_src_addr = EXT_MEM_BASE + (next_idx * IMAGE_STRIDE_WORDS * 4u);
        LOG_HIGH("[DEBUG] Image %u: Starting DMA Transfer, src_addr 0x%x, dst_words 0x%x, word_count %u, is_label %u\n", 
                 next_idx, next_src_addr, (uint32_t)fetch_buffer, PIXEL_WORD_COUNT, 0);
        dma_start_transfer(next_src_addr, fetch_buffer, PIXEL_WORD_COUNT, 0);
    }

    // 2. Unpack CURRENT image using the CPU
    uint64_t unpack_start = neorv32_clint_time_get();
    unpack_pixels_to_bits(active_buffer, pixel_bits);
    LOG_MID("Image %u: unpacked and thresholded.\n", img_idx);
    t_total_unpack_ticks += (neorv32_clint_time_get() - unpack_start);

    // 3. Wait for TPU to finish PREVIOUS image
    if (img_idx > 0) {
        uint32_t prev_idx = img_idx - 1;
        LOG_HIGH("[DEBUG] Waiting for TPU inference of Image %u to complete.\n", prev_idx);
        
        status_value = neorv32_cfs_wait_for_result_irq(&prediction);
        
        LOG_MID("Image %u: Inference Complete.\n", prev_idx); 
        LOG_MID("Image %u: Prediction=%u, Label=%u, Status=0x%x\n",
                prev_idx, prediction, true_labels[prev_idx], status_value);
        
        // Update test hash
        test_hash = upd_hash(test_hash, status_value);
        test_hash = upd_hash(test_hash, prediction);
        LOG_MID("Image %u: Post Infer HASH = %x\n", prev_idx, test_hash);
                
        predictions[prev_idx] = (uint8_t)prediction;
        if (prediction != true_labels[prev_idx]) {
          LOG_MID("[ERROR] Mismatch for image %u.\n\n", prev_idx);
          err_cnt++;
        } else {
          LOG_MID("[SUCCESS] Prediction matches true label.\n\n");
        }
    }

    // 4. Load & Start TPU inference for CURRENT image
    neorv32_cfs_clear_frame();
    neorv32_cfs_load_image(pixel_bits, PIXEL_COUNT);
    LOG_HIGH("[DEBUG] Image %u: loaded to TPU.\n", img_idx);

    neorv32_cfs_start_inference();
    LOG_HIGH("[DEBUG] Image %u: Start sent to TPU.\n", img_idx);

    // 5. Wait for the NEXT image's DMA transfer to finish before looping back to unpack it
    if (img_idx + 1 < IMAGE_COUNT) {
        uint32_t next_idx = img_idx + 1;
        LOG_HIGH("[DEBUG] Image %u: initiating wait for DMA fetch of Image %u.\n", img_idx, next_idx);
        dma_wait_for_done();
        LOG_HIGH("[DEBUG] Image %u: DMA done.\n", next_idx);
    }
  }

  /* -------------------------------------------------------------
   * Epilogue: Collect Final Image
   * ------------------------------------------------------------- */
  status_value = neorv32_cfs_wait_for_result_irq(&prediction);
  predictions[IMAGE_COUNT - 1] = (uint8_t)prediction;
  LOG_MID("Image %u: Inference Complete.\n", IMAGE_COUNT - 1); 
  LOG_MID("Image %u: Prediction=%u, Label=%u, Status=0x%x\n",
          IMAGE_COUNT - 1, prediction, true_labels[IMAGE_COUNT - 1], status_value);
  // Update test hash
  test_hash = upd_hash(test_hash, status_value);
  test_hash = upd_hash(test_hash, prediction);
  LOG_MID("Image %u: Post Infer HASH = %x\n", IMAGE_COUNT - 1, test_hash);
  
  if (prediction != true_labels[IMAGE_COUNT - 1]) {
      LOG_MID("[ERROR] Mismatch for image %u.\n\n", IMAGE_COUNT - 1);
      err_cnt++;
    }
    else LOG_MID("[SUCCESS] Prediction matches true label.\n\n");

  t_pipeline_end = neorv32_clint_time_get();

  /* -------------------------------------------------------------
   * Timing Math (Converting Ticks to Microseconds)
   * ------------------------------------------------------------- */
  uint64_t tpu_total_ticks = neorv32_cfs_get_total_ticks();

  uint32_t time_labels_us   = (uint32_t)((dma_total_label_ticks * 1000000ULL) / sys_freq);
  uint32_t time_images_us   = (uint32_t)((dma_total_image_ticks * 1000000ULL) / sys_freq);
  uint32_t time_unpack_us   = (uint32_t)((t_total_unpack_ticks  * 1000000ULL) / sys_freq);
  uint32_t time_tpu_us      = (uint32_t)((tpu_total_ticks       * 1000000ULL) / sys_freq);
  uint32_t time_pipeline_us = (uint32_t)(((t_pipeline_end - t_pipeline_start) * 1000000ULL) / sys_freq);

  // Note: Since DMA runs in parallel, Total Pipeline Time < (Images + Unpack + TPU)
  
  /* -------------------------------------------------------------
   * Report Generation
   * ------------------------------------------------------------- */
  /* -------------------------------------------------------------
   * Prediction Accuracy Metrics
   * ------------------------------------------------------------- */
  uint32_t crct_cnt = IMAGE_COUNT - err_cnt;
  uint32_t acc_scaled = (crct_cnt * 10000u) / IMAGE_COUNT;
  uint32_t acc_int = acc_scaled / 100u;
  uint32_t acc_frac = acc_scaled % 100u;

  uint32_t err_scaled = (err_cnt * 10000u) / IMAGE_COUNT;
  uint32_t err_int = err_scaled / 100u;
  uint32_t err_frac = err_scaled % 100u;

  LOG_LOW("\n=======================================================\n");
  LOG_LOW("          MNIST NEO_TPU CLASSIFICATION REPORT            \n");
  LOG_LOW("=======================================================\n");
  LOG_LOW("Total Images Evaluated  : %u\n", IMAGE_COUNT);
  LOG_LOW("Correct Classifications : %u\n", crct_cnt);
  LOG_LOW("Misclassifications      : %u\n", err_cnt);
  LOG_LOW("Accuracy                : %u.%u%u%%\n", acc_int, acc_frac / 10u, acc_frac % 10u);
  LOG_LOW("Error Rate (1 - Acc)    : %u.%u%u%%\n", err_int, err_frac / 10u, err_frac % 10u);
  LOG_LOW("-------------------------------------------------------\n");

  if (err_cnt > 0u) {
    LOG_LOW("Misclassified Samples Details:\n");
    LOG_LOW("Image Index | Predicted | True Label\n");
    LOG_LOW("------------+-----------+-----------\n");
    for (uint32_t i = 0u; i < IMAGE_COUNT; ++i) {
      if (predictions[i] != true_labels[i]) {
        LOG_LOW("     %u     |     %u     |     %u\n", i, predictions[i], true_labels[i]);
      }
    }
  } else {
    LOG_LOW("Perfect Classification! 0 errors encountered.\n");
  }
  LOG_LOW("=======================================================\n\n");

  LOG_LOW("\n=======================================================\n");
  LOG_LOW("                   TIMING REPORT                       \n");
  LOG_LOW("=======================================================\n");
  LOG_LOW("Label DMA Time         : %u us (Total for %u labels)\n", time_labels_us, IMAGE_COUNT);
  LOG_LOW("Image DMA Time         : %u us (Total for %u imgs)\n", time_images_us, IMAGE_COUNT);
  LOG_LOW("CPU Unpack Time        : %u us (Total for %u imgs)\n", time_unpack_us, IMAGE_COUNT);
  LOG_LOW("TPU Operation Time     : %u us (Total for %u imgs)\n", time_tpu_us, IMAGE_COUNT);
  LOG_LOW("-------------------------------------------------------\n");
  LOG_LOW("Total Pipeline Latency : %u us\n", time_pipeline_us);
  LOG_LOW("-------------------------------------------------------\n");
  LOG_LOW("Avg TPU Latency        : %u us / inference\n", time_tpu_us / IMAGE_COUNT);
  LOG_LOW("Avg Pipeline Throughput: %u inferences / second\n", (1000000u * IMAGE_COUNT) / time_pipeline_us);
  LOG_LOW("=======================================================\n\n");

  test_hash = upd_hash(test_hash, (uint32_t)time_pipeline_us);
  test_hash = upd_hash(test_hash, err_cnt);

  // Transmit exactly 4 bytes (8 hex characters) to Python
  LOG_HASH("HASH:%x\n", test_hash);

  neorv32_cfs_irq_disable();
  neorv32_gpio_pin_set(0, 1);
  return 0;
}