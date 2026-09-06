/**
 * @file neorv32_cfs.c
 * @brief Custom Functions Subsystem (CFS) hardware driver source.
 */

#include "neorv32_cfs.h"

volatile uint32_t neorv32_cfs_irq_status = 0u;
volatile uint32_t neorv32_cfs_irq_prediction = 0u;
volatile uint32_t neorv32_cfs_irq_pending = 0u;

// Hardware Active Time Accumulators
volatile uint64_t tpu_start_tick = 0;
volatile uint64_t tpu_total_ticks = 0;

const char* cfs_reg_names[] = {
  "CFS_REG_CONTROL",
  "CFS_REG_STATUS",
  "CFS_REG_RESULT",
  "CFS_REG_VERSION"
};

static inline volatile uint32_t *neorv32_cfs_word_base(void) {
  return (volatile uint32_t *)(uintptr_t)NEORV32_CFS_BASE;
}

int neorv32_cfs_available(void) {
  return (int)(NEORV32_SYSINFO->SOC & (1u << SYSINFO_SOC_IO_CFS));
}

void neorv32_cfs_write_reg(uint32_t reg, uint32_t value) {
  volatile uint32_t *base = neorv32_cfs_word_base();
  base[reg] = value;
}

uint32_t neorv32_cfs_read_reg(uint32_t reg) {
  volatile uint32_t *base = neorv32_cfs_word_base();
  return base[reg];
}

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
}

void neorv32_cfs_irq_handler(void) {
  // Immediately capture the end time
  uint64_t end_tick = neorv32_clint_time_get();
  tpu_total_ticks += (end_tick - tpu_start_tick);

  neorv32_cfs_irq_status = neorv32_cfs_read_reg(CFS_REG_STATUS);
  neorv32_cfs_irq_prediction = neorv32_cfs_read_reg(CFS_REG_RESULT);
  neorv32_cfs_irq_pending = 1u;
  neorv32_cfs_clear_irq();
}

void neorv32_cfs_start_inference(void) {
  neorv32_cfs_irq_pending = 0u;
  // Capture the start time immediately before writing the start bit
  tpu_start_tick = neorv32_clint_time_get();
  neorv32_cfs_write_reg(CFS_REG_CONTROL, CFS_CTRL_START_BIT);
}

void neorv32_cfs_load_image(const uint8_t *pixel_array, uint32_t pixel_cnt) {
  volatile uint32_t *image = neorv32_cfs_word_base() + CFS_IMAGE_WORD_BASE;
  
  uint32_t byte_cnt = (pixel_cnt + 7u) / 8u; 
  uint32_t word_cnt = byte_cnt / 4u;         
  uint32_t leftover_bytes = byte_cnt % 4u;     

  const uint8_t *src = pixel_array; 

  for (uint32_t w = 0; w < word_cnt; ++w) {
    uint32_t packed = ((uint32_t)src[0])        | 
                      ((uint32_t)src[1] << 8)   | 
                      ((uint32_t)src[2] << 16)  | 
                      ((uint32_t)src[3] << 24);   
    image[w] = packed;
    src += 4; 
  }

  if (leftover_bytes > 0) {
    uint32_t packed = 0u;
    for (uint32_t i = 0; i < leftover_bytes; ++i) {
      packed |= ((uint32_t)src[i] << (i * 8));
    }
    image[word_cnt] = packed; 
  }
}

uint32_t neorv32_cfs_busy_wait_result(uint32_t *prediction) {
  while ((neorv32_cfs_read_reg(CFS_REG_STATUS) & CFS_STATUS_DONE_BIT) == 0u) {}
  *prediction = neorv32_cfs_read_reg(CFS_REG_RESULT);
  return neorv32_cfs_read_reg(CFS_REG_STATUS);
}

uint32_t neorv32_cfs_wait_for_result_irq(uint32_t *prediction) {
  // neorv32_cfs_irq_pending = 0u;
  while (neorv32_cfs_irq_pending == 0u) {
    neorv32_cpu_sleep(); // wfi
  }
  *prediction = neorv32_cfs_irq_prediction;
  return neorv32_cfs_irq_status;
}

// Custom Getter for main.c
uint64_t neorv32_cfs_get_total_ticks(void) {
  return tpu_total_ticks;
}