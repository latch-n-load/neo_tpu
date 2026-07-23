// ================================================================================ //
// The NEORV32 RISC-V Processor - https://github.com/stnolting/neorv32              //
// Copyright (c) NEORV32 contributors.                                              //
// Copyright (c) 2020 - 2026 Stephan Nolting. All rights reserved.                  //
// Licensed under the BSD-3-Clause license, see LICENSE for details.                //
// SPDX-License-Identifier: BSD-3-Clause                                            //
// ================================================================================ //

/**
 * @file neorv32_cfs.h
 * @brief Custom Functions Subsystem (CFS) HW driver header file.
 */

#ifndef NEORV32_CFS_H
#define NEORV32_CFS_H

#include <neorv32.h>
#include <stdint.h>

/**********************************************************************//**
 * @name IO Device: Custom Functions Subsystem (CFS)
 **************************************************************************/
/**@{*/
/** CFS module prototype */
typedef volatile struct __attribute__((packed, aligned(4))) {
  uint32_t CONTROL;        /**< Offset 0x00: Control register */
  uint32_t STATUS;         /**< Offset 0x04: Status register */
  uint32_t RESULT;         /**< Offset 0x08: Result register */
  uint32_t VERSION_VALUE;  /**< Offset 0x0C: Version register */
  uint32_t RESERVED[((64u * 1024u) / 4u) - 4u];
} neorv32_cfs_t;

/** CFS module hardware handle */
#define NEORV32_CFS ((neorv32_cfs_t *)(uintptr_t)NEORV32_CFS_BASE)

/* Register indices (word based, matching the HDL register map) */
enum {
  CFS_REG_CONTROL = 0u,
  CFS_REG_STATUS = 1u,
  CFS_REG_RESULT = 2u,
  CFS_REG_VERSION = 3u
};

/* Control register bits */
#define CFS_CTRL_START_BIT         (1u << 0)
#define CFS_CTRL_CLEAR_FRAME_BIT   (1u << 1)
#define CFS_CTRL_CLEAR_DONE_BIT    (1u << 2)
#define CFS_CTRL_CLEAR_ERR_BIT     (1u << 3)
#define CFS_CTRL_CLEAR_IRQ_BIT     (1u << 4)

/* Image packing constants */
#define CFS_IMAGE_BYTE_COUNT       (98u)
#define CFS_IMAGE_BIT_THRESHOLD    (127u)

/* Status register bits */
#define CFS_STATUS_BUSY_BIT        (1u << 0)
#define CFS_STATUS_DONE_BIT        (1u << 1)
#define CFS_STATUS_FRAME_LOADED_BIT (1u << 2)
#define CFS_STATUS_WRITE_BUSY_BIT  (1u << 3)
#define CFS_STATUS_IRQ_PENDING_BIT (1u << 4)

#define CFS_VERSION_VALUE          0x4D4E4953u
#define CFS_IMAGE_WORD_BASE        0x40u
#define CFS_IMAGE_WORD_COUNT       ((784u + 31u) / 32u)
/**@}*/

/**********************************************************************//**
 * @name Prototypes
 **************************************************************************/
/**@{*/
int neorv32_cfs_available(void);
void neorv32_cfs_write_reg(uint32_t reg, uint32_t value);
uint32_t neorv32_cfs_read_reg(uint32_t reg);
void neorv32_cfs_clear_frame(void);
void neorv32_cfs_clear_done(void);
void neorv32_cfs_clear_error(void);
void neorv32_cfs_clear_irq(void);
void neorv32_cfs_irq_enable(void);
void neorv32_cfs_irq_disable(void);
void neorv32_cfs_irq_handler(void);
void neorv32_cfs_start_inference(void);
void neorv32_cfs_load_image(const uint8_t *pixel_array, uint32_t pixel_count);
uint32_t neorv32_cfs_busy_wait_result(uint32_t *prediction);
uint32_t neorv32_cfs_wait_for_result_irq(uint32_t *prediction);
/**@}*/

#endif
