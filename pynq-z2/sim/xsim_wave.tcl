# Only a slightly modified copy of vsim_wave.tcl
# DOESN'T WORK FOR XSIM ########################


# config wave -signalnamewidth 1

# add wave -divider "NEORV32 CFS"
# # Add the specified signals to the wave window
# add_wave {{/tb_neo_tpu_npynq/uut/neo_tpu_pynq_i/neorv32_vivado_ip_0/U0/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/pixel_data_s}}
# add_wave {{/tb_neo_tpu_npynq/uut/neo_tpu_pynq_i/neorv32_vivado_ip_0/U0/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/pixel_addr_s}}
# add_wave {{/tb_neo_tpu_npynq/uut/neo_tpu_pynq_i/neorv32_vivado_ip_0/U0/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/pixel_addr_int}}
# add_wave {{/tb_neo_tpu_npynq/uut/neo_tpu_pynq_i/neorv32_vivado_ip_0/U0/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/done_reg}}
# add_wave {{/tb_neo_tpu_npynq/uut/neo_tpu_pynq_i/neorv32_vivado_ip_0/U0/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/prediction_reg}}
# add_wave {{/tb_neo_tpu_npynq/uut/neo_tpu_pynq_i/neorv32_vivado_ip_0/U0/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/status_reg}}

# add wave -divider "TPU Classifier"
# add_wave {{/tb_neo_tpu_npynq/uut/neo_tpu_pynq_i/neorv32_vivado_ip_0/U0/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/classifier_inst/ub_wr_host_data_in_0}}
# add_wave {{/tb_neo_tpu_npynq/uut/neo_tpu_pynq_i/neorv32_vivado_ip_0/U0/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/classifier_inst/ub_wr_host_data_in_1}}
# add_wave {{/tb_neo_tpu_npynq/uut/neo_tpu_pynq_i/neorv32_vivado_ip_0/U0/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/classifier_inst/ub_rd_weight_data_out_0}}
# add_wave {{/tb_neo_tpu_npynq/uut/neo_tpu_pynq_i/neorv32_vivado_ip_0/U0/neorv32_top_inst/io_system/neorv32_cfs_enabled/neorv32_cfs_inst/classifier_inst/ub_rd_weight_data_out_1}}

# run all

# wave zoom full