-- ================================================================================ --
-- NEORV32 SoC - Custom Functions Subsystem (CFS)                                   --
-- -------------------------------------------------------------------------------- --

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library neorv32;
use neorv32.neorv32_package.all;

entity neorv32_cfs is
  generic ( 
    PIXELS : integer := 784;
    PIXEL_ADDR_WIDTH : integer := index_size_f(784);
    PIXEL_BASE_ADDR_REG : integer := 16#100#; -- Pixel base register at 0x100
    W1_INIT_FILE : string := "/home/ale/tesi/tesi_git/tiny-tpu/mnist_demo/data/model/reference/w1_tiled_q8_8.memh";
    B1_INIT_FILE : string := "/home/ale/tesi/tesi_git/tiny-tpu/mnist_demo/data/model/reference/b1_q8_8.memh";
    W2_INIT_FILE : string := "/home/ale/tesi/tesi_git/tiny-tpu/mnist_demo/data/model/reference/w2_tiled_q8_8.memh";
    B2_INIT_FILE : string := "/home/ale/tesi/tesi_git/tiny-tpu/mnist_demo/data/model/reference/b2_q8_8.memh"
  );
  port (
    -- global control --
    clk_i     : in  std_ulogic; -- global clock line
    rstn_i    : in  std_ulogic; -- global reset line, low-active, async
    -- CPU access --
    bus_req_i : in  bus_req_t; -- bus request
    bus_rsp_o : out bus_rsp_t; -- bus response
    -- CPU interrupt --
    irq_o     : out std_ulogic; -- interrupt request
    -- external IO --
    cfs_in_i  : in  std_ulogic_vector(255 downto 0); -- custom inputs conduit
    cfs_out_o : out std_ulogic_vector(255 downto 0) -- custom outputs conduit
  );
end neorv32_cfs;

architecture neorv32_cfs_rtl of neorv32_cfs is

  -- CFS register map -------------------------------------------------------------
  -- The NEORV32 CFS example uses a simple 32-bit word-addressed register space.
  -- We keep the first four words for control/status/result/version and use the
  -- image base address to store the packed 784-bit MNIST frame.
  constant ctrl_word_addr_c     : std_logic_vector(13 downto 0) := "00000000000000";
  constant status_word_addr_c   : std_logic_vector(13 downto 0) := "00000000000001";
  constant result_word_addr_c   : std_logic_vector(13 downto 0) := "00000000000010";
  constant version_word_addr_c  : std_logic_vector(13 downto 0) := "00000000000011";
  constant image_base_word_addr_c : natural := PIXEL_BASE_ADDR_REG / 4;
  constant image_word_count_c     : natural := (PIXELS + 31) / 32; -- Since 784 % 32 != 0, ceil to nearest word integer
  constant image_last_word_addr_c : natural := image_base_word_addr_c + image_word_count_c - 1;
  constant version_value_c        : std_logic_vector(31 downto 0) := x"4D4E4953";
  constant q8_8_one_c             : std_logic_vector(15 downto 0) := x"0100";
  constant q8_8_zero_c            : std_logic_vector(15 downto 0) := x"0000";

  -- Generate 4 (read) and 4 (write) 32-bit registers
  type cfs_regs_t is array (0 to 3) of std_logic_vector(31 downto 0);
  signal cfs_reg_wr : cfs_regs_t;
  signal cfs_reg_rd : cfs_regs_t;

  -- Image storage and loading state --------------------------------------------
  signal frame_bits      : std_logic_vector(PIXELS-1 downto 0);
  signal frame_valid     : std_logic_vector(PIXELS-1 downto 0); -- Indicates which pixels in frame_bits are valid (like a cache)
  signal pixel_load_count : natural range 0 to PIXELS;
  signal frame_loaded_reg : std_logic;
  signal write_while_busy_reg : std_logic;

  -- Classifier interface -------------------------------------------------------
  signal classifier_start_s : std_logic;
  signal classifier_busy_s  : std_logic;
  signal classifier_done_s  : std_logic;
  signal classifier_prediction_s : std_logic_vector(3 downto 0);
  signal pixel_data_s       : std_logic_vector(15 downto 0);
  signal pixel_addr_s       : std_logic_vector(PIXEL_ADDR_WIDTH-1 downto 0) := (others => '0');
  signal pixel_addr_int     : integer range 0 to PIXELS;
  signal done_reg           : std_logic;
  signal prediction_reg     : std_logic_vector(3 downto 0);
  signal status_reg         : std_logic_vector(31 downto 0);
  signal irq_pending_reg    : std_logic;

  -- mnist_classifier_core component declaration --------------------------------
  component mnist_classifier_core is
    generic (
      PIXELS               : integer;
      PIXEL_ADDR_WIDTH     : integer;
      HIDDEN_NEURONS       : integer;
      HIDDEN_ADDR_WIDTH    : integer;
      OUTPUT_NEURONS       : integer;
      OUTPUT_ADDR_WIDTH    : integer;
      TILE_WIDTH           : integer;
      UNIFIED_BUFFER_WIDTH : integer;
      PRELOAD_MODEL        : integer;
      W1_INIT_FILE         : string;
      B1_INIT_FILE         : string;
      W2_INIT_FILE         : string;
      B2_INIT_FILE         : string      
    );
    port (
      clk            : in  std_logic;
      rst            : in  std_logic;
      start          : in  std_logic;
      pixel_data_in  : in  std_logic_vector(15 downto 0);
      pixel_addr_out : out std_logic_vector(PIXEL_ADDR_WIDTH-1 downto 0);
      busy           : out std_logic;
      done           : out std_logic;
      prediction_out : out std_logic_vector(3 downto 0)
    );
  end component;

begin

  -- CFS IOs 
  cfs_out_o <= (others => '0');
  irq_o <= irq_pending_reg;

  -- Status register bit assignments
  status_reg(0) <= classifier_busy_s;
  status_reg(1) <= done_reg;
  status_reg(2) <= frame_loaded_reg;
  status_reg(3) <= write_while_busy_reg;
  status_reg(4) <= irq_pending_reg;
  status_reg(31 downto 5) <= (others => '0');

  -- CFS READONLY registers: 0: Control (Write Only so Reads 0), 1: Status, 2: Result, 3: Version
  cfs_reg_rd(0) <= (others => '0');
  cfs_reg_rd(1) <= status_reg;
  cfs_reg_rd(2) <= std_logic_vector(resize(unsigned(prediction_reg), 32));
  cfs_reg_rd(3) <= version_value_c;

  -- Bus interface --------------------------------------------------------------
  bus_access: process(rstn_i, clk_i)
    variable addr_word_v : natural;
    variable pixel_base_v : natural;
    variable pixels_this_word_v : natural;
    variable bit_idx_v : integer;
  begin
    if (rstn_i = '0') then
      cfs_reg_wr <= (others => (others => '0'));
      bus_rsp_o  <= rsp_terminate_c;
      frame_bits <= (others => '0');
      frame_valid <= (others => '0'); -- Intialize pixel frame as invalid (like cache)
      pixel_load_count <= 0;
      frame_loaded_reg <= '0';
      write_while_busy_reg <= '0';
      done_reg <= '0';
      prediction_reg <= (others => '0');
      irq_pending_reg <= '0';
      classifier_start_s <= '0';
    elsif rising_edge(clk_i) then
      -- Default values for one-cycle responses.
      bus_rsp_o.ack <= '0';
      bus_rsp_o.err <= '0';
      bus_rsp_o.data <= (others => '0');
      classifier_start_s <= '0';

      -- Keep the sticky done flag until explicitly cleared by software.
      if (classifier_done_s = '1') then
        done_reg <= '1';
        prediction_reg <= classifier_prediction_s;
        report "DEBUG HW: @" & to_string(now) & " | done_reg:" & std_logic'image(classifier_done_s) 
                & " | prediction_reg:" & integer'image(to_integer(unsigned(std_logic_vector(classifier_prediction_s))));
        irq_pending_reg <= '1';
      end if;

      -- Acknowledge bus strobe
      if (bus_req_i.stb = '1') then
        bus_rsp_o.ack <= '1';

        -- Read access ----------------------------------------------------------
        if (bus_req_i.rw = '0') then -- rw = 0 for read_en
          -- Check bus_req_i.addr lower half (excluding 2lsbs) for addressed register
          -- report "Time: " & to_string(now) & " | bus_req.addr: 0x" & to_hstring(bus_req_i.addr) 
          -- & " | Data: 0x" & to_hstring(bus_req_i.data) & " | RW: " & to_string(bus_req_i.rw) 
          -- & " | BEN: " & to_string(bus_req_i.ben) & " | STB: " & to_string(bus_req_i.stb);
          case std_logic_vector(bus_req_i.addr(15 downto 2)) is 
            when ctrl_word_addr_c    => bus_rsp_o.data <= cfs_reg_rd(0);
            when status_word_addr_c  => bus_rsp_o.data <= cfs_reg_rd(1);
            when result_word_addr_c  => 
              bus_rsp_o.data <= cfs_reg_rd(2);
              
              -- report "DEBUG HW: @" & to_string(now) & " Received Addr: " & 
              -- integer'image (to_integer(unsigned(bus_req_i.addr(15 downto 2)))) & " | Data: "
              -- & integer'image(to_integer(unsigned(cfs_reg_rd(2))));
              report "DEBUG HW: @" & to_string(now) & " Received Addr: " & 
              to_hstring(bus_req_i.addr(15 downto 2)) & " | Data: " & to_hstring(cfs_reg_rd(2));

            when version_word_addr_c => bus_rsp_o.data <= cfs_reg_rd(3);
            when others              => bus_rsp_o.data <= (others => '0');
          end case;

        -- Write access ---------------------------------------------------------
        else -- rw = 1 for write_en
          -- If control_reg is accessed
          if (std_logic_vector(bus_req_i.addr(15 downto 2)) = ctrl_word_addr_c) then
            cfs_reg_wr(0) <= bus_req_i.data;

            -- Control bits:
            -- bit0 : start inference (only once image is fully loaded and core is idle)
            -- bit1 : clear image frame and flags
            -- bit2 : clear done flag
            -- bit3 : clear write-while-busy flag
            if (bus_req_i.data(0) = '1' and frame_loaded_reg = '1' and classifier_busy_s = '0') then
              classifier_start_s <= '1';
              report "DEBUG HW: @" & to_string(now) & " Sending start = " & std_logic'image(bus_req_i.data(0));
            end if;

            if (bus_req_i.data(1) = '1') then
              frame_bits <= (others => '0');
              frame_valid <= (others => '0');
              pixel_load_count <= 0;
              frame_loaded_reg <= '0';
              write_while_busy_reg <= '0';
              done_reg <= '0'; -- Clear done, clear irq_o
              prediction_reg <= (others => '0');
            end if;

            if (bus_req_i.data(2) = '1') then
              done_reg <= '0';
            end if;

            if (bus_req_i.data(3) = '1') then
              write_while_busy_reg <= '0';
            end if;

            if (bus_req_i.data(4) = '1') then
              irq_pending_reg <= '0';
            end if;

          -- Test: Allows writing to status, result and version
          -- elsif (std_logic_vector(bus_req_i.addr(15 downto 2)) = status_word_addr_c) then
          --   cfs_reg_wr(1) <= bus_req_i.data;
          -- elsif (std_logic_vector(bus_req_i.addr(15 downto 2)) = result_word_addr_c) then
          --   cfs_reg_wr(2) <= bus_req_i.data;
          -- elsif (std_logic_vector(bus_req_i.addr(15 downto 2)) = version_word_addr_c) then
          --   cfs_reg_wr(3) <= bus_req_i.data;

          -- If bus_req_i.addr is NOT ctrl, status, result or version -> Write to image frame
          else
            -- Get word address and check if it is within the image frame range
            addr_word_v := to_integer(unsigned(std_logic_vector(bus_req_i.addr(15 downto 2))));
            if (addr_word_v >= image_base_word_addr_c and addr_word_v <= image_last_word_addr_c) then
              -- Prevent writing to image frame when classifier busy
              if (classifier_busy_s = '1') then
                write_while_busy_reg <= '1';
                irq_pending_reg <= '1';
              else
                -- If classifier idle, get pixel id (not absolute address) being written
                pixel_base_v := (addr_word_v - image_base_word_addr_c) * 32; -- Each successive access jumps by 32 because
                pixels_this_word_v := 32; -- Each word has 32 1b pixels
                -- When writing last pixels, check if written word exceeds total PIXELS
                if (pixel_base_v + pixels_this_word_v > PIXELS) then
                  pixels_this_word_v := PIXELS - pixel_base_v; -- So pixels_this_word_v < 32
                end if;

                -- Loop to extract 1b pixel from 32b pixels_this_word
                report "DEBUG HW: @" & to_string(now) & " | Received Addr:" & integer'image(addr_word_v) 
                & " | Data:" & to_hstring(bus_req_i.data) & " | Pixel Base:" & to_string(pixel_base_v);
                for bit_idx_v in 0 to 31 loop
                  if (bit_idx_v < pixels_this_word_v) then -- Check if pixel_this_word < 32
                    if (frame_valid(pixel_base_v + bit_idx_v) = '0') then -- Check if current pixel is invalid
                      frame_bits(pixel_base_v + bit_idx_v) <= bus_req_i.data(bit_idx_v); -- Write pixel bit from bus to frame

                      -- report "DEBUG HW: @" & to_string(now) & " | frame_bits (" & to_string(pixel_base_v + bit_idx_v) 
                      -- & ") = " & to_string(bus_req_i.data(bit_idx_v));
                      
                      frame_valid(pixel_base_v + bit_idx_v) <= '1'; -- Set valid bit 
                    end if;
                  end if;
                end loop;
                
                -- Check if all pixels loaded
                if (pixel_load_count + pixels_this_word_v >= PIXELS) then
                  pixel_load_count <= PIXELS;
                  frame_loaded_reg <= '1';
                else
                  pixel_load_count <= pixel_load_count + pixels_this_word_v;
                end if;
                
                -- Set frame_loaded_reg 
                if (pixel_load_count + pixels_this_word_v >= PIXELS) then
                  frame_loaded_reg <= '1';
                end if;
              end if;
            end if;
          end if;
        end if;
      end if;
    end if;
  end process bus_access;

  -- Pixel data formatting ------------------------------------------------------
  -- The classifier expects Q8.8 values. A binary '1' pixel is mapped to 0x0100,
  -- while '0' is mapped to 0x0000
  pixel_addr_int <= to_integer(unsigned(pixel_addr_s)); --Convert to int for lookup
  pixel_data_s <= q8_8_one_c when (pixel_addr_int < PIXELS and frame_bits(pixel_addr_int) = '1') else q8_8_zero_c;

  -- Instantiate MNIST classifier core ------------------------------------------
  classifier_inst : mnist_classifier_core
    generic map (
      PIXELS               => PIXELS,
      PIXEL_ADDR_WIDTH     => PIXEL_ADDR_WIDTH,
      HIDDEN_NEURONS       => 64,
      HIDDEN_ADDR_WIDTH    => 6,
      OUTPUT_NEURONS       => 10,
      OUTPUT_ADDR_WIDTH    => 4,
      TILE_WIDTH           => 2,
      UNIFIED_BUFFER_WIDTH => 128,
      PRELOAD_MODEL        => 1,
      W1_INIT_FILE         => W1_INIT_FILE,
      B1_INIT_FILE         => B1_INIT_FILE,
      W2_INIT_FILE         => W2_INIT_FILE,
      B2_INIT_FILE         => B2_INIT_FILE
    )
    port map (
      clk            => clk_i,
      rst            => not rstn_i,
      start          => classifier_start_s,
      pixel_data_in  => pixel_data_s,
      pixel_addr_out => pixel_addr_s,
      busy           => classifier_busy_s,
      done           => classifier_done_s,
      prediction_out => classifier_prediction_s
    );

end neorv32_cfs_rtl;
