import re
from collections import defaultdict

# --- Configuration ---
LL_FILE = "/home/a_akif/tesi/neo_tpu_pynq2/neo_tpu_pynq2.runs/impl_1/neo_tpu_pynq_wrapper.ll"
EBD_FILE = "/home/a_akif/tesi/neo_tpu_pynq2/neo_tpu_pynq2.runs/impl_1/neo_tpu_pynq_wrapper.ebd" 
OUTPUT_FILE = "ll_ebd_overlap_summary.txt"

def generate_overlap_summary():
    print("[*] Extracting sets...")
    
    # 1. Build the .ebd set
    ebd_bits = set()
    try:
        with open(EBD_FILE, 'r') as f:
            ebd_data = ""
            for line in f:
                clean = line.strip()
                if clean.startswith('0') or clean.startswith('1'):
                    ebd_data += clean
        for idx, bit_char in enumerate(ebd_data):
            if bit_char == '1':
                ebd_bits.add(idx)
    except FileNotFoundError:
        print(f"[!] ERROR: {EBD_FILE} not found.")
        return

    # 2. Build the .ll set
    ll_bits = set()
    ll_regex = re.compile(r"^Bit\s+(\d+)\s+")
    try:
        with open(LL_FILE, 'r') as f:
            for line in f:
                match = ll_regex.match(line.strip())
                if match:
                    ll_bits.add(int(match.group(1)))
    except FileNotFoundError:
        print(f"[!] ERROR: {LL_FILE} not found.")
        return

    # 3. Find the overlapping bits (In .ll AND in .ebd)
    overlap_bits = ll_bits.intersection(ebd_bits)
    print(f"[*] Found {len(overlap_bits)} overlapping bits between .ll and .ebd")
    print("[*] Aggregating summary...")

    # Regex patterns for extracting specific key-value pairs from the info string
    block_re = re.compile(r"Block=([^ \n]+)")
    latch_re = re.compile(r"Latch=([^ \n]+)")
    ram_re = re.compile(r"Ram=([^ \n]+)")
    net_re = re.compile(r"Net=([^\n]+)")

    summary_counts = defaultdict(int)

    # 4. Parse the .ll file again to extract details for just the overlapping bits
    with open(LL_FILE, "r") as f:
        for line in f:
            match = ll_regex.match(line)
            if not match:
                continue
                
            abs_offset = int(match.group(1))
            
            # Only process if this bit is in the intersection
            if abs_offset in overlap_bits:
                info_str = line[match.end():] # Everything after the Frame Offset
                
                block_match = block_re.search(info_str)
                latch_match = latch_re.search(info_str)
                ram_match = ram_re.search(info_str)
                net_match = net_re.search(info_str)
                
                block_val = block_match.group(1) if block_match else "UNKNOWN"
                
                # Construct the grouping key based on available data
                if net_match:
                    net_full = net_match.group(1)
                    # Chop off the leaf node to group by parent
                    # if '/' in net_full:
                    #     net_parent = net_full.rsplit('/', 1)[0] + '/'
                    # else:
                    net_parent = net_full
                        
                    if latch_match:
                        key = f"Block={block_val} Latch={latch_match.group(1)} Net={net_parent}"
                    elif ram_match:
                        ram_base = ram_match.group(1).split(':')[0] 
                        key = f"Block={block_val} Ram={ram_base} Net={net_parent}"
                    else:
                        key = f"Block={block_val} Net={net_parent}"
                else:
                    key = f"Block={block_val}"
                    
                summary_counts[key] += 1

    # 5. Write the sorted summary report
    print(f"[*] Writing report to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as out_file:
        out_file.write(f"SUMMARY OF OVERLAPPING BITS (.LL INTERSECT .EBD)\n")
        out_file.write(f"Total Overlapping Bits: {len(overlap_bits)}\n")
        out_file.write("=" * 80 + "\n\n")
        
        # Sort alphabetically by block name for easy reading
        for key, count in sorted(summary_counts.items()):
            out_file.write(f"Bits: {count:<8} {key}\n")

    print("[+] Done!")

if __name__ == "__main__":
    generate_overlap_summary()