set fileId [open "/home/a_akif/tesi/tesi_git/pynq-z2/fault_test/target_nets/inst_list_ll_leaf.txt" "w"] 
# Get hierarchy of sequential instances
foreach my_instance [get_cells -hierarchical -filter {IS_SEQUENTIAL == 1}] { 
    # Split string by '/' get very last element
    set leaf_name [lindex [split $my_instance "/"] end]
    # Add name to list
    lappend inst_list $leaf_name
} 
# Filter the list to remove all duplicates and sort it alphabetically
set unique_insts [lsort -unique $inst_list]
# Write unique instance names to file
foreach p $unique_insts {
    puts $fileId $p
}
close $fileId