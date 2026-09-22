set fileId [open "/home/a_akif/tesi/tesi_git/pynq-z2/fault_test/target_nets/inst_list_ll_parent.txt" "w"] 
set parent_list [list]
foreach my_instance [get_cells -hierarchical -filter {IS_SEQUENTIAL == 1}] { 
    # Grab the full parent hierarchy string
    set parent_inst [get_property PARENT $my_instance]
    # Split the string by '/' and grab the very last element
    set parent_leaf_name [lindex [split $parent_inst "/"] end]
    # Add just the isolated name to our temporary list
    lappend parent_list $parent_leaf_name
} 
# Filter the list to remove all duplicates and sort it alphabetically
set unique_parents [lsort -unique $parent_list]
# Write the clean, unique parent instance names to the file
foreach p $unique_parents {
    puts $fileId $p
}
close $fileId