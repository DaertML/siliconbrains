# tpu_core.py (Most Optimized for Fixed-Point and Multi-Cycle Operations towards Synthesizability)
from myhdl import block, always_seq, Signal, intbv, instances, enum, concat, always_comb, ResetSignal

from fixed_point_utils import fixed_mul, fixed_add, fixed_relu, \
                             FIXED_POINT_F_BITS, FIXED_POINT_TOTAL_BITS, \
                             ACCUMULATOR_TOTAL_BITS, PRODUCT_ACC_BITS # NEW IMPORTS

# --- State Machine Enum (MOVED TO GLOBAL SCOPE) ---
TPUState = enum('FETCH', 'DECODE_EXECUTE',
                'MATMUL_PREP', 'MATMUL_LOOP_ROWS', 'MATMUL_LOOP_COLS', 'MATMUL_MAC_INNER_LOOP',
                'ADD_ELEM_PREP', 'ADD_ELEM_LOOP',
                'RELU_PREP', 'RELU_LOOP',
                'HALT', 'ERROR')
DATA_WIDTH = 32

@block
def tpu_core(
    clk,
    reset,
    program_rom, # List of instruction tuples
    # NEW: Accept memory signals directly as parameters
    wmem,        # Weight Memory
    imem,        # Input Memory
    smem,        # Scratchpad Memory
    omem,        # Output Memory
    halted_out,     # Output signal indicating if TPU is halted
    pc_out,         # Output for Program Counter
    state_out,      # Output for FSM State
    # Parameters (made explicit for MyHDL conversion)
    MEM_ADDR_BITS=12,    # Address width (e.g., 2^12 = 4096 addresses)
    MEM_BANK_ENUM_BITS=2, # Bits for memory bank ID (WMEM, IMEM, OMEM, SMEM)
    SHAPE_DIM_BITS=8,    # Bits for each dimension value (e.g., 2^8 = 256)
    NUM_TENSOR_REGS=16,  # Number of tensor pointer registers
    NUM_SCALAR_REGS=8,   # Number of scalar value registers
    WMEM_SIZE=4096,      # Size of Weight Memory
    IMEM_SIZE=4096,      # Size of Input Memory
    OMEM_SIZE=4096,      # Size of Output Memory
    SMEM_SIZE=8192       # Size of Scratchpad Memory
):
    """
    An optimized MyHDL model of a TPU core with fixed-point arithmetic,
    explicit memory addressing, multi-cycle operations for all relevant ops,
    and synthesizable memory interfaces.
    """

    # --- Internal Signals ---
    pc = Signal(intbv(0)[MEM_ADDR_BITS:]) # Program Counter
    ir = Signal(tuple()) # Current instruction
    src_val_sig = Signal(intbv(0)[DATA_WIDTH:]) # Use a distinct name to avoid confusion with local python var
    result_sig = Signal(intbv(0)[DATA_WIDTH:]) 
    # State Machine
    state = Signal(TPUState.FETCH) # Initialize with an enum member

    # --- Register Files (remain internal to the block) ---
    regs_tensor_ptr = Signal(tuple(concat(intbv(0)[MEM_BANK_ENUM_BITS:], intbv(0)[MEM_ADDR_BITS:]) for _ in range(NUM_TENSOR_REGS)))
    regs_scalar = Signal(tuple(intbv(0)[FIXED_POINT_TOTAL_BITS:] for _ in range(NUM_SCALAR_REGS)))

    # --- Multi-cycle Operation Registers (e.g., for MATMUL, ADD_ELEM, RELU) ---
    matmul_dest_ptr = Signal(concat(intbv(0)[MEM_BANK_ENUM_BITS:], intbv(0)[MEM_ADDR_BITS:]))
    matmul_A_ptr = Signal(concat(intbv(0)[MEM_BANK_ENUM_BITS:], intbv(0)[MEM_ADDR_BITS:]))
    matmul_B_ptr = Signal(concat(intbv(0)[MEM_BANK_ENUM_BITS:], intbv(0)[MEM_ADDR_BITS:]))
    matmul_A_rows = Signal(intbv(0)[SHAPE_DIM_BITS:])
    matmul_A_cols = Signal(intbv(0)[SHAPE_DIM_BITS:]) # Also B_rows
    matmul_B_cols = Signal(intbv(0)[SHAPE_DIM_BITS:])
    matmul_row_idx = Signal(intbv(0)[SHAPE_DIM_BITS:])
    matmul_col_idx = Signal(intbv(0)[SHAPE_DIM_BITS:])
    matmul_k_idx = Signal(intbv(0)[SHAPE_DIM_BITS:])
    matmul_mac_acc = Signal(intbv(0)[ACCUMULATOR_TOTAL_BITS:])


    # Add_Elem specific registers
    addelem_dest_ptr = Signal(concat(intbv(0)[MEM_BANK_ENUM_BITS:], intbv(0)[MEM_ADDR_BITS:]))
    addelem_A_ptr = Signal(concat(intbv(0)[MEM_BANK_ENUM_BITS:], intbv(0)[MEM_ADDR_BITS:]))
    addelem_B_ptr = Signal(concat(intbv(0)[MEM_BANK_ENUM_BITS:], intbv(0)[MEM_ADDR_BITS:]))
    addelem_A_rows = Signal(intbv(0)[SHAPE_DIM_BITS:])
    addelem_A_cols = Signal(intbv(0)[SHAPE_DIM_BITS:])
    addelem_B_shape_len = Signal(intbv(0)[2:]) # 1 or 2
    addelem_B_dim1 = Signal(intbv(0)[SHAPE_DIM_BITS:]) # For bias broadcasting
    addelem_idx = Signal(intbv(0)[MEM_ADDR_BITS:]) # Linear index for element processing
    addelem_total_elements = Signal(intbv(0)[MEM_ADDR_BITS+1:]) # Total elements in tensor

    # ReLU specific registers
    relu_dest_ptr = Signal(concat(intbv(0)[MEM_BANK_ENUM_BITS:], intbv(0)[MEM_ADDR_BITS:]))
    relu_src_ptr = Signal(concat(intbv(0)[MEM_BANK_ENUM_BITS:], intbv(0)[MEM_ADDR_BITS:]))
    relu_idx = Signal(intbv(0)[MEM_ADDR_BITS:])
    relu_total_elements = Signal(intbv(0)[MEM_ADDR_BITS+1:])

    # --- Initial Memory Loading (Synthesizable at Reset) ---
    # Memory initial values are now loaded in the testbench.
    # This block only handles internal register resets.
    @always_seq(clk.posedge, reset=reset)
    def init_reset_logic():
        if reset:
            # ONLY RESET INTERNAL REGISTERS AND STATE, MEMORIES ARE PRE-LOADED BY TESTBENCH
            regs_tensor_ptr.next = tuple(concat(intbv(0)[MEM_BANK_ENUM_BITS:], intbv(0)[MEM_ADDR_BITS:]) for _ in range(NUM_TENSOR_REGS))
            regs_scalar.next = tuple(intbv(0)[FIXED_POINT_TOTAL_BITS:] for _ in range(NUM_SCALAR_REGS))
            pc.next = 0
            state.next = TPUState.FETCH
            halted_out.next = False
            # Reset all multi-cycle op registers
            matmul_row_idx.next = 0
            matmul_col_idx.next = 0
            matmul_k_idx.next = 0
            matmul_mac_acc.next = 0
            addelem_idx.next = 0
            relu_idx.next = 0
        # Else (not reset), do nothing here. Main FSM takes over.

    # --- Output Assignment ---
    @always_comb
    def assign_outputs():
        pc_out.next = pc
        state_out.next = state

    # --- Main FSM for Instruction Execution ---
    @always_seq(clk.posedge, reset=reset)
    def tpu_fsm():
        if reset:
            pass # Reset handled by init_reset_logic
        else:
            # Default assignments for signals to avoid multiple drivers or latches
            # If a signal is not assigned in a state, it implicitly holds its value (sequential logic)
            # However, if it's assigned in *some* states, it needs to be explicitly assigned
            # or given a default in *all* relevant states, or driven by a dedicated logic block.

            # We need to explicitly manage the next values of the loop counters
            # matmul_k_idx_next, matmul_row_idx_next, matmul_col_idx_next, matmul_mac_acc_next
            # these will be the *proposed* next values based on the current state.
            # Then, at the end of the FSM, we assign them to the actual signals.

            # Default to current value if not explicitly updated in this cycle
            _next_pc = pc.next
            _next_state = state.next
            _next_halted_out = halted_out.next
            _next_matmul_row_idx = matmul_row_idx.next
            _next_matmul_col_idx = matmul_col_idx.next
            _next_matmul_k_idx = matmul_k_idx.next
            _next_matmul_mac_acc = matmul_mac_acc.next
            _next_addelem_idx = addelem_idx.next
            _next_relu_idx = relu_idx.next
        
            # --- State Machine Logic ---
            if state == TPUState.FETCH:
                if pc < len(program_rom):
                    ir.next = program_rom[pc]
                    _next_state = TPUState.DECODE_EXECUTE
                    # _next_pc remains pc for FETCH
                else:
                    _next_state = TPUState.HALT
                    _next_halted_out = True

            elif state == TPUState.DECODE_EXECUTE:
                opcode = ir[0]

                if opcode == "LOADI":
                    dest_reg_str =  ir[1]
                    value = ir[2]
                    dest_reg_idx = int(dest_reg_str[2:]) # e.g., 'SC0' -> 0
                    regs_scalar.next[dest_reg_idx] = intbv(value)[FIXED_POINT_TOTAL_BITS:] # Access using .next[idx]
                    _next_pc = pc + 1
                    _next_state = TPUState.FETCH

                elif opcode == "LOAD_TENSOR":
                    dest_reg_str = ir[1]
                    mem_type_str = ir[2]
                    base_addr = ir[3]
                    shape_list = ir[4]
                    dest_reg_idx = int(dest_reg_str[1:]) # e.g., 'R0' -> 0
                    
                    # Convert mem_type_str to ID (0:WMEM, 1:IMEM, 2:OMEM, 3:SMEM)
                    mem_bank_id = 0 # Default to WMEM
                    if mem_type_str == 'WMEM': mem_bank_id = 0
                    elif mem_type_str == 'IMEM': mem_bank_id = 1
                    elif mem_type_str == 'OMEM': mem_bank_id = 2
                    elif mem_type_str == 'SMEM': mem_bank_id = 3
                    else: _next_state = TPUState.ERROR # Unknown memory type

                    pointer_width = MEM_BANK_ENUM_BITS + MEM_ADDR_BITS
                    new_pointer_val = concat(intbv(mem_bank_id)[MEM_BANK_ENUM_BITS:], intbv(base_addr)[MEM_ADDR_BITS:])

                    regs_tensor_ptr.next = (
                        new_pointer_val if 0 == dest_reg_idx else intbv(regs_tensor_ptr[0])[pointer_width:],
                        new_pointer_val if 1 == dest_reg_idx else intbv(regs_tensor_ptr[1])[pointer_width:],
                        new_pointer_val if 2 == dest_reg_idx else intbv(regs_tensor_ptr[2])[pointer_width:],
                        new_pointer_val if 3 == dest_reg_idx else intbv(regs_tensor_ptr[3])[pointer_width:],
                        new_pointer_val if 4 == dest_reg_idx else intbv(regs_tensor_ptr[4])[pointer_width:],
                        new_pointer_val if 5 == dest_reg_idx else intbv(regs_tensor_ptr[5])[pointer_width:],
                        new_pointer_val if 6 == dest_reg_idx else intbv(regs_tensor_ptr[6])[pointer_width:],
                        new_pointer_val if 7 == dest_reg_idx else intbv(regs_tensor_ptr[7])[pointer_width:],
                        new_pointer_val if 8 == dest_reg_idx else intbv(regs_tensor_ptr[8])[pointer_width:],
                        new_pointer_val if 9 == dest_reg_idx else intbv(regs_tensor_ptr[9])[pointer_width:],
                        new_pointer_val if 10 == dest_reg_idx else intbv(regs_tensor_ptr[10])[pointer_width:],
                        new_pointer_val if 11 == dest_reg_idx else intbv(regs_tensor_ptr[11])[pointer_width:],
                        new_pointer_val if 12 == dest_reg_idx else intbv(regs_tensor_ptr[12])[pointer_width:],
                        new_pointer_val if 13 == dest_reg_idx else intbv(regs_tensor_ptr[13])[pointer_width:],
                        new_pointer_val if 14 == dest_reg_idx else intbv(regs_tensor_ptr[14])[pointer_width:],
                        new_pointer_val if 15 == dest_reg_idx else intbv(regs_tensor_ptr[15])[pointer_width:],
                        # Add more lines here if NUM_TENSOR_REGS is greater than 16
                    )
                    
                    _next_pc = pc + 1
                    _next_state = TPUState.FETCH

                elif opcode == "ALLOC_TENSOR":
                    dest_reg_str = ir[1] 
                    mem_type_str = ir[2] 
                    base_addr = ir[3]
                    shape_list = ir[4]
                    dest_reg_idx = int(dest_reg_str[1:])

                    mem_bank_id = 0
                    if mem_type_str == 'WMEM': mem_bank_id = 0
                    elif mem_type_str == 'IMEM': mem_bank_id = 1
                    elif mem_type_str == 'OMEM': mem_bank_id = 2
                    elif mem_type_str == 'SMEM': mem_bank_id = 3
                    else: _next_state = TPUState.ERROR

                    pointer_width = MEM_BANK_ENUM_BITS + MEM_ADDR_BITS
                    new_pointer_val = concat(intbv(mem_bank_id)[MEM_BANK_ENUM_BITS:], intbv(base_addr)[MEM_ADDR_BITS:])

                    # Manually unroll the tuple construction for regs_tensor_ptr.next
                    regs_tensor_ptr.next = (
                        new_pointer_val if 0 == dest_reg_idx else intbv(regs_tensor_ptr[0])[pointer_width:],
                        new_pointer_val if 1 == dest_reg_idx else intbv(regs_tensor_ptr[1])[pointer_width:],
                        new_pointer_val if 2 == dest_reg_idx else intbv(regs_tensor_ptr[2])[pointer_width:],
                        new_pointer_val if 3 == dest_reg_idx else intbv(regs_tensor_ptr[3])[pointer_width:],
                        new_pointer_val if 4 == dest_reg_idx else intbv(regs_tensor_ptr[4])[pointer_width:],
                        new_pointer_val if 5 == dest_reg_idx else intbv(regs_tensor_ptr[5])[pointer_width:],
                        new_pointer_val if 6 == dest_reg_idx else intbv(regs_tensor_ptr[6])[pointer_width:],
                        new_pointer_val if 7 == dest_reg_idx else intbv(regs_tensor_ptr[7])[pointer_width:],
                        new_pointer_val if 8 == dest_reg_idx else intbv(regs_tensor_ptr[8])[pointer_width:],
                        new_pointer_val if 9 == dest_reg_idx else intbv(regs_tensor_ptr[9])[pointer_width:],
                        new_pointer_val if 10 == dest_reg_idx else intbv(regs_tensor_ptr[10])[pointer_width:],
                        new_pointer_val if 11 == dest_reg_idx else intbv(regs_tensor_ptr[11])[pointer_width:],
                        new_pointer_val if 12 == dest_reg_idx else intbv(regs_tensor_ptr[12])[pointer_width:],
                        new_pointer_val if 13 == dest_reg_idx else intbv(regs_tensor_ptr[13])[pointer_width:],
                        new_pointer_val if 14 == dest_reg_idx else intbv(regs_tensor_ptr[14])[pointer_width:],
                        new_pointer_val if 15 == dest_reg_idx else intbv(regs_tensor_ptr[15])[pointer_width:],
                        # Add more lines here if NUM_TENSOR_REGS is > 16
                    )
                    # For synthesizable memory, allocation implies getting a pointer.
                    # Clearing memory is done here behaviorally, could be skipped for efficiency.
                    num_elements = 1
                    # Assuming shape_list has a known, fixed maximum number of dimensions
                    # and you can unroll or otherwise handle it.
                    # For example, if max 4 dimensions:
                    if len(shape_list) > 0:
                        num_elements *= shape_list[0]
                    if len(shape_list) > 1:
                        num_elements *= shape_list[1]
                    if len(shape_list) > 2:
                        num_elements *= shape_list[2]
                    if len(shape_list) > 3:
                        num_elements *= shape_list[3]

                    _next_pc = pc + 1
                    _next_state = TPUState.FETCH

                elif opcode == "MATMUL":
                    dest_reg_str = ir[1]
                    a_reg_str = ir[2]
                    b_reg_str = ir[3]
                    a_shape_list = ir[4] # This is likely a Python list from your IR
                    b_shape_list = ir[5] # This is likely a Python list from your IR
                    dest_reg_idx = int(dest_reg_str[1:])
                    a_reg_idx = int(a_reg_str[1:])
                    b_reg_idx = int(b_reg_str[1:])

                    # Corrected: Accessing the current value of regs_tensor_ptr
                    # Simply use regs_tensor_ptr[idx] to get the current value of that element
                    matmul_dest_ptr.next = regs_tensor_ptr[dest_reg_idx]
                    matmul_A_ptr.next = regs_tensor_ptr[a_reg_idx]
                    matmul_B_ptr.next = regs_tensor_ptr[b_reg_idx]

                    # Assuming a_shape_list and b_shape_list are Python lists
                    # and their elements are fixed at conversion time or represent
                    # literal constants for the hardware.
                    # If these are dynamic and need to be hardware signals, they should be defined as such.
                    # If they are just Python integers from your IR for configuration:
                    matmul_A_rows.next = a_shape_list[0]
                    matmul_A_cols.next = a_shape_list[1]
                    matmul_B_cols.next = b_shape_list[1]

                    # Initialize for MATMUL:
                    _next_matmul_row_idx = 0
                    _next_matmul_col_idx = 0
                    _next_matmul_k_idx = 0 # <--- Initialized here
                    _next_matmul_mac_acc = 0
                    _next_state = TPUState.MATMUL_PREP

                elif opcode == "ADD_ELEM":
                    dest_reg_str = ir[1]
                    a_reg_str = ir[2]
                    b_reg_str = ir[3]
                    a_shape_list = ir[4]
                    b_shape_list = ir[5]
                    dest_reg_idx = int(dest_reg_str[1:])
                    a_reg_idx = int(a_reg_str[1:])
                    b_reg_idx = int(b_reg_str[1:])

                    addelem_dest_ptr.next = regs_tensor_ptr[dest_reg_idx]
                    addelem_A_ptr.next = regs_tensor_ptr[a_reg_idx]
                    addelem_B_ptr.next = regs_tensor_ptr[b_reg_idx]

                    addelem_A_rows.next = a_shape_list[0]
                    addelem_A_cols.next = a_shape_list[1]
                    addelem_B_shape_len.next = len(b_shape_list) # len() is fine if b_shape_list is Python list
                    if len(b_shape_list) == 1:
                        addelem_B_dim1.next = b_shape_list[0]
                    else:
                        addelem_B_dim1.next = 0 # Not used for 2D B

                    _next_addelem_idx = 0 # Initialized here
                    addelem_total_elements.next = a_shape_list[0] * a_shape_list[1]
                    _next_state = TPUState.ADD_ELEM_PREP

                elif opcode == "RELU":
                    dest_reg_str = ir[1]
                    src_reg_str = ir[2]
                    shape_list = ir[3] # This is likely a Python list from your IR
                    dest_reg_idx = int(dest_reg_str[1:])
                    src_reg_idx = int(src_reg_str[1:])

                    relu_dest_ptr.next = regs_tensor_ptr[dest_reg_idx]
                    relu_src_ptr.next = regs_tensor_ptr[src_reg_idx]
                    
                    _next_relu_idx = 0 # Initialized here
                    relu_total_elements.next = shape_list[0] * shape_list[1] # Assuming 2D for now
                    _next_state = TPUState.RELU_PREP

                elif opcode == "HALT":
                    _next_halted_out = True
                    _next_state = TPUState.HALT
                else:
                    _next_state = TPUState.ERROR # Unknown opcode

            # --- MATMUL State Machine ---
            elif state == TPUState.MATMUL_PREP:
                _next_matmul_row_idx = 0 # Initialized here
                _next_state = TPUState.MATMUL_LOOP_ROWS

            elif state == TPUState.MATMUL_LOOP_ROWS:
                if matmul_row_idx < matmul_A_rows:
                    _next_matmul_col_idx = 0 # Initialized here
                    _next_matmul_mac_acc = 0 # Initialized here
                    _next_state = TPUState.MATMUL_LOOP_COLS
                else:
                    _next_pc = pc + 1
                    _next_state = TPUState.FETCH

            elif state == TPUState.MATMUL_LOOP_COLS:
                if matmul_col_idx < matmul_B_cols:
                    _next_matmul_k_idx = 0 # Initialized here
                    _next_matmul_mac_acc = 0 # Initialized here
                    _next_state = TPUState.MATMUL_MAC_INNER_LOOP
                else:
                    _next_matmul_row_idx = matmul_row_idx + 1
                    _next_state = TPUState.MATMUL_LOOP_ROWS

            elif state == TPUState.MATMUL_MAC_INNER_LOOP:
                # Get memory arrays based on stored pointers
                a_mem_bank_id = matmul_A_ptr[MEM_ADDR_BITS + MEM_BANK_ENUM_BITS:MEM_ADDR_BITS]
                a_base_addr = matmul_A_ptr[MEM_ADDR_BITS:]
                b_mem_bank_id = matmul_B_ptr[MEM_ADDR_BITS + MEM_BANK_ENUM_BITS:MEM_ADDR_BITS]
                b_base_addr = matmul_B_ptr[MEM_ADDR_BITS:]
                
                assert a_mem_bank_id == 0 or a_mem_bank_id == 1 or a_mem_bank_id == 2 or a_mem_bank_id == 3, "Invalid A memory bank ID"
                assert b_mem_bank_id == 0 or b_mem_bank_id == 1 or b_mem_bank_id == 2 or b_mem_bank_id == 3, "Invalid B memory bank ID"

                if matmul_k_idx < matmul_A_cols:
                    # Read A[row_idx, k_idx]
                    if a_mem_bank_id == 0: src_val_sig.next = wmem[a_base_addr + matmul_row_idx * matmul_A_cols + matmul_k_idx]
                    elif a_mem_bank_id == 1: src_val_sig.next = imem[a_base_addr + matmul_row_idx * matmul_A_cols + matmul_k_idx]
                    elif a_mem_bank_id == 2: src_val_sig.next = omem[a_base_addr + matmul_row_idx * matmul_A_cols + matmul_k_idx]
                    elif a_mem_bank_id == 3: src_val_sig.next = smem[a_base_addr + matmul_row_idx * matmul_A_cols + matmul_k_idx]

                    # Read B[k_idx, col_idx]
                    # Note: Using result_sig as a temporary for b_val, if fixed_mul accepts Signals
                    # If fixed_mul needs intbv, you might need a dedicated b_val_sig as well.
                    if b_mem_bank_id == 0: result_sig.next = wmem[b_base_addr + matmul_k_idx * matmul_B_cols + matmul_col_idx]
                    elif b_mem_bank_id == 1: result_sig.next = imem[b_base_addr + matmul_k_idx * matmul_B_cols + matmul_col_idx]
                    elif b_mem_bank_id == 2: result_sig.next = omem[b_base_addr + matmul_k_idx * matmul_B_cols + matmul_col_idx]
                    elif b_mem_bank_id == 3: result_sig.next = smem[b_base_addr + matmul_k_idx * matmul_B_cols + matmul_col_idx]

                    # Fixed_mul and fixed_add now operate on the current values of signals
                    # and the accumulator is updated with its next value
                    _next_matmul_mac_acc = fixed_add(matmul_mac_acc, fixed_mul(src_val_sig, result_sig))
                    _next_matmul_k_idx = matmul_k_idx + 1 # <--- Incremented here
                    # state remains MATMUL_MAC_INNER_LOOP
                else:
                    # Inner loop (k) finished, write result
                    dest_mem_bank_id = matmul_dest_ptr[MEM_ADDR_BITS + MEM_BANK_ENUM_BITS:MEM_ADDR_BITS]
                    dest_base_addr = matmul_dest_ptr[MEM_ADDR_BITS:]

                    assert dest_mem_bank_id == 0 or dest_mem_bank_id == 1 or dest_mem_bank_id == 2 or dest_mem_bank_id == 3, "Invalid Destination memory bank ID"

                    write_addr = dest_base_addr + matmul_row_idx * matmul_B_cols + matmul_col_idx
                    if dest_mem_bank_id == 0:
                        wmem[write_addr].next = matmul_mac_acc
                    elif dest_mem_bank_id == 1:
                        imem[write_addr].next = matmul_mac_acc
                    elif dest_mem_bank_id == 2:
                        omem[write_addr].next = matmul_mac_acc
                    elif dest_mem_bank_id == 3:
                        smem[write_addr].next = matmul_mac_acc

                    _next_matmul_col_idx = matmul_col_idx + 1 # <--- Incremented here
                    _next_state = TPUState.MATMUL_LOOP_COLS

            # --- ADD_ELEM State Machine ---
            elif state == TPUState.ADD_ELEM_PREP:
                _next_addelem_idx = 0 # Initialized here
                _next_state = TPUState.ADD_ELEM_LOOP

            elif state == TPUState.ADD_ELEM_LOOP:
                if addelem_idx < addelem_total_elements:
                    # These pointers are MyHDL signals, access their current value directly
                    dest_mem_bank_id = addelem_dest_ptr[MEM_ADDR_BITS + MEM_BANK_ENUM_BITS:MEM_ADDR_BITS]
                    dest_base_addr = addelem_dest_ptr[MEM_ADDR_BITS:]
                    a_mem_bank_id = addelem_A_ptr[MEM_ADDR_BITS + MEM_BANK_ENUM_BITS:MEM_ADDR_BITS]
                    a_base_addr = addelem_A_ptr[MEM_ADDR_BITS:]
                    b_mem_bank_id = addelem_B_ptr[MEM_ADDR_BITS + MEM_BANK_ENUM_BITS:MEM_ADDR_BITS]
                    b_base_addr = addelem_B_ptr[MEM_ADDR_BITS:]
                    
                    assert dest_mem_bank_id == 0 or dest_mem_bank_id == 1 or dest_mem_bank_id == 2 or dest_mem_bank_id == 3, "Invalid ADD_ELEM Dest memory bank ID"
                    assert a_mem_bank_id == 0 or a_mem_bank_id == 1 or a_mem_bank_id == 2 or a_mem_bank_id == 3, "Invalid ADD_ELEM A memory bank ID"
                    assert b_mem_bank_id == 0 or b_mem_bank_id == 1 or b_mem_bank_id == 2 or b_mem_bank_id == 3, "Invalid ADD_ELEM B memory bank ID"

                    # Reading A value into src_val_sig
                    if a_mem_bank_id == 0: src_val_sig.next = wmem[a_base_addr + addelem_idx]
                    elif a_mem_bank_id == 1: src_val_sig.next = imem[a_base_addr + addelem_idx]
                    elif a_mem_bank_id == 2: src_val_sig.next = omem[a_base_addr + addelem_idx]
                    elif a_mem_bank_id == 3: src_val_sig.next = smem[a_base_addr + addelem_idx]

                    # Reading B value into result_sig (used as temporary for b_val)
                    if addelem_B_shape_len == 2: # 2D tensor B
                        r = addelem_idx // addelem_A_cols
                        c = addelem_idx % addelem_A_cols
                        if b_mem_bank_id == 0: result_sig.next = wmem[b_base_addr + r * addelem_B_dim1 + c]
                        elif b_mem_bank_id == 1: result_sig.next = imem[b_base_addr + r * addelem_B_dim1 + c]
                        elif b_mem_bank_id == 2: result_sig.next = omem[b_base_addr + r * addelem_B_dim1 + c]
                        elif b_mem_bank_id == 3: result_sig.next = smem[b_base_addr + r * addelem_B_dim1 + c]
                    else: # 1D tensor B (bias), broadcast across rows
                        c = addelem_idx % addelem_A_cols # Column index for B
                        if b_mem_bank_id == 0: result_sig.next = wmem[b_base_addr + c]
                        elif b_mem_bank_id == 1: result_sig.next = imem[b_base_addr + c]
                        elif b_mem_bank_id == 2: result_sig.next = omem[b_base_addr + c]
                        elif b_mem_bank_id == 3: result_sig.next = smem[b_base_addr + c]

                    # Perform addition on current values of signals, assign to result_sig
                    result_sig.next = fixed_add(src_val_sig, result_sig)

                    write_addr = dest_base_addr + addelem_idx
                    # Write the current value of result_sig to memory
                    if dest_mem_bank_id == 0:
                        wmem[write_addr].next = result_sig
                    elif dest_mem_bank_id == 1:
                        imem[write_addr].next = result_sig
                    elif dest_mem_bank_id == 2:
                        omem[write_addr].next = result_sig
                    elif dest_mem_bank_id == 3:
                        smem[write_addr].next = result_sig
                        
                    _next_addelem_idx = addelem_idx + 1 # <--- Incremented here
                    # state remains ADD_ELEM_LOOP
                else:
                    _next_pc = pc + 1
                    _next_state = TPUState.FETCH

            # --- RELU State Machine ---
            elif state == TPUState.RELU_PREP:
                _next_relu_idx = 0 # Initialized here
                _next_state = TPUState.RELU_LOOP

            elif state == TPUState.RELU_LOOP:
                if relu_idx < relu_total_elements:
                    src_mem_bank_id = relu_src_ptr[MEM_ADDR_BITS + MEM_BANK_ENUM_BITS:MEM_ADDR_BITS]
                    src_base_addr = relu_src_ptr[MEM_ADDR_BITS:]
                    dest_mem_bank_id = relu_dest_ptr[MEM_ADDR_BITS + MEM_BANK_ENUM_BITS:MEM_ADDR_BITS]
                    dest_base_addr = relu_dest_ptr[MEM_ADDR_BITS:]

                    assert src_mem_bank_id == 0 or src_mem_bank_id == 1 or src_mem_bank_id == 2 or src_mem_bank_id == 3, "Invalid RELU Src memory bank ID"
                    assert dest_mem_bank_id == 0 or dest_mem_bank_id == 1 or dest_mem_bank_id == 2 or dest_mem_bank_id == 3, "Invalid RELU Dest memory bank ID"

                    if src_mem_bank_id == 0: src_val_sig.next = wmem[src_base_addr + relu_idx]
                    elif src_mem_bank_id == 1: src_val_sig.next = imem[src_base_addr + relu_idx]
                    elif src_mem_bank_id == 2: src_val_sig.next = omem[src_base_addr + relu_idx]
                    elif src_mem_bank_id == 3: src_val_sig.next = smem[src_base_addr + relu_idx]
                    
                    result_sig.next = fixed_relu(src_val_sig)

                    write_addr = dest_base_addr + relu_idx
                    if dest_mem_bank_id == 0:
                        wmem[write_addr].next = result_sig
                    elif dest_mem_bank_id == 1:
                        imem[write_addr].next = result_sig
                    elif dest_mem_bank_id == 2:
                        omem[write_addr].next = result_sig
                    elif dest_mem_bank_id == 3:
                        smem[write_addr].next = result_sig

                    _next_relu_idx = relu_idx + 1 # <--- Incremented here
                    # state remains RELU_LOOP
                else:
                    _next_pc = pc + 1
                    _next_state = TPUState.FETCH

            elif state == TPUState.HALT:
                _next_halted_out = True
                _next_state = TPUState.HALT # Stay halted
            elif state == TPUState.ERROR:
                _next_halted_out = True # Halt on error
                print("TPU Entered ERROR state due to unexpected condition.")
                _next_state = TPUState.ERROR # Stay in error

            # --- Final Assignments for the current clock cycle ---
            # These assignments must happen exactly once per clock cycle for each Signal.
            pc.next = _next_pc
            state.next = _next_state
            halted_out.next = _next_halted_out
            matmul_row_idx.next = _next_matmul_row_idx
            matmul_col_idx.next = _next_matmul_col_idx
            matmul_k_idx.next = _next_matmul_k_idx
            matmul_mac_acc.next = _next_matmul_mac_acc
            addelem_idx.next = _next_addelem_idx
            relu_idx.next = _next_relu_idx

    return instances()