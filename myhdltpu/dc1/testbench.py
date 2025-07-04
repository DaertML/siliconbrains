# testbench.py (Most Optimized for Fixed-Point MyHDL)
from myhdl import Signal, intbv, delay, instance, StopSimulation, Simulation, toVerilog, toVHDL, ResetSignal, instances # ADD instancesimport numpy as np
import numpy as np
from tpu_core import tpu_core, TPUState # Optimized MyHDL TPU core
from tpu_compiler import TPUCompiler # Optimized Python compiler
from fixed_point_utils import fixed_to_float, float_to_fixed, \
                             FIXED_POINT_TOTAL_BITS, FIXED_POINT_F_BITS

# Define parameters for the testbench (must match tpu_core defaults or be explicitly passed)
# These values are passed to both compiler and tpu_core for consistency
_MEM_ADDR_BITS = 12
_MEM_BANK_ENUM_BITS = 2
_SHAPE_DIM_BITS = 8
_NUM_TENSOR_REGS = 16
_NUM_SCALAR_REGS = 8
_WMEM_SIZE = 4096
_IMEM_SIZE = 4096
_OMEM_SIZE = 4096
_SMEM_SIZE = 8192

def test_tpu_core_optimized():
    # --- 1. Setup Host Data and Compile Program ---
    batch_size = 4
    input_features = 10
    output_features = 5

    # Original float host data
    host_input_data = (np.random.rand(batch_size, input_features) * 2 - 1).astype(np.float32)
    host_weight_data = (np.random.rand(input_features, output_features) * 2 - 1).astype(np.float32)
    host_bias_data = (np.random.rand(output_features) * 2 - 1).astype(np.float32)

    # Instantiate compiler with consistent parameters
    compiler = TPUCompiler(wmem_size=_WMEM_SIZE, imem_size=_IMEM_SIZE, omem_size=_OMEM_SIZE, smem_size=_SMEM_SIZE,
                           num_tensor_regs=_NUM_TENSOR_REGS, num_scalar_regs=_NUM_SCALAR_REGS)
    
    tpu_program, initial_mem_content_raw, logical_to_physical_map, allocated_tensors_info = \
        compiler.compile_linear_relu_layer(
            input_shape=host_input_data.shape,
            weight_shape=host_weight_data.shape,
            bias_shape=host_bias_data.shape,
            output_shape=(batch_size, output_features),
            host_input_data=host_input_data,
            host_weight_data=host_weight_data,
            host_bias_data=host_bias_data
        )

    # --- MyHDL Signals ---
    clk = Signal(bool(0))
    reset = ResetSignal(0, active=1, isasync=True) # Reset: 0 for inactive, 1 for active high, asynchronous
    halted = Signal(bool(0))
    pc_out = Signal(intbv(0)[_MEM_ADDR_BITS:]) # To observe PC from testbench
    state_out = Signal(TPUState.FETCH) # Initialize with an enum member for clarity and type

    wmem = Signal(tuple(intbv(val)[FIXED_POINT_TOTAL_BITS:] for val in initial_mem_content_raw['WMEM']))
    imem = Signal(tuple(intbv(val)[FIXED_POINT_TOTAL_BITS:] for val in initial_mem_content_raw['IMEM']))
    omem = Signal(tuple(intbv(val)[FIXED_POINT_TOTAL_BITS:] for val in initial_mem_content_raw['OMEM']))
    smem = Signal(tuple(intbv(val)[FIXED_POINT_TOTAL_BITS:] for val in initial_mem_content_raw['SMEM']))

    # --- Instantiate MyHDL TPU Core ---
    # Pass the actual memory signals to the tpu_core block
    tpu_inst = tpu_core(
        clk=clk,
        reset=reset,
        program_rom=tuple(tpu_program), # MyHDL expects tuples for ROM
        # NEW: Pass the pre-initialized memory signals directly
        wmem=wmem,
        imem=imem,
        smem=smem,
        omem=omem,
        halted_out=halted,
        pc_out=pc_out,
        state_out=state_out,
        MEM_ADDR_BITS=_MEM_ADDR_BITS,
        MEM_BANK_ENUM_BITS=_MEM_BANK_ENUM_BITS,
        SHAPE_DIM_BITS=_SHAPE_DIM_BITS,
        NUM_TENSOR_REGS=_NUM_TENSOR_REGS,
        NUM_SCALAR_REGS=_NUM_SCALAR_REGS,
        WMEM_SIZE=_WMEM_SIZE,
        IMEM_SIZE=_IMEM_SIZE,
        OMEM_SIZE=_OMEM_SIZE,
        SMEM_SIZE=_SMEM_SIZE
    )

    # --- Testbench Logic ---
    @instance
    def stimulus():
        print("\n--- Starting MyHDL Fixed-Point Simulation (Synthesizable I/O) ---")
        reset.next = 1
        yield delay(10) # Hold reset for a few cycles
        reset.next = 0
        yield clk.negedge # Allow one cycle for initial memory load

        max_cycles = 10000 # Increased cycles significantly for multi-cycle ops and larger models
        cycle_count = 0
        while not halted and cycle_count < max_cycles:
            yield clk.negedge # Advance clock for state transition
            cycle_count += 1
            if cycle_count % 50 == 0:
                print(f"Cycle {cycle_count}, PC: {pc_out}, State: {state_out}")
            if state_out == TPUState.ERROR: # Use the directly imported TPUState
                print("TPU entered ERROR state. Halting simulation.")
                break # Break if error state is reached

        yield clk.negedge # Ensure last state is captured

        if halted:
            print(f"\nMyHDL TPU Halted after {cycle_count} cycles. PC: {pc_out}")
        else:
            print(f"\nMyHDL TPU did not halt within {max_cycles} cycles. Last state: {state_out}")

        # --- 5. Verify Results ---
        print("\n--- Verifying MyHDL Results ---")

        output_logical_reg_name = "R2"
        # output_physical_reg_name = logical_to_physical_map[output_logical_reg_name] # Not needed now

        # Get the actual allocated tensor info for the output from the compiler's record
        # This tells us which memory bank and base address the output is stored at
        output_mem_type, output_base_addr, output_shape_list = allocated_tensors_info[output_logical_reg_name]
        
        # Determine which external MyHDL memory array to read from
        tpu_output_mem_array = None # Renamed to avoid confusion with internal signals
        if output_mem_type == 'OMEM':
            tpu_output_mem_array = omem # Access the testbench's omem signal directly
        elif output_mem_type == 'SMEM':
            tpu_output_mem_array = smem # Access the testbench's smem signal directly
        elif output_mem_type == 'WMEM':
            tpu_output_mem_array = wmem # Access the testbench's wmem signal directly
        elif output_mem_type == 'IMEM':
            tpu_output_mem_array = imem # Access the testbench's imem signal directly

        if tpu_output_mem_array is not None:
            num_elements = int(np.prod(output_shape_list))
            # Access elements from the tuple-based Signal array's value
            tpu_output_fixed_list = [int(tpu_output_mem_array._val[output_base_addr + i]) for i in range(num_elements)]
            tpu_output_float_flat = [fixed_to_float(val) for val in tpu_output_fixed_list]
            tpu_output = np.array(tpu_output_float_flat, dtype=np.float32).reshape(output_shape_list)
            print("MyHDL TPU Computed Output (first 3 rows):\n", tpu_output[:3])

            # --- Calculate CPU reference (using original float data) ---
            cpu_intermediate = np.dot(host_input_data, host_weight_data)
            cpu_output = np.maximum(0, cpu_intermediate + host_bias_data)
            print("CPU Computed Output (first 3 rows):\n", cpu_output[:3])

            tolerance = 1e-3 # Tolerance for fixed-point
            if np.allclose(tpu_output, cpu_output, atol=tolerance):
                print(f"\nVerification SUCCESS! MyHDL TPU output matches CPU output (tolerance: {tolerance}).")
            else:
                print("\nVerification FAILED! MyHDL TPU output does NOT match CPU output.")
                diff = np.abs(tpu_output - cpu_output)
                print("Max absolute difference:", np.max(diff))
        else:
            print(f"Error: Could not retrieve output from internal memory. Output was allocated in {output_mem_type}.")

        raise StopSimulation # End simulation

    # Clock generator
    @instance
    def clkgen():
        while True:
            clk.next = not clk
            yield delay(5) # 10ns clock period

    return instances()

def convert_to_hdl_optimized():
    # Dummy signals and values for conversion signature.
    clk = Signal(bool(0))
    reset = ResetSignal(0, active=1, isasync=True) # Needs to be ResetSignal for conversion
    halted = Signal(bool(0))
    pc_out = Signal(intbv(0)[_MEM_ADDR_BITS:])
    state_out = Signal(TPUState.FETCH) # Use the directly imported TPUState

    dummy_program = (("HALT",),)
    dummy_fixed_val = intbv(0)[FIXED_POINT_TOTAL_BITS:]

    # Declare dummy memory signals for conversion
    wmem_dummy = Signal(tuple(dummy_fixed_val for _ in range(_WMEM_SIZE)))
    imem_dummy = Signal(tuple(dummy_fixed_val for _ in range(_IMEM_SIZE)))
    smem_dummy = Signal(tuple(dummy_fixed_val for _ in range(_SMEM_SIZE)))
    omem_dummy = Signal(tuple(dummy_fixed_val for _ in range(_OMEM_SIZE)))

    tpu_core_conversion_inst = tpu_core(
        clk=clk, # Assuming 'clk' here is the dummy_clk for conversion
        reset=reset, # Assuming 'reset' here is the dummy_reset for conversion
        program_rom=dummy_program,
        wmem=wmem_dummy,
        imem=imem_dummy,
        smem=smem_dummy,
        omem=omem_dummy,
        halted_out=halted, # This 'halted' is the dummy output signal
        pc_out=pc_out,   # This 'pc_out' is the dummy output signal
        state_out=state_out, # This 'state_out' is the dummy output signal
        # IMPORTANT: Pass all parameters explicitly, as they are part of the module's interface
        MEM_ADDR_BITS=_MEM_ADDR_BITS,
        MEM_BANK_ENUM_BITS=_MEM_BANK_ENUM_BITS,
        SHAPE_DIM_BITS=_SHAPE_DIM_BITS,
        NUM_TENSOR_REGS=_NUM_TENSOR_REGS,
        NUM_SCALAR_REGS=_NUM_SCALAR_REGS,
        WMEM_SIZE=_WMEM_SIZE,
        IMEM_SIZE=_IMEM_SIZE,
        OMEM_SIZE=_OMEM_SIZE,
        SMEM_SIZE=_SMEM_SIZE
    )

    # --- Verilog Conversion ---
    print("--- Attempting MyHDL to Verilog Conversion (Synthesizable I/O) ---")
    try:
        # Pass the instantiated block to toVerilog
        toVerilog(tpu_core_conversion_inst)
        print("Verilog conversion successful.")
    except Exception as e:
        print(f"Verilog conversion failed: {e}")

    # --- VHDL Conversion ---
    print("\n--- Attempting MyHDL to VHDL Conversion (Synthesizable I/O) ---")
    try:
        # Pass the instantiated block to toVHDL
        toVHDL(tpu_core_conversion_inst)
        print("VHDL conversion successful.")
    except Exception as e:
        print(f"VHDL conversion failed: {e}")

if __name__ == "__main__":
    # Run the simulation
    sim = Simulation(test_tpu_core_optimized())
    sim.run()

    # Attempt HDL conversion
    convert_to_hdl_optimized()
