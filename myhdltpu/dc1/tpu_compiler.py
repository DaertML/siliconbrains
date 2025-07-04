# tpu_compiler.py (More Optimized for MyHDL conversion)
import numpy as np
from fixed_point_utils import float_to_fixed, FIXED_POINT_TOTAL_BITS

# --- 1. Compiler's Internal Memory and Register Abstractions ---
class CompilerTPUMemory:
    def __init__(self, wmem_size, imem_size, omem_size, smem_size):
        self.allocated_tensors = {} # {logical_reg_name: (memory_type, base_addr, shape)}
        self.next_addr = {
            'WMEM': 0,
            'IMEM': 0,
            'OMEM': 0,
            'SMEM': 0
        }
        self.bank_sizes = {
            'WMEM': wmem_size,
            'IMEM': imem_size,
            'OMEM': omem_size,
            'SMEM': smem_size
        }
        # Initial raw memory content (will be filled with fixed-point integers)
        self.initial_mem_content = {
            'WMEM': [0] * wmem_size,
            'IMEM': [0] * imem_size,
            'OMEM': [0] * omem_size,
            'SMEM': [0] * smem_size
        }

    def alloc_tensor(self, logical_reg_name, shape, mem_type='SMEM'):
        num_elements = int(np.prod(shape))
        base_addr = self.next_addr[mem_type]
        if base_addr + num_elements > self.bank_sizes[mem_type]:
            raise MemoryError(f"{mem_type} exhausted. Needed {num_elements}, available {self.bank_sizes[mem_type] - base_addr}")
        self.next_addr[mem_type] += num_elements
        self.allocated_tensors[logical_reg_name] = (mem_type, base_addr, shape)
        return base_addr, shape # Return allocated info

    def store_initial_data(self, mem_type, base_addr, data_flat):
        """Stores fixed-point data into the compiler's initial memory content buffer."""
        for i, val in enumerate(data_flat):
            self.initial_mem_content[mem_type][base_addr + i] = val

    def get_tensor_info(self, logical_reg_name):
        return self.allocated_tensors.get(logical_reg_name)

class CompilerTPURegisters:
    def __init__(self, num_tensor_regs, num_scalar_regs):
        self.logical_to_physical_map = {} # {logical_name: physical_reg_name}
        self.next_tensor_reg_idx = 0
        self.next_scalar_reg_idx = 0
        self.num_tensor_regs = num_tensor_regs
        self.num_scalar_regs = num_scalar_regs

    def get_next_tensor_reg(self, logical_name=None):
        if self.next_tensor_reg_idx >= self.num_tensor_regs:
            raise RuntimeError(f"Exceeded available tensor registers (R0-R{self.num_tensor_regs-1}).")
        physical_reg_name = f'R{self.next_tensor_reg_idx}'
        self.next_tensor_reg_idx += 1
        if logical_name:
            self.logical_to_physical_map[logical_name] = physical_reg_name
        return physical_reg_name

    def get_next_scalar_reg(self, logical_name=None):
        if self.next_scalar_reg_idx >= self.num_scalar_regs:
            raise RuntimeError(f"Exceeded available scalar registers (SC0-SC{self.num_scalar_regs-1}).")
        physical_reg_name = f'SC{self.next_scalar_reg_idx}'
        self.next_scalar_reg_idx += 1
        if logical_name:
            self.logical_to_physical_map[logical_name] = physical_reg_name
        return physical_reg_name

    def get_physical_reg(self, logical_name):
        return self.logical_to_physical_map.get(logical_name)

# --- 2. TPU Compiler (Optimized for MyHDL conversion) ---

class TPUCompiler:
    def __init__(self, wmem_size=4096, imem_size=4096, omem_size=4096, smem_size=8192,
                 num_tensor_regs=16, num_scalar_regs=8):
        self.wmem_size = wmem_size
        self.imem_size = imem_size
        self.omem_size = omem_size
        self.smem_size = smem_size
        self.num_tensor_regs = num_tensor_regs
        self.num_scalar_regs = num_scalar_regs

        self.tpu_program = []
        self.compiler_mem = CompilerTPUMemory(wmem_size, imem_size, omem_size, smem_size)
        self.compiler_regs = CompilerTPURegisters(num_tensor_regs, num_scalar_regs)


    def compile_linear_relu_layer(self, input_shape, weight_shape, bias_shape, output_shape,
                                  host_input_data, host_weight_data, host_bias_data):
        self.tpu_program = []
        # Re-initialize compiler memory and registers for a new compilation run
        self.compiler_mem = CompilerTPUMemory(self.wmem_size, self.imem_size, self.omem_size, self.smem_size)
        self.compiler_regs = CompilerTPURegisters(self.num_tensor_regs, self.num_scalar_regs)

        # Allocate physical registers and map logical names to them
        input_reg = self.compiler_regs.get_next_tensor_reg("input")
        weight_reg = self.compiler_regs.get_next_tensor_reg("weight")
        bias_reg = self.compiler_regs.get_next_tensor_reg("bias")
        matmul_out_reg = self.compiler_regs.get_next_tensor_reg("matmul_intermediate")
        add_out_reg = self.compiler_regs.get_next_tensor_reg("add_intermediate")
        relu_out_reg = self.compiler_regs.get_next_tensor_reg("output") # This will be the final output tensor

        print("\n--- Compiling Linear+ReLU Layer for Optimized MyHDL ---")
        print(f"Input: {input_shape} -> {input_reg}")
        print(f"Weight: {weight_shape} -> {weight_reg}")
        print(f"Bias: {bias_shape} -> {bias_reg}")
        print(f"Output: {output_shape} -> {relu_out_reg}")

        # 1. Allocate memory for Input, Weight, Bias and store fixed-point data
        input_base_addr, _ = self.compiler_mem.alloc_tensor(input_reg, input_shape, "IMEM")
        self.compiler_mem.store_initial_data('IMEM', input_base_addr, [float_to_fixed(x) for x in host_input_data.flatten()])
        # Instruction: LOAD_TENSOR (target_reg, mem_type_str, base_addr, shape_list)
        self.tpu_program.append(("LOAD_TENSOR", input_reg, "IMEM", input_base_addr, list(input_shape)))

        weight_base_addr, _ = self.compiler_mem.alloc_tensor(weight_reg, weight_shape, "WMEM")
        self.compiler_mem.store_initial_data('WMEM', weight_base_addr, [float_to_fixed(x) for x in host_weight_data.flatten()])
        self.tpu_program.append(("LOAD_TENSOR", weight_reg, "WMEM", weight_base_addr, list(weight_shape)))

        bias_base_addr, _ = self.compiler_mem.alloc_tensor(bias_reg, bias_shape, "WMEM")
        self.compiler_mem.store_initial_data('WMEM', bias_base_addr, [float_to_fixed(x) for x in host_bias_data.flatten()])
        self.tpu_program.append(("LOAD_TENSOR", bias_reg, "WMEM", bias_base_addr, list(bias_shape)))

        # 2. Allocate memory for intermediate MatMul result
        matmul_out_shape = [input_shape[0], weight_shape[1]]
        matmul_base_addr, _ = self.compiler_mem.alloc_tensor(matmul_out_reg, matmul_out_shape, "SMEM")
        # ALLOC_TENSOR no longer requires pre-clearing in the compiler; MyHDL handles it
        self.tpu_program.append(("ALLOC_TENSOR", matmul_out_reg, "SMEM", matmul_base_addr, matmul_out_shape))

        # 3. Perform Matrix Multiplication: matmul_out = Input * Weight
        self.tpu_program.append(("MATMUL", matmul_out_reg, input_reg, weight_reg, list(input_shape), list(weight_shape)))

        # 4. Allocate memory for intermediate Add result
        add_out_shape = list(output_shape)
        add_base_addr, _ = self.compiler_mem.alloc_tensor(add_out_reg, add_out_shape, "SMEM")
        self.tpu_program.append(("ALLOC_TENSOR", add_out_reg, "SMEM", add_base_addr, add_out_shape))

        # 5. Perform Element-wise Addition: add_out = matmul_out + Bias
        self.tpu_program.append(("ADD_ELEM", add_out_reg, matmul_out_reg, bias_reg, list(matmul_out_shape), list(bias_shape)))

        # 6. Allocate memory for ReLU result (final output)
        relu_base_addr, _ = self.compiler_mem.alloc_tensor(relu_out_reg, output_shape, "OMEM")
        self.tpu_program.append(("ALLOC_TENSOR", relu_out_reg, "OMEM", relu_base_addr, list(output_shape)))

        # 7. Perform ReLU Activation: relu_out = ReLU(add_out)
        self.tpu_program.append(("RELU", relu_out_reg, add_out_reg, list(output_shape)))

        # 8. Mark Final Output Location (Implicit store from relu_out_reg's assigned OMEM location)
        # No explicit STORE_TENSOR instruction needed here if relu_out_reg is already in OMEM.
        # The testbench will read from the OMEM at the address pointed to by relu_out_reg.

        self.tpu_program.append(("HALT",))
        print("--- Compilation Complete ---")

        # Return initial memory contents for MyHDL simulation
        return self.tpu_program, self.compiler_mem.initial_mem_content, self.compiler_regs.logical_to_physical_map, self.compiler_mem.allocated_tensors