import numpy as np

# --- 1. Simplified TPU Memory and Register Abstractions ---

class TPUMemory:
    def __init__(self, wmem_size=1024*1024, imem_size=1024*1024,
                 omem_size=1024*1024, smem_size=1024*1024):
        self.wmem = np.zeros(wmem_size // 4, dtype=np.float32) # Stored in FP32
        self.imem = np.zeros(imem_size // 4, dtype=np.float32)
        self.omem = np.zeros(omem_size // 4, dtype=np.float32)
        self.smem = np.zeros(smem_size // 4, dtype=np.float32)
        self.allocated_tensors = {} # {register_name: (memory_type, base_addr, shape, dtype)}
        self.next_wmem_addr = 0
        self.next_imem_addr = 0
        self.next_omem_addr = 0
        self.next_smem_addr = 0
        print("TPU Memory Initialized.")

    def _get_mem_bank(self, mem_type):
        if mem_type == 'WMEM': return self.wmem
        if mem_type == 'IMEM': return self.imem
        if mem_type == 'OMEM': return self.omem
        if mem_type == 'SMEM': return self.smem
        raise ValueError(f"Unknown memory type: {mem_type}")

    def _allocate_in_bank(self, mem_type, size_in_floats):
        mem_bank = self._get_mem_bank(mem_type)
        if mem_type == 'WMEM':
            current_addr = self.next_wmem_addr
            if current_addr + size_in_floats > len(mem_bank):
                raise MemoryError(f"WMEM exhausted. Needed {size_in_floats}, available {len(mem_bank) - current_addr}")
            self.next_wmem_addr += size_in_floats
        elif mem_type == 'IMEM':
            current_addr = self.next_imem_addr
            if current_addr + size_in_floats > len(mem_bank):
                raise MemoryError(f"IMEM exhausted. Needed {size_in_floats}, available {len(mem_bank) - current_addr}")
            self.next_imem_addr += size_in_floats
        elif mem_type == 'OMEM':
            current_addr = self.next_omem_addr
            if current_addr + size_in_floats > len(mem_bank):
                raise MemoryError(f"OMEM exhausted. Needed {size_in_floats}, available {len(mem_bank) - current_addr}")
            self.next_omem_addr += size_in_floats
        elif mem_type == 'SMEM':
            current_addr = self.next_smem_addr
            if current_addr + size_in_floats > len(mem_bank):
                raise MemoryError(f"SMEM exhausted. Needed {size_in_floats}, available {len(mem_bank) - current_addr}")
            self.next_smem_addr += size_in_floats
        else:
            raise ValueError(f"Unknown memory type for allocation: {mem_type}")
        return current_addr

    def alloc_tensor(self, reg_name, shape, mem_type='SMEM', dtype=np.float32):
        num_elements = int(np.prod(shape))
        size_in_floats = num_elements # Assuming dtype is float32
        base_addr = self._allocate_in_bank(mem_type, size_in_floats)
        self.allocated_tensors[reg_name] = (mem_type, base_addr, shape, dtype)
        print(f"Allocated {shape} tensor for {reg_name} at {mem_type}:{base_addr}")
        return base_addr

    def load_tensor_from_host(self, reg_name, host_data: np.ndarray, mem_type):
        num_elements = host_data.size
        size_in_floats = num_elements
        base_addr = self._allocate_in_bank(mem_type, size_in_floats)
        mem_bank = self._get_mem_bank(mem_type)
        mem_bank[base_addr : base_addr + num_elements] = host_data.flatten()
        self.allocated_tensors[reg_name] = (mem_type, base_addr, host_data.shape, host_data.dtype)
        print(f"Loaded {host_data.shape} tensor into {reg_name} at {mem_type}:{base_addr}")
        return base_addr

    def get_tensor(self, reg_name):
        if reg_name not in self.allocated_tensors:
            raise ValueError(f"Tensor {reg_name} not allocated or loaded.")
        mem_type, base_addr, shape, dtype = self.allocated_tensors[reg_name]
        mem_bank = self._get_mem_bank(mem_type)
        num_elements = int(np.prod(shape))
        tensor_data = mem_bank[base_addr : base_addr + num_elements].reshape(shape)
        return tensor_data

    def set_tensor(self, reg_name, data: np.ndarray):
        if reg_name not in self.allocated_tensors:
            raise ValueError(f"Tensor {reg_name} not allocated. Cannot set data.")
        mem_type, base_addr, shape, dtype = self.allocated_tensors[reg_name]
       
        # Abort if shape mismatch
        #if data.shape != shape:
        #    raise ValueError(f"Shape mismatch for {reg_name}. Expected {shape}, got {data.shape}")

        # Continue if shape mismatch
        if data.shape != shape:
            # A simplified broadcasting check: if the smaller tensor can be broadcasted to the larger one.
            # This is not a full numpy broadcasting logic, but handles the (N, M) + (M,) case.
            if len(shape) == 2 and len(data.shape) == 1 and shape[1] == data.shape[0]:
                data_to_store = np.tile(data, (shape[0], 1))
            else:
                # If it's not the simple N,M + M case, just fail or implement full broadcasting
                raise ValueError(f"Complex broadcasting not supported for setting {reg_name}. Expected {shape}, got {data.shape}")
        else:
            data_to_store = data

        mem_bank = self._get_mem_bank(mem_type)
        mem_bank[base_addr : base_addr + data.size] = data.flatten()
        print(f"Set data for {reg_name}.")

    def reset(self):
        self.__init__() # Simple reset
        print("TPU Memory Reset.")

class TPURegisters:
    def __init__(self):
        self.tensor_regs = {f'R{i}': None for i in range(16)} # Holds tensor references (names)
        self.scalar_regs = {f'SC{i}': None for i in range(8)} # Holds scalar values
        print("TPU Registers Initialized.")

    def set_tensor_reg(self, reg_name, tensor_ref_name):
        if reg_name not in self.tensor_regs:
            raise ValueError(f"Invalid tensor register: {reg_name}")
        self.tensor_regs[reg_name] = tensor_ref_name

    def get_tensor_reg(self, reg_name):
        if reg_name not in self.tensor_regs:
            raise ValueError(f"Invalid tensor register: {reg_name}")
        return self.tensor_regs[reg_name]

    def set_scalar_reg(self, reg_name, value):
        if reg_name not in self.scalar_regs:
            raise ValueError(f"Invalid scalar register: {reg_name}")
        self.scalar_regs[reg_name] = value

    def get_scalar_reg(self, reg_name):
        if reg_name not in self.scalar_regs:
            raise ValueError(f"Invalid scalar register: {reg_name}")
        return self.scalar_regs[reg_name]

    def reset(self):
        self.__init__() # Simple reset
        print("TPU Registers Reset.")

# --- 2. Simple TPU Processor ---

class SimpleTPU:
    def __init__(self):
        self.memory = TPUMemory()
        self.registers = TPURegisters()
        self.program_counter = 0
        self.program = []
        self.halted = False
        print("SimpleTPU Processor Initialized.")

    def load_program(self, program_instructions):
        self.program = program_instructions
        self.program_counter = 0
        self.halted = False
        print(f"Program loaded: {len(program_instructions)} instructions.")

    def _execute_instruction(self, instruction):
        opcode = instruction[0]
        operands = instruction[1:]
        # print(f"Executing: {instruction}") # Uncomment for verbose execution

        if opcode == "LOADI":
            dest_reg, value = operands
            self.registers.set_scalar_reg(dest_reg, value)
        elif opcode == "LOAD_TENSOR":
            dest_reg, host_data_name, shape_str, mem_type = operands
            # For simulation, host_data_name is a placeholder to retrieve from some context
            # In a real scenario, this would be a direct memory address to host memory
            # We assume host_data_name maps to an actual numpy array here.
            host_data = self._get_host_data_by_name(host_data_name) # Requires context
            if host_data.shape != tuple(shape_str):
                raise ValueError(f"LOAD_TENSOR shape mismatch for {dest_reg}. Expected {shape_str}, got {host_data.shape}")
            self.memory.load_tensor_from_host(dest_reg, host_data, mem_type)
            self.registers.set_tensor_reg(dest_reg, dest_reg) # Register points to itself (its name)
        elif opcode == "STORE_TENSOR":
            src_reg, host_data_name = operands
            tensor_ref = self.registers.get_tensor_reg(src_reg)
            tensor_data = self.memory.get_tensor(tensor_ref)
            # In a real scenario, this would write to a host memory address
            # For simulation, we'll store it back to a global dictionary
            self._set_host_data_by_name(host_data_name, tensor_data) # Requires context
        elif opcode == "ALLOC_TENSOR":
            dest_reg, shape_str, mem_type = operands
            self.memory.alloc_tensor(dest_reg, tuple(shape_str), mem_type)
            self.registers.set_tensor_reg(dest_reg, dest_reg) # Register points to itself (its name)
        elif opcode == "MATMUL":
            dest_reg, a_reg, b_reg = operands
            a_tensor_ref = self.registers.get_tensor_reg(a_reg)
            b_tensor_ref = self.registers.get_tensor_reg(b_reg)
            A = self.memory.get_tensor(a_tensor_ref)
            B = self.memory.get_tensor(b_tensor_ref)

            # --- Systolic Array Simulation (conceptual) ---
            # In a real TPU, this is where the hardware performs the high-speed MAC operations
            # We simulate it with numpy.
            if A.ndim != 2 or B.ndim != 2:
                raise ValueError("MATMUL expects 2D tensors.")
            if A.shape[1] != B.shape[0]:
                raise ValueError(f"MATMUL dimension mismatch: A:{A.shape}, B:{B.shape}")

            result = np.dot(A, B)
            dest_tensor_ref = self.registers.get_tensor_reg(dest_reg)
            self.memory.set_tensor(dest_tensor_ref, result)
            print(f"MATMUL {dest_reg} = {a_reg} x {b_reg} (result shape: {result.shape})")

        elif opcode == "ADD_ELEM":
            dest_reg, a_reg, b_reg = operands
            a_tensor_ref = self.registers.get_tensor_reg(a_reg)
            b_tensor_ref = self.registers.get_tensor_reg(b_reg)
            A = self.memory.get_tensor(a_tensor_ref)
            B = self.memory.get_tensor(b_tensor_ref)

            # --- FIX: Implement simplified broadcasting for ADD_ELEM ---
            # This handles the common case of (N, M) + (M,)
            # A real TPU might have a dedicated broadcasting unit or require explicit copy/tile operations.
            if A.shape == B.shape:
                result = A + B
            elif len(A.shape) == 2 and len(B.shape) == 1 and A.shape[1] == B.shape[0]:
                # Broadcas B to match A's first dimension
                result = A + B[np.newaxis, :] # Add a new axis for broadcasting
            elif len(B.shape) == 2 and len(A.shape) == 1 and B.shape[1] == A.shape[0]:
                # Broadcas A to match B's first dimension (less common for bias)
                result = A[np.newaxis, :] + B
            else:
                # If more complex broadcasting is needed, a real TPU compiler
                # would need to emit explicit reshape/tile/broadcast ops
                raise ValueError(f"ADD_ELEM shape mismatch or unsupported broadcasting: A:{A.shape}, B:{B.shape}")
            # --- END FIX ---

            dest_tensor_ref = self.registers.get_tensor_reg(dest_reg)
            self.memory.set_tensor(dest_tensor_ref, result)
            print(f"ADD_ELEM {dest_reg} = {a_reg} + {b_reg}")

        elif opcode == "RELU":
            dest_reg, src_reg = operands
            src_tensor_ref = self.registers.get_tensor_reg(src_reg)
            Src = self.memory.get_tensor(src_tensor_ref)

            result = np.maximum(0, Src)
            dest_tensor_ref = self.registers.get_tensor_reg(dest_reg)
            self.memory.set_tensor(dest_tensor_ref, result)
            print(f"RELU {dest_reg} = ReLU({src_reg})")

        elif opcode == "HALT":
            self.halted = True
            print("TPU Halted.")
        else:
            raise ValueError(f"Unknown opcode: {opcode}")

    def execute_program(self, host_data_context=None):
        if host_data_context is None:
            self._host_data_context = {}
        else:
            self._host_data_context = host_data_context

        self.program_counter = 0
        self.halted = False
        print("Starting TPU program execution...")
        while not self.halted and self.program_counter < len(self.program):
            instruction = self.program[self.program_counter]
            self._execute_instruction(instruction)
            self.program_counter += 1
        print("TPU program execution finished.")

    def _get_host_data_by_name(self, name):
        if name not in self._host_data_context:
            raise ValueError(f"Host data '{name}' not found in context.")
        return self._host_data_context[name]

    def _set_host_data_by_name(self, name, data):
        self._host_data_context[name] = data
        print(f"Stored result back to host context as '{name}'.")

    def reset(self):
        self.memory.reset()
        self.registers.reset()
        self.program = []
        self.program_counter = 0
        self.halted = False
        self._host_data_context = {}
        print("SimpleTPU Processor Reset.")

# --- 3. TPU Compiler (Very Basic) ---

class TPUCompiler:
    def __init__(self):
        self.tpu_program = []
        self.reg_alloc_map = {} # {logical_name: physical_reg_name}
        self.next_tensor_reg = 0
        self.next_scalar_reg = 0

    def _get_next_tensor_reg(self):
        if self.next_tensor_reg >= 16:
            raise RuntimeError("Exceeded available tensor registers (R0-R15).")
        reg_name = f'R{self.next_tensor_reg}'
        self.next_tensor_reg += 1
        return reg_name

    def _get_next_scalar_reg(self):
        if self.next_scalar_reg >= 8:
            raise RuntimeError("Exceeded available scalar registers (SC0-SC7).")
        reg_name = f'SC{self.next_scalar_reg}'
        self.next_scalar_reg += 1
        return reg_name

    def compile_linear_relu_layer(self, input_shape, weight_shape, bias_shape, output_shape,
                                  input_name="input", weight_name="weight", bias_name="bias", output_name="output"):
        self.tpu_program = []
        self.reg_alloc_map = {}
        self.next_tensor_reg = 0
        self.next_scalar_reg = 0

        # Allocate registers for inputs, weights, bias, and intermediates
        input_reg = self._get_next_tensor_reg()
        weight_reg = self._get_next_tensor_reg()
        bias_reg = self._get_next_tensor_reg()
        matmul_out_reg = self._get_next_tensor_reg()
        add_out_reg = self._get_next_tensor_reg()
        relu_out_reg = self._get_next_tensor_reg()

        self.reg_alloc_map[input_name] = input_reg
        self.reg_alloc_map[weight_name] = weight_reg
        self.reg_alloc_map[bias_name] = bias_reg
        self.reg_alloc_map["matmul_intermediate"] = matmul_out_reg
        self.reg_alloc_map["add_intermediate"] = add_out_reg
        self.reg_alloc_map[output_name] = relu_out_reg


        print("\n--- Compiling Linear+ReLU Layer ---")
        print(f"Input: {input_name} ({input_shape}) -> {input_reg}")
        print(f"Weight: {weight_name} ({weight_shape}) -> {weight_reg}")
        print(f"Bias: {bias_name} ({bias_shape}) -> {bias_reg}")
        print(f"Output: {output_name} ({output_shape}) -> {relu_out_reg}")

        # 1. Load Input, Weight, Bias from Host to TPU Memory
        self.tpu_program.append(["LOAD_TENSOR", input_reg, input_name, list(input_shape), "IMEM"])
        self.tpu_program.append(["LOAD_TENSOR", weight_reg, weight_name, list(weight_shape), "WMEM"])
        self.tpu_program.append(["LOAD_TENSOR", bias_reg, bias_name, list(bias_shape), "WMEM"])

        # 2. Allocate memory for intermediate MatMul result
        self.tpu_program.append(["ALLOC_TENSOR", matmul_out_reg, [input_shape[0], weight_shape[1]], "SMEM"])

        # 3. Perform Matrix Multiplication: matmul_out = Input * Weight
        self.tpu_program.append(["MATMUL", matmul_out_reg, input_reg, weight_reg])

        # 4. Allocate memory for intermediate Add result
        self.tpu_program.append(["ALLOC_TENSOR", add_out_reg, list(output_shape), "SMEM"]) # Bias is added to MatMul result

        # 5. Perform Element-wise Addition: add_out = matmul_out + Bias
        self.tpu_program.append(["ADD_ELEM", add_out_reg, matmul_out_reg, bias_reg])

        # 6. Allocate memory for ReLU result (which is our final output)
        self.tpu_program.append(["ALLOC_TENSOR", relu_out_reg, list(output_shape), "OMEM"])

        # 7. Perform ReLU Activation: relu_out = ReLU(add_out)
        self.tpu_program.append(["RELU", relu_out_reg, add_out_reg])

        # 8. Store Final Output back to Host Memory
        self.tpu_program.append(["STORE_TENSOR", relu_out_reg, output_name])

        self.tpu_program.append(["HALT"])
        print("--- Compilation Complete ---")
        return self.tpu_program

# --- Example Usage ---

if __name__ == "__main__":
    # Initialize TPU components
    tpu = SimpleTPU()
    compiler = TPUCompiler()

    # Define a simple linear layer
    batch_size = 4
    input_features = 10
    output_features = 5

    # Host Data (simulated)
    host_input_data = np.random.rand(batch_size, input_features).astype(np.float32)
    host_weight_data = np.random.rand(input_features, output_features).astype(np.float32)
    host_bias_data = np.random.rand(output_features).astype(np.float32) # Bias is 1D, will be broadcast by ADD_ELEM
    host_output_buffer = np.zeros((batch_size, output_features), dtype=np.float32) # Buffer to store result

    # Context for host data (passed to TPU for LOAD/STORE_TENSOR)
    host_data_context = {
        "input": host_input_data,
        "weight": host_weight_data,
        "bias": host_bias_data,
        "output": host_output_buffer # The TPU will write into this
    }

    # Compile the layer operation
    tpu_program = compiler.compile_linear_relu_layer(
        input_shape=host_input_data.shape,
        weight_shape=host_weight_data.shape,
        bias_shape=host_bias_data.shape,
        output_shape=host_output_buffer.shape,
        input_name="input",
        weight_name="weight",
        bias_name="bias",
        output_name="output"
    )

    # Load and execute the compiled program on the TPU
    tpu.load_program(tpu_program)
    tpu.execute_program(host_data_context)

    # Retrieve the result from the host_data_context
    tpu_output = host_data_context["output"]
    print("\nTPU Computed Output (first 3 rows):\n", tpu_output[:3])

    # --- Verify with NumPy (CPU reference) ---
    print("\n--- Verifying with NumPy (CPU) ---")
    cpu_intermediate = np.dot(host_input_data, host_weight_data)
    cpu_output = np.maximum(0, cpu_intermediate + host_bias_data)
    print("CPU Computed Output (first 3 rows):\n", cpu_output[:3])

    # Compare results
    tolerance = 1e-6
    if np.allclose(tpu_output, cpu_output, atol=tolerance):
        print(f"\nVerification SUCCESS! TPU output matches CPU output (tolerance: {tolerance}).")
    else:
        print("\nVerification FAILED! TPU output does NOT match CPU output.")
        diff = np.abs(tpu_output - cpu_output)
        print("Max absolute difference:", np.max(diff))

    # Demonstrate reset
    print("\n--- Demonstrating TPU Reset ---")
    tpu.reset()
    print("TPU state after reset:", tpu.memory.allocated_tensors, tpu.registers.tensor_regs)

    # You could now re-run the program or load a new one
