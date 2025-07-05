# siliconbrains
Open Source TPUs for the masses

![alt text](siliconbrain.webp "A silicon brain")

# Introduction
This repo attempts on the design of open source TPUs (Tensor Processing Units) and other kinds of ASIC and hardware designed to accelerate the execution of neural networks.

The main goal focuses on doing the forward pass; as usually we would dedicate other resources to train them. As a first attempt, achieving something like what Google did with TPU Coral, but open source, and supporting other kinds of architectures of neural networks and following different datapaths... would be a win.

The first approach is to use bare Python to make the first design of the TPU, then move to behavioral design based on a framework with hardware abstractions like MyHDL, and finally transforming such into a synthesizable design, burn it into an FPGA and consider future ASIC creation.

# Chips
## DC-1: Daert Chip 1
The first chip that will get designed is a simple chip that will do simple MLPs: focus on dense and ReLUs, which can bring us quite far for many nowadays problems.

### Design
The DC-1 follows the Systolic-Array Architecture, as other TPU solutions like Google TPUs, one of the main focus is to be efficient in the data transfer between registers and memory segments, so that, matrix multiplications can be done fast.

#### Registers
- R0 - R15: General-purpose tensor registers. These hold references to tensors in memory.
- SC0 - SC7: Scalar registers for immediate values or control flow.

#### Memory
- Weight Memory (WMEM): Stores model weights.
- Input Memory (IMEM): Stores input activations.
- Output Memory (OMEM): Stores output activations.
- Scratchpad Memory (SMEM): Temporary storage for intermediate results.

#### Data Types:
- FP32: 32-bit Floating Point (our primary data type)
- INT32: 32-bit Integer (for indexing/control)

#### Instruction format:
OPCODE Dest_Reg, Src1_Reg, Src2_Reg, [Immediate_Value/Mem_Address]

#### Tensor representation:
Row major order.
- Base Address in Memory
- Dimensions

### ISA
A. Memory Operations:
- LOADI (Load Immediate):
  - Syntax: LOADI Dest_Reg, Value
  - Description: Loads an immediate scalar value into a scalar register.
  - Example: LOADI SC0, 10 (SC0 = 10)

- LOAD_TENSOR (Load Tensor from Host/Global Memory):
  - Syntax: LOAD_TENSOR Dest_Reg, Mem_Address, Dim0, Dim1, ...
  - Description: Loads a tensor from the host's global memory into a TPU memory bank and assigns a register to point to it. (Simulated by copying data).
  - Example: LOAD_TENSOR R0, IMEM_ADDR_0, 64, 128 (R0 points to a 64x128 tensor at IMEM_ADDR_0)

- STORE_TENSOR (Store Tensor to Host/Global Memory):
  - Syntax: STORE_TENSOR Src_Reg, Mem_Address
  - Description: Stores a tensor from a TPU memory bank back to the host's global memory.
  - Example: STORE_TENSOR R5, OMEM_ADDR_0

- ALLOC_TENSOR (Allocate Tensor in TPU Memory):
  - Syntax: ALLOC_TENSOR Dest_Reg, Dim0, Dim1, ...
  - Description: Allocates space for a new tensor in scratchpad/output memory and assigns a register to it.
  - Example: ALLOC_TENSOR R3, 64, 128 (Allocates a 64x128 tensor in SMEM for R3)

B. Tensor Operations (Core):
- MATMUL (Matrix Multiplication):
  - Syntax: MATMUL Dest_Reg, A_Reg, B_Reg
  - Description: Performs Dest=AtimesB. Assumes compatible dimensions. Efficiently executed on the systolic array.
  - Example: MATMUL R2, R0, R1 (R2 = R0 * R1)

- ADD_ELEM (Element-wise Addition):
  - Syntax: ADD_ELEM Dest_Reg, A_Reg, B_Reg
  - Description: Performs element-wise addition Dest=A+B. Assumes same dimensions.
  - Example: ADD_ELEM R4, R2, R3

- RELU (Rectified Linear Unit):
  - Syntax: RELU Dest_Reg, Src_Reg
  - Description: Applies ReLU activation Dest=max(0,Src) element-wise.
  - Example: RELU R5, R4

C. Control Flow (Simplified):
- HALT:
  - Syntax: HALT
  - Description: Stops execution.
