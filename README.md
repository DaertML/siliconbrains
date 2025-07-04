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
TODO
