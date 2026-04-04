# ML Systems Accelerator Lab

A collection of ML systems projects exploring the **software stack behind modern AI accelerators**, including graph compilation, kernel generation, and accelerator runtime simulation.

These projects demonstrate how high-level machine learning workloads are **lowered, optimized, and executed** across different layers of an ML accelerator stack.

---

## Stack Overview

```
PyTorch Model
      |
      ▼
TensorDescent
(Graph Compiler)
      |
      ▼
KernelForge
(Kernel Lowering + Fusion)
      |
      ▼
AccelSim
(Runtime Simulation + Performance Analysis)
```

This repository acts as a **hub** connecting the individual projects that implement each layer of the stack.

---

## Projects

### [TensorDescent](https://github.com/pkarakala/FX2Accel)

A PyTorch-to-IR compiler pipeline that lowers, optimizes, and maps neural network computations toward hardware execution.

**Key ideas:**
- Graph IR representation
- Operator fusion
- Graph traversal and scheduling
- Lowering ML workloads into kernel operations

---

### [KernelForge](https://github.com/pkarakala/KernelForge)

A minimal Triton-style kernel compiler that implements kernel-level optimizations and code generation.

**Key ideas:**
- Intermediate representation (IR)
- Kernel fusion
- Tiling optimizations
- CUDA-style kernel code generation
- Execution simulation

---

### [AccelSim](https://github.com/pkarakala/AccelSim)

A lightweight neural accelerator simulator that models instruction execution, memory traffic, and cycle-level performance for compiler-generated tensor operation streams.

**Key ideas:**
- Runtime execution scheduling
- Simulated GPU-style execution
- Performance benchmarking
- Kernel performance analysis

---

## Motivation

Modern AI workloads require highly optimized software stacks to execute efficiently on specialized hardware such as GPUs, TPUs, and custom AI accelerators.

These projects explore the key stages involved in that stack:

- **Graph compilation** — transforming high-level ML models
- **Kernel lowering and optimization** — for accelerator execution
- **Runtime simulation** — for analyzing performance characteristics

Together they illustrate the core concepts behind ML systems infrastructure used in modern AI hardware platforms.

---

## Why This Matters

ML accelerator stacks are critical for scaling modern AI systems. Companies building these systems include NVIDIA, AMD, AWS (Annapurna Labs), Google (TPU), Cerebras, Lightmatter, and Etched.

Understanding the interaction between **compiler layers, kernel execution, and runtime behavior** is essential for building efficient AI hardware and infrastructure.

---

## Repository Structure

```
ml-systems-lab
├── README.md
└── assets
```

Individual projects live in their own repositories and are linked above.

---

## Future Work

- Scheduling optimizations
- Memory planning
- Auto-tuning kernel parameters
- Multi-device execution simulation
- Inference optimization passes

---

## Author

**Pranav Reddy**  
Electrical Engineering — UC Santa Barbara

Interested in: ML systems, AI accelerators, compiler infrastructure, high-performance computing, and hardware–software co-design.

GitHub: [github.com/pkarakala](https://github.com/pkarakala)
