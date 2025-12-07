# Symbolic Code Generation & Kernel Fusion Plan

## 1. Mathematical Analysis & Speedup Calculation

This section analyzes the theoretical performance benefits of replacing the current "Vectorized + Sparse Matrix" approach with a "Hardcoded Fused Kernel" approach, specifically targeting CPU architectures (AVX2/AVX-512).

### 1.1 Current Implementation (Vectorized + Sparse Matmul)

The current `carbox` ODE evaluation `dy/dt = f(y)` consists of three distinct passes:

1.  **Rate Evaluation:** $R_k = k(T) \cdot \prod y_j^{\nu_{jk}}$
    *   Input: State $y$ ($N_{spec}$), Constants $K$ ($N_{rxn} \times 3$).
    *   Output: Intermediate array `rates` ($N_{rxn}$).
    *   Memory: Writes $N_{rxn}$ floats to memory (L2/L3 cache).
2.  **Rate-Abundance Product:**
    *   Input: `rates`, $y$.
    *   Memory: Reads `rates`, Reads $y$ (scattered access via `reactant_multipliers`), Writes modified `rates`.
3.  **Sparse Matrix Multiplication:** $dy = S \times R$
    *   Input: Sparse Matrix $S$ (Indices + Values), `rates`.
    *   Operation: `dy[i] += S[i, k] * rates[k]`
    *   Memory: Reads `rates` ($N_{rxn}$), Reads Sparse Indices ($2 \times N_{nonzero\_S}$), Writes $dy$ ($N_{spec}$).

**Arithmetic Intensity (AI):** Low.
The sparse matrix multiplication is memory-bound. For every non-zero element in the stoichiometry matrix, we perform 1 FMA (Fused Multiply-Add) but fetch 1 float (rate) + 1 integer index.

**Memory Traffic Estimate (per step):**
*   $N_{rxn} \approx 1500$, $N_{spec} \approx 160$.
*   Intermediate `rates` array: $1500 \times 4$ bytes = 6 KB (Fits in L1, but forces load/store traffic).
*   Sparse Indices: $N_{nonzero} \approx 4 \times N_{rxn} \approx 6000$ integers = 24 KB.
*   **Total Data Moved:** ~30 KB per substep.

### 1.2 Proposed Fused Kernel (Hardcoded ODE)

Instead of generic matrix operations, we generate a specific Python function for the exact chemical network at hand.

$$ \frac{dy_i}{dt} = \sum_{k \in \text{reactions producing } i} R_k(y) - \sum_{k \in \text{reactions consuming } i} R_k(y) $$

The generated code looks like:
```python
def fused_ode(y, T, ...):
    # Load frequently used species into registers (compiler optimization)
    y0 = y[0]; y1 = y[1]; ...

    # Compute rates (temporaries in registers, no array write)
    r0 = 1.3e-10 * y1 * y4
    r1 = 4.5e-11 * y2 * y8 * T_pow_05
    ...

    # Accumulate derivatives (register accumulation)
    dy0 = -r0 + r5
    dy1 = -r0 - r1 + r8
    ...
    return jnp.array([dy0, dy1, ...])
```

**Theoretical Speedup Factors:**

1.  **Elimination of Memory Traffic (Infinite AI):**
    *   The intermediate `rates` array is **never written to memory**. It exists only in CPU registers (YMM/ZMM).
    *   The sparse matrix indices are **baked into the instruction stream**. We don't load "row index 5, col index 10"; the code just says `dy[5] += ...`.
    *   **Traffic Reduction:** Reduces memory bandwidth usage by ~60-80%.

2.  **Instruction Level Parallelism (ILP):**
    *   The current `vmap` approach forces the CPU to process reactions in batches of the same type (Arrhenius vs. Cosmic Ray).
    *   The fused approach allows the compiler (XLA/LLVM) to interleave instructions from different reaction types to maximize pipeline utilization.

3.  **Constant Folding:**
    *   Reaction coefficients (alpha, beta, gamma) are currently loaded from arrays.
    *   In the fused kernel, these become immediate values in the assembly code (e.g., `VMULPS zmm0, zmm1, [rip+const_val]`), reducing register pressure and load slots.

**Estimated CPU Gain:**
*   Memory Bandwidth: **3x - 4x** improvement (removing sparse indices + intermediate arrays).
*   Instruction Count: **1.5x** improvement (removing loop overhead and index calculation).
*   **Total Estimated Speedup:** **2x - 5x** for the ODE evaluation function.

---

## 2. Implementation Plan

We will build a `CodeGenerator` class that takes a `carbox.Network` and emits a string containing valid JAX code.

### Phase 1: The Transpiler (`carbox/codegen.py`)

**Goal:** Create a class that iterates over the network and produces a string.

1.  **Header Generation:**
    *   Imports (`jax`, `jax.numpy`).
    *   Function signature `def static_ode(t, y, args):`.

2.  **Rate Unrolling:**
    *   Iterate through `network.reactions`.
    *   For each reaction $k$, generate a line of code:
        `r{k} = {rate_constant_expression} * y[{reactant1_idx}] * y[{reactant2_idx}]`
    *   *Optimization:* Pre-calculate temperature-dependent terms if T is constant over the step.

3.  **Derivative Accumulation:**
    *   Initialize `dy = [0.0] * n_species`.
    *   Iterate through `network.reactions` again (or use the pre-computed incidence).
    *   For reaction $k$ consuming $S_i$ and producing $S_j$:
        *   Add string `dy[{i}] -= r{k}`
        *   Add string `dy[{j}] += r{k}`

4.  **Return Statement:**
    *   `return jnp.array(dy)`

### Phase 2: JAX Integration

**Goal:** Compile the string into a callable function.

1.  **Dynamic Execution:**
    *   Use Python's `exec()` to compile the string into a function object within a local namespace.
    *   Pass this function to `jax.jit`.

2.  **Replacement:**
    *   Add a method `Network.get_fused_ode()` that returns this JIT-compiled function.
    *   The solver will use this function instead of `network.get_ode()`.

### Phase 3: Validation & Benchmarking

1.  **Correctness:**
    *   Compare `fused_ode(y, t)` vs `original_ode(y, t)` for random state vectors.
    *   Ensure relative error is within float32/float64 epsilon.

2.  **Benchmarks:**
    *   Run `timeit` on both versions for N=163 (UMIST) and N=500+ networks.
    *   Measure compilation time (Fused kernel will take significantly longer to compile, which is a trade-off).

### Code Structure Preview

```python
class CodeGenerator:
    def __init__(self, network: Network):
        self.network = network

    def generate_rate_expr(self, reaction, idx):
        # Returns string like "r5 = 1.5e-10 * y[2] * y[5]"
        pass

    def generate_code(self) -> str:
        # Orchestrates the full string generation
        pass

def compile_network(network: Network):
    code = CodeGenerator(network).generate_code()
    scope = {}
    exec(code, globals(), scope)
    return jax.jit(scope['fused_ode'])
```

## 3. Risks & Mitigations

1.  **Compilation Time:**
    *   **Risk:** For very large networks (N > 2000), the generated Python function might be 10,000+ lines long. JAX/XLA might take minutes to compile this.
    *   **Mitigation:** This approach is best for "production runs" where the network is fixed and run for millions of steps. For interactive exploration, use the old vectorized method.

2.  **Code Size Limit:**
    *   **Risk:** Python has limits on function size (bytecode size).
    *   **Mitigation:** If N is too large, we may need to chunk the function into sub-functions (e.g., `calc_rates_batch_1`, `calc_rates_batch_2`).

3.  **Precision:**
    *   **Risk:** Summation order changes.
    *   **Mitigation:** The fused approach actually usually *increases* precision because we can use Kahan summation or just rely on registers being 80-bit extended precision (on some architectures, though less relevant for AVX).
