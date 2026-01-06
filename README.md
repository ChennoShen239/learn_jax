## learn_jax

This repo is a scratchpad for doing **quantitative macro** with **JAX**, where:

- **state space** can be big,  
- **value / policy iteration** is embarrassingly parallel, and  
- we want everything **JIT-compiled on the metal** instead of crawling in pure Python.

### What lives here?

- **`main.py`**: quick experiments / playground.
- **`aiyagari_jax.*`**: Aiyagari-style heterogeneous-agent model in JAX  
  - vectorized Bellman / EGM style operations  
  - `jit`-compiled stationary equilibrium search
- **`VFI.ipynb`**, **`opt_savings_2.ipynb`**, **`OPI_HPI.ipynb`**, etc.: assorted notebooks for
  - value function iteration benchmarks,
  - optimal savings problems,
  - playing with JAX autodiff and linear algebra for macro.

### Tech stack (roughly)

- **Python + JAX**: `jit`, `vmap`, `grad` for fast numerical macro
- **Quarto + Jupyter**: literate programming / notes (`*.qmd`, `*.ipynb`)

### Getting started

```bash
# (optional) create a venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install dependencies (uv or pip)
uv sync  # if you use uv
# or
pip install -e .
```

Then open any notebook (e.g. `aiyagari_jax.ipynb`) and run the cells, or run:

```bash
python main.py
```

### Why JAX for macro?

- **XLA-compiled** numerical kernels instead of Python loops  
- Cheap **autodiff** for gradients / Jacobians in macro models  
- Easy **GPU/TPU** offloading if things get ridiculous

If something breaks, it's probably because the model got **too big, too non-linear, or too clever**. In that case, reduce the state space, re-run, and pretend it was a deliberate robustness check.