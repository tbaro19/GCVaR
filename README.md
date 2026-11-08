# GCVaR: Exact Tail-Risk Evaluation and Stochastic Optimization

This repository provides the reference implementation and experimental reproducibility suite for Geometric Conditional Value-at-Risk (GCVaR) and planar skyband core screening in risk-averse stochastic optimization.

The codebase implements exact geometric screening algorithms, mathematical formulations for extensive and cutting-plane optimization models, and query-streaming baselines on both synthetic factor distributions and real-world operational datasets.

---

## 1. Repository Structure

```text
GCVaR/
│
├── src/                  # Core algorithms and optimization modules
│   ├── core.py           # O(N log N) planar 2D skyband screening via Fenwick tree
│   ├── exact.py          # Arbitrary-precision rational arithmetic (Fraction) for Top-k and CVaR
│   ├── optimization.py   # HiGHS LP models: Extensive Form (EF-F, EF-C) & Outer Approximation (OA)
│   ├── oracles.py        # Query streaming baselines: FULL-SELECT, SKYBAND-SELECT, SORTED-LAZY
│   ├── onion.py          # Planar convex onion layers and directional extreme search
│   ├── kinetic.py        # Exact rational kinetic spectral risk profiles
│   ├── synthetic.py      # Bivariate scenario generators (Gamma, Lognormal, Copula, Student-t)
│   ├── datasets.py       # Operational dataset adapters and chronological splitting
│   └── __init__.py       # Package initialization
│
├── tests/                # Automated verification and mathematical certification
│   └── test_exactness.py # Certification test suite (20 tests, >10,500 fuzzing instances)
│
├── data/                 # Directory for operational benchmark datasets
│
├── download_data.py      # Automated data acquisition script for public benchmark datasets
├── run_reproduce.py      # Main benchmark runner for optimization and retrieval workflows
└── requirements.txt      # Dependency specification
```

---

## 2. Requirements & Setup

### Environment
- **Python**: Version 3.9 or newer (tested on Python 3.10, 3.11, 3.12, 3.13).
- **Core Dependencies**:
  - `numpy` (>= 1.23.0): Numerical matrix operations.
  - `scipy` (>= 1.10.0): HiGHS linear programming solver backend (`scipy.optimize.linprog`).
  - `pandas` (>= 1.5.0): Time-series data manipulation and split management.
  - `pytest` (>= 7.0.0): Automated testing harness.

### Installation
Install all required dependencies via `pip`:

```bash
pip install -r requirements.txt
```

---

## 3. Data Acquisition (Benchmark Datasets)

Synthetic factor distributions are generated algorithmically on-the-fly and require no external downloads.

To evaluate real-world operational benchmarks, download and extract the public datasets using either automated or manual methods below:

### Automated Download
Run the included data acquisition utility to download and extract datasets into `data/`:

```bash
python download_data.py
```

To download a specific dataset only:
```bash
python download_data.py --dataset bike
python download_data.py --dataset tetouan
python download_data.py --dataset electricity
```

### Manual Download & Placement
Alternatively, download the official archives directly from the public UCI Machine Learning Repository:

1. **UCI Bike Sharing Dataset (ID 275)**:
   - Source URL: `https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip`
   - Target location: `data/bike/hour.csv`

2. **Power Consumption of Tetouan City (ID 849)**:
   - Source URL: `https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip`
   - Target location: `data/tetouan/Tetuan City power consumption.csv`

3. **Electricity Load Diagrams 2011–2014 (ID 321)**:
   - Source URL: `https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip`
   - Target location: `data/electricity/LD2011_2014.txt`

---

## 4. Mathematical Verification Suite

To verify the mathematical exactness of the algorithms—including rational arithmetic correctness, degeneracy fuzzing, cone invariance, and equivalence between the full and screened optimization formulations:

```bash
python tests/test_exactness.py
```

Alternatively, run via `pytest`:

```bash
pytest tests/test_exactness.py -v
```

This suite executes 20 rigorous tests covering:
- Exhaustive fuzz testing (>10,500 trials) across arbitrary slope vectors, ties, and duplicate scenarios.
- Extreme geometry tests (monotone chains, antichains, and degeneracies).
- Cone invariance and out-of-cone ineligibility certification.
- Rational kinetic spectral profiles and directional onion traversals.
- Zero optimality gap between the full Extensive Formulation (EF-F) and the Skyband Core Formulation (EF-C).

---

## 5. Reproducing Empirical Benchmarks

The master runner `run_reproduce.py` executes end-to-end experiments, evaluates scenario reduction rates, measures wall-clock solve times, and records solution metrics.

### A. Quick Validation Run (~10–15 seconds)
Validates the execution pipeline across synthetic configurations and operational data subsets:

```bash
python run_reproduce.py --mode smoke
```

### B. Full Benchmark Run
Executes the comprehensive evaluation grid across all synthetic and operational benchmark instances:

```bash
python run_reproduce.py
```

### C. Selective Evaluation by Dataset
Target specific benchmark datasets independently:

- **Synthetic instances only:**
  ```bash
  python run_reproduce.py --dataset synthetic
  ```

- **UCI Bike Sharing only:**
  ```bash
  python run_reproduce.py --dataset bike
  ```

- **Tetouan City Power Consumption only:**
  ```bash
  python run_reproduce.py --dataset tetouan
  ```

- **Include Outer Approximation (OA-F and OA-C):**
  ```bash
  python run_reproduce.py --include-oa
  ```

### Generated Artifacts
When executed, output files are created in the `results/` directory:
- `results/optimization_reproduce.csv`: Per-instance solve times, optimality gaps, constraint dimensions, iterations, and objective values.
- `results/query_stream_reproduce.csv`: Query batch throughput, amortized retrieval speedups, and exactness verification percentages.

---

## 6. Module Descriptions

| Module | Primary Role |
| :--- | :--- |
| **`src/core.py`** | Implements the $O(N \log N)$ planar $k$-skyband screening algorithm using coordinate discretization and a Fenwick tree. Preserves original scenario identifiers and multiplicities. |
| **`src/exact.py`** | Provides exact reference implementations using Python's `fractions.Fraction` for fractional CVaR calculation, top-$k$ selection, and supporting cut separation. |
| **`src/optimization.py`** | Constructs sparse linear programs for CVaR-constrained stochastic optimization under both Extensive Formulations (EF-F, EF-C) and Outer Approximation cutting-plane algorithms (OA-F, OA-C), solved via HiGHS. |
| **`src/oracles.py`** | Implements query streaming baselines, including linear scan partition (`full_select`), full sorting (`full_sort`), component-wise unions (`component_union`), and lazy stream merging (`sorted_lazy`). |
| **`src/onion.py`** | Builds convex hull layer structures (Onion Index) with counter-clockwise extreme vertex tracking and directional binary search. |
| **`src/kinetic.py`** | Models continuous rational kinetic spectral profiles under parametric trajectory changes, handling simultaneous topological events and permanent coincidences. |
| **`src/synthetic.py`** | Generates bivariate synthetic uncertainty clouds under varying dependency structures (independent Gamma, correlated lognormal $\rho = \pm 0.9$, heavy-tailed Student-t, copula mixtures, chains, and antichains). |
| **`src/datasets.py`** | Deterministic data ingestion adapters for UCI operational datasets, enforcing chronological 60/20/20 train/validation/holdout splits without out-of-sample data leakage. |
| **`download_data.py`** | Automated data acquisition utility downloading and extracting public benchmark datasets into standard directory structures. |
| **`run_reproduce.py`** | Top-level benchmark execution script coordinating end-to-end reproducibility pipelines. |
| **`tests/test_exactness.py`** | Comprehensive certification test suite verifying theoretical and numerical exactness. |
