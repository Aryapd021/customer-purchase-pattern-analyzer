# Customer Purchase Pattern & Profile Analyzer

An end-to-end Python framework implementing a strictly sequential **Linear Algebra & Vector Calculus** computational pipeline to uncover latent customer purchasing preferences, evaluate pairwise similarity, reduce dimensional space via orthogonal projections, and synthesize behavioral customer archetypes.

---

## Computational Pipeline Architecture

```
[ Raw Purchase Data (CSV / JSON / Synthetic) ]
                      │
                      ▼
┌───────────────────────────────────────────┐
│ Stage 1: Data Ingestion & Matrix Setup    │ ──► Dense Matrix A ∈ ℝ^(m × n)
└───────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│ Stage 2: Rank & Fundamental Subspaces     │ ──► RREF, Rank r, Bases for R(A), C(A), N(A)
└───────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│ Stage 3: Vector Similarity Engine         │ ──► Gram Matrix G = A*A^T, Cosine Matrix S
└───────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│ Stage 4: Covariance & Spectral PCA        │ ──► Covariance C, Eigenvalues λ, Eigenvectors V
└───────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│ Stage 5: Subspace Orthogonal Projection   │ ──► Operator P = V_k*V_k^T, Coordinates Y = X*V_k
└───────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│ Stage 6: Profile Formation & Archetypes   │ ──► Centroid Re-projection c_orig, Radar Profiles
└───────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│ Stage 7: Verification Engine & Scipy Comp │ ──► Benchmark A_test, Identities, scipy.linalg
└───────────────────────────────────────────┘
```

---

## Key Pipeline Modules & Mathematical Formulations

### Stage 1: Data Ingestion & Matrix Setup
* **Controllable-Rank Synthetic Generator**: Creates matrix $A \in \mathbb{R}^{m \times n}$ with exact linear rank $r < \min(m, n)$ via low-rank factorization $A = W \cdot H$.
* **Robust Imputation**: Configurable `zero`, `mean`, `median`, and `mode` strategies.
* **Normalization**:
  - **L2 Normalization**: $\mathbf{a}_i / \|\mathbf{a}_i\|_2$ maps customer vectors onto the unit hypersphere $S^{n-1}$.
  - **Mean Centering**: $X = A - \mathbf{1}\boldsymbol{\mu}^T$ where $\boldsymbol{\mu} = \frac{1}{m} \sum \mathbf{a}_i$.

### Stage 2: Matrix Rank & Basis Extraction
* **Custom RREF**: Gaussian Elimination with Partial Pivoting.
* **Rank-Nullity Theorem**: $\dim \mathcal{C}(A) + \dim \mathcal{N}(A) = n$.
* **Subspace Bases**:
  - $\mathcal{R}(A)$: Non-zero rows of $\text{RREF}(A)$.
  - $\mathcal{C}(A)$: Pivot columns of the original matrix $A$.
  - $\mathcal{N}(A)$: Parameterization of free variables satisfying $A \mathbf{v} = \mathbf{0}$.

### Stage 3: Vector Inner Product & Cosine Similarity
* **Inner Product**: $\langle \mathbf{a}_i, \mathbf{a}_j \rangle = \sum_{k=1}^n A_{ik} A_{jk}$.
* **Gram Matrix**: $G = A A^T \in \mathbb{R}^{m \times m}$.
* **Cosine Similarity**: $S = D^{-1} G D^{-1}$ where $D = \operatorname{diag}(\|\mathbf{a}_1\|_2, \dots, \|\mathbf{a}_m\|_2)$.

### Stage 4: Covariance & Spectral Decomposition (PCA)
* **Sample Covariance**: $C = \frac{1}{m - 1} X^T X \in \mathbb{R}^{n \times n}$.
* **First-Principles Solvers**:
  - Householder Tridiagonalization with Wilkinson Shift and Submatrix Deflation (QR algorithm).
  - Cyclic Jacobi Eigenvalue Algorithm (Givens similarity rotations).
  - Cardano trigonometric cubic polynomial solver for $n \le 3$.
* **Variance Retention**: Cumulative variance ratio $R(k) = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^n \lambda_j}$.

### Stage 5: Subspace Orthogonal Projection
* **Projection Matrix**: $P = V_k V_k^T \in \mathbb{R}^{n \times n}$ ($P^T = P$, $P^2 = P$).
* **Subspace Coordinates**: $Y = X V_k \in \mathbb{R}^{m \times k}$.
* **Reconstruction**: $\hat{X} = Y V_k^T = X P$, with relative information loss $\mathcal{L}(k) = 1 - R(k)$.

### Stage 6: Profile Formation & Archetype Extraction
* **Subspace Proximity**: Euclidean distance clustering $d(\mathbf{y}_i, \mathbf{y}_j) = \|\mathbf{y}_i - \mathbf{y}_j\|_2$.
* **Centroid Re-projection**: $\mathbf{c}_{\text{orig}, g} = \mathbf{c}_{\text{proj}, g} V_k^T + \boldsymbol{\mu} \in \mathbb{R}^n$.
* **Behavioral Synthesis**: Category lift ranking and archetype profile labeling.

### Stage 7: Mathematical Verification Engine
* **Hardcoded Benchmark Matrix $A_{\text{test}}$**: Validates Row 3 = 2 * Row 1, Rank 2, Pivot Columns [0, 1], RREF, Norms, Inner Product 50.0, Cosine Similarity 1.0.
* **Theoretical Identities**: $\operatorname{Trace}(C) = \sum \lambda_i$, $\det(C) = \prod \lambda_i$, $V^T V = I$, $C \mathbf{v}_k = \lambda_k \mathbf{v}_k$.
* **Scipy Validation**: Numerical stability check vs `scipy.linalg` reference routines.

---

## Installation & Setup

```bash
# Clone and enter directory
cd splendid-heisenberg

# Setup virtual environment (Python 3.10+)
python -m venv .venv

# Windows activation
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage: CLI & Interactive Dashboard

### 1. Run Complete Sequential Pipeline (CLI Mode)
Executes all stages sequentially, prints rich mathematical tables, validates against benchmarks, and renders plots:
```bash
python main.py
```

### 2. Launch Interactive Streamlit Presentation Dashboard
Launches the full interactive academic presentation dashboard with LaTeX formulas, 2D/3D Plotly visualizers, and Viva Voce defense mode:
```bash
streamlit run app.py
```
*(Or run `python main.py --ui`)*

---

## Automated Verification Suite

Run all 43 automated unit and theoretical identity tests:
```bash
pytest -v
```
