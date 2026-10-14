# Customer Purchase Pattern & Profile Analyzer

An end-to-end Python framework implementing a sequential Linear Algebra and Vector Calculus data pipeline to analyze customer purchase patterns, uncover latent behavioral dimensions, calculate customer profile similarity, and reduce dimensional space via orthogonal projections.

---

## Computational Pipeline Architecture

```
[ Raw Purchase Data (CSV / JSON) ]
                 │
                 ▼
┌─────────────────────────────────┐
│ Stage 1: Data Ingestion Module  │ ──► Dense Matrix A ∈ ℝ^(m × n), L2 Normalized, Mean Centered
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Stage 2: Rank & Basis Analyzer  │ ──► RREF, Matrix Rank r, Column/Null Space Bases
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Stage 3: Vector Similarity      │ ──► Inner Product Gram Matrix G, Cosine Similarity S
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Stage 4: Spectral / PCA Engine  │ ──► Covariance C, Eigenvalues λ, Eigenvectors V
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Stage 5: Subspace Projection    │ ──► Orthogonal Projection P, Reduced Matrix Y ∈ ℝ^(m × k)
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Stage 6: Profile Reconstruction │ ──► Archetype Profiles & Centroid Interpretations
└─────────────────────────────────┘
```

---

## Stage 1: Data Ingestion & Matrix Formulation

The first module (`src.ingestion`) provides:
1. **Synthetic Data Generator with Controllable Rank**:
   - Creates a synthetic customer-category matrix $A \in \mathbb{R}^{m \times n}$ with a strictly controllable linear rank $r < \min(m, n)$.
   - Constructed via low-rank matrix decomposition:
     $$A = W \cdot H$$
     where $W \in \mathbb{R}^{m \times r}$ represents customer archetype mixture loadings and $H \in \mathbb{R}^{r \times n}$ represents category archetype bases.
   - Guaranteed $\text{rank}(A) = r$ when noise is zero, with singular values $s_{r+1} = \dots = s_{\min(m, n)} = 0$.
2. **Robust Ingestion & Imputation**:
   - Ingests tabular data from CSV and JSON files (or buffers and DataFrames).
   - Handles missing values via configurable imputation: `zero`, `mean`, `median`, `mode`.
   - Formulates a dense NumPy Matrix $A_{m \times n}$ with float64 precision.
3. **Mathematical Normalization**:
   - **L2 Normalization**: Row-wise normalization $\mathbf{a}_i / \|\mathbf{a}_i\|_2$ projecting customer vectors onto the unit hypersphere for directional cosine similarity.
   - **Mean Centering**: Column-wise centering $X = A - \mathbf{1}\boldsymbol{\mu}^T$ where $\boldsymbol{\mu} = \frac{1}{m} \sum_{i=1}^m \mathbf{a}_i$, preparing data for sample covariance matrix construction $C = \frac{1}{m-1} X^T X$.

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested with Python 3.13 / 3.14)
- Virtual environment (recommended)

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Running the Stage 1 Pipeline

Execute the demonstration script:

```bash
python main.py
```

### Running Tests

Execute the automated test suite with pytest:

```bash
pytest tests/ -v
```
