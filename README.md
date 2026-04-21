# hlda-cv

Harmonic Linear Discriminant Analysis utilities.

Input data:
- `X_A`: `list[list[float]]` with one descriptor row per sample in state A
- `X_B`: `list[list[float]]` with one descriptor row per sample in state B
- `desc_cols`: `list[str]` with descriptor names in column order

## Install

```bash
pip install "hlda-cv @ git+https://github.com/MendelsResearchGroup/hlda-cv.git"
```

## Usage

```python
from hlda_cv import fit_hlda

weights, eigenvalue, full_weights = fit_hlda(
    X_A=state_a_descriptors,
    X_B=state_b_descriptors,
    desc_cols=descriptor_names,
    prune_threshold=0.93,
    include_pruned_weights=True,
)

print(eigenvalue)
print(weights)
print(full_weights)
```

For workflows that already compute state means and covariance matrices, use
`hlda_from_moments(...)` directly.

## Example

The repository includes a compact example dataset for chignolin and two mutants.
Each pickle contains only distance descriptors for folded and unfolded states.

Run the example with:

```bash
PYTHONPATH=src python examples/run_peptide_example.py
```

## Tests

Run the synthetic validation tests with:

```bash
PYTHONPATH=src pytest tests
```

The test suite checks that HLDA prefers the obvious separating direction on a
tiny toy system and that descriptor pruning removes redundant variables.
