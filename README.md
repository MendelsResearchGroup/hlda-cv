# hlda-cv

Harmonic Linear Discriminant Analysis utilities.

Inputs:
- `X_A, X_B`: Trajectory of descriptors for state A and B
- `desc_cols`: descriptor names
- `prune_threshold`: optional pruning threshold between `0` and `1`
- `correlation_method`: optional pruning correlation, `"spearman"` or `"pearson"`

Outputs:
- `weights`: HLDA eigenvector weights
- `eigenvalue`: HLDA eigenvalue
- `full_weights`: optional weights for all descriptors when pruning is enabled

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
    correlation_method="spearman",
    include_pruned_weights=True,
)

print(eigenvalue)
print(weights)
print(full_weights)
```

For workflows that already compute state means and covariance matrices, use
`hlda_from_moments(...)` directly.

Pruning uses Spearman correlation by default. Pearson correlation is also
supported by setting `correlation_method="pearson"`. When pruning is enabled,
`prune_threshold` must be between `0` and `1`.

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
tiny toy system, that descriptor pruning removes redundant variables, and that
pruning validates the threshold range.
