# hlda-cv

Harmonic Linear Discriminant Analysis utilities.

## Install

```bash
pip install "hlda-cv @ git+https://github.com/MendelsResearchGroup/hlda-cv.git"
```

## Usage

`fit_hlda(...)` computes an HLDA collective variable from descriptor samples for
two states.

Required inputs:

- `X_A`: descriptor values for state A, with shape
  `(number of samples, number of descriptors)`.
- `X_B`: descriptor values for state B, with shape
  `(number of samples, number of descriptors)`. It must have the same number of
  descriptors as `X_A`.
- `desc_cols`: descriptor names, in the same order as the columns in `X_A` and
  `X_B`.

Minimal example:

```python
from hlda_cv import fit_hlda

state_a_descriptors = [
    [0.0, 1.0],
    [0.1, 1.1],
    [0.2, 0.9],
]
state_b_descriptors = [
    [1.0, 2.0],
    [1.1, 2.1],
    [0.9, 1.9],
]
descriptor_names = ["d1", "d2"]

weights, eigenvalue = fit_hlda(
    X_A=state_a_descriptors,
    X_B=state_b_descriptors,
    desc_cols=descriptor_names,
)
```

Optional inputs:

- `prune_threshold`: correlation threshold for pruning similar descriptors. Must
  be between `0` and `1`. Leave unset to disable pruning.
- `correlation_method`: pruning method, either `"spearman"` or `"pearson"`.
  Defaults to `"spearman"`.
- `include_pruned_weights`: when `True`, also return weights for descriptors
  removed by pruning. Defaults to `False`.
- `ridge`: non-negative diagonal regularization added to each covariance matrix
  before inversion. Defaults to `1e-8`.

Returns:

- `weights`: `pandas.Series` of HLDA eigenvector weights indexed by descriptor
  name. If pruning is enabled, this contains only retained descriptors.
- `eigenvalue`: float eigenvalue for the selected HLDA direction.
- `full_weights`: returned only when `include_pruned_weights=True` and
  `prune_threshold` is set. This contains weights for all original descriptors.

## Chignolin peptide example

The repository includes a compact real peptide example for chignolin and two
mutants. The data files are available to download in `examples/data/`. Each
pickle contains distance descriptors for folded and unfolded states.

```python
import pickle
from pathlib import Path

from hlda_cv import fit_hlda

with Path("examples/data/WT.pkl").open("rb") as handle:
    sample = pickle.load(handle)

weights, eigenvalue, full_weights = fit_hlda(
    X_A=sample["folded"],
    X_B=sample["unfolded"],
    desc_cols=sample["desc_cols"],
    prune_threshold=0.93,
    correlation_method="spearman",
    include_pruned_weights=True,
    ridge=1e-8,
)

print(f"eigenvalue = {eigenvalue:.6f}")
print(weights.head())
print(f"full descriptor count = {len(full_weights)}")
```

Expected output starts with:

```text
eigenvalue = 12042.986742
d03    120.102451
d04    -94.681273
d05     39.896318
d06    -84.115497
d09     21.472177
dtype: float64
full descriptor count = 72
```

For workflows that already compute state means and covariance matrices, use
`hlda_from_moments(...)` directly.

You can also run the full peptide example script:

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
