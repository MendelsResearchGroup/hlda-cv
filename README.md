# hlda-cv

Harmonic Linear Discriminant Analysis utilities.

## Install

```bash
pip install "hlda-cv @ git+https://github.com/MendelsResearchGroup/hlda-cv.git"
```

## Usage

`fit_hlda(...)` computes an HLDA collective variable from descriptor samples for
two states.

Parameters:

- `X_A`: sample-by-descriptor matrix for state A. Accepts a nested Python list or
  a NumPy array with shape `(n_samples_A, n_descriptors)`.
- `X_B`: sample-by-descriptor matrix for state B. Accepts a nested Python list or
  a NumPy array with shape `(n_samples_B, n_descriptors)`. It must have the same
  number of descriptor columns as `X_A`.
- `desc_cols`: descriptor names in the same column order as `X_A` and `X_B`.
  The length must equal the number of descriptor columns.
- `prune_threshold`: optional float between `0` and `1`. When set, descriptors
  with absolute correlation greater than this threshold are pruned before HLDA is
  fitted. Use `None` to disable pruning.
- `correlation_method`: correlation method used only when `prune_threshold` is
  set. Supported values are `"spearman"` and `"pearson"`. Defaults to
  `"spearman"`.
- `include_pruned_weights`: when `False`, return weights only for descriptors
  retained after pruning. When `True` and pruning is enabled, also return a
  dictionary that maps every original descriptor name to a weight.
- `ridge`: non-negative diagonal regularization added to each covariance matrix
  before inversion. Defaults to `1e-8`.

Returns:

- `weights`: `pandas.Series` of HLDA eigenvector weights indexed by descriptor
  name. If pruning is enabled, this contains only retained descriptors.
- `eigenvalue`: float eigenvalue for the selected HLDA direction.
- `full_weights`: returned only when `include_pruned_weights=True` and
  `prune_threshold` is set. This is a dictionary containing weights for all
  original descriptors, including descriptors removed during pruning.

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

Pruning uses Spearman correlation by default. Pearson correlation is also
supported by setting `correlation_method="pearson"`. When pruning is enabled,
`prune_threshold` must be between `0` and `1`.

## Example

The repository includes a compact example dataset for chignolin and two mutants.
The data files are available to download in `examples/data/`. Each pickle
contains only distance descriptors for folded and unfolded states.

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
