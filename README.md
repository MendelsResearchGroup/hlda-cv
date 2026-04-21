# hlda-cv

Minimal reusable Harmonic Linear Discriminant Analysis utilities.

## Install

```bash
pip install "hlda-cv @ git+https://github.com/<owner>/hlda-cv.git"
```

## Usage

```python
from hlda_cv import fit_hlda

result = fit_hlda(
    X_A=state_a_descriptors,
    X_B=state_b_descriptors,
    desc_cols=descriptor_names,
    prune_threshold=0.93,
    include_pruned_weights=True,
)

print(result.eigenvalue)
print(result.weights)
print(result.full_weights)
```

For workflows that already compute state means and covariance matrices, use
`hlda_from_moments(...)` directly.
