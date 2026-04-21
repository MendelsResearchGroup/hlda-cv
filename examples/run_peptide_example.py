from __future__ import annotations

import pickle
from pathlib import Path

from hlda_cv import fit_hlda


DATA_DIR = Path(__file__).resolve().parent / "data"
MUTANTS = ["WT", "T7D", "Y0A"]


def main() -> None:
    for mutant in MUTANTS:
        sample_path = DATA_DIR / f"{mutant}.pkl"
        with sample_path.open("rb") as handle:
            sample = pickle.load(handle)

        weights, eigenvalue = fit_hlda(
            X_A=sample["folded"],
            X_B=sample["unfolded"],
            desc_cols=sample["desc_cols"],
            prune_threshold=0.93,
        )

        print(f"{sample['mutant']}:")
        print(f"  eigenvalue = {eigenvalue:.6f}")
        print("  eigenvector weights:")
        for desc, weight in weights.items():
            print(f"    {desc}: {weight:.6f}")
        print()


if __name__ == "__main__":
    main()
