"""Print a ranked summary of v37-v60 experiments by pooled OOS R2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results/research")


def main() -> None:
    """Aggregate experiment CSVs into one ranked table."""
    rows: list[dict[str, object]] = []
    for csv_path in sorted(RESULTS_DIR.glob("v[3-6][0-9]_*results.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: {csv_path.name}: {exc}")
            continue

        if "benchmark" not in df.columns or "r2" not in df.columns:
            continue

        pooled = df[df["benchmark"] == "POOLED"]
        if pooled.empty:
            pooled = df.head(1)

        for _, row in pooled.iterrows():
            variant = row.get("variant", row.get("alpha", "default"))
            rows.append(
                {
                    "file": csv_path.name,
                    "variant": str(variant),
                    "r2": round(float(row["r2"]), 4),
                    "ic": round(float(row.get("ic", float("nan"))), 4),
                    "hit_rate": round(float(row.get("hit_rate", float("nan"))), 4),
                    "sigma_ratio": round(float(row.get("sigma_ratio", float("nan"))), 4),
                }
            )

    if not rows:
        print("No results CSVs found. Run experiments first.")
        return

    summary = pd.DataFrame(rows).sort_values("r2", ascending=False)
    print("\n" + "=" * 75)
    print("v37-v60 Experiment Results - Ranked by OOS R2")
    print("=" * 75)
    print(summary.to_string(index=False))
    print("=" * 75)
    best = summary.iloc[0]
    print(f"\nBest: {best['file']} / {best['variant']} -> R2={best['r2']:+.4f}")


if __name__ == "__main__":
    main()
