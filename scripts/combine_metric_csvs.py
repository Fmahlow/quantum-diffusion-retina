from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path

import pandas as pd


DEFAULT_KEYS = ["Model", "Generator_Label", "Real_Label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine metric CSV files into one table.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/combined_metrics.csv"))
    return parser.parse_args()


def merge_two(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = [key for key in DEFAULT_KEYS if key in left.columns and key in right.columns]
    if not keys:
        raise ValueError("No common merge keys found. Expected Model, Generator_Label, and/or Real_Label.")
    return left.merge(right, on=keys, how="outer", suffixes=("", "_dup"))


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_columns = [column for column in df.columns if column.endswith("_dup")]
    return df.drop(columns=duplicate_columns)


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(path) for path in args.inputs]
    combined = reduce(merge_two, frames)
    combined = drop_duplicate_columns(combined)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()

