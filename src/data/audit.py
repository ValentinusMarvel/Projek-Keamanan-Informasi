from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".json", ".xlsx", ".xls", ".pkl", ".pickle"}


@dataclass
class DatasetAuditResult:
    file_path: Path
    shape: tuple[int, int] | None
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    duplicate_rows: int | None
    head_preview: list[dict[str, Any]]


def discover_dataset_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_TABULAR_SUFFIXES
    )


def load_tabular_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        try:
            return pd.read_csv(file_path)
        except Exception:
            return pd.read_csv(file_path, sep="\t")
    if suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]
    return normalized


def summarize_frame(frame: pd.DataFrame, max_rows: int = 5) -> DatasetAuditResult:
    preview = frame.head(max_rows).replace({np.nan: None}).to_dict(orient="records")
    return DatasetAuditResult(
        file_path=Path(),
        shape=frame.shape,
        columns=list(frame.columns),
        dtypes={column: str(dtype) for column, dtype in frame.dtypes.items()},
        missing_values=frame.isna().sum().astype(int).to_dict(),
        duplicate_rows=int(frame.duplicated().sum()),
        head_preview=preview,
    )


def infer_candidate_columns(columns: list[str]) -> dict[str, list[str]]:
    lower_columns = [column.lower() for column in columns]

    def pick_by_keywords(keywords: list[str]) -> list[str]:
        return [column for column in lower_columns if any(keyword in column for keyword in keywords)]

    return {
        "user": pick_by_keywords(["user", "subject", "participant", "person", "id"]),
        "session": pick_by_keywords(["session", "sess", "trial", "block"]),
        "timestamp": pick_by_keywords(["time", "timestamp", "press", "release", "event"]),
        "key": pick_by_keywords(["key", "char", "stroke", "press", "release"]),
        "label": pick_by_keywords(["label", "target", "class", "auth"]),
    }


def audit_dataset_file(file_path: Path) -> DatasetAuditResult:
    frame = normalize_column_names(load_tabular_file(file_path))
    result = summarize_frame(frame)
    result.file_path = file_path
    return result


def audit_first_available_dataset(raw_dir: Path) -> DatasetAuditResult | None:
    files = discover_dataset_files(raw_dir)
    if not files:
        return None
    return audit_dataset_file(files[0])
