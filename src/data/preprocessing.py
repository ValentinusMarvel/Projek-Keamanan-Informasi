from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class SequenceBundle:
    features: np.ndarray
    labels: np.ndarray
    sequence_ids: list[str]
    session_ids: np.ndarray | None = None
    rep_ids: np.ndarray | None = None


def handle_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    numeric_columns = cleaned.select_dtypes(include=[np.number]).columns
    categorical_columns = [column for column in cleaned.columns if column not in numeric_columns]

    for column in numeric_columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in categorical_columns:
        mode = cleaned[column].mode(dropna=True)
        if not mode.empty:
            cleaned[column] = cleaned[column].fillna(mode.iloc[0])
        else:
            cleaned[column] = cleaned[column].fillna("")

    return cleaned


def filter_outliers(frame: pd.DataFrame, numeric_columns: Iterable[str] | None = None, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
    cleaned = frame.copy()
    if len(cleaned) < 10:
        return cleaned.reset_index(drop=True)
    if numeric_columns is None:
        numeric_columns = cleaned.select_dtypes(include=[np.number]).columns.tolist()

    mask = pd.Series(True, index=cleaned.index)
    for column in numeric_columns:
        if column not in cleaned.columns:
            continue
        lower_bound = cleaned[column].quantile(lower_quantile)
        upper_bound = cleaned[column].quantile(upper_quantile)
        mask &= cleaned[column].between(lower_bound, upper_bound, inclusive="both")

    filtered = cleaned.loc[mask].reset_index(drop=True)
    if filtered.empty:
        return cleaned.reset_index(drop=True)
    return filtered


def infer_sequence_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    lower_columns = [column.lower() for column in frame.columns]

    def pick(keywords: list[str]) -> list[str]:
        return [column for column in frame.columns if any(keyword in column.lower() for keyword in keywords)]

    return {
        "group": pick(["user", "subject", "participant", "person", "id"]),
        "session": pick(["session", "sess", "trial", "block"]),
        "label": pick(["label", "target", "class", "auth"]),
        "numeric": [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])],
        "timing": pick(["press", "release", "dwell", "flight", "latency", "speed", "rhythm", "time"]),
    }


def segment_sessions(frame: pd.DataFrame, session_column: str | None = None) -> pd.DataFrame:
    segmented = frame.copy()
    if session_column and session_column in segmented.columns:
        segmented = segmented.sort_values([session_column]).reset_index(drop=True)
    return segmented


def build_temporal_sequences(frame: pd.DataFrame, group_column: str | None = None, label_column: str | None = None, feature_columns: list[str] | None = None):
    if feature_columns is None:
        feature_columns = frame.select_dtypes(include=[np.number]).columns.tolist()

    if group_column and group_column in frame.columns:
        groups = frame.groupby(group_column, sort=False)
    else:
        groups = [("sequence_0", frame)]

    sequences = []
    labels = []
    sequence_ids = []

    for sequence_id, group in groups:
        if group.empty:
            continue
        sequence = group[feature_columns].to_numpy(dtype=float)
        sequences.append(sequence)
        sequence_ids.append(str(sequence_id))
        if label_column and label_column in group.columns:
            labels.append(group[label_column].iloc[0])
        else:
            labels.append(sequence_id)

    return sequences, np.asarray(labels), sequence_ids


def normalize_sequences(sequences: list[np.ndarray]) -> list[np.ndarray]:
    if not sequences:
        return []

    combined = np.vstack(sequences)
    scaler = StandardScaler()
    scaler.fit(combined)
    return [scaler.transform(sequence) for sequence in sequences]


def window_and_pad_sequences(sequences: list[np.ndarray], window_size: int = 128) -> np.ndarray:
    if not sequences:
        return np.empty((0, window_size, 0))

    feature_dim = sequences[0].shape[1]
    padded = []
    for sequence in sequences:
        truncated = sequence[:window_size]
        if len(truncated) < window_size:
            padding = np.zeros((window_size - len(truncated), feature_dim))
            truncated = np.vstack([truncated, padding])
        padded.append(truncated)
    return np.stack(padded, axis=0)


def _build_temporal_keystroke_sample(row: pd.Series, password_steps: list[str], include_terminal_key: bool = False) -> np.ndarray:
    sample = []
    previous_step = None

    # Detect if the Series index is lowercase or uppercase
    is_lowercase = any(k.islower() for k in row.index if isinstance(k, str) and (k.startswith("h.") or k.startswith("ud.") or k.startswith("dd.")))

    for step_index, step_name in enumerate(password_steps):
        if is_lowercase:
            hold_column = f"h.{step_name}"
        else:
            hold_column = f"H.{step_name}"
        hold_value = float(row.get(hold_column, row.get(hold_column.upper() if hasattr(hold_column, 'upper') else hold_column, 0.0)))

        if step_index == 0:
            sample.append([hold_value, 0.0, 0.0])
        else:
            if is_lowercase:
                ud_column = f"ud.{previous_step}.{step_name}"
                dd_column = f"dd.{previous_step}.{step_name}"
            else:
                ud_column = f"UD.{previous_step}.{step_name}"
                dd_column = f"DD.{previous_step}.{step_name}"
            
            ud_value = float(row.get(ud_column, row.get(ud_column.upper() if hasattr(ud_column, 'upper') else ud_column, 0.0)))
            dd_value = float(row.get(dd_column, row.get(dd_column.upper() if hasattr(dd_column, 'upper') else dd_column, 0.0)))
            sample.append([hold_value, ud_value, dd_value])

        previous_step = step_name

    if include_terminal_key:
        if is_lowercase:
            ret_h = "h.return"
            ret_ud = "ud.l.return"
            ret_dd = "dd.l.return"
        else:
            ret_h = "H.Return"
            ret_ud = "UD.l.Return"
            ret_dd = "DD.l.Return"

        h_val = float(row.get(ret_h, row.get(ret_h.upper() if hasattr(ret_h, 'upper') else ret_h, 0.0)))
        ud_val = float(row.get(ret_ud, row.get(ret_ud.upper() if hasattr(ret_ud, 'upper') else ret_ud, 0.0)))
        dd_val = float(row.get(ret_dd, row.get(ret_dd.upper() if hasattr(ret_dd, 'upper') else ret_dd, 0.0)))
        sample.append([h_val, ud_val, dd_val])

    return np.asarray(sample, dtype=float)


def build_password_temporal_sequences(
    frame: pd.DataFrame,
    label_column: str = "subject",
    session_column: str = "sessionindex",
    rep_column: str = "rep",
    include_terminal_key: bool = False,
):
    password_steps = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l"]

    sequences = []
    labels = []
    sequence_ids = []
    session_ids = []
    rep_ids = []

    for _, row in frame.iterrows():
        sequence = _build_temporal_keystroke_sample(row, password_steps=password_steps, include_terminal_key=include_terminal_key)
        sequences.append(sequence)
        labels.append(row[label_column] if label_column in frame.columns else "")

        subject_value = str(row[label_column]) if label_column in frame.columns else "unknown"
        session_value = str(row[session_column]) if session_column in frame.columns else "0"
        rep_value = str(row[rep_column]) if rep_column in frame.columns else "0"
        sequence_ids.append(f"{subject_value}_{session_value}_{rep_value}")
        session_ids.append(session_value)
        rep_ids.append(rep_value)

    return sequences, np.asarray(labels), sequence_ids, np.asarray(session_ids), np.asarray(rep_ids)


def split_train_validation_test(features: np.ndarray, labels: np.ndarray, train_ratio: float = 0.7, validation_ratio: float = 0.15, random_state: int = 42):
    if len(features) == 0:
        return (np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,)))

    from sklearn.model_selection import train_test_split

    x_train, x_temp, y_train, y_temp = train_test_split(features, labels, test_size=1 - train_ratio, random_state=random_state, stratify=labels if len(np.unique(labels)) > 1 else None)
    validation_size = validation_ratio / (1 - train_ratio)
    x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=1 - validation_size, random_state=random_state, stratify=y_temp if len(np.unique(y_temp)) > 1 else None)
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def prepare_sequence_bundle(frame: pd.DataFrame, group_column: str | None = None, session_column: str | None = None, label_column: str | None = None, window_size: int = 128, normalize: bool = False) -> SequenceBundle:
    cleaned = handle_missing_values(frame)
    cleaned = filter_outliers(cleaned)
    cleaned = segment_sessions(cleaned, session_column=session_column)

    sequence_columns = infer_sequence_columns(cleaned)
    feature_columns = sequence_columns["numeric"]
    if not feature_columns:
        feature_columns = cleaned.select_dtypes(include=[np.number]).columns.tolist()

    sequences, labels, sequence_ids = build_temporal_sequences(
        cleaned,
        group_column=group_column,
        label_column=label_column,
        feature_columns=feature_columns,
    )
    if normalize:
        sequences = normalize_sequences(sequences)
    padded_features = window_and_pad_sequences(sequences, window_size=window_size)
    return SequenceBundle(features=padded_features, labels=labels, sequence_ids=sequence_ids)


def prepare_password_sequence_bundle(frame: pd.DataFrame, label_column: str = "subject", session_column: str = "sessionindex", rep_column: str = "rep", include_terminal_key: bool = False, normalize: bool = False) -> SequenceBundle:
    cleaned = handle_missing_values(frame)
    cleaned = filter_outliers(cleaned)
    cleaned = segment_sessions(cleaned, session_column=session_column)

    sequences, labels, sequence_ids, session_ids, rep_ids = build_password_temporal_sequences(
        cleaned,
        label_column=label_column,
        session_column=session_column,
        rep_column=rep_column,
        include_terminal_key=include_terminal_key,
    )

    if normalize:
        sequences = normalize_sequences(sequences)

    features = np.stack(sequences, axis=0) if sequences else np.empty((0, 10, 3))
    return SequenceBundle(features=features, labels=labels, sequence_ids=sequence_ids, session_ids=session_ids, rep_ids=rep_ids)
