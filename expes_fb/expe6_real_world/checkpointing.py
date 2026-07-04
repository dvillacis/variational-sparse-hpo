"""Reusable DataFrame checkpoint helpers for Experiment 6 runners."""

import json
from pathlib import Path

import pandas as pd


def load_dataframe_checkpoint(checkpoint_path, meta_path, config, log=None):
    """Load a DataFrame checkpoint if its saved config matches."""
    checkpoint_path = Path(checkpoint_path)
    meta_path = Path(meta_path)
    if not checkpoint_path.exists() or not meta_path.exists():
        return pd.DataFrame()

    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        if log is not None:
            log("checkpoint metadata is unreadable; ignoring old checkpoint")
        return pd.DataFrame()

    if meta.get("config") != config:
        if log is not None:
            log("checkpoint config mismatch; ignoring old checkpoint")
        return pd.DataFrame()

    try:
        df = pd.read_pickle(checkpoint_path)
    except Exception:
        if log is not None:
            log("checkpoint file is unreadable; ignoring old checkpoint")
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        if log is not None:
            log("checkpoint payload is not a DataFrame; ignoring old checkpoint")
        return pd.DataFrame()

    if log is not None:
        log(f"loaded checkpoint with {len(df)} rows from {checkpoint_path}")
    return df


def save_dataframe_checkpoint(df, checkpoint_path, meta_path, config):
    """Persist a DataFrame checkpoint and matching metadata."""
    checkpoint_path = Path(checkpoint_path)
    meta_path = Path(meta_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(checkpoint_path)
    meta_path.write_text(json.dumps({"config": config}, indent=2) + "\n")


def completed_key_set(df, key_cols):
    """Return a set of completed keys from a checkpoint DataFrame."""
    if df.empty:
        return set()
    missing = [col for col in key_cols if col not in df.columns]
    if missing:
        return set()
    return {
        tuple(row[col] for col in key_cols)
        for _, row in df[key_cols].drop_duplicates().iterrows()
    }
