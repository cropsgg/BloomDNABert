"""Generic stratified splits with duplicate-sequence leakage removal."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from bloom_seq.alphabets import Alphabet


def sanitize_labeled_frame(
    df: pd.DataFrame,
    alphabet: Alphabet,
    min_length: int = 40,
    label_values: Tuple[int, ...] = (0, 1),
) -> pd.DataFrame:
    """Drop invalid rows; normalize case; restrict to alphabet symbols only."""
    if df.empty:
        return df
    pat = alphabet.strict_pattern()
    out = df.dropna(subset=["sequence", "label"]).copy()
    out["sequence"] = out["sequence"].astype(str).str.upper().str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out = out[out["label"].isin(label_values)]
    out = out[out["sequence"].str.len() >= min_length]
    out = out[out["sequence"].apply(lambda s: bool(pat.match(s)))]
    before = len(out)
    out = out.drop_duplicates(subset=["sequence"], keep="first")
    dropped = before - len(out)
    if dropped:
        print(f"  Sanitize: dropped {dropped} duplicate sequences (same window text)")
    return out.reset_index(drop=True)


def stratified_train_val_test(
    df_labeled: pd.DataFrame,
    *,
    val_split: float = 0.2,
    test_split: float = 0.2,
    random_state: int = 42,
    stratify_key: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test with no duplicate ``sequence`` across splits.

    ``df_labeled`` must already be sanitized (binary labels only).
    """
    if df_labeled.empty:
        raise ValueError("No labeled rows to split.")

    def _default_stratify(d: pd.DataFrame):
        if "variant_type" not in d.columns:
            return d["label"]
        vc = d["variant_type"].value_counts()
        if len(vc) < 2 or vc.min() < 2 or len(vc) > 100:
            return d["label"]
        return d["variant_type"]

    strat_fn = stratify_key or _default_stratify

    def _split_stratified(d: pd.DataFrame, test_sz: float, seed: int):
        st = strat_fn(d)
        try:
            return train_test_split(
                d, test_size=test_sz, random_state=seed, stratify=st
            )
        except ValueError:
            return train_test_split(d, test_size=test_sz, random_state=seed, shuffle=True)

    train_val_df, test_df = _split_stratified(df_labeled, test_split, random_state)
    relative_val = val_split / (1 - test_split)
    train_df, val_df = _split_stratified(train_val_df, relative_val, random_state)

    train_seqs = set(train_df["sequence"].tolist())
    val_keep = val_df[~val_df["sequence"].isin(train_seqs)]
    combined_seen = train_seqs | set(val_keep["sequence"].tolist())
    test_keep = test_df[~test_df["sequence"].isin(combined_seen)]

    if len(val_keep) < len(val_df) or len(test_keep) < len(test_df):
        removed = (len(val_df) - len(val_keep)) + (len(test_df) - len(test_keep))
        print(f"  Removed {removed} duplicate sequences across splits")

    return train_df.reset_index(drop=True), val_keep.reset_index(drop=True), test_keep.reset_index(drop=True)
