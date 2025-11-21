"""
Converted from a Kaggle notebook into a runnable script.
Usage examples:
  python Classification_tasks.py \
    --train /kaggle/input/lmsys-chatbot-arena/train.csv \
    --test /kaggle/input/lmsys-chatbot-arena/test.csv \
    --output submission.csv

This script performs basic preprocessing, trains a simple logistic regression pipeline,
reports cross-validated log-loss, fits to full training data and (optionally) writes a
submission CSV with probability columns for winner_model_a, winner_model_b, winner_tie.

Notes:
- This is a lightweight conversion of the original notebook. It focuses on reproducible
  CLI usage, modular functions, and basic validation. For heavy grid search or large
  max_features values, run on a machine with enough memory.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss


def load_data(train_path: str, test_path: str | None = None):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path else None
    return train, test


def make_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    # create combined text column used for simple pipelines
    df = df.copy()
    df["text"] = (
        "[PROMPT] " + df["prompt"].fillna("").astype(str) +
        " [ANS_A] " + df["response_a"].fillna("").astype(str) +
        " [ANS_B] " + df["response_b"].fillna("").astype(str)
    )
    return df


def build_vectorizer(max_p_word: int = 100_000, max_a_char: int = 100_000, max_b_char: int = 100_000):
    # ColumnTransformer mapping the three text columns to separate vectorizers
    vec = ColumnTransformer(
        transformers=[
            ("p_word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9,
                                       sublinear_tf=True, max_features=max_p_word,
                                       dtype=np.float32), "prompt"),
            ("a_char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                       min_df=2, max_features=max_a_char,
                                       dtype=np.float32), "response_a"),
            ("b_char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                       min_df=2, max_features=max_b_char,
                                       dtype=np.float32), "response_b"),
        ],
        sparse_threshold=1.0,
    )
    return vec


def prepare_targets(df: pd.DataFrame) -> np.ndarray:
    # convert one-hot winner columns into a 1D integer target
    y = np.argmax(df[["winner_model_a", "winner_model_b", "winner_tie"]].values, axis=1)
    return y


def train_and_submit(train: pd.DataFrame,
                     test: pd.DataFrame | None,
                     sample: int | None,
                     drop_ties: bool,
                     output: str | None):

    # optionally filter ties and sample
    df = train.copy()
    df = make_text_columns(df)
    y = prepare_targets(df)

    if drop_ties:
        mask = (y != 2)
        df = df[mask]
        y = y[mask]

    if sample is not None and sample > 0 and len(df) > sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(df), size=sample, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        y = y[idx]

    # Build vectorizer and classifier
    vec = build_vectorizer(max_p_word=100_000, max_a_char=100_000, max_b_char=100_000)
    clf = LogisticRegression(solver="saga", multi_class="multinomial", C=4.0, max_iter=2000)
    pipe = Pipeline([("vec", vec), ("clf", clf)])

    # cross-validated log-loss
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = df[["prompt", "response_a", "response_b"]].fillna("")

    print("Computing 5-fold cross-validated log-loss (this may take a while)...")
    try:
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="neg_log_loss", n_jobs=-1)
        print(f"5-fold CV log_loss: {-scores.mean():.4f} ± {scores.std():.4f}")
    except Exception as e:
        print("Warning: cross_val_score failed:", e, file=sys.stderr)

    # Fit full model and (optionally) create submission
    print("Fitting final model on full training set...")
    pipe.fit(X, y)

    if test is not None and output is not None:
        test_df = test.copy()
        test_df = make_text_columns(test_df)
        X_test = test_df[["prompt", "response_a", "response_b"]].fillna("")
        probs = pipe.predict_proba(X_test)
        submission = pd.DataFrame({
            "id": test_df["id"],
            "winner_model_a": probs[:, 0],
            "winner_model_b": probs[:, 1],
            "winner_tie": probs[:, 2],
        })
        submission.to_csv(output, index=False)
        print(f"Submission written to: {output}")
    else:
        print("No test path or output provided — skipping submission creation.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a classifier for LMSYS Chatbot Arena")
    p.add_argument("--train", required=True, help="Path to train CSV")
    p.add_argument("--test", required=False, help="Path to test CSV (optional)")
    p.add_argument("--output", default="submission.csv", help="Submission CSV path")
    p.add_argument("--sample", type=int, default=8000, help="Number of training rows to sample (or 0/None to use all)")
    p.add_argument("--drop-ties", action="store_true", help="Drop tie rows from training")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sample = None if (args.sample is None or args.sample <= 0) else int(args.sample)

    train_path = args.train
    test_path = args.test
    output = args.output

    train, test = load_data(train_path, test_path)
    train_and_submit(train, test, sample=sample, drop_ties=args.drop_ties, output=output)


if __name__ == "__main__":
    main()
