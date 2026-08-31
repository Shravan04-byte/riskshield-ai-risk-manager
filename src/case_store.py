"""ChromaDB-backed precedent search over historical transactions.

Builds a normalized embedding index from the temporal training split so the
agent can retrieve similar past cases when explaining or reviewing a transaction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from explain import (
    RANDOM_STATE,
    _find_confusion_examples,
    _temporal_split,
    load_model,
)

DEFAULT_COLLECTION_NAME = "fraud_cases"
SCALER_FILENAME = "scaler.joblib"
STORE_CONFIG_FILENAME = "store_config.json"
CHROMA_SUBDIR = "chroma"


def _chroma_path(persist_dir: Path) -> Path:
    return persist_dir / CHROMA_SUBDIR


def _store_config_path(persist_dir: Path) -> Path:
    return persist_dir / STORE_CONFIG_FILENAME


def _scaler_path(persist_dir: Path) -> Path:
    return persist_dir / SCALER_FILENAME


def _stratified_train_sample(
    train_df: pd.DataFrame,
    sample_size: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Keep all fraud rows, then sample legitimate rows to reach ``sample_size``."""
    fraud_df = train_df[train_df["Class"] == 1]
    legit_df = train_df[train_df["Class"] == 0]

    n_fraud = len(fraud_df)
    if n_fraud >= sample_size:
        # Degenerate case: more fraud rows than the cap (unlikely on this dataset).
        return fraud_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    n_legit_needed = sample_size - n_fraud
    legit_sample = legit_df.sample(
        n=min(n_legit_needed, len(legit_df)),
        random_state=random_state,
    )

    sample_df = pd.concat([fraud_df, legit_sample], ignore_index=True)
    return sample_df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def build_case_store(
    data_path: str | Path,
    persist_dir: str | Path,
    feature_cols: list[str],
    sample_size: int = 5000,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> chromadb.Collection:
    """Build and persist a ChromaDB index from the 55% temporal training split.

  Steps:
    1. Stratified sample (all fraud + random legitimate rows up to ``sample_size``)
    2. Fit StandardScaler on feature columns — required before L2/cosine similarity
    3. Upsert normalized vectors with class/amount/index metadata

    Also saves the scaler and store config alongside the Chroma persist directory.
    """
    data_path = Path(data_path)
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    chroma_path = _chroma_path(persist_dir)
    chroma_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    train_df, _val_df, _test_df = _temporal_split(df)
    sample_df = _stratified_train_sample(train_df, sample_size)

    X = sample_df[feature_cols].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # L2 on scaled features; cosine is equivalent when vectors are unit-normalized,
    # but L2 on StandardScaler output is the straightforward choice here.
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "l2"},
    )

    ids = [f"case_{i}" for i in range(len(sample_df))]
    embeddings = X_scaled.tolist()
    metadatas = [
        {
            "class": int(row["Class"]),
            "amount": float(row["Amount"]),
            "index": int(idx),
        }
        for idx, row in sample_df.iterrows()
    ]

    # Replace any prior build (idempotent rebuild).
    existing_ids = collection.get(include=[])["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    joblib.dump(scaler, _scaler_path(persist_dir))
    store_config = {
        "feature_cols": feature_cols,
        "collection_name": collection_name,
        "sample_size": sample_size,
        "n_indexed": len(sample_df),
        "n_fraud_indexed": int((sample_df["Class"] == 1).sum()),
        "random_state": RANDOM_STATE,
    }
    with _store_config_path(persist_dir).open("w", encoding="utf-8") as f:
        json.dump(store_config, f, indent=2)

    return collection


def load_case_store(
    persist_dir: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> tuple[chromadb.Collection, StandardScaler, list[str]]:
    """Reconnect to a persisted ChromaDB collection and load the fitted scaler."""
    persist_dir = Path(persist_dir)
    chroma_path = _chroma_path(persist_dir)

    if not chroma_path.exists():
        raise FileNotFoundError(f"Chroma persist directory not found: {chroma_path}")
    if not _scaler_path(persist_dir).exists():
        raise FileNotFoundError(f"Scaler not found: {_scaler_path(persist_dir)}")

    with _store_config_path(persist_dir).open(encoding="utf-8") as f:
        store_config = json.load(f)

    scaler: StandardScaler = joblib.load(_scaler_path(persist_dir))
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=collection_name)

    return collection, scaler, store_config["feature_cols"]


def retrieve_similar_cases(
    collection: chromadb.Collection,
    scaler: StandardScaler,
    feature_cols: list[str],
    transaction_features: dict[str, Any],
    k: int = 3,
) -> list[dict[str, Any]]:
    """Return the k nearest historical cases to a query transaction.

    The query is normalized with the scaler saved at index-build time so distances
    are comparable to stored vectors.
    """
    missing = set(feature_cols) - set(transaction_features)
    if missing:
        raise ValueError(f"Transaction missing required features: {sorted(missing)}")

    query_vector = scaler.transform(
        np.array([[transaction_features[col] for col in feature_cols]], dtype=float)
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["metadatas", "distances"],
    )

    neighbors: list[dict[str, Any]] = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        neighbors.append(
            {
                "class": int(metadata["class"]),
                "amount": float(metadata["amount"]),
                "distance": float(distance),
            }
        )

    # Chroma returns nearest-first; sort explicitly for a stable contract.
    return sorted(neighbors, key=lambda item: item["distance"])


def _case_store_exists(persist_dir: Path) -> bool:
    return (
        _chroma_path(persist_dir).exists()
        and _scaler_path(persist_dir).exists()
        and _store_config_path(persist_dir).exists()
    )


if __name__ == "__main__":
    import pprint

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "creditcard.csv"
    model_path = project_root / "models" / "fraud_xgb.joblib"
    persist_dir = project_root / "data" / "case_store"

    model, model_config = load_model(model_path)
    feature_cols: list[str] = model_config["feature_cols"]

    if not _case_store_exists(persist_dir):
        print(f"Case store not found at {persist_dir} — building now...\n")
        build_case_store(data_path, persist_dir, feature_cols, sample_size=5000)
        print("Case store built.\n")
    else:
        print(f"Loading existing case store from {persist_dir}\n")

    collection, scaler, loaded_feature_cols = load_case_store(persist_dir)
    assert loaded_feature_cols == feature_cols

    # Same three sanity-check transactions as explain.py (TP / FP / FN from test set).
    df = pd.read_csv(data_path)
    _train_df, _val_df, test_df = _temporal_split(df)
    examples = _find_confusion_examples(
        model, test_df, feature_cols, model_config["threshold"]
    )

    for label, transaction in examples.items():
        neighbors = retrieve_similar_cases(
            collection, scaler, feature_cols, transaction, k=3
        )
        actual_class = 0 if label == "false_positive" else 1
        fraud_neighbors = sum(1 for n in neighbors if n["class"] == 1)

        print(f"=== {label.replace('_', ' ').title()} (actual class={actual_class}) ===")
        print(f"  Fraud neighbors in top-3: {fraud_neighbors}/3")
        for i, neighbor in enumerate(neighbors, start=1):
            tag = "fraud" if neighbor["class"] == 1 else "legit"
            print(
                f"  #{i}  class={neighbor['class']} ({tag})  "
                f"amount={neighbor['amount']:.2f}  distance={neighbor['distance']:.4f}"
            )
        print()
