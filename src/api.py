"""FastAPI demo service wrapping the fraud-assessment pipeline.

Internal analyst-tool demo — no authentication. Load the model, SHAP explainer,
case store, and Groq client once at startup and reuse them on every request.
"""

from __future__ import annotations

import math
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent import assess_transaction, create_llm_client, log_decision
from case_store import DEFAULT_COLLECTION_NAME, load_case_store, retrieve_similar_cases
from explain import (
    CONFIG_ATTR,
    _find_confusion_examples,
    _temporal_split,
    build_explainer,
    explain_transaction,
    load_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_xgb.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
CASE_STORE_DIR = PROJECT_ROOT / "data" / "case_store"
DECISIONS_LOG_PATH = PROJECT_ROOT / "logs" / "decisions.jsonl"

FALLBACK_JUSTIFICATION = (
    "The language-model justification service was unavailable, so this decision "
    "used a rule-based fallback: HOLD when the model risk score meets or exceeds "
    "the configured threshold, otherwise APPROVE. Re-run the assessment when the "
    "LLM is back, or send this case to a human analyst if the score is close to "
    "the threshold."
)


def _finite_float(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("must be a finite number")
    return value


class TransactionRequest(BaseModel):
    """One credit-card transaction in the training feature schema."""

    model_config = ConfigDict(extra="forbid")

    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    @field_validator("*")
    @classmethod
    def _reject_non_finite(cls, value: float) -> float:
        return _finite_float(value)


class ShapFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    shap_value: float
    direction: str


class SimilarCase(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    class_: int = Field(alias="class")
    amount: float
    distance: float


class AssessmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_score: float
    flagged: bool
    top_factors: list[ShapFactor]
    similar_cases: list[SimilarCase]
    neighbor_fraud_rate: float
    justification: str
    action: str
    timestamp: str
    llm_fallback_used: bool = False


class ExampleTransaction(BaseModel):
    category: str
    actual_class: int
    transaction: TransactionRequest


class ExampleTransactionsResponse(BaseModel):
    examples: list[ExampleTransaction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    case_store_loaded: bool


def _features_as_floats(features: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in features.items()}


def _rule_based_assessment(
    transaction_features: dict[str, Any],
    model,
    shap_explainer,
    case_collection,
    scaler,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Score + retrieve precedent, then HOLD/APPROVE from the model threshold."""
    config = getattr(model, CONFIG_ATTR, None)
    if config is None:
        raise ValueError("Model config missing — load via load_model().")
    threshold: float = config["threshold"]

    explanation = explain_transaction(model, shap_explainer, transaction_features)
    similar_cases = retrieve_similar_cases(
        case_collection, scaler, feature_cols, transaction_features, k=3
    )
    neighbor_fraud_rate = (
        sum(case["class"] == 1 for case in similar_cases) / len(similar_cases)
        if similar_cases
        else 0.0
    )
    action = "HOLD" if explanation["risk_score"] >= threshold else "APPROVE"

    return {
        "risk_score": explanation["risk_score"],
        "flagged": explanation["flagged"],
        "top_factors": explanation["top_factors"],
        "similar_cases": similar_cases,
        "neighbor_fraud_rate": round(neighbor_fraud_rate, 4),
        "justification": FALLBACK_JUSTIFICATION,
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llm_fallback_used": True,
    }


def _load_example_transactions(model, feature_cols: list[str], threshold: float) -> list[dict[str, Any]]:
    df = pd.read_csv(DATA_PATH)
    _train_df, _val_df, test_df = _temporal_split(df)
    raw_examples = _find_confusion_examples(model, test_df, feature_cols, threshold)

    labeled: list[dict[str, Any]] = []
    for category, features in raw_examples.items():
        labeled.append(
            {
                "category": category,
                "actual_class": 0 if category == "false_positive" else 1,
                "transaction": _features_as_floats(features),
            }
        )
    return labeled


@asynccontextmanager
async def lifespan(app: FastAPI):
    # override=True so .env wins over a stale shell-exported GROQ_API_KEY / GROQ_MODEL.
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    print("[startup] Loading XGBoost model and config...")
    model, model_config = load_model(MODEL_PATH)
    print(
        f"[startup] Model loaded (threshold={model_config['threshold']:.2f}, "
        f"features={len(model_config['feature_cols'])})"
    )

    print("[startup] Building SHAP explainer...")
    explainer = build_explainer(model)
    print("[startup] SHAP explainer ready")

    print("[startup] Loading case store collection and scaler...")
    collection, scaler, feature_cols = load_case_store(
        CASE_STORE_DIR, collection_name=DEFAULT_COLLECTION_NAME
    )
    print(f"[startup] Case store loaded (feature_cols={len(feature_cols)})")

    print("[startup] Creating Groq LLM client...")
    try:
        llm_client = create_llm_client()
        print("[startup] Groq LLM client ready")
    except Exception as exc:
        llm_client = None
        print(
            f"[startup] Groq LLM client unavailable ({exc}); "
            "assessments will use the rule-based fallback"
        )

    print("[startup] Caching example transactions from the test split...")
    example_transactions = _load_example_transactions(
        model, feature_cols, model_config["threshold"]
    )
    print(f"[startup] Cached {len(example_transactions)} example transactions")
    print("[startup] RiskShield API ready")

    app.state.model = model
    app.state.model_config = model_config
    app.state.explainer = explainer
    app.state.collection = collection
    app.state.scaler = scaler
    app.state.feature_cols = feature_cols
    app.state.llm_client = llm_client
    app.state.example_transactions = example_transactions

    yield


app = FastAPI(
    title="RiskShield Fraud Assessment API",
    description="Internal analyst-tool demo wrapping the fraud-assessment pipeline.",
    lifespan=lifespan,
)

# Permissive CORS for this local demo. In a real deployment this would be
# restricted to the analyst UI origin (same discipline as a JWT-backed app —
# this project simply has no auth because it is not a public multi-user product).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    return HealthResponse(
        status="ok",
        model_loaded=getattr(state, "model", None) is not None,
        case_store_loaded=getattr(state, "collection", None) is not None,
    )


@app.get("/example-transactions", response_model=ExampleTransactionsResponse)
def example_transactions(request: Request) -> ExampleTransactionsResponse:
    return ExampleTransactionsResponse(examples=request.app.state.example_transactions)


@app.post("/assess-transaction", response_model=AssessmentResponse)
def assess_transaction_route(
    payload: TransactionRequest,
    request: Request,
) -> AssessmentResponse:
    state = request.app.state
    transaction_features = payload.model_dump()
    llm_fallback_used = False

    try:
        if state.llm_client is None:
            raise RuntimeError("Groq LLM client was not initialized at startup")
        decision = assess_transaction(
            transaction_features,
            state.model,
            state.explainer,
            state.collection,
            state.scaler,
            state.feature_cols,
            state.llm_client,
        )
        decision["llm_fallback_used"] = False
    except Exception as exc:
        llm_fallback_used = True
        print(f"[audit] LLM call failed; using rule-based fallback: {exc}")
        decision = _rule_based_assessment(
            transaction_features,
            state.model,
            state.explainer,
            state.collection,
            state.scaler,
            state.feature_cols,
        )
        decision["llm_error"] = str(exc)

    log_decision(decision, log_path=DECISIONS_LOG_PATH)
    return AssessmentResponse(
        risk_score=decision["risk_score"],
        flagged=decision["flagged"],
        top_factors=decision["top_factors"],
        similar_cases=decision["similar_cases"],
        neighbor_fraud_rate=decision["neighbor_fraud_rate"],
        justification=decision["justification"],
        action=decision["action"],
        timestamp=decision["timestamp"],
        llm_fallback_used=llm_fallback_used,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
