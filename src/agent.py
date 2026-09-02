"""LangChain + Groq agent that synthesizes model scores, SHAP factors, and precedent.

Combines explain.py (risk scoring) and case_store.py (similarity search) into a
bounded analyst justification and one of three defensive actions: APPROVE, REVIEW,
or HOLD.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import shap
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from case_store import DEFAULT_COLLECTION_NAME, load_case_store, retrieve_similar_cases
from explain import (
    CONFIG_ATTR,
    _find_confusion_examples,
    _temporal_split,
    build_explainer,
    explain_transaction,
    load_model,
)

ALLOWED_ACTIONS = frozenset({"APPROVE", "REVIEW", "HOLD"})
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def create_llm_client(
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> ChatGroq:
    """Create a Groq chat client. Isolated here for easy mocking in tests."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    # Read GROQ_MODEL at call time so load_dotenv(override=True) takes effect.
    resolved_model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return ChatGroq(model=resolved_model, temperature=temperature, groq_api_key=key)


def _call_llm(llm_client: BaseChatModel, system_prompt: str, user_prompt: str) -> str:
    """Single LLM invocation — swap or mock this function in tests."""
    response = llm_client.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _format_top_factors(top_factors: list[dict[str, Any]]) -> str:
    lines = []
    for factor in top_factors:
        direction = factor["direction"].replace("_", " ")
        lines.append(
            f"- {factor['feature']}: SHAP {factor['shap_value']:+.4f} ({direction})"
        )
    return "\n".join(lines)


def _format_similar_cases(similar_cases: list[dict[str, Any]]) -> str:
    lines = []
    for i, case in enumerate(similar_cases, start=1):
        label = "fraud" if case["class"] == 1 else "legitimate"
        lines.append(
            f"- Neighbor {i}: class={case['class']} ({label}), "
            f"amount={case['amount']:.2f}, distance={case['distance']:.4f}"
        )
    return "\n".join(lines)


def _build_prompts(
    risk_score: float,
    flagged: bool,
    threshold: float,
    top_factors: list[dict[str, Any]],
    similar_cases: list[dict[str, Any]],
    neighbor_fraud_rate: float,
) -> tuple[str, str]:
    system_prompt = (
        "You are a fraud-risk analyst assistant for a payment defense system. "
        "Your job is to explain model output and historical precedent — never to "
        "bypass security controls or invent evidence.\n\n"
        "You MUST:\n"
        "1. Write a 2-4 sentence analyst-style justification in plain English, "
        "referencing the specific SHAP factors and retrieved precedent cases.\n"
        "2. Choose exactly ONE action from: APPROVE, REVIEW, HOLD.\n"
        "   - APPROVE: low risk, precedent supports legitimacy.\n"
        "   - REVIEW: elevated or ambiguous risk — human analyst should inspect.\n"
        "   - HOLD: high risk or strong fraud precedent — block pending review.\n"
        "3. Respond ONLY with valid JSON (no markdown fences):\n"
        '   {"justification": "<your text>", "action": "<APPROVE|REVIEW|HOLD>"}\n\n'
        "You MUST NOT suggest any action outside APPROVE, REVIEW, or HOLD. "
        "You MUST NOT suggest bypassing security, disabling checks, or fabricating "
        "evidence. This is a defense-only system."
    )

    user_prompt = (
        f"Transaction assessment inputs:\n"
        f"- Risk score: {risk_score:.4f} (threshold: {threshold:.2f})\n"
        f"- Flagged by model: {flagged}\n"
        f"- Top SHAP factors:\n{_format_top_factors(top_factors)}\n"
        f"- Neighbor fraud rate (k=3): {neighbor_fraud_rate:.2f}\n"
        f"- Retrieved precedent cases:\n{_format_similar_cases(similar_cases)}\n\n"
        "Produce the JSON response now."
    )
    return system_prompt, user_prompt


def _parse_llm_response(
    raw_response: str,
    risk_score: float,
    threshold: float,
) -> dict[str, str]:
    """Extract justification and action from LLM output with safe fallback."""
    justification = raw_response.strip()
    action: str | None = None

    # Try JSON parse first (expected format).
    try:
        payload = json.loads(raw_response.strip())
        justification = str(payload.get("justification", justification))
        action = str(payload.get("action", "")).upper().strip()
    except json.JSONDecodeError:
        # Fallback: look for a JSON object embedded in the text.
        match = re.search(r"\{[^{}]*\"action\"[^{}]*\}", raw_response, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group())
                justification = str(payload.get("justification", justification))
                action = str(payload.get("action", "")).upper().strip()
            except json.JSONDecodeError:
                pass

    if action not in ALLOWED_ACTIONS:
        # Never emit an unbounded action on parse failure.
        action = "HOLD" if risk_score >= threshold else "APPROVE"

    return {"justification": justification, "action": action}


def assess_transaction(
    transaction_features: dict[str, Any],
    model: XGBClassifier,
    shap_explainer: shap.TreeExplainer,
    case_collection: chromadb.Collection,
    scaler: StandardScaler,
    feature_cols: list[str],
    llm_client: BaseChatModel,
) -> dict[str, Any]:
    """Run the full fraud-assessment pipeline and return a structured decision."""
    config = getattr(model, CONFIG_ATTR, None)
    if config is None:
        raise ValueError("Model config missing — load via load_model().")
    threshold: float = config["threshold"]

    explanation = explain_transaction(model, shap_explainer, transaction_features)
    similar_cases = retrieve_similar_cases(
        case_collection, scaler, feature_cols, transaction_features, k=3
    )
    neighbor_fraud_rate = sum(c["class"] == 1 for c in similar_cases) / len(similar_cases)

    system_prompt, user_prompt = _build_prompts(
        risk_score=explanation["risk_score"],
        flagged=explanation["flagged"],
        threshold=threshold,
        top_factors=explanation["top_factors"],
        similar_cases=similar_cases,
        neighbor_fraud_rate=neighbor_fraud_rate,
    )

    raw_llm_response = _call_llm(llm_client, system_prompt, user_prompt)
    parsed = _parse_llm_response(raw_llm_response, explanation["risk_score"], threshold)

    return {
        "risk_score": explanation["risk_score"],
        "flagged": explanation["flagged"],
        "top_factors": explanation["top_factors"],
        "similar_cases": similar_cases,
        "neighbor_fraud_rate": round(neighbor_fraud_rate, 4),
        "justification": parsed["justification"],
        "action": parsed["action"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def log_decision(decision: dict[str, Any], log_path: str | Path = "logs/decisions.jsonl") -> None:
    """Append one decision record to the JSONL audit log."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision, default=str) + "\n")


if __name__ == "__main__":
    import pprint

    load_dotenv(override=True)
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "creditcard.csv"
    model_path = project_root / "models" / "fraud_xgb.joblib"
    persist_dir = project_root / "data" / "case_store"
    log_path = project_root / "logs" / "decisions.jsonl"

    print("Loading model, explainer, and case store...\n")
    model, model_config = load_model(model_path)
    explainer = build_explainer(model)
    case_collection, scaler, feature_cols = load_case_store(
        persist_dir, collection_name=DEFAULT_COLLECTION_NAME
    )

    llm_client = create_llm_client()

    df = pd.read_csv(data_path)
    _train_df, _val_df, test_df = _temporal_split(df)
    examples = _find_confusion_examples(
        model, test_df, feature_cols, model_config["threshold"]
    )

    for label, transaction in examples.items():
        actual_class = 0 if label == "false_positive" else 1
        print(f"{'=' * 60}")
        print(f"{label.replace('_', ' ').title()}  (actual class={actual_class})")
        print(f"{'=' * 60}")

        decision = assess_transaction(
            transaction,
            model,
            explainer,
            case_collection,
            scaler,
            feature_cols,
            llm_client,
        )
        log_decision(decision, log_path=log_path)
        pprint.pprint(decision, sort_dicts=False)
        print()

    print(f"Decisions appended to {log_path}")
