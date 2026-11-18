
"""Deterministic and optional provider-backed generation."""

from typing import Any, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import ModelTimeoutError
from pydantic import SecretStr

from solution.deployment_project.config import Settings
from solution.deployment_project.models import GenerateRequest, ProviderResult, Usage


NORTHSTAR_SYSTEM_PROMPT = """
You are Northstar's customer support assistant.

Your task is to provide concise, accurate, and policy-safe
customer support guidance.

GENERAL RULES:
- Answer clearly and professionally.
- Keep responses short, ideally 3 to 5 sentences.
- Do not invent company policies, procedures, or guarantees.
- Never claim that an action has already been completed
  unless explicitly confirmed by the provided information.
- Do not invent transaction details, refund amounts,
  processing times, or customer account information.
- If information is insufficient, explain what must be verified.
- If a policy cannot be confirmed, state this explicitly.
- Do not promise outcomes that require human approval.

NORTHSTAR SUPPORT GUIDELINES:

1. DUPLICATE PAYMENTS
- Request the receipt and transaction identifier.
- Verify the alleged duplicate payment before recommending action.
- Refer refund approval and processing to the authorized team.
- Never claim that a refund has already been issued.

2. LOCKED ACCOUNTS
- Require identity verification before account recovery.
- Refer customers to the approved password reset process.
- Never recommend bypassing authentication or security controls.

3. PRIVACY AND DATA DELETION
- Treat personal data requests as sensitive.
- Route privacy and deletion requests to human review.
- Never claim that personal data has already been deleted.

4. INCIDENTS AND OUTAGES
- Communicate only confirmed information.
- Refer customers to the official status page.
- Never invent outage causes or resolution times.

5. DISCOUNTS AND COMMERCIAL PROMISES
- Never promise unapproved discounts or benefits.
- Do not invent special offers or contractual guarantees.
- Refer requests requiring approval to the responsible team.

RESPONSE STYLE:
- Be helpful, professional, and concise.
- Prefer plain text over long lists or tables.
- Focus on the next appropriate action.
- Avoid unnecessary explanations.
- Never present hypothetical actions as completed facts.
"""


class Provider(Protocol):
    model: str

    async def generate(self, request: GenerateRequest) -> ProviderResult: ...


class DryProvider:
    """Produce stable policy-shaped answers for the shared cases."""

    model = "dry-northstar-v1"

    async def generate(self, request: GenerateRequest) -> ProviderResult:
        prompt = request.prompt.lower()

        if "duplicate-payment" in prompt:
            text = (
                "Collect the receipt and transaction identifier "
                "before review."
            )
        elif "locked account" in prompt:
            text = "Verify identity, then issue the approved reset flow."
        elif "delete all personal" in prompt:
            text = "Route the privacy request to human review."
        elif "active incident" in prompt:
            text = "Share only confirmed updates from the status page."
        else:
            text = "I cannot confirm this from approved policy."

        return ProviderResult(
            text=text,
            model=self.model,
            usage=Usage(
                input_tokens=len(request.prompt.split()),
                output_tokens=len(text.split()),
            ),
        )


class ChatProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.api_key:
            raise ValueError(
                f"{settings.provider.upper()}_API_KEY is required for live mode"
            )

        self.model = settings.model

        self.token_limit_key = (
            "max_output_tokens"
            if settings.provider == "google"
            else "max_tokens"
        )

        fallback = settings.openrouter_fallback_model.strip()

        options: dict[str, Any] = (
            {
                "model_kwargs": {
                    "retries": None,
                    "models": (
                        [settings.model, fallback]
                        if fallback
                        else None
                    ),
                },
                "openrouter_provider": {
                    "require_parameters": True
                },
            }
            if settings.provider == "openrouter"
            else {"vertexai": False}
            if settings.provider == "google"
            else {"reasoning_effort": "none"}
            if settings.provider == "openai"
            else {}
        )

        self.client = init_chat_model(
            model=self.model,
            model_provider={
                "mistral": "mistralai",
                "google": "google_genai",
            }.get(settings.provider, settings.provider),
            api_key=SecretStr(settings.api_key),
            timeout=(
                int(settings.request_timeout_seconds * 1000)
                if settings.provider == "openrouter"
                else settings.request_timeout_seconds
            ),
            max_retries=0,
            **options,
        )

    async def generate(
        self,
        request: GenerateRequest,
    ) -> ProviderResult:

        try:
            response = await self.client.ainvoke(
                [
                    ("system", NORTHSTAR_SYSTEM_PROMPT),
                    ("user", request.prompt),
                ],
                temperature=request.temperature,
                **{
                    self.token_limit_key: request.max_tokens
                },
            )

        except ModelTimeoutError as error:
            raise TimeoutError(
                "Model request timed out"
            ) from error

        except IndexError as error:
            raise ValueError(
                "Provider returned no choices"
            ) from error

        finish_reason = str(
            response.response_metadata.get("finish_reason")
        ).lower()

        if finish_reason != "stop":
            raise ValueError(
                "Provider returned an incomplete answer"
            )

        text = response.text

        if not text.strip():
            raise ValueError(
                "Provider returned no answer"
            )

        usage = response.usage_metadata or {}

        return ProviderResult(
            text=text,
            model=(
                response.response_metadata.get("model_name")
                or self.model
            ),
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
        )


def build_provider(settings: Settings) -> Provider:
    if settings.provider == "dry":
        return DryProvider()

    if settings.provider in {
        "groq",
        "mistral",
        "openrouter",
        "google",
        "openai",
    }:
        return ChatProvider(settings)

    raise ValueError(
        f"Unsupported provider: {settings.provider}"
    )
