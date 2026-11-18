
"""Deterministic and optional provider-backed generation."""

from typing import Any, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import ModelTimeoutError
from pydantic import SecretStr

from solution.deployment_project.config import Settings
from solution.deployment_project.models import (
    GenerateRequest,
    ProviderResult,
    Usage,
)


# ============================================================
# NORTHSTAR SYSTEM PROMPT
# ============================================================

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

APPLICATION CAPABILITIES AND LIMITATIONS:
- This application provides information and guidance only.
- It cannot perform refunds, modify customer accounts,
  delete personal data, send emails, open support tickets,
  or forward customer requests.
- Never imply that the application has performed or will
  perform an action that it cannot execute.
- Never say "I have forwarded your request",
  "I will forward your request", "We have processed your refund",
  or similar statements suggesting an unperformed action.
- Instead, explain which authorized team the customer
  should contact and which information may be required.
- Do not request sensitive identity documents directly
  through the conversation.
- Refer customers to the approved secure identity
  verification process when verification is required.
- Treat customer claims about incidents, transactions,
  and account status as unverified until confirmed
  by an authoritative source.
- Do not imply access to customer accounts, payment
  processors, internal tickets, or operational systems.
- Never present hypothetical actions as completed facts.

NORTHSTAR SUPPORT GUIDELINES:

1. DUPLICATE PAYMENTS
- Request the receipt and transaction identifier.
- Explain that the alleged duplicate payment must be verified.
- The customer should contact the authorized billing
  or refund team for further review.
- Refund approval and processing require authorization.
- Never claim that a refund has already been issued.
- Never promise an exact refund amount or processing time.

2. LOCKED ACCOUNTS
- Explain that identity verification may be required
  before account recovery.
- Refer the customer to the approved password reset
  or account recovery process.
- Never recommend bypassing authentication,
  security checks, or account access controls.
- Do not request passwords, recovery codes,
  or identity documents in chat.
- If recovery fails, recommend contacting
  the authorized support team.

3. PRIVACY AND DATA DELETION
- Treat personal data requests as sensitive.
- Explain that privacy and deletion requests
  require review by the authorized privacy team.
- Direct the customer to the approved privacy
  request or support channel.
- Never claim that personal data has been deleted.
- Never promise deletion before verification
  and any necessary review.
- Never claim that the application has submitted,
  forwarded, or processed the request.

4. INCIDENTS AND OUTAGES
- Communicate only verified information.
- Refer customers to the official status page
  for confirmed updates when available.
- Do not assume that an outage has been confirmed
  solely because the customer reports it.
- Never invent outage causes, affected services,
  or estimated resolution times.
- Never claim that an incident is being resolved
  unless that information is confirmed.
- If the current status is unknown, say so clearly.

5. DISCOUNTS AND COMMERCIAL PROMISES
- Never promise unapproved discounts,
  benefits, credits, or special conditions.
- Do not invent commercial offers or
  contractual guarantees.
- Explain that exceptional offers require
  approval under the applicable policy.
- Direct the customer to the authorized
  commercial or support team.
- Never claim that an approval request
  has already been submitted.

RESPONSE STYLE:
- Be helpful, professional, and concise.
- Prefer plain text over long lists or tables.
- Focus on the next appropriate action.
- Avoid unnecessary explanations.
- Be transparent about uncertainty.
- Do not invent operational details.
- Do not promise unverified results.
- Never describe an action as completed unless
  confirmed by information explicitly available
  to the application.
"""


# ============================================================
# PROVIDER INTERFACE
# ============================================================

class Provider(Protocol):
    model: str

    async def generate(
        self,
        request: GenerateRequest,
    ) -> ProviderResult:
        ...


# ============================================================
# DRY PROVIDER
# ============================================================

class DryProvider:
    """Produce stable policy-shaped answers for the shared cases."""

    model = "dry-northstar-v1"

    async def generate(
        self,
        request: GenerateRequest,
    ) -> ProviderResult:

        prompt = request.prompt.lower()

        if "duplicate-payment" in prompt:
            text = (
                "Collect the receipt and transaction identifier "
                "before review."
            )

        elif "locked account" in prompt:
            text = (
                "Verify identity, then issue the approved reset flow."
            )

        elif "delete all personal" in prompt:
            text = (
                "Route the privacy request to human review."
            )

        elif "active incident" in prompt:
            text = (
                "Share only confirmed updates from the status page."
            )

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


# ============================================================
# LIVE CHAT PROVIDER
# ============================================================

class ChatProvider:

    def __init__(self, settings: Settings) -> None:

        if not settings.api_key:
            raise ValueError(
                f"{settings.provider.upper()}_API_KEY "
                "is required for live mode"
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
                    "require_parameters": True,
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
            }.get(
                settings.provider,
                settings.provider,
            ),
            api_key=SecretStr(settings.api_key),
            timeout=(
                int(settings.request_timeout_seconds * 1000)
                if settings.provider == "openrouter"
                else settings.request_timeout_seconds
            ),
            max_retries=0,
            **options,
        )

    # --------------------------------------------------------
    # GENERATE RESPONSE
    # --------------------------------------------------------

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

        # Reject incomplete generations.
        finish_reason = str(
            response.response_metadata.get("finish_reason")
        ).lower()

        if finish_reason != "stop":
            raise ValueError(
                "Provider returned an incomplete answer"
            )

        # Extract the response text.
        text = response.text

        if not text.strip():
            raise ValueError(
                "Provider returned no answer"
            )

        # Extract token usage.
        usage = response.usage_metadata or {}

        return ProviderResult(
            text=text,
            model=(
                response.response_metadata.get("model_name")
                or self.model
            ),
            usage=Usage(
                input_tokens=usage.get(
                    "input_tokens",
                    0,
                ),
                output_tokens=usage.get(
                    "output_tokens",
                    0,
                ),
            ),
        )


# ============================================================
# PROVIDER FACTORY
# ============================================================

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
