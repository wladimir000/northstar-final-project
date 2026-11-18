"""API, provider, trace, and evaluation contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=8000)
    max_tokens: int = Field(default=1024, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0, le=2)


class Usage(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ProviderResult(BaseModel):
    text: str
    model: str
    usage: Usage


class GenerateResponse(BaseModel):
    text: str
    model: str
    usage: Usage
    cache_hit: bool
    latency_ms: float = Field(ge=0)
    quality_score: float = Field(ge=0, le=1)
    trace_id: str


class EvaluationCase(BaseModel):
    case_id: str
    prompt: str
    expected_terms: list[str]
    repeat: int = Field(ge=1)


class CaseResult(BaseModel):
    case_id: str
    passed: bool
    cache_verified: bool | None
    quality_score: float
    latency_ms: float


class TraceRecord(BaseModel):
    trace_id: str
    name: str = "northstar-support-request"
    model: str
    prompt: str
    output: str
    latency_ms: float
    cache_hit: bool
    usage: Usage
    quality_score: float
    release: str = "local"
    environment: str = "development"
    status: Literal["ok", "error"] = "ok"
    spans: list[str] = Field(default_factory=list)
