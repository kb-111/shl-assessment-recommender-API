"""
Pydantic models that define the strict API contract.
Schema deviations fail the automated evaluator — never add extra fields.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def last_message_is_user(cls, v: list[Message]) -> list[Message]:
        if not v or v[-1].role != "user":
            raise ValueError("Last message must be from user")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"


class CatalogItem(BaseModel):
    """Internal representation of a scraped SHL catalog entry."""
    name: str
    url: str
    description: str = ""
    test_type: str = ""          # e.g. K, A, B, P, S, C, E
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    duration: Optional[str] = None
    remote_testing: bool = False
    adaptive: bool = False
    skills: list[str] = Field(default_factory=list)