# """
# FastAPI application entry point.

# Endpoints:
#   GET  /health  — liveness check
#   POST /chat    — stateless conversational agent

# Cold start: embedding model + FAISS index are loaded on first /chat request,
# not at import time, so /health responds immediately.
# """

# import logging
# from contextlib import asynccontextmanager

# from fastapi import FastAPI, HTTPException, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse

# from app.models import ChatRequest, ChatResponse, HealthResponse, Recommendation
# from app.agent import process_chat
# from app.config import settings
# from fastapi import FastAPI
# import traceback
# import logging

# from app.agent import process_chat
# from app.models import ChatRequest



# logging.basicConfig(
#     level=getattr(logging, settings.log_level.upper(), logging.INFO),
#     format="%(asctime)s %(levelname)s %(name)s — %(message)s",
# )
# log = logging.getLogger(__name__)
# app = FastAPI()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Pre-warm retrieval index at startup to avoid cold-start latency on first /chat."""
#     log.info("Starting SHL Assessment Recommender…")
#     try:
#         # Import triggers model + index load
#         from app.retriever import _load_resources
#         _load_resources()
#         log.info("Retrieval resources loaded.")
#     except Exception as exc:
#         log.error("Startup resource loading failed (will retry on first request): %s", exc)
#     yield
#     log.info("Shutting down.")


# app = FastAPI(
#     title="SHL Assessment Recommender",
#     version="1.0.0",
#     description="Conversational SHL assessment selection API",
#     lifespan=lifespan,
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["GET", "POST"],
#     allow_headers=["*"],
# )


# @app.get("/health", response_model=HealthResponse)
# async def health() -> HealthResponse:
#     """Readiness check. Always returns 200 with status ok."""
#     return HealthResponse(status="ok")


# @app.post("/chat", response_model=ChatResponse)
# async def chat(request: ChatRequest) -> ChatResponse:
#     """
#     Stateless conversational endpoint.

#     Accepts full conversation history.
#     Returns agent reply + optional recommendations.
#     Schema is non-negotiable — never deviates.
#     """
#     log.info(
#         "POST /chat — %d messages, last user: %.80s",
#         len(request.messages),
#         request.messages[-1].content,
#     )

#     try:
#         result = process_chat(request.messages)
#     except Exception as exc:
#         log.exception("Unhandled error in process_chat: %s", exc)
#         raise HTTPException(status_code=500, detail="Internal server error") from exc

#     # Build and validate recommendations list
#     recommendations: list[Recommendation] = []
#     raw_recs = result.get("recommendations", [])

#     for rec in raw_recs:
#         if not isinstance(rec, dict):
#             continue
#         name = rec.get("name", "")
#         url = rec.get("url", "")
#         test_type = rec.get("test_type", "K")

#         # Final guard: must have both name and a real SHL URL
#         if not name or not url:
#             continue
#         if not url.startswith("https://www.shl.com"):
#             log.warning("Non-SHL URL filtered: %s", url)
#             continue

#         recommendations.append(Recommendation(name=name, url=url, test_type=test_type))

#     # Hard cap
#     recommendations = recommendations[:10]

#     response = ChatResponse(
#         reply=result.get("reply", "I'm sorry, something went wrong."),
#         recommendations=recommendations,
#         end_of_conversation=bool(result.get("end_of_conversation", False)),
#     )

#     log.info(
#         "Response — mode implied by recs=%d, end=%s, reply=%.80s",
#         len(recommendations),
#         response.end_of_conversation,
#         response.reply,
#     )
#     return response

# # @app.post("/chat")
# # def chat(req: ChatRequest):
# #     try:
# #         return process_chat(req.messages)

# #     except Exception as e:
# #         # NEVER expose raw error to client
# #         log.error("CHAT_FATAL_ERROR:\n%s", traceback.format_exc())

# #         return {
# #             "reply": "I’m having trouble processing your request right now. Please try again in a moment.",
# #             "recommendations": [],
# #             "end_of_conversation": False
# #         }
    
# @app.exception_handler(Exception)
# async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
#     log.exception("Unhandled exception: %s", exc)
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "Internal server error"},
#     )
"""
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models import ChatRequest, ChatResponse, HealthResponse, Recommendation
from app.agent import process_chat
from app.config import settings


# ---------------- LOGGING ----------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

log = logging.getLogger(__name__)


# ---------------- LIFESPAN ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting SHL Assessment Recommender…")
    try:
        from app.retriever import _load_resources
        _load_resources()
        log.info("Retrieval resources loaded.")
    except Exception as exc:
        log.error("Startup failed (will retry later): %s", exc)

    yield

    log.info("Shutting down.")


# ---------------- APP (ONLY ONCE) ----------------
app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    description="Conversational SHL assessment selection API",
    lifespan=lifespan,
)

# ---------------- MIDDLEWARE ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- ROUTES ----------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    log.info("POST /chat — %d messages", len(request.messages))

    try:
        result = process_chat(request.messages)

    except Exception as exc:
        log.exception("CHAT ERROR")
        raise HTTPException(status_code=500, detail="Internal server error")

    recommendations = []
    for rec in result.get("recommendations", []):
        if isinstance(rec, dict):
            if rec.get("name") and rec.get("url"):
                recommendations.append(
                    Recommendation(
                        name=rec["name"],
                        url=rec["url"],
                        test_type=rec.get("test_type", "K"),
                    )
                )

    return ChatResponse(
        reply=result.get("reply", ""),
        recommendations=recommendations[:10],
        end_of_conversation=result.get("end_of_conversation", False),
    )


# ---------------- GLOBAL ERROR HANDLER ----------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )