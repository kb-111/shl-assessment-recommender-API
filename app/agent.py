# """
# Agent orchestration: stateless, reconstructs intent from full message history.

# Pipeline per request:
#   1. Guardrail check (injection / off-topic)
#   2. Extract structured constraints via LLM
#   3. Decide mode: clarify | recommend | compare | refuse
#   4. Retrieve catalog candidates (semantic + reranked)
#   5. Format catalog context
#   6. Generate reply via LLM
#   7. Validate and sanitize output against catalog
#   8. Return ChatResponse-shaped dict
# """

# import logging
# from typing import Any

# from app.guardrails import classify_message, REFUSAL_INJECTION, REFUSAL_OFF_TOPIC
# from app.llm import extract_constraints, generate_reply
# from app.models import Message, CatalogItem
# from app.retriever import retrieve, get_by_names, Constraints

# log = logging.getLogger(__name__)

# # Count how many clarifying questions the assistant has already asked.
# # We derive this from conversation history to remain stateless.
# def _count_clarifications(messages: list[Message]) -> int:
#     """
#     Count assistant turns that ended without recommendations.
#     Used to avoid infinite clarification loops.
#     """
#     count = 0
#     for m in messages:
#         if m.role == "assistant":
#             # Heuristic: if assistant message is a question, it was clarifying
#             if "?" in m.content:
#                 count += 1
#     return count


# def _format_catalog_context(items: list[CatalogItem]) -> str:
#     """Render catalog items as a numbered plain-text list for the LLM context window."""
#     if not items:
#         return "No matching assessments found in catalog."

#     lines = []
#     for i, item in enumerate(items, 1):
#         line_parts = [f"{i}. Name: {item.name}"]
#         line_parts.append(f"   URL: {item.url}")
#         line_parts.append(f"   Type: {item.test_type}")
#         if item.description:
#             line_parts.append(f"   Description: {item.description[:200]}")
#         if item.duration:
#             line_parts.append(f"   Duration: {item.duration}")
#         if item.job_levels:
#             line_parts.append(f"   Levels: {', '.join(item.job_levels)}")
#         if item.skills:
#             line_parts.append(f"   Skills: {', '.join(item.skills[:10])}")
#         remote_str = "Yes" if item.remote_testing else "Not specified"
#         adaptive_str = "Yes" if item.adaptive else "No"
#         line_parts.append(f"   Remote: {remote_str} | Adaptive: {adaptive_str}")
#         lines.append("\n".join(line_parts))

#     return "\n\n".join(lines)


# def _extract_comparison_names(text: str) -> list[str]:
#     """
#     Heuristically extract assessment names from a comparison query.
#     Looks for patterns like "between OPQ and GSA" or "OPQ vs GSA".
#     """
#     import re
#     # Pattern: "between X and Y"
#     m = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)", text, re.IGNORECASE)
#     if m:
#         return [m.group(1).strip(), m.group(2).strip()]
#     # Pattern: "X vs Y" or "X versus Y"
#     m = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)", text, re.IGNORECASE)
#     if m:
#         return [m.group(1).strip(), m.group(2).strip()]
#     return []


# def _validate_recommendations(
#     raw_recs: list[dict[str, Any]],
#     catalog_items: list[CatalogItem],
# ) -> list[dict[str, Any]]:
#     """
#     Hallucination prevention: only pass through recommendations whose URLs
#     appear in the catalog items we retrieved. This is a hard filter.
#     """
#     valid_urls = {item.url for item in catalog_items}
#     valid_names = {item.name.lower() for item in catalog_items}

#     # Build lookup for fast catalog access
#     url_to_item = {item.url: item for item in catalog_items}
#     name_to_item = {item.name.lower(): item for item in catalog_items}

#     validated: list[dict[str, Any]] = []
#     for rec in raw_recs:
#         if not isinstance(rec, dict):
#             continue
#         url = rec.get("url", "")
#         name = rec.get("name", "")
#         name_lower = name.lower()

#         # Check URL match first (most reliable)
#         if url in valid_urls:
#             item = url_to_item[url]
#             validated.append({
#                 "name": item.name,
#                 "url": item.url,
#                 "test_type": item.test_type,
#             })
#         # Fallback: fuzzy name match
#         elif name_lower in valid_names:
#             item = name_to_item[name_lower]
#             validated.append({
#                 "name": item.name,
#                 "url": item.url,
#                 "test_type": item.test_type,
#             })
#         else:
#             # Try substring match
#             matched = False
#             for item in catalog_items:
#                 if name_lower and (name_lower in item.name.lower() or item.name.lower() in name_lower):
#                     validated.append({
#                         "name": item.name,
#                         "url": item.url,
#                         "test_type": item.test_type,
#                     })
#                     matched = True
#                     break
#             if not matched:
#                 log.warning("Hallucinated recommendation filtered out: name=%s url=%s", name, url)

#     # Deduplicate by URL
#     seen: set[str] = set()
#     deduped: list[dict[str, Any]] = []
#     for rec in validated:
#         if rec["url"] not in seen:
#             seen.add(rec["url"])
#             deduped.append(rec)

#     return deduped[:10]  # hard cap


# def process_chat(messages: list[Message]) -> dict[str, Any]:
#     """
#     Main agent entry point. Stateless.
#     Takes full conversation history, returns response dict.
#     """
#     last_user_message = messages[-1].content

#     # -----------------------------------------------------------------------
#     # Step 1: Guardrail check on the last user message
#     # -----------------------------------------------------------------------
#     classification = classify_message(last_user_message)

#     if classification == "injection":
#         return {
#             "reply": REFUSAL_INJECTION,
#             "recommendations": [],
#             "end_of_conversation": False,
#         }

#     if classification == "off_topic":
#         return {
#             "reply": REFUSAL_OFF_TOPIC,
#             "recommendations": [],
#             "end_of_conversation": False,
#         }

#     # -----------------------------------------------------------------------
#     # Step 2: Extract structured constraints from full conversation
#     # -----------------------------------------------------------------------
#     constraint_dict = extract_constraints(messages)
#     constraints = Constraints.from_dict(constraint_dict)

#     # -----------------------------------------------------------------------
#     # Step 3: Determine mode
#     # -----------------------------------------------------------------------
#     clarifications_so_far = _count_clarifications(messages)

#     if classification == "comparison":
#         mode = "compare"
#     elif _is_vague(messages, constraints, clarifications_so_far):
#         mode = "clarify"
#     else:
#         mode = "recommend"

#     # -----------------------------------------------------------------------
#     # Step 4: Retrieve catalog candidates
#     # -----------------------------------------------------------------------
#     catalog_items: list[CatalogItem] = []

#     if mode == "compare":
#         names = _extract_comparison_names(last_user_message)
#         if names:
#             catalog_items = get_by_names(names)
#         if not catalog_items:
#             # Fall back to semantic retrieval if name extraction failed
#             query = constraints.to_query_string() or last_user_message
#             catalog_items = retrieve(query, constraints, top_k=10)

#     elif mode == "recommend":
#         query = constraints.to_query_string()
#         if not query.strip() or query == "assessment":
#             query = last_user_message
#         catalog_items = retrieve(query, constraints, top_k=10)

#     # For clarify/refuse modes, we still retrieve a few items to have context
#     # but the LLM won't recommend them
#     else:
#         query = last_user_message
#         catalog_items = retrieve(query, constraints, top_k=5)

#     # -----------------------------------------------------------------------
#     # Step 5: Format catalog context for LLM
#     # -----------------------------------------------------------------------
#     catalog_context = _format_catalog_context(catalog_items)

#     # -----------------------------------------------------------------------
#     # Step 6: Generate reply
#     # -----------------------------------------------------------------------
#     result = generate_reply(messages, catalog_context, mode=mode)

#     # -----------------------------------------------------------------------
#     # Step 7: Validate recommendations (hallucination prevention)
#     # -----------------------------------------------------------------------
#     raw_recs = result.get("recommendations", [])
#     if raw_recs and catalog_items:
#         validated_recs = _validate_recommendations(raw_recs, catalog_items)
#     elif raw_recs:
#         # LLM returned recommendations but we have no catalog items — strip them
#         log.warning("LLM returned recommendations with no catalog context — stripping.")
#         validated_recs = []
#     else:
#         validated_recs = []

#     # In clarify mode, force empty recommendations
#     if mode in ("clarify",):
#         validated_recs = []

#     return {
#         "reply": result["reply"],
#         "recommendations": validated_recs,
#         "end_of_conversation": result.get("end_of_conversation", False),
#     }


# def _is_vague(
#     messages: list[Message],
#     constraints: Constraints,
#     clarifications_so_far: int,
# ) -> bool:
#     """
#     Decide if we need to clarify before recommending.

#     Not vague if:
#       - We have a role, OR
#       - We have technical skills, OR
#       - The user pasted a job description (>100 chars in last message), OR
#       - We have already asked 2+ clarifying questions (avoid infinite loops).
#     """
#     # Don't keep clarifying forever
#     if clarifications_so_far >= 2:
#         return False

#     # If we have extracted meaningful constraints, proceed
#     has_role = bool(constraints.role and constraints.role.strip())
#     has_skills = bool(constraints.technical_skills)
#     has_type_preference = any([
#         constraints.needs_personality,
#         constraints.needs_cognitive,
#         constraints.needs_situational,
#         constraints.needs_knowledge,
#     ])

#     if has_role or has_skills or has_type_preference:
#         return False

#     # If last user message is a long job description, extract from it
#     last_msg = messages[-1].content if messages else ""
#     if len(last_msg) > 120:
#         return False  # Assume a long message has enough context

#     return True

"""
Agent orchestration: stateless, reconstructs intent from full message history.

Pipeline per request:
  1. Guardrail check (injection / off-topic)
  2. Extract structured constraints via LLM
  3. Decide mode: clarify | recommend | compare | refuse
  4. Retrieve catalog candidates (semantic + reranked)
  5. Format catalog context
  6. Generate reply via LLM
  7. Validate and sanitize output against catalog
  8. Return ChatResponse-shaped dict
"""

import logging
from typing import Any

from app.guardrails import classify_message, REFUSAL_INJECTION, REFUSAL_OFF_TOPIC
from app.llm import extract_constraints, generate_reply
from app.models import Message, CatalogItem
from app.retriever import retrieve, get_by_names, Constraints

log = logging.getLogger(__name__)


def _count_clarifications(messages: list[Message]) -> int:
    count = 0
    for m in messages:
        if m.role == "assistant" and "?" in m.content:
            count += 1
    return count


def _format_catalog_context(items: list[CatalogItem]) -> str:
    if not items:
        return "No matching assessments found in catalog."
    lines = []
    for i, item in enumerate(items, 1):
        parts = [f"{i}. Name: {item.name}"]
        parts.append(f"   URL: {item.url}")
        parts.append(f"   Type: {item.test_type}")
        if item.description:
            parts.append(f"   Description: {item.description[:200]}")
        if item.duration:
            parts.append(f"   Duration: {item.duration}")
        if item.job_levels:
            parts.append(f"   Levels: {', '.join(item.job_levels)}")
        if item.skills:
            parts.append(f"   Skills: {', '.join(item.skills[:10])}")
        parts.append(f"   Remote: {'Yes' if item.remote_testing else 'Not specified'} | Adaptive: {'Yes' if item.adaptive else 'No'}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _extract_comparison_names(text: str) -> list[str]:
    import re
    m = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)", text, re.IGNORECASE)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    m = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)", text, re.IGNORECASE)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    return []


def _validate_recommendations(
    raw_recs: list[dict[str, Any]],
    catalog_items: list[CatalogItem],
) -> list[dict[str, Any]]:
    """
    Hallucination prevention: match LLM recommendations back to catalog items.

    Key fix: test_type is ALWAYS taken from the catalog, not from the LLM output.
    This means even if Gemini omits test_type, the recommendation is still returned
    correctly using the ground-truth catalog value.
    """
    # Build lookups (ground truth)
    url_to_item: dict[str, CatalogItem] = {}
    for item in catalog_items:
        url_to_item[item.url] = item
        url_to_item[item.url.rstrip("/")] = item  # normalised

    name_to_item: dict[str, CatalogItem] = {
        item.name.lower(): item for item in catalog_items
    }

    validated: list[dict[str, Any]] = []

    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue

        url = rec.get("url", "").rstrip("/")
        name = rec.get("name", "").strip()
        name_lower = name.lower()
        matched_item: CatalogItem | None = None

        # 1. URL match (exact or trailing-slash normalised)
        if url in url_to_item:
            matched_item = url_to_item[url]

        # 2. Exact name match
        if matched_item is None and name_lower in name_to_item:
            matched_item = name_to_item[name_lower]

        # 3. Substring name match (Gemini sometimes slightly rephrases)
        if matched_item is None and name_lower:
            for item in catalog_items:
                if name_lower in item.name.lower() or item.name.lower() in name_lower:
                    matched_item = item
                    break

        if matched_item is not None:
            # Always use catalog ground truth — never trust LLM for these values
            validated.append({
                "name": matched_item.name,
                "url": matched_item.url,
                "test_type": matched_item.test_type,
            })
        else:
            log.warning("Recommendation not in catalog — dropped: name=%r url=%r", name, url)

    # Deduplicate by URL, preserve order
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for rec in validated:
        if rec["url"] not in seen:
            seen.add(rec["url"])
            deduped.append(rec)

    log.info("Validated %d/%d LLM recommendations against catalog", len(deduped), len(raw_recs))
    return deduped[:10]


def _is_vague(
    messages: list[Message],
    constraints: Constraints,
    clarifications_so_far: int,
) -> bool:
    if clarifications_so_far >= 2:
        return False
    has_role = bool(constraints.role and constraints.role.strip())
    has_skills = bool(constraints.technical_skills)
    has_type_preference = any([
        constraints.needs_personality,
        constraints.needs_cognitive,
        constraints.needs_situational,
        constraints.needs_knowledge,
    ])
    if has_role or has_skills or has_type_preference:
        return False
    last_msg = messages[-1].content if messages else ""
    if len(last_msg) > 120:
        return False
    return True


def process_chat(messages: list[Message]) -> dict[str, Any]:
    """
    Main agent entry point. Stateless — full conversation history per call.
    """
    last_user_message = messages[-1].content

    # Step 1: Guardrails
    classification = classify_message(last_user_message)

    if classification == "injection":
        return {"reply": REFUSAL_INJECTION, "recommendations": [], "end_of_conversation": False}

    if classification == "off_topic":
        return {"reply": REFUSAL_OFF_TOPIC, "recommendations": [], "end_of_conversation": False}

    # Step 2: Extract structured constraints from full conversation
    constraint_dict = extract_constraints(messages)
    constraints = Constraints.from_dict(constraint_dict)

    # Step 3: Determine mode
    clarifications_so_far = _count_clarifications(messages)

    if classification == "comparison":
        mode = "compare"
    elif _is_vague(messages, constraints, clarifications_so_far):
        mode = "clarify"
    else:
        mode = "recommend"

    # Step 4: Retrieve catalog candidates
    catalog_items: list[CatalogItem] = []

    if mode == "compare":
        names = _extract_comparison_names(last_user_message)
        if names:
            catalog_items = get_by_names(names)
        if not catalog_items:
            query = constraints.to_query_string() or last_user_message
            catalog_items = retrieve(query, constraints, top_k=10)
    elif mode == "recommend":
        query = constraints.to_query_string()
        if not query.strip() or query == "assessment":
            query = last_user_message
        catalog_items = retrieve(query, constraints, top_k=10)
    else:
        catalog_items = retrieve(last_user_message, constraints, top_k=5)

    # Step 5: Format catalog context for LLM
    catalog_context = _format_catalog_context(catalog_items)

    # Step 6: Generate reply
    result = generate_reply(messages, catalog_context, mode=mode)

    # Step 7: Validate recommendations (hallucination prevention)
    raw_recs = result.get("recommendations", [])
    if raw_recs and catalog_items:
        validated_recs = _validate_recommendations(raw_recs, catalog_items)
    else:
        validated_recs = []

    # Force empty recommendations in clarify mode
    if mode == "clarify":
        validated_recs = []

    return {
        "reply": result["reply"],
        "recommendations": validated_recs,
        "end_of_conversation": result.get("end_of_conversation", False),
    }