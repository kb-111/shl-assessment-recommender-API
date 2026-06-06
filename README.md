# SHL Conversational Assessment Recommender

An AI-powered conversational recommendation system that helps recruiters and hiring managers discover the most relevant SHL assessments through natural language interactions.

## Features

* Conversational assessment recommendation
* Clarification of vague hiring requirements
* Context-aware refinement of recommendations
* Assessment comparison using catalog data
* Protection against hallucinated recommendations
* Prompt injection resistance
* Stateless API design
* Semantic search using vector embeddings
* FastAPI-based REST API

## Architecture

The system combines:

* SHL assessment catalog scraping
* Vector embeddings and semantic retrieval
* LLM-powered dialogue management
* Constraint extraction from conversation history
* Metadata-aware recommendation ranking

### Workflow

1. User submits conversation history.
2. System reconstructs hiring requirements from previous messages.
3. Relevant assessments are retrieved from the SHL catalog.
4. Results are reranked using extracted constraints.
5. Agent either:

   * asks a clarification question,
   * returns recommendations,
   * compares assessments,
   * or refuses out-of-scope requests.

## Tech Stack

* Python
* FastAPI
* Sentence Transformers
* FAISS
* Gemini API
* BeautifulSoup
* Pydantic
* Uvicorn

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Chat Endpoint

```http
POST /chat
```

Request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a Java developer"
    }
  ]
}
```

Response:

```json
{
  "reply": "Here are some recommended assessments.",
  "recommendations": [
    {
      "name": "Assessment Name",
      "url": "https://...",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd shl-recommender
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

## Running Locally

```bash
uvicorn app.main:app --reload
```

Server:

```text
http://localhost:8000
```

API Docs:

```text
http://localhost:8000/docs
```


