import json
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_service import get_rag_answer


class QueryRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    source_page: List[int]
    confidence_score: float
    token_count: int


app = FastAPI(
    title="RAG Query API",
    description="An API that uses a RAG pipeline to answer questions and reports token usage.",
    version="1.1.0",
)


@app.get("/")
async def read_root():
    return {"status": "ok", "message": "RAG API is running."}


@app.post("/query", response_model=AnswerResponse)
async def process_query(request: QueryRequest):

    try:
        print(f"INFO:     Received query: {request.question}")
        response_dict = get_rag_answer(request.question)
        print(f"INFO:     Returning structured response: {response_dict}")
        return response_dict

    except Exception as e:
        print(f"ERROR:    An unexpected error occurred in the endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {str(e)}"
        )