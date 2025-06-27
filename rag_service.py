import os
import json
from typing import List, Tuple, Dict, Any
from pydantic_settings import BaseSettings
from gpt4all import Embed4All
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import SparseVector
import google.generativeai as genai


class Settings(BaseSettings):
    qdrant_url: str
    collection_name: str
    google_api_key: str

    class Config:
        env_file = ".env"


print("INFO:     Executing rag_service.py module level code...")
settings = Settings()

print("INFO:     Loading dense embedding model (Embed4All)...")
dense_encoder = Embed4All()

print("INFO:     Loading sparse embedding model (BM25)...")
sparse_model = SparseTextEmbedding("Qdrant/bm25", device="cpu")

print("INFO:     Connecting to Qdrant...")
qdrant_client = QdrantClient(url=settings.qdrant_url)

print("INFO:     Configuring Google Gemini...")
genai.configure(api_key=settings.google_api_key)
llm = genai.GenerativeModel('gemini-1.5-flash')

print("INFO:     RAG Service module initialized successfully.")


def get_dense_vector(text: str) -> List[float]:
    return dense_encoder.embed(text)


def get_sparse_vector(text: str) -> SparseVector:
    sparse_embeddings = list(sparse_model.embed([text]))
    sparse_embedding = sparse_embeddings[0]
    return SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist()
    )


def find_points(query: str) -> Tuple[str, List[int]]:
    dense_query_vector = get_dense_vector(query)
    sparse_query_vector = get_sparse_vector(query)

    prefetch = [
        models.Prefetch(query=dense_query_vector, using="gpt4all", limit=20),
        models.Prefetch(query=sparse_query_vector, using="bm25", limit=20)
    ]
    query_result = qdrant_client.query_points(
        collection_name=settings.collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=10
    )
    points = query_result.points
    sorted_points = sorted(points, key=lambda x: x.payload.get('chunk_order', 0))
    source_pages = sorted(list(set(p.payload.get('page') for p in sorted_points if p.payload.get('page') is not None)))
    knowledge = "\n".join(
        f"PAGE NUMBER: {hit.payload['page']}, TOPIC: {hit.payload['topic']}, SUB-TOPIC: {hit.payload['subtopic']}, CHUNK CONTENT: {hit.payload['content']}"
        for hit in sorted_points
    )
    return knowledge, source_pages


def get_rag_answer(query: str) -> Dict[str, Any]:
    knowledge_base, source_pages = find_points(query)
    token_count = llm.count_tokens(knowledge_base).total_tokens
    print(f"INFO:     Retrieved context has {token_count} tokens.")


    prompt = f"""
    You are an expert at answering questions based on the provided context.
    Your entire response must be a single, valid JSON object and nothing else.

    **Context**:
    {knowledge_base}

    **Question**:
    {query}

    **Answer Format (strictly follow this JSON structure)**:
    {{
      "answer": "Your detailed answer based *only* on the context provided.",
      "source_page": [list_of_integer_page_numbers_referenced_in_the_answer],
      "confidence_score": <A float score between 0.0 and 1.0 indicating how confident you are in the answer based on the context>
    }}

    Important: Your response must start with `{{` and end with `}}`. Do not include any text, code block markers, or formatting outside of the JSON object.
    """

    response = llm.generate_content(prompt)
    cleaned_text = response.text.strip().lstrip("```json").rstrip("```")

    try:
        response_data = json.loads(cleaned_text)
        response_data['token_count'] = token_count
        return response_data
    except json.JSONDecodeError:
        print(f"WARNING:  LLM did not return valid JSON. Response: {cleaned_text}")
        return {
            "answer": "I was unable to generate a structured answer. The raw response is: " + cleaned_text,
            "source_page": [],
            "confidence_score": 0.0,
            "token_count": token_count
        }