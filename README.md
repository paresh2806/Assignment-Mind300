# Project Setup and Usage

This README outlines the steps to get the application up and running locally.

## Prerequisites

* **Python** 3.12 installed
* **Docker** and **Docker Compose** installed
* Git client (optional, but recommended)

## 1. Clone the Repository (if not already)

```bash
git clone https://github.com/paresh2806/Assignment-Mind300.git
cd Assignment-Mind300
```

## 2. Create and Activate a Python Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`
```

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 4. Start Qdrant via Docker Compose

```bash
docker compose up -d
```

This will:

* Pull the Qdrant image
* Start the Qdrant service exposed on port **6333**

## 5. Load Snapshot into Qdrant

1. Open your browser and navigate to:
   `http://localhost:6333/dashboard`
2. Click **Upload snapshot**.
3. Enter **Collection name**: `medicare_chunks_hybrid`.
4. In the file selector, choose:

   ```
   qudrent-snapshot/medicare_chunks_hybrid-8071113743018817-2025-06-27-12-02-37.snapshot
   ```
5. Confirm — the collection `medicare_chunks_hybrid` will be created and populated.

## 6. Run the FastAPI Application

In a new terminal (with your virtual environment still active):

```bash
uvicorn main:app --port 8000 --reload
```

* The API will be available on port **8000**.
* `--reload` enables auto-restart on code changes.

## 7. Explore the API Documentation

Open your browser and go to:

```
http://localhost:8000/docs
```

Here you can:

* View all available endpoints
* Try out the `/query` endpoint interactively

## 8. Usage: Query Endpoint

1. In the `/query` section of the Swagger UI, click **Try it out**.
2. Provide your query parameters, e.g.:

   ```json
   {
     "question": "What is the Medicare enrollment period?",
     "collection": "medicare_chunks_hybrid"
   }
   ```
3. Click **Execute** to see the response from the RAG system.

---

## Workflow Showcase

1. **PDF Readers**: PyPDF2 and pdfplumber (conventional readers) could not handle the complex PDF structure as medicare.
2. **LLMAParse**: Parsed the PDF into JSON (`Notebook/output_v2.json`) and Markdown (`Notebook/output_v2.md`) via LLMAparse.

   * Note: The Markdown dump lacks page metadata; the JSON dump includes page numbers so was not able to utilise md.
3. **Post-Processing Script**: Generates dynamic, topic-wise chunks with metadata (topic, subtopic, content, page) and outputs to `Notebook/topic_chunks.json`.
4. **Indexing Notebook**: Demonstrates detailed indexing steps in `Notebook/Chunk.ipynb`.
5. **Embedding Strategy**: Hybrid approach combining:

   * **Dense Embeddings** via GPT4All (`all-MiniLM-L6-v2`, embedding size 384).
   * **Sparse Embeddings** via BM25.
6. **RAG Orchestration**: Utilized `gemini-2.5-flash` model to orchestrate the retrieval-augmented generation pipeline.

#