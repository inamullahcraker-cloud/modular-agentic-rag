import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.chunking import ChunkerConfig, TextChunker
from src.contracts import IngestionSummary, RAGResponse
from src.embedding_service import EmbeddingConfig, EmbeddingService
from src.generation import GenerationConfig, GenerationService
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.retriever import Retriever, RetrieverConfig
from src.vectordatabase import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Modular Production RAG API",
    version="1.0.0",
    description="Production-ready RAG backend powered by FastAPI, LangChain, FAISS, and Ollama.",
)

# Enable CORS for local client development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pipeline & Component Singletons Initialization
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate modular components
chunker = TextChunker(config=ChunkerConfig(chunk_size=400, chunk_overlap=50))
embedding_service = EmbeddingService(config=EmbeddingConfig())
# 384 is the standard dimensionality for sentence-transformers/all-MiniLM-L6-v2
vector_store = VectorStore(dimension=384)
retriever = Retriever(
    embedding_service=embedding_service,
    vector_store=vector_store,
    config=RetrieverConfig(top_k=3),
)
generation_service = GenerationService(config=GenerationConfig(model_name="mistral"))

# Assemble unified RAG pipeline with ingestion support
pipeline = RAGPipeline(
    retriever=retriever,
    generation_service=generation_service,
    vector_store=vector_store,
    embedding_service=embedding_service,
    chunker=chunker,
    config=RAGPipelineConfig(),
)


# ---------------------------------------------------------------------------
# API Request Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language question for the RAG pipeline.")
    top_k: Optional[int] = Field(default=None, gt=0, description="Optional custom top-k override.")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/ingest", response_model=IngestionSummary)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """Uploads document files (TXT, PDF, etc.), saves them locally, and executes RAG ingestion."""
    saved_paths = []
    try:
        for file in files:
            if not file.filename:
                continue
            file_path = DATA_DIR / file.filename
            contents = await file.read()
            file_path.write_bytes(contents)
            saved_paths.append(file_path)
            logger.info("Saved uploaded file: %s", file_path)

        if not saved_paths:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid files provided.")

        summary = pipeline.ingest_documents(saved_paths)
        
        # Map pipeline IngestionSummary to match API response schema & frontend expectations
        return IngestionSummary(
            total_documents_loaded=getattr(summary, "loaded_documents", len(saved_paths)),
            total_chunks_created=getattr(summary, "chunks_created", 0),
            vector_dimension=vector_store.dimension,
        )
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

@app.post("/api/query", response_model=RAGResponse)
async def query_rag(payload: QueryRequest):
    """Executes a query against the knowledge base and returns generated answers with source citations."""
    try:
        response = pipeline.run(query=payload.query, top_k=payload.top_k)
        
        # Robustly extract fields regardless of internal model naming ('question' vs 'query', 'sources' vs 'retrieved_sources')
        query_text = getattr(response, "query", getattr(response, "question", payload.query))
        answer_text = getattr(response, "answer", "")
        raw_sources = getattr(response, "retrieved_sources", getattr(response, "sources", []))
        
        formatted_sources = []
        for src in raw_sources:
            if isinstance(src, dict):
                formatted_sources.append({
                    "file_name": src.get("file_name", src.get("source", "Unknown Source")),
                    "page_number": src.get("page_number", 1),
                    "score": src.get("score", 0.0),
                    "content": src.get("content", ""),
                })
            else:
                formatted_sources.append({
                    "file_name": getattr(src, "file_name", getattr(src, "source", "Unknown Source")),
                    "page_number": getattr(src, "page_number", 1),
                    "score": getattr(src, "score", 0.0),
                    "content": getattr(src, "content", ""),
                })

        return RAGResponse(
            query=query_text,
            answer=answer_text,
            retrieved_sources=formatted_sources,
        )
    except Exception as exc:
        logger.exception("Query execution failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves an interactive web dashboard for document uploading and real-time Q&A chat."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Modular RAG Assistant</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-50 text-gray-800 font-sans min-h-screen flex flex-col">
        <header class="bg-indigo-600 text-white shadow-md p-4">
            <div class="container mx-auto flex justify-between items-center">
                <h1 class="text-xl font-bold tracking-wide">Modular Production RAG System</h1>
                <span class="text-xs bg-indigo-500 px-2 py-1 rounded">FastAPI + Ollama + LangChain</span>
            </div>
        </header>

        <main class="container mx-auto p-4 flex-1 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl">
            <!-- Left Column: Ingestion Panel -->
            <div class="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col h-fit">
                <h2 class="text-lg font-semibold text-gray-700 mb-3 border-b pb-2">1. Ingest Documents</h2>
                <p class="text-sm text-gray-500 mb-4">Upload document files to build your vector knowledge index.</p>
                
                <input type="file" id="fileInput" multiple class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 mb-4 cursor-pointer">
                
                <button onclick="uploadFiles()" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 px-4 rounded-lg transition duration-200">Upload & Ingest</button>
                
                <div id="ingestStatus" class="mt-4 text-xs font-mono p-3 bg-gray-50 rounded border hidden"></div>
            </div>

            <!-- Right Column: Interactive Chat / Q&A Panel -->
            <div class="md:col-span-2 bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col h-[650px]">
                <h2 class="text-lg font-semibold text-gray-700 mb-3 border-b pb-2">2. Ask Questions</h2>
                
                <!-- Chat History Box -->
                <div id="chatHistory" class="flex-1 overflow-y-auto space-y-4 pr-2 mb-4 bg-gray-50 p-4 rounded-lg border">
                    <div class="text-center text-gray-400 text-sm my-auto">Upload documents first, then ask questions about them here.</div>
                </div>

                <!-- Input Controls -->
                <div class="flex gap-2">
                    <input type="text" id="queryInput" placeholder="Ask a question about your documents..." class="flex-1 border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500" onkeypress="if(event.key === 'Enter') sendQuery()">
                    <button onclick="sendQuery()" class="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-5 py-2 rounded-lg text-sm transition duration-200">Ask</button>
                </div>
            </div>
        </main>

        <script>
            async function uploadFiles() {
                const fileInput = document.getElementById('fileInput');
                const statusDiv = document.getElementById('ingestStatus');
                
                if (fileInput.files.length === 0) {
                    alert('Please select at least one file.');
                    return;
                }

                const formData = new FormData();
                for (let i = 0; i < fileInput.files.length; i++) {
                    formData.append('files', fileInput.files[i]);
                }

                statusDiv.classList.remove('hidden');
                statusDiv.innerText = 'Uploading and indexing documents...';

                try {
                    const response = await fetch('/api/ingest', { method: 'POST', body: formData });
                    const result = await response.json();
                    if (response.ok) {
                        statusDiv.innerHTML = `<span class="text-green-600 font-bold">Success!</span><br>Loaded: ${result.total_documents_loaded} doc(s)<br>Chunks: ${result.total_chunks_created}<br>Vector Dim: ${result.vector_dimension}`;
                    } else {
                        statusDiv.innerHTML = `<span class="text-red-600 font-bold">Error:</span> ${result.detail}`;
                    }
                } catch (err) {
                    statusDiv.innerHTML = `<span class="text-red-600 font-bold">Network Error:</span> ${err.message}`;
                }
            }

            async function sendQuery() {
                const queryInput = document.getElementById('queryInput');
                const chatHistory = document.getElementById('chatHistory');
                const query = queryInput.value.trim();

                if (!query) return;

                if (chatHistory.querySelector('.text-center')) {
                    chatHistory.innerHTML = '';
                }

                // Append user bubble
                chatHistory.innerHTML += `
                    <div class="flex flex-col items-end">
                        <div class="bg-indigo-600 text-white rounded-2xl rounded-tr-none px-4 py-2 max-w-[80%] text-sm shadow-sm">${escapeHtml(query)}</div>
                        <span class="text-[10px] text-gray-400 mt-1">You</span>
                    </div>
                `;
                queryInput.value = '';
                chatHistory.scrollTop = chatHistory.scrollHeight;

                // Append temporary thinking bubble
                const loadingId = 'loading-' + Date.now();
                chatHistory.innerHTML += `
                    <div id="${loadingId}" class="flex flex-col items-start">
                        <div class="bg-gray-200 text-gray-600 rounded-2xl rounded-tl-none px-4 py-2 text-sm animate-pulse">Thinking...</div>
                    </div>
                `;
                chatHistory.scrollTop = chatHistory.scrollHeight;

                try {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query })
                    });
                    const result = await response.json();
                    
                    document.getElementById(loadingId).remove();

                    if (response.ok) {
                        let sourcesHtml = '';
                        if (result.retrieved_sources && result.retrieved_sources.length > 0) {
                            sourcesHtml = '<div class="mt-2 pt-2 border-t border-gray-200 text-[11px] text-gray-500"><strong>Sources:</strong><ul>';
                            result.retrieved_sources.forEach(src => {
                                sourcesHtml += `<li>📄 ${escapeHtml(src.file_name)} (Page ${src.page_number}) — Score: ${src.score.toFixed(3)}</li>`;
                            });
                            sourcesHtml += '</ul></div>';
                        }

                        chatHistory.innerHTML += `
                            <div class="flex flex-col items-start">
                                <div class="bg-white border text-gray-800 rounded-2xl rounded-tl-none px-4 py-3 max-w-[85%] text-sm shadow-sm">
                                    <div class="whitespace-pre-wrap">${escapeHtml(result.answer)}</div>
                                    ${sourcesHtml}
                                </div>
                                <span class="text-[10px] text-gray-400 mt-1">AI Assistant</span>
                            </div>
                        `;
                    } else {
                        chatHistory.innerHTML += `
                            <div class="flex flex-col items-start">
                                <div class="bg-red-50 border border-red-200 text-red-700 rounded-2xl rounded-tl-none px-4 py-2 text-sm">${escapeHtml(result.detail)}</div>
                            </div>
                        `;
                    }
                } catch (err) {
                    document.getElementById(loadingId).remove();
                    chatHistory.innerHTML += `
                        <div class="flex flex-col items-start">
                            <div class="bg-red-50 border border-red-200 text-red-700 rounded-2xl rounded-tl-none px-4 py-2 text-sm">Network error: ${escapeHtml(err.message)}</div>
                        </div>
                    `;
                }
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }

            function escapeHtml(text) {
                const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
                return text.replace(/[&<>"']/g, m => map[m]);
            }
        </script>
    </body>
    </html>
    """