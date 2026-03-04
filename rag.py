# rag.py
import os
import requests
import chromadb
from chromadb.api.types import Documents, Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import fitz  # PyMuPDF
import re

class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom ChromaDB embedding function that strictly calls local Ollama."""
    def __init__(self, model_name, host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                response = requests.post(
                    f"{self.host}/api/embeddings",
                    json={"model": self.model_name, "prompt": text}
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            except requests.exceptions.RequestException as e:
                print(f"Embedding failed: {e}")
                # Fallback zero-vector to prevent ChromaDB from crashing
                embeddings.append([0.0] * 768) 
        return embeddings

class HybridRAG:
    def __init__(self, db_path="./data/chroma_db", embed_model="nomic-embed-text"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.emb_fn = OllamaEmbeddingFunction(model_name=embed_model)
        self.collection_name = "academic_pdfs"
        
        # Get or create vector collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, 
            embedding_function=self.emb_fn
        )
        self.bm25 = None
        self.bm25_docs = []
        self.bm25_metadata = []
        
        self._initialize_bm25()

    def _tokenize(self, text):
        """Simple tokenizer for BM25 keyword search."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def _initialize_bm25(self):
        """Rebuilds the BM25 index from ChromaDB documents."""
        try:
            all_data = self.collection.get()
            if all_data and all_data['documents']:
                self.bm25_docs = all_data['documents']
                self.bm25_metadata = all_data['metadatas']
                tokenized_docs = [self._tokenize(doc) for doc in self.bm25_docs]
                self.bm25 = BM25Okapi(tokenized_docs)
        except Exception as e:
            print(f"Could not initialize BM25: {e}")

    def embed_folder(self, pdf_folder, chunk_size=1000, chunk_overlap=200):
        """Chunks and embeds PDFs using LangChain's smart text splitter."""
        if not os.path.exists(pdf_folder):
            return 0

        # Smart chunking that respects paragraphs and sentences
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        docs, metadatas, ids = [], [], []
        doc_id = 0
        
        # Clear existing to prevent duplicates during testing
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name, 
                embedding_function=self.emb_fn
            )
        except ValueError:
            pass

        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_folder, filename)
                doc = fitz.open(filepath)
                text = "".join([page.get_text() for page in doc])
                
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    docs.append(chunk)
                    metadatas.append({"filename": filename, "filepath": filepath})
                    ids.append(f"chunk_{doc_id}")
                    doc_id += 1
                    
        if docs:
            self.collection.add(documents=docs, metadatas=metadatas, ids=ids)
            self._initialize_bm25() # Update keyword index
            
        return len(docs)

    def hybrid_search(self, query, top_k=5, alpha=0.5):
        """
        Combines Semantic (ChromaDB) and Keyword (BM25) search via Reciprocal Rank Fusion.
        Alpha controls the weight (0.5 = equal weight).
        """
        if not self.bm25 or not self.bm25_docs:
            return []

        # 1. Semantic Search (ChromaDB)
        semantic_results = self.collection.query(query_texts=[query], n_results=top_k * 2)
        semantic_docs = semantic_results['documents'][0]
        semantic_metas = semantic_results['metadatas'][0]
        
        # 2. Keyword Search (BM25)
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        # Score Semantic
        for rank, text in enumerate(semantic_docs):
            if text not in rrf_scores:
                rrf_scores[text] = {"score": 0.0, "meta": semantic_metas[rank]}
            rrf_scores[text]["score"] += alpha * (1 / (rank + 60)) # 60 is the standard RRF constant
            
        # Score Keyword
        for rank, idx in enumerate(bm25_top_indices):
            text = self.bm25_docs[idx]
            if text not in rrf_scores:
                rrf_scores[text] = {"score": 0.0, "meta": self.bm25_metadata[idx]}
            rrf_scores[text]["score"] += (1 - alpha) * (1 / (rank + 60))

        # Sort and return top_k
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]
        
        final_results = []
        for text, data in sorted_results:
            final_results.append({
                "text": text,
                "filename": data["meta"]["filename"],
                "filepath": data["meta"]["filepath"],
                "rrf_score": data["score"]
            })
            
        return final_results