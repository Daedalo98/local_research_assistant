# rag.py
import os
import re
import fitz  # PyMuPDF
import chromadb
from chromadb.api.types import Documents, Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from gemini_manager import GeminiManager

class DynamicGeminiEmbedding(chromadb.EmbeddingFunction):
    """
    Custom ChromaDB embedding function. 
    It dynamically updates which model it uses based on the GUI selection.
    """
    def __init__(self):
        self.manager = GeminiManager()
        self.active_model = None

    def __call__(self, input: Documents) -> Embeddings:
        if not self.active_model:
            raise ValueError("active_model must be set before generating embeddings.")
        return [self.manager.get_embedding(text, self.active_model) for text in input]


class AdvancedRAG:
    def __init__(self, db_path=None):
        if db_path is None:
            # Always resolve absolute path from rag.py location
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "data", "chroma_db")
            
        self.client = chromadb.PersistentClient(path=db_path)
        self.emb_fn = DynamicGeminiEmbedding()
        self.manager = GeminiManager()
        
        # Keyword search (BM25) initialization
        self.collection = None
        self.collection_name = ""
        self.bm25 = None
        self.bm25_docs = []
        self.bm25_metadata = []

    def _set_active_collection(self, active_model, collection_id=None):
        """Sets the collection and bm25 indices based on the active embedding model's dimensions"""
        self.emb_fn.active_model = active_model
        safe_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', active_model)
        self.collection_name = f"academic_pdfs_{safe_model_name}"
        if collection_id:
            self.collection_name += f"_{collection_id}"
        
        if active_model == "gemini-embedding-001":
            try:
                legacy_col = self.client.get_collection("academic_pdfs_gemini")
                try:
                    new_col = self.client.get_collection(self.collection_name)
                    if new_col.count() == 0:
                        self.client.delete_collection(self.collection_name)
                        legacy_col.modify(name=self.collection_name)
                except Exception:
                    legacy_col.modify(name=self.collection_name)
            except Exception:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, 
            embedding_function=self.emb_fn
        )
        self._initialize_bm25()

    def _tokenize(self, text):
        """Simple tokenizer for BM25 keyword search."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def _initialize_bm25(self):
        """Rebuilds the BM25 index from existing ChromaDB documents."""
        try:
            all_data = self.collection.get()
            if all_data and all_data['documents']:
                self.bm25_docs = all_data['documents']
                self.bm25_metadata = all_data['metadatas']
                tokenized_docs = [self._tokenize(doc) for doc in self.bm25_docs]
                self.bm25 = BM25Okapi(tokenized_docs)
        except Exception as e:
            print(f"Could not initialize BM25: {e}")

    def embed_folder(self, pdf_folder, chunk_size, chunk_overlap, active_model, force_rebuild=False, separators=None, progress_callback=None):
        """Chunks and embeds PDFs, using the model currently selected in the GUI."""
        if not os.path.exists(pdf_folder):
            return 0

        self._set_active_collection(active_model)

        if separators is None:
            separators = ["\n\n", "\n", ".", " ", ""]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        
        # Determine existing files to skip if not forcing rebuild
        existing_filenames = set()
        if force_rebuild:
            try:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name, 
                    embedding_function=self.emb_fn
                )
            except ValueError: pass
        else:
            try:
                existing_data = self.collection.get()
                if existing_data and existing_data['metadatas']:
                    existing_filenames = {m['filename'] for m in existing_data['metadatas'] if m and 'filename' in m}
            except Exception as e:
                print(f"Failed to check existing collection data: {e}")

        # Gather files to process
        all_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
        files_to_process = [f for f in all_files if force_rebuild or f not in existing_filenames]
        
        if not files_to_process:
            if progress_callback:
                progress_callback(1, 1, "All files already embedded.")
            return 0
            
        total_files = len(files_to_process)
        docs, metadatas, ids = [], [], []
        
        # To ensure unique IDs across incremental builds, figure out max existing ID
        doc_id = 0
        try:
             existing_data = self.collection.get()
             if existing_data and existing_data['ids']:
                 id_nums = [int(orig_id.split('_')[1]) for orig_id in existing_data['ids'] if '_' in orig_id]
                 if id_nums:
                     doc_id = max(id_nums) + 1
        except Exception: pass

        # Parse PDFs into chunks
        for idx, filename in enumerate(files_to_process):
            if progress_callback:
                progress_callback(idx, total_files, f"Parsing {filename} ({idx+1}/{total_files})...")
                
            filepath = os.path.join(pdf_folder, filename)
            try:
                doc = fitz.open(filepath)
                text = "".join([page.get_text() for page in doc])
                
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    docs.append(chunk)
                    metadatas.append({"filename": filename, "filepath": filepath})
                    ids.append(f"chunk_{doc_id}")
                    doc_id += 1
            except Exception as e:
                print(f"Failed to parse {filename}: {e}")
                
        # Inject into VectorDB using batching for speed
        total_chunks = len(docs)
        if total_chunks > 0:
            batch_size = 100
            for i in range(0, total_chunks, batch_size):
                if progress_callback:
                    progress_callback(i, total_chunks, f"Embedding chunk {i} parameters out of {total_chunks}...")
                
                batch_docs = docs[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                # Pre-calculate embeddings if using get_embeddings_batch (Chromadb takes care of this via the emb_fn usually, 
                # but to guarantee batch efficiency we invoke it manually and pass to add())
                embeddings = self.manager.get_embeddings_batch(batch_docs, active_model)
                
                self.collection.add(
                    documents=batch_docs, 
                    embeddings=embeddings,
                    metadatas=batch_metas, 
                    ids=batch_ids
                )
                
            if progress_callback:
                progress_callback(1, 1, "Updating Keyword index...")
            self._initialize_bm25()
            
        return total_chunks

    def embed_zotero_library(self, zotero_client, active_model, collection_id=None, limit=50, chunk_size=1000, chunk_overlap=200, force_rebuild=False, separators=None, progress_callback=None):
        import asyncio
        import json
        
        self._set_active_collection(active_model, collection_id)
        if separators is None: separators = ["\n\n", "\n", ".", " ", ""]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
        
        existing_filepaths = set()
        if force_rebuild:
            try:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(name=self.collection_name, embedding_function=self.emb_fn)
            except ValueError: pass
        else:
            try:
                existing_data = self.collection.get()
                if existing_data and existing_data['metadatas']:
                    existing_filepaths = {m.get('filepath') for m in existing_data['metadatas'] if m and 'filepath' in m}
            except Exception as e:
                print(f"Failed to check existing collection data: {e}")
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if progress_callback: progress_callback(0, 1, "Fetching items from Zotero...")
        
        items = []
        try:
            kwargs = {}
            if collection_id: kwargs["collection"] = collection_id
            for start in range(0, limit, 100):
                current_limit = min(100, limit - start)
                items_str = loop.run_until_complete(zotero_client.search_items(query="", limit=current_limit, start=start, **kwargs))
                new_items = json.loads(items_str)
                if not isinstance(new_items, list):
                    new_items = new_items.get("items", []) if isinstance(new_items, dict) else []
                if not new_items:
                    break
                items.extend(new_items)
        except Exception as e:
            if progress_callback: progress_callback(1, 1, f"Failed to fetch Zotero items: {e}")
            return 0
            
        # Also specifically fetch PDF attachments to get their text!
        attachments = []
        try:
            kwargs = {"itemType": "attachment"}
            if collection_id: kwargs["collection"] = collection_id
            for start in range(0, limit, 100):
                current_limit = min(100, limit - start)
                at_str = loop.run_until_complete(zotero_client.search_items(query="", limit=current_limit, start=start, **kwargs))
                at_json = json.loads(at_str)
                new_atts = at_json.get("items", []) if isinstance(at_json, dict) else (at_json if isinstance(at_json, list) else [])
                if not new_atts: break
                attachments.extend(new_atts)
        except Exception as e:
            pass

        docs, metadatas, ids = [], [], []
        doc_id = 0
        try:
             existing_data = self.collection.get()
             if existing_data and existing_data['ids']:
                 id_nums = [int(orig_id.split('_')[1]) for orig_id in existing_data['ids'] if '_' in orig_id]
                 if id_nums: doc_id = max(id_nums) + 1
        except Exception: pass
        
        total_items = len(items) + len(attachments)
        if total_items == 0:
            if progress_callback: progress_callback(1, 1, "No items found in Zotero.")
            return 0

        # Step 1: Process parent items (Metadata)
        processed_count = 0
        for item in items:
            processed_count += 1
            title = item.get("title") or item.get("data", {}).get("title", "Unknown Title")
            key = item.get("key", "unknown")
            item_type = item.get("itemType") or item.get("data", {}).get("itemType", "")
            
            # Skip attachments here, we do them next
            if item_type == "attachment":
                continue
                
            if progress_callback: progress_callback(processed_count, total_items, f"Processing Metadata context: {title[:30]}...")
            
            abstract = item.get("abstractNote") or item.get("data", {}).get("abstractNote", "")
            creators = item.get("creators") or item.get("data", {}).get("creators", [])
            authors = ", ".join([c.get("lastName", "") for c in creators if "lastName" in c])
            text = f"Title: {title}\nAuthors: {authors}\nAbstract: {abstract}"
            
            # Make sure we don't embed empty tombstones
            if not title and not abstract:
                continue
                
            # Skip if already embedded natively
            if f"zotero://{key}" in existing_filepaths and not force_rebuild:
                if progress_callback: progress_callback(processed_count, total_items, f"Skipping (Already Indexed): {title[:30]}...")
                continue
                
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                docs.append(chunk)
                metadatas.append({"filename": f"Zotero: {title}", "filepath": f"zotero://{key}"})
                ids.append(f"chunk_{doc_id}")
                doc_id += 1

        # Step 2: Process Attachments (Extract Full PDF Text)
        for att in attachments:
            processed_count += 1
            title = att.get("title") or att.get("data", {}).get("title", "PDF Document")
            key = att.get("key", "unknown")
            parent_key = att.get("parentItem") or att.get("data", {}).get("parentItem", key)
            
            if f"zotero://{parent_key}" in existing_filepaths and not force_rebuild:
                if progress_callback: progress_callback(processed_count, total_items, f"Skipping PDF (Already Indexed): {title[:30]}...")
                continue
            
            if progress_callback: progress_callback(processed_count, total_items, f"Extracting PDF text: {title[:30]}...")
            
            # Try to grab the full text via the extract_pdf_text MCP tool!
            try:
                pdf_text_json = loop.run_until_complete(zotero_client.extract_pdf_text(key))
                parsed_pdf = json.loads(pdf_text_json)
                pdf_content = parsed_pdf.get("content", "")
                
                if pdf_content:
                    chunks = text_splitter.split_text(pdf_content)
                    for chunk in chunks:
                        docs.append(chunk)
                        # We use the parent item key but append the attachment key for robust local lookups!
                        metadatas.append({"filename": f"Zotero PDF: {title}", "filepath": f"zotero://{parent_key}/{key}"})
                        ids.append(f"chunk_{doc_id}")
                        doc_id += 1
            except Exception as e:
                pass
                
        # Inject into VectorDB
        total_chunks = len(docs)
        if total_chunks > 0:
            batch_size = 100
            for i in range(0, total_chunks, batch_size):
                if progress_callback: progress_callback(i, total_chunks, f"Embedding chunk {i} out of {total_chunks}...")
                batch_docs = docs[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                embeddings = self.manager.get_embeddings_batch(batch_docs, active_model)
                self.collection.add(documents=batch_docs, embeddings=embeddings, metadatas=batch_metas, ids=batch_ids)
                
            if progress_callback: progress_callback(1, 1, "Updating Keyword index...")
            self._initialize_bm25()
            
        return total_chunks

    def generate_multi_queries(self, original_query, n_variations, active_model):
        """Node 1: Generates alternative search queries to improve retrieval."""
        prompt = f"Generate {n_variations} alternative academic search queries for: '{original_query}'. Return ONLY the queries separated by newlines."
        system = "You are an expert research librarian. Do not include introductory text."
        
        response = self.manager.generate_sync(prompt, system, active_model)
        queries = [q.strip("- ") for q in response.split('\n') if q.strip()]
        
        # Return the original query plus the requested number of variations
        return [original_query] + queries[:n_variations]

    def rerank_results(self, query, results, top_k, active_model, score_key='score'):
        """Node 3: Zero-shot LLM Re-ranking to score final relevance."""
        scored_results = []
        for res in results:
            prompt = f"Query: {query}\nDocument: {res['text']}\nScore the relevance from 0 to 10. Return ONLY the integer."
            try:
                score_str = self.manager.generate_sync(prompt, "You are a relevance scorer.", active_model)
                match = re.search(r'\d+', score_str)
                score = int(match.group()) if match else 5
            except Exception:
                score = 5 # Default tie-breaker if parsing fails
                
            # Weight the original score by the LLM's explicit relevance score
            base_score = res.get(score_key, 1.0)
            res['final_score'] = base_score * (score / 10)
            scored_results.append(res)
            
        return sorted(scored_results, key=lambda x: x['final_score'], reverse=True)[:top_k]

    def decompose_query(self, query, active_model):
        """Node 1b: Break a complex query into simpler sub-queries."""
        prompt = f"Decompose the following complex query into 2 to 3 simpler, distinct sub-queries for document retrieval. Return ONLY the sub-queries separated by newlines.\nQuery: '{query}'"
        system = "You are an expert research assistant. Do not include introductory text."
        response = self.manager.generate_sync(prompt, system, active_model)
        queries = [q.strip("- ") for q in response.split('\n') if q.strip()]
        return queries

    def advanced_search(self, query, top_k, use_multiquery, n_queries, use_decomposition, use_reranking, alpha, text_model, embed_model, status_callback=None):
        """
        Executes Pure Keyword and Hybrid (Context + Keyword) search.
        Retrieves chunks, combines them by file, and returns 'pure_keyword' and 'hybrid' result lists.
        """
        if status_callback: status_callback(f"Connecting to '{embed_model}' ChromaDB collection...")
        self._set_active_collection(embed_model)
        
        if not self.bm25 or not self.bm25_docs:
            if status_callback: status_callback("No documents found in the database. Aborting.")
            return {"pure_keyword": [], "hybrid": []}
            
        # Extract explicit keywords
        keyword_match = re.search(r'(?i)<keywords>(.*?)</keywords>', query, re.DOTALL)
        if keyword_match:
            keyword_query = keyword_match.group(1).strip()
            semantic_query = query.replace(keyword_match.group(0), "").strip()
        else:
            keyword_match = re.search(r'(?i)keywords:\s*(.*)', query)
            if keyword_match:
                keyword_query = keyword_match.group(1).strip()
                semantic_query = query.replace(keyword_match.group(0), "").strip()
            else:
                keyword_query = query
                semantic_query = query
                
        if not semantic_query.strip():
            semantic_query = keyword_query
            
        # Step 1: Query Expansion & Decomposition
        semantic_queries = [semantic_query]
        if use_decomposition:
            if status_callback: status_callback("Decomposing semantic query...")
            semantic_queries.extend(self.decompose_query(semantic_query, text_model))
            
        if use_multiquery:
            if status_callback: status_callback(f"Generating {n_queries} conceptual variations for Semantic Search...")
            expanded = []
            for sq in semantic_queries:
                expanded.extend(self.generate_multi_queries(sq, n_queries, text_model))
            semantic_queries = expanded
            
        # Step 2: Semantic Retrieval
        semantic_aggregated = {}
        if status_callback: status_callback("Performing Semantic Search...")
        for q in semantic_queries:
            semantic_res = self.collection.query(query_texts=[q], n_results=top_k * 2)
            if semantic_res['documents'] and semantic_res['documents'][0]:
                for rank, text in enumerate(semantic_res['documents'][0]):
                    if text not in semantic_aggregated:
                        meta = semantic_res['metadatas'][0][rank]
                        semantic_aggregated[text] = {"text": text, "filename": meta["filename"], "filepath": meta["filepath"], "semantic_score": 1 / (rank + 60)}
                    else:
                        semantic_aggregated[text]["semantic_score"] += 1 / (rank + 60)
                        
        # Step 3: Keyword Retrieval
        keyword_aggregated = {}
        if status_callback: status_callback("Performing Keyword Search...")
        tokenized_query = self._tokenize(keyword_query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
        
        for rank, idx in enumerate(bm25_top_indices):
            if bm25_scores[idx] > 0:
                text = self.bm25_docs[idx]
                if text not in keyword_aggregated:
                    meta = self.bm25_metadata[idx]
                    keyword_aggregated[text] = {"text": text, "filename": meta["filename"], "filepath": meta["filepath"], "keyword_score": bm25_scores[idx], "keyword_rrf": 1 / (rank + 60)}

        # Combine items by filepath
        def group_by_file(items_list, score_key):
            grouped = {}
            for item in items_list:
                fp = item['filepath']
                if fp not in grouped:
                    grouped[fp] = dict(item)
                else:
                    if item['text'] not in grouped[fp]['text']:
                        grouped[fp]['text'] += f"\n\n[...]\n\n{item['text']}"
                    grouped[fp][score_key] = max(grouped[fp][score_key], item[score_key])
            return list(grouped.values())

        # Build Pure Keyword list
        pure_keyword_list = group_by_file(list(keyword_aggregated.values()), 'keyword_score')
        pure_keyword_list = sorted(pure_keyword_list, key=lambda x: x['keyword_score'], reverse=True)[:top_k]

        # Build Hybrid list (RRF of Semantic and Keyword)
        hybrid_aggregated = {}
        for text, chunk in semantic_aggregated.items():
            if text not in hybrid_aggregated:
                hybrid_aggregated[text] = {"text": chunk["text"], "filename": chunk["filename"], "filepath": chunk["filepath"], "hybrid_score": 0.0}
            hybrid_aggregated[text]["hybrid_score"] += alpha * chunk['semantic_score']
            
        for text, chunk in keyword_aggregated.items():
            if text not in hybrid_aggregated:
                hybrid_aggregated[text] = {"text": chunk["text"], "filename": chunk["filename"], "filepath": chunk["filepath"], "hybrid_score": 0.0}
            hybrid_aggregated[text]["hybrid_score"] += (1 - alpha) * chunk['keyword_rrf']
            
        hybrid_list = group_by_file(list(hybrid_aggregated.values()), 'hybrid_score')
        hybrid_list = sorted(hybrid_list, key=lambda x: x['hybrid_score'], reverse=True)[:top_k]

        # Step 4: LLM Re-ranking
        if use_reranking:
            if status_callback: status_callback("Re-ranking Pure Keyword results...")
            pure_keyword_list = self.rerank_results(keyword_query, pure_keyword_list, top_k, text_model, score_key='keyword_score')
            if status_callback: status_callback("Re-ranking Hybrid results...")
            hybrid_list = self.rerank_results(query, hybrid_list, top_k, text_model, score_key='hybrid_score')
            
        return {"pure_keyword": pure_keyword_list, "hybrid": hybrid_list}