import streamlit as st
import os
import json
import requests
import fitz  # PyMuPDF
import chromadb
import time
import base64
import pandas as pd

from audio import Transcriber
from ollama_utils import get_ollama_models, refine_text_stream

# --- Configuration & Initialization ---
st.set_page_config(page_title="Academic Assistant", layout="wide")

# Initialize all Session States
if "raw_transcript" not in st.session_state:
    st.session_state.raw_transcript = ""
if "refined_text" not in st.session_state:
    st.session_state.refined_text = ""
if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []
if "citations" not in st.session_state:
    st.session_state.citations = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "system_logs" not in st.session_state:
    st.session_state.system_logs = []

# --- System Logger ---
def log_action(msg):
    """Logs actions to the sidebar and creates a popup toast."""
    timestamp = time.strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {msg}"
    if log_msg not in st.session_state.system_logs:
        st.session_state.system_logs.insert(0, log_msg)
        st.toast(msg)

# --- Core Functions ---

PROMPTS_FILE = "data/prompts.json"

def load_prompts():
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            log_action("WARNING: Prompts file empty, loading default.")
            pass
    return {"Default Academic": "Translate my speech into formal academic prose. Fix any grammar and structure."}

def save_prompt(name, prompt_text):
    log_action(f"Saving new prompt: '{name}'")
    prompts = load_prompts()
    prompts[name] = prompt_text
    os.makedirs(os.path.dirname(PROMPTS_FILE), exist_ok=True)
    with open(PROMPTS_FILE, "w") as f:
        json.dump(prompts, f, indent=4)

    
def highlight_and_render_pdf(filepath, search_text):
    """Highlights text in a PDF and returns base64 string for embedding."""
    log_action(f"Opening PDF for highlighting: {os.path.basename(filepath)}")
    doc = fitz.open(filepath)
    highlight_count = 0
    
    for page in doc:
        text_instances = page.search_for(search_text)
        for inst in text_instances:
            page.add_highlight_annot(inst)
            highlight_count += 1
            
    temp_path = f"./data/temp_highlighted.pdf"
    doc.save(temp_path)
    log_action(f"Created temporary highlighted PDF with {highlight_count} annotations.")
    
    with open(temp_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
    return base64_pdf, highlight_count

class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom ChromaDB embedding function that calls local Ollama."""
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        embeddings = []
        for text in input:
            try:
                response = requests.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": self.model_name, "prompt": text}
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to generate embedding: {e}")
        return embeddings

def embed_documents(pdf_folder, chunk_size, chunk_overlap, embed_model_name):
    """Chunks and embeds PDFs into ChromaDB via Ollama on command."""
    log_action("Initializing ChromaDB vector store...")
    client = chromadb.PersistentClient(path="./data/chroma_db")
    emb_fn = OllamaEmbeddingFunction(model_name=embed_model_name)
    
    log_action("Clearing old vector collection (if exists)...")
    try:
        client.delete_collection(name="academic_pdfs")
    except ValueError:
        pass 
        
    collection = client.create_collection(name="academic_pdfs", embedding_function=emb_fn)
    
    if not os.path.exists(pdf_folder):
        st.error(f"Folder not found: {pdf_folder}")
        log_action("ERROR: PDF folder missing.")
        return
        
    docs, metadatas, ids = [], [], []
    doc_id = 0
    
    log_action(f"Scanning folder '{pdf_folder}' for PDFs...")
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            try:
                doc = fitz.open(filepath)
                text = "".join([page.get_text() for page in doc])
                
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    chunk = text[start:end]
                    docs.append(chunk)
                    metadatas.append({"filename": filename, "filepath": filepath})
                    ids.append(f"chunk_{doc_id}")
                    doc_id += 1
                    start += (chunk_size - chunk_overlap)
                    
            except Exception as e:
                st.warning(f"Could not read {filename}: {e}")
                
    if docs:
        log_action(f"Embedding {len(docs)} chunks using model: {embed_model_name}. This may take time...")
        collection.add(documents=docs, metadatas=metadatas, ids=ids)
        log_action(f"Successfully created vector DB with {len(docs)} chunks.")
        st.success(f"Successfully embedded {len(docs)} chunks using {embed_model_name}.")

def extract_doi_and_cite(filename):
    """Extracts DOI from filename and fetches APA citation."""
    log_action(f"Extracting DOI from filename: {filename}")
    try:
        doi = filename.split('_')[0]
        if not doi.startswith("10."):
            log_action(f"Skipped citation: '{filename}' does not contain a DOI.")
            return f"Filename '{filename}' does not start with a valid DOI."
            
        url = f"https://citation.doi.org/{doi}"
        headers = {"Accept": "text/bibliography; style=apa"}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            log_action(f"Successfully fetched citation for {doi}.")
            return response.text.strip()
        else:
            return f"Could not fetch citation for DOI {doi}."
    except Exception as e:
        return f"Error extracting citation from {filename}: {e}"

def get_vosk_models(base_path="./models"):
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        return []
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
# --- Sidebar Configurations & Logging---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Track state changes for logging purposes
    def on_emb_change(): log_action(f"Embedding Model changed to: {st.session_state.emb_select}")
    def on_k_change(): log_action(f"Retrieval Chunks (k) changed to: {st.session_state.k_select}")
  
    st.subheader("Speech-to-Text")
    st.info("Using Faster-Whisper for local transcription. No internet required for STT.")

    st.subheader("Ollama Models")
    available_models = get_ollama_models()
    
    embed_model = st.selectbox("Embedding Model", available_models if available_models else ["No models found"], key="emb_select", on_change=on_emb_change)

    st.subheader("RAG Parameters")
    k_chunks = st.number_input("Retrieval Chunks (k)", min_value=1, value=5, step=1, key="k_select", on_change=on_k_change)
    chunk_size = st.number_input("Chunk Size", min_value=100, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, value=200, step=50)
    
    pdf_path = st.text_input("PDF Library Folder", "./data/pdf_library")

    st.divider()
    st.subheader("📋 System Logs")
    with st.container(height=200):
        for msg in st.session_state.system_logs:
            st.caption(msg)

# --- Main UI Workflow ---
st.title("🎙️ Academic Speech-to-Text & Citation Assistant")

tab1, tab2, tab3 = st.tabs(["1. Draft & Refine", "2. Source Highlights", "3. Final Output"])

# Initialize Session States
if "raw_transcript" not in st.session_state:
    st.session_state.raw_transcript = ""
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

# Initialize the Transcriber securely in session state so it doesn't reload
if "transcriber" not in st.session_state:
    with st.spinner("Loading Faster-Whisper STT Model..."):
        st.session_state.transcriber = Transcriber(model_size="base")

st.title("🎙️ Academic Speech-to-Text")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dictation")
    
    # Toggle Recording State
    if st.button("⏹️ Stop" if st.session_state.is_recording else "🔴 Start Recording"):
        st.session_state.is_recording = not st.session_state.is_recording
        
        if st.session_state.is_recording:
            st.session_state.transcriber.start_recording()
        else:
            st.session_state.transcriber.stop_recording()
            # Run one final transcription sweep when stopping
            final_text, _ = st.session_state.transcriber.transcribe_chunk()
            if final_text:
                st.session_state.raw_transcript += f" {final_text}"
        st.rerun()

    chart_placeholder = st.empty()
    text_placeholder = st.empty()

    # The UI update loop
    if st.session_state.is_recording:
        # Pull new audio and transcribe it in near real-time
        new_text, waveform = st.session_state.transcriber.transcribe_chunk()
        
        if len(waveform) > 0:
            df = pd.DataFrame({"Top": waveform, "Bottom": -waveform})
            chart_placeholder.bar_chart(df, height=150, color=["#333333", "#333333"])

        if new_text:
            st.session_state.raw_transcript += f" {new_text}"

        # Display current text
        text_placeholder.text_area("Live Transcript", st.session_state.raw_transcript, height=300, disabled=True)
        
        # Sleep briefly to avoid maxing out CPU on the UI thread, then rerun just this block
        time.sleep(1)
        st.rerun() 
        
    else:
        # Editable state when not recording
        st.session_state.raw_transcript = st.text_area("Edit Transcript", st.session_state.raw_transcript, height=300)

with col2:
    st.subheader("🧠 Stage 2: Ollama Refinement")
    
    # Model and Prompt Configuration
    available_models = get_ollama_models()
    ollama_model = st.selectbox(
        "Select Ollama Model", 
        available_models if available_models else ["No models found (Check Ollama status)"]
    )
    
    st.subheader("Prompt Library")
    prompts_dict = load_prompts()
    selected_prompt_name = st.selectbox("Choose a Prompt", list(prompts_dict.keys()))
    system_prompt = st.text_area("Edit Prompt", prompts_dict[selected_prompt_name], height=100)
    
    with st.expander("Save as New Prompt"):
        new_prompt_name = st.text_input("New Prompt Name")
        if st.button("Save Prompt"):
            if new_prompt_name and system_prompt:
                save_prompt(new_prompt_name, system_prompt)
                st.success(f"Saved '{new_prompt_name}'!")
                st.rerun()
    
    # Initialize refined_text in session state if it doesn't exist
    if "refined_text" not in st.session_state:
        st.session_state.refined_text = ""
    
    if st.button("✨ Refine Transcript"):
        if not st.session_state.raw_transcript.strip():
            st.warning("No transcript to refine. Please dictate or type something first.")
        elif ollama_model.startswith("No models"):
            st.error("Cannot connect to Ollama. Please ensure the service is running.")
        else:
            st.markdown("### Generating Refinement...")
            # Stream the output directly to the UI
            full_refined_text = st.write_stream(
                refine_text_stream(
                    st.session_state.raw_transcript, 
                    system_prompt, 
                    ollama_model
                )
            )
            # Save the final result to session state and force a rerun to load the tabs cleanly
            st.session_state.refined_text = full_refined_text
            st.rerun()
                
    # Display the split text block ONLY if we have refined text
    if st.session_state.refined_text:
        st.divider()
        edit_tab, preview_tab = st.tabs(["✏️ Edit Markdown", "👁️ Rendered Preview"])
        
        with edit_tab:
            # Callback to update the session state whenever the user hits Ctrl+Enter
            def update_text():
                st.session_state.refined_text = st.session_state.text_editor
                
            st.text_area(
                "Edit your draft here:", 
                value=st.session_state.refined_text, 
                height=400,
                key="text_editor",  # Binds the widget to st.session_state.text_editor
                on_change=update_text # Triggers when the user clicks out or hits Ctrl+Enter
            )
            
        with preview_tab:
            # Renders the markdown cleanly, uneditable
            st.markdown(st.session_state.refined_text)
    

with tab2:
    st.subheader("Stage 3: Source Indexing & Retrieval")
    
    if st.button("📚 Embed Documents"):
        log_action("Triggered Document Embedding process.")
        with st.spinner(f"Chunking and embedding with {embed_model}... This may take a while."):
            embed_documents(pdf_path, chunk_size, chunk_overlap, embed_model)
            
    st.divider()
            
    if st.button("🔍 Find Sources"):
        if st.session_state.refined_text:
            log_action(f"Querying VectorDB for top {k_chunks} sources...")
            with st.spinner("Searching VectorDB..."):
                client = chromadb.PersistentClient(path="./data/chroma_db")
                emb_fn = OllamaEmbeddingFunction(model_name=embed_model)
                try:
                    collection = client.get_collection(name="academic_pdfs", embedding_function=emb_fn)
                    results = collection.query(query_texts=[st.session_state.refined_text], n_results=int(k_chunks))
                    
                    retrieved = []
                    if results['documents']:
                        for i in range(len(results['documents'][0])):
                            retrieved.append({
                                "text": results['documents'][0][i],
                                "filename": results['metadatas'][0][i]['filename'],
                                "filepath": results['metadatas'][0][i]['filepath']
                            })
                    st.session_state.retrieved_chunks = retrieved
                    log_action(f"Successfully retrieved {len(retrieved)} chunks.")
                except ValueError:
                    st.error("Collection not found. Please click 'Embed Documents' first.")
                    log_action("ERROR: VectorDB collection missing.")
        else:
            st.warning("Please refine your text first before finding sources.")

    if st.session_state.retrieved_chunks:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("### Relevant Chunks")
            for idx, chunk in enumerate(st.session_state.retrieved_chunks):
                with st.expander(f"Source {idx + 1}: {chunk['filename']}", expanded=(idx==0)):
                    st.write(chunk['text'])
                    if st.button(f"📄 View PDF {idx+1}", key=f"view_btn_{idx}"):
                        log_action(f"Selected PDF to view: {chunk['filename']}")
                        st.session_state.view_pdf_path = chunk['filepath']
                        st.session_state.view_pdf_text = chunk['text']
                        
        with col_right:
            st.markdown("### Document Viewer")
            if "view_pdf_path" in st.session_state:
                with st.spinner("Applying highlights..."):
                    b64_pdf, count = highlight_and_render_pdf(
                        st.session_state.view_pdf_path, 
                        st.session_state.view_pdf_text
                    )
                    st.caption(f"Found and highlighted {count} instances.")
                    pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
            else:
                st.info("Click 'View PDF' on a chunk to display the document here.")

with tab3:
    st.subheader("Stage 4: Final Document & Citations")
    if st.button("📑 Generate Citations"):
        log_action("Triggered Citation Generation.")
        st.session_state.citations = []
        if st.session_state.retrieved_chunks:
            with st.spinner("Fetching APA Citations..."):
                seen_filenames = set()
                for chunk in st.session_state.retrieved_chunks:
                    fname = chunk['filename']
                    if fname not in seen_filenames:
                        citation = extract_doi_and_cite(fname)
                        st.session_state.citations.append(citation)
                        seen_filenames.add(fname)
                log_action("Completed Citation Generation.")
        else:
             st.warning("No sources retrieved yet.")

    st.markdown("### Final Academic Draft")
    st.write(st.session_state.refined_text)
    
    st.markdown("### References")
    for cite in st.session_state.citations:
        st.info(cite)