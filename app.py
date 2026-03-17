import streamlit as st
import os
import json
import time
import base64
import pandas as pd
import fitz

from gemini_manager import GeminiManager
from audio import Transcriber, generate_tts
from rag import AdvancedRAG
from trust_pipeline import TrustPipeline

st.set_page_config(page_title="Local Academic Assistant", layout="wide", page_icon="🧠")


PROMPTS_FILE = "data/prompts.json"

def load_prompts():
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r") as f: return json.load(f)
        except json.JSONDecodeError: pass
    return {"Default Academic": "Translate my speech into formal academic prose in Markdown format. Fix grammar and structure."}

def save_prompt(name, prompt_text):
    prompts = load_prompts()
    prompts[name] = prompt_text
    os.makedirs(os.path.dirname(PROMPTS_FILE), exist_ok=True)
    with open(PROMPTS_FILE, "w") as f: json.dump(prompts, f, indent=4)

@st.cache_resource
def get_gemini_manager(): return GeminiManager()

@st.cache_resource
def get_transcriber(): return Transcriber(model_size="base", silence_threshold=0.01)

@st.cache_resource
def get_rag_pipeline(): return AdvancedRAG()

@st.cache_data
def get_pages_to_highlight(filepath, target_texts_tuple):
    """Fast pre-filter: returns list of page numbers that contain the text."""
    try:
        doc = fitz.open(filepath)
        search_lines = set()
        for text in target_texts_tuple:
            lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
            search_lines.update(lines)
            
        search_lines = list(search_lines)
        pages_to_keep = []
        if search_lines:
            for page in doc:
                page_text = page.get_text()
                for line in search_lines:
                    if line in page_text:
                        pages_to_keep.append(page.number)
                        break
        return pages_to_keep
    except Exception as e:
        print(f"Error getting pages: {e}")
        return []

@st.cache_data
def get_highlighted_pdf(filepath, target_texts_tuple, pages_to_keep):
    """Highlights and extracts specific pages from a PDF."""
    try:
        doc = fitz.open(filepath)
        search_lines = set()
        for text in target_texts_tuple:
            lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
            search_lines.update(lines)
            
        search_lines = list(search_lines)
        if search_lines and pages_to_keep:
            for page_num in pages_to_keep:
                page = doc[page_num]
                for line in search_lines:
                    text_instances = page.search_for(line)
                    for inst in text_instances:
                        annot = page.add_highlight_annot(inst)
                        annot.update()
            
            doc.select(pages_to_keep)
        return doc.write()
    except Exception as e:
        print(f"Error highlighting PDF: {e}")
        return None
        
def init_session_state():
    defaults = {
        "raw_transcript": "",
        "enhanced_text": "",
        "is_recording": False,
        "is_editing": False,
        "retrieved_chunks": [],
        "system_logs": []
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session_state()
manager = get_gemini_manager()
transcriber = get_transcriber()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Master Controls")
    
    available_models = manager.get_models()
    default_models = ["No models found"] if not available_models else available_models

    # Safely find default indices
    text_idx = default_models.index("gemini-2.5-flash") if "gemini-2.5-flash" in default_models else 0
    embed_idx = default_models.index("gemini-embedding-001") if "gemini-embedding-001" in default_models else 0
    
    with st.expander("🧠 Active Models & Gen Options", expanded=True):
        text_model = st.selectbox("Text Generation Model", default_models, index=text_idx)
        embed_model = st.selectbox("Vector Embedding Model", default_models, index=embed_idx)
        st.divider()
        temperature = st.slider("Temperature", 0.0, 100.0, 1.0, 0.1, help="Higher values make output more random")
        max_tokens = st.number_input("Max Tokens (Verbosity)", 100, 8000, 2000, 100)
        streaming_on = st.toggle("Streaming Generation", value=True)
    
    # Initialize RAG without tying it to a specific model
    rag = get_rag_pipeline()

    with st.expander("📚 RAG Search Parameters", expanded=True):
        pdf_path = st.text_input("PDF Folder", "./data/pdf_library")
        chunk_size = st.number_input("Chunk Size", 100, 2000, 1000, 100)
        chunk_overlap = st.number_input("Chunk Overlap", 0, 500, 200, 50)
        k_chunks = st.number_input("Top-K Retrieval", 1, 20, 5)
        display_k = st.number_input("Display Top-K Sources", 1, 20, 3)
    
    with st.expander("⚙️ Vector Builder Utilities", expanded=False):
        force_rebuild = st.toggle("Force Rebuild Database", value=False, help="Delete existing embeddings and start fresh.")
        custom_separators = st.text_input("Custom Separators", value="\\n\\n, \\n, ., , \u200b", help="Comma-separated strings to split on.")

    with st.expander("🔬 Search Engine Toggles", expanded=False):
        use_decomposition = st.toggle("Query Decomposition", value=True, help="Break complex query into sub-queries.")
        use_multiquery = st.toggle("Multi-Query Expansion", value=False)
        n_queries = st.slider("Variations (n)", 1, 5, 3) if use_multiquery else 1
        use_reranking = st.toggle("LLM Zero-Shot Re-ranking", value=True)
        alpha = st.slider("Hybrid Search Alpha (0=Keyword, 1=Semantic)", 0.0, 1.0, 0.5, 0.1)

    st.divider()
    if st.button("🗑️ Wipe Session Memory", use_container_width=True, type="secondary"):
        st.session_state.raw_transcript = ""
        st.session_state.enhanced_text = ""
        st.session_state.retrieved_chunks = []
        st.rerun()

# --- Main UI ---
st.title("🎙️ Local Research Assistant")

tab1, tab2, tab3 = st.tabs(["1. Dictation & Enhancement", "2. VectorDB & RAG", "3. Trust Pipeline Output"])

# --- TAB 1: Dictation ---
with tab1:
    col_audio, col_text = st.columns([1, 2])
    
    with col_audio:
        st.subheader("Audio Ingestion")
        # Recording Button
        if st.button("🔴 Stop Recording" if st.session_state.is_recording else "🎙️ Start Recording", use_container_width=True):
            st.session_state.is_recording = not st.session_state.is_recording
            if st.session_state.is_recording:
                st.session_state.is_editing = False # Force out of edit mode
                transcriber.start_recording()
            else:
                transcriber.stop_recording()
                final_text, _ = transcriber.transcribe_chunk()
                if final_text: st.session_state.raw_transcript += f" {final_text}"
            st.rerun()

        # Prompt Manager
        st.divider()
        st.subheader("System Prompts")
        prompts_dict = load_prompts()
        selected_prompt_name = st.selectbox("Active Prompt", list(prompts_dict.keys()))
        system_prompt = st.text_area("Edit Current Prompt", prompts_dict[selected_prompt_name], height=150)
        
        with st.expander("Save as New Prompt"):
            new_prompt_name = st.text_input("New Prompt Name")
            if st.button("Save"):
                if new_prompt_name and system_prompt:
                    save_prompt(new_prompt_name, system_prompt)
                    st.success("Saved!")
                    st.rerun()

    with col_text:
        st.subheader("Transcript Pipeline")
        
        # Live Text Flow (No graphics)
        if st.session_state.is_recording:
            new_text, _ = transcriber.transcribe_chunk()
            if new_text: st.session_state.raw_transcript += f" {new_text}"
            st.markdown(f"> *Listening...*\n\n{st.session_state.raw_transcript}")
            time.sleep(0.5)
            st.rerun()
        else:
            # Manual Edit State
            if st.session_state.is_editing:
                st.session_state.raw_transcript = st.text_area("Modify Transcript", st.session_state.raw_transcript, height=200)
                if st.button("✅ Save Changes"):
                    st.session_state.is_editing = False
                    st.rerun()
            else:
                st.markdown(f"**Raw Transcript:**\n\n{st.session_state.raw_transcript}")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✏️ Modify Transcript", use_container_width=True):
                        st.session_state.is_editing = True
                        st.rerun()

                with col_btn2:
                    if st.button("✨ Enhance Text", use_container_width=True, type="primary"):
                        st.session_state.enhanced_text = ""
                        st.markdown("### Enhanced Output (.md)")
                        
                        text_placeholder = st.empty()
                        
                        if streaming_on:
                            # THE SPEED HACK: Bypass st.write_stream
                            full_text = ""
                            buffer = ""
                            with st.status(f"Generative pass using {text_model}...", expanded=True) as status:
                                status.write("Streaming tokens from API...")
                                for chunk in manager.generate_stream(st.session_state.raw_transcript, system_prompt, text_model, temperature, max_tokens):
                                    full_text += chunk
                                    buffer += chunk
                                    # Update UI only when buffer gets large or hits a natural break, freeing the React thread
                                    if len(buffer) > 30 or "\n" in buffer:
                                        text_placeholder.markdown(full_text + "▌") 
                                        buffer = ""
                                status.update(label="Text Enhancement Complete!", state="complete", expanded=False)
                            text_placeholder.empty() # Clear placeholder to avoid double display
                            st.session_state.enhanced_text = full_text
                        else:
                            with st.status(f"Generating Output with {text_model}...", expanded=True) as status:
                                status.write("Waiting for complete payload from API...")
                                full_text = manager.generate_sync(st.session_state.raw_transcript, system_prompt, text_model, temperature, max_tokens)
                                status.update(label="Text Enhancement Complete!", state="complete", expanded=False)
                                text_placeholder.empty()
                                st.session_state.enhanced_text = full_text
            
            # Show Enhanced text if it exists
            if st.session_state.enhanced_text and not st.session_state.is_editing:
                st.divider()
                st.markdown("### Enhanced Output (.md)")
                
                # Setup editable and renderable view
                mode = st.toggle("Edit Enhanced Text Mode")
                
                if mode:
                    edited = st.text_area("Edit Enhanced Text", st.session_state.enhanced_text, height=300, label_visibility="collapsed")
                    if edited != st.session_state.enhanced_text:
                        st.session_state.enhanced_text = edited
                else:
                    st.markdown(st.session_state.enhanced_text)

                col_dl, col_tts = st.columns(2)
                with col_dl:
                    st.download_button(
                        label="💾 Download as .md",
                        data=st.session_state.enhanced_text,
                        file_name="enhanced_transcript.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with col_tts:
                    if st.button("🔊 Read Aloud (Edge-TTS)", use_container_width=True):
                        with st.spinner("Generating Neural Audio..."):
                            audio_file = generate_tts(st.session_state.enhanced_text)
                            if audio_file:
                                st.audio(audio_file, format="audio/mp3")
                            else:
                                st.warning("Text is empty.")

# --- TAB 2: RAG Configuration ---
with tab2:
    st.subheader("Master Vector Database Controls")
    
    if st.button("📚 1. Embed Document Library", use_container_width=True):
        if "embed" not in embed_model.lower():
            st.warning(f"Warning: You are about to embed using '{embed_model}'. It is highly recommended to select an embedding model (like gemini-embedding-001) first.")
        else:
            separators = [s.strip().replace('\\n', '\n') for s in custom_separators.split(',') if s.strip()]
            if not separators:
                separators = ["\n\n", "\n", ".", " ", ""]
            
            with st.status("Embedding Library", expanded=True) as status:
                status.write(f"Connecting to {embed_model} isolated database...")
                progress_bar = st.progress(0, text="Preparing to embed documents...")
                
                def update_progress(current, total, msg):
                    progress = current / total if total > 0 else 1.0
                    progress = min(max(progress, 0.0), 1.0) # Clamp between 0 and 1
                    progress_bar.progress(progress, text=msg)
                    
                count = rag.embed_folder(
                    pdf_path, 
                    chunk_size, 
                    chunk_overlap, 
                    embed_model,
                    force_rebuild=force_rebuild,
                    separators=separators,
                    progress_callback=update_progress
                )
                progress_bar.progress(1.0, text="Complete!")
                status.update(label=f"Successfully vectorized {count} chunks using {embed_model}.", state="complete", expanded=False)
                
    if st.button("🔍 2. Test Advanced Search Strategy", use_container_width=True):
        target_text = st.session_state.enhanced_text if st.session_state.enhanced_text else st.session_state.raw_transcript
        if target_text:
            text_type = "Enhanced Text" if st.session_state.enhanced_text else "Raw Transcript"
            st.info(f"Querying using: **{text_type}**")
            
            with st.status("Executing RAG Pipeline...", expanded=True) as status:
                st.write(f"Querying using '{text_model}' and '{embed_model}'...")
                
                def rag_status(msg):
                    status.write(msg)
                    
                st.session_state.retrieved_chunks = rag.advanced_search(
                    target_text, top_k=k_chunks, use_multiquery=use_multiquery, 
                    n_queries=n_queries, use_decomposition=use_decomposition, use_reranking=use_reranking,
                    alpha=alpha, text_model=text_model, embed_model=embed_model, status_callback=rag_status
                )
                status.update(label="RAG Search Complete!", state="complete", expanded=False)
        else:
            st.warning("Provide text in Tab 1 first.")

    if st.session_state.retrieved_chunks and isinstance(st.session_state.retrieved_chunks, dict):
        st.markdown("### Retrieved Sources")
        
        for section_title, section_key, chunks_list in [
            ("🧠 Hybrid Search (Context + Keyword)", "hybrid", st.session_state.retrieved_chunks.get("hybrid", [])[:display_k]),
            ("🎯 Pure Keyword Search", "pure_keyword", st.session_state.retrieved_chunks.get("pure_keyword", [])[:display_k])
        ]:
            if not chunks_list:
                continue
            st.markdown(f"#### {section_title}")
            for idx, chunk in enumerate(chunks_list):
                unique_key = f"{section_key}_{idx}"
                # Get current chunk scoring
                score = chunk.get('final_score', chunk.get('semantic_score', chunk.get('keyword_score', 0)))
                with st.expander(f"[{idx+1}] {chunk['filename']} (Score: {score:.3f})"):
                    st.checkbox("Select this source for upgrading text", key=f"source_sel_{unique_key}")
                    st.markdown(chunk['text'])
                    
                    # File chunk pre-calculation logic
                    # All chunks for this file to know highlighting pages across the search
                    all_chunks_in_dict = []
                    all_chunks_in_dict.extend(st.session_state.retrieved_chunks.get("hybrid", []))
                    all_chunks_in_dict.extend(st.session_state.retrieved_chunks.get("pure_keyword", []))
                    file_chunks = [c['text'] for c in all_chunks_in_dict if c['filepath'] == chunk['filepath']]
                    
                    pages_to_keep = get_pages_to_highlight(chunk['filepath'], tuple(file_chunks))
                    num_pages = len(pages_to_keep)
                    
                    st.write(f"**Pages to be rendered:** {num_pages}")
                    if num_pages > 20:
                        st.warning("Suggestion: We recommend not rendering more than 20 pages as it may slow down your system.")
                        
                    if num_pages > 0 and st.button("📄 View Highlighted PDF", key=f"view_pdf_{unique_key}"):
                        with st.status(f"Loading {chunk['filename']}...", expanded=True) as status:
                            status.write("Injecting optimized highlights into vector bounds...")
                            pdf_bytes = get_highlighted_pdf(chunk['filepath'], tuple(file_chunks), pages_to_keep)
                            
                            if pdf_bytes:
                                status.write("Generating interactive IFrame encoded view...")
                                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#view=FitH" width="100%" height="800" type="application/pdf"></iframe>'
                                status.update(label="Highlight generation fully rendered!", state="complete", expanded=False)
                                st.markdown(pdf_display, unsafe_allow_html=True)
                            else:
                                status.update(label="Render failure", state="error", expanded=False)
                                st.error("Could not load or highlight PDF.")
                        
        st.divider()
        if st.button("✨ Upgrade Text with Selected Sources", type="primary"):
            selected_texts = []
            for sect_key in ["hybrid", "pure_keyword"]:
                for i, c in enumerate(st.session_state.retrieved_chunks.get(sect_key, [])[:display_k]):
                    if st.session_state.get(f"source_sel_{sect_key}_{i}", False):
                        selected_texts.append(c['text'])
            
            if not selected_texts:
                st.warning("Please select at least one source using the checkboxes above.")
            else:
                target_text = st.session_state.enhanced_text if st.session_state.enhanced_text else st.session_state.raw_transcript
                if not target_text:
                    st.warning("No base text available to upgrade. Please transcribe some audio in Tab 1 first.")
                else:
                    sources_combined = "\n\n---\n\n".join(selected_texts)
                    upgrade_prompt = f"Original Text:\n{target_text}\n\nAdditional Knowledge Sources:\n{sources_combined}\n\nPlease expertly integrate the additional knowledge into the original text. Improve the structure and flow."
                    system_msg = "You are an expert editor and academic writer."
                    
                    st.markdown("### Upgraded Output (.md)")
                    text_placeholder = st.empty()
                    
                    if streaming_on:
                        full_text = ""
                        buffer = ""
                        with st.status(f"Integrating knowledge via {text_model}...", expanded=True) as status:
                            status.write("Streaming synthesis...")
                            for text_chunk in manager.generate_stream(upgrade_prompt, system_msg, text_model, temperature, max_tokens):
                                full_text += text_chunk
                                buffer += text_chunk
                                if len(buffer) > 30 or "\n" in buffer:
                                    text_placeholder.markdown(full_text + "▌") 
                                    buffer = ""
                                    
                            status.update(label="Knowledge successfully fused!", state="complete", expanded=False)
                        text_placeholder.markdown(full_text)
                        st.session_state.enhanced_text = full_text.replace("▌", "")
                    else:
                        with st.status(f"Integrating knowledge via {text_model}...", expanded=True) as status:
                            status.write("Awaiting complete synthesis block...")
                            full_text = manager.generate_sync(upgrade_prompt, system_msg, text_model, temperature, max_tokens)
                            status.update(label="Knowledge successfully fused!", state="complete", expanded=False)
                        text_placeholder.markdown(full_text)
                        st.session_state.enhanced_text = full_text

# --- TAB 3: Trust Pipeline Output ---
with tab3:
    st.subheader("🛡️ Trust Pipeline: Academic Synthesis")
    
    if not st.session_state.retrieved_chunks:
        st.info("No sources retrieved yet. Please run a search in Tab 2 first.")
    else:
        st.markdown("### 1. Identify Sources for Pipeline")
        pipeline_chunks = []
        if isinstance(st.session_state.retrieved_chunks, dict):
            all_chunks_with_keys = []
            for sect_key in ["hybrid", "pure_keyword"]:
                for i, chunk in enumerate(st.session_state.retrieved_chunks.get(sect_key, [])):
                    all_chunks_with_keys.append((chunk, f"{sect_key}_{i}", sect_key))
                    
            for chunk, unique_key, sect_key in all_chunks_with_keys:
                label = "Hybrid" if sect_key == "hybrid" else "Keyword"
                if st.checkbox(f"[{label}] Include {chunk['filename']} in Trust Pipeline", value=st.session_state.get(f"source_sel_{unique_key}", False), key=f"trust_sel_{unique_key}"):
                    pipeline_chunks.append(chunk)
                
        if not pipeline_chunks:
            st.warning("Please select at least one source to continue.")
        else:
            base_text = st.session_state.enhanced_text if st.session_state.enhanced_text else st.session_state.raw_transcript
            
            st.markdown("### 2. Configure Output Style")
            tp = TrustPipeline(text_model)
            prompts = tp.get_versatile_prompts()
            selected_style = st.selectbox("Pipeline Generation Style:", list(prompts.keys()))
            prompt_instruction = st.text_area("Custom Instructions (Optional):", value=prompts[selected_style], height=100)
            
            if "trust_step_1_done" not in st.session_state:
                st.session_state.trust_step_1_done = False
                st.session_state.tp_enhanced_text = ""
                st.session_state.tp_citations_list = []
                
            if st.button("🚀 Step 1: Enhance Text & Extract DOIs", type="primary", use_container_width=True):
                if not base_text:
                    st.error("No base text provided. Please go to Tab 1 to dictate or type some text.")
                else:
                    st.markdown("---")
                    st.markdown("### Pipeline Execution Logs")
                    with st.status("Running Step 1...", expanded=True) as pt_status:
                        pt_status.write("Step A: Enhancing base text with highlighted chunks...")
                        sources_text = tp.format_sources(pipeline_chunks)
                        st.session_state.tp_enhanced_text = tp.step_a_enhance_text(base_text, sources_text, prompt_instruction)
                        pt_status.write("✓ Base text enhanced.")
                        
                        pt_status.write("Step B: Scanning for DOIs and retrieving citations via DOI API...")
                        st.session_state.tp_citations_list = tp.step_b_retrieve_citations(pipeline_chunks)
                        pt_status.write(f"✓ Found {len(st.session_state.tp_citations_list)} valid DOIs with citations.")
                        pt_status.update(label="Step 1 Complete!", state="complete", expanded=False)
                    st.session_state.trust_step_1_done = True
                    st.rerun()

            if st.session_state.get("trust_step_1_done"):
                st.markdown("### 3. Review & Add Citations")
                
                # Display current list of citations
                if st.session_state.tp_citations_list:
                    st.write("**Found Citations:**")
                    for idx, c in enumerate(st.session_state.tp_citations_list):
                        st.markdown(f"- {c}")
                else:
                    st.info("No DOIs found in the selected chunks.")
                    
                # Manual DOI entry
                col_doi1, col_doi2 = st.columns([3, 1])
                with col_doi1:
                    manual_doi = st.text_input("Add manual DOI (e.g. 10.1038/s41586-020-2649-2)", key="manual_doi_input")
                with col_doi2:
                    st.write("") # spacing
                    st.write("")
                    if st.button("Fetch & Add", use_container_width=True):
                        if manual_doi:
                            with st.spinner("Fetching citation..."):
                                cit = tp.fetch_citation(manual_doi.strip())
                                st.session_state.tp_citations_list.append(f"[Manual Entry - {manual_doi}] {cit}")
                            st.rerun()
                            
                st.divider()
                if st.button("🚀 Step 2: Generate Final Academic Paragraph", type="primary", use_container_width=True):
                    with st.status("Running Step 2: Final Synthesis...", expanded=True) as pt_status:
                        citations_text = "\n".join(st.session_state.tp_citations_list) if st.session_state.tp_citations_list else "No citations provided."
                        final_academic_text = tp.step_c_create_academic_paragraph(base_text, st.session_state.tp_enhanced_text, citations_text, prompt_instruction)
                        pt_status.update(label="Trust Pipeline Completed Successfully!", state="complete", expanded=False)
                    st.session_state.trust_pipeline_output = final_academic_text

            if st.session_state.get("trust_pipeline_output"):
                st.markdown("### Final Academic Output")
                
                edit_mode = st.toggle("Edit Trust Pipeline Output Mode")
                if edit_mode:
                    st.session_state.trust_pipeline_output = st.text_area(
                        "Edit Output", 
                        st.session_state.trust_pipeline_output, 
                        height=400, 
                        label_visibility="collapsed"
                    )
                else:
                    st.markdown(st.session_state.trust_pipeline_output)
                    
                st.download_button(
                    label="💾 Download Trust Pipeline Output as .md",
                    data=st.session_state.trust_pipeline_output,
                    file_name="trust_pipeline_academic_output.md",
                    mime="text/markdown",
                    use_container_width=True
                )