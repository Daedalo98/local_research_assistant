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
    default_prompts = {
        "Default Academic": "Translate my speech into formal academic prose in Markdown format. Fix grammar and structure.",
        "Expand & Integrate": "Expand upon the user's base idea by deeply integrating information from the provided sources. Ensure all source contributions are cited inline.",
        "Scientific Article Style": "Rewrite the discourse to make it sound like a high-quality scientific or academic article. Improve the vocabulary, structure, and academic tone while synthesizing the sources.",
        "Retrieval & Summarization Focus": "Focus heavily on summarizing the retrieved information. Present the core findings of the sources clearly and concisely, using the base text only as context.",
        "Fully Cited Introduction": "Transform the text and sources into a fully cited academic introduction. Set up the context, outline the problem using the sources, and conclude with the primary research objective or thesis.",
        "Standard Academic Integration": "Integrate the knowledge seamlessly into the text, maintaining a formal academic tone.",
        "Literature Review Style": "Write this as a literature review section, comparing and contrasting the sources against the user's base idea.",
        "Critical Analysis": "Critically analyze the base text in light of the sources, pointing out strengths, potential gaps, and backing claims with citations.",
        "Direct Support": "Use the sourced information primarily to strongly back and validate the claims made in the user's text."
    }
    
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r") as f: 
                loaded = json.load(f)
                return loaded if loaded else default_prompts
        except json.JSONDecodeError: pass
    
    # Save default if not exists
    os.makedirs(os.path.dirname(PROMPTS_FILE), exist_ok=True)
    with open(PROMPTS_FILE, "w") as f: json.dump(default_prompts, f, indent=4)
    return default_prompts

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

@st.cache_data(ttl=300)
def get_zotero_collections():
    try:
        from mcp_client import ZoteroMCPClient
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        zc = ZoteroMCPClient()
        res = loop.run_until_complete(zc.manage_collections(action="list"))
        import json
        cols = json.loads(res)
        return {c.get("name"): c.get("key") for c in cols} if isinstance(cols, list) else {}
    except Exception as e:
        print(f"Failed to fetch collections: {e}")
        return {}

def resolve_zotero_local_path(filepath):
    if not filepath.startswith("zotero://"): return filepath
    parts = filepath.replace("zotero://", "").split("/")
    if len(parts) > 1:
        att_key = parts[1]
        pot_dir = os.path.expanduser(f"~/Zotero/storage/{att_key}")
        if os.path.exists(pot_dir):
            for f in os.listdir(pot_dir):
                if f.endswith(".pdf"):
                    return os.path.join(pot_dir, f)
    return ""

@st.cache_data
def get_pages_to_highlight(filepath, target_texts_tuple):
    """Fast pre-filter: returns list of page numbers that contain the text."""
    try:
        real_path = resolve_zotero_local_path(filepath)
        if not real_path or not os.path.exists(real_path):
            return []
            
        doc = fitz.open(real_path)
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
        real_path = resolve_zotero_local_path(filepath)
        if not real_path or not os.path.exists(real_path):
            return None
            
        doc = fitz.open(real_path)
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
        "is_editing_base": False,
        "is_editing_enhanced": False,
        "retrieved_chunks": [],
        "system_logs": [],
        "tp_enhanced_text": "",
        "tp_citations_list": [],
        "trust_step_1_done": False,
        "trust_pipeline_output": "",
        "is_editing_final": False
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session_state()
manager = get_gemini_manager()
transcriber = get_transcriber()
rag = get_rag_pipeline()

from mcp_client import ZoteroMCPClient
zclient = ZoteroMCPClient()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Master Controls")
    
    available_models = manager.get_models()
    default_models = ["No models found"] if not available_models else available_models

    text_idx = default_models.index("gemini-2.5-flash") if "gemini-2.5-flash" in default_models else 0
    embed_idx = default_models.index("gemini-embedding-001") if "gemini-embedding-001" in default_models else 0
    
    with st.expander("🧠 Active Models & Gen Options", expanded=True):
        text_model = st.selectbox("Text Generation Model", default_models, index=text_idx)
        embed_model = st.selectbox("Vector Embedding Model", default_models, index=embed_idx)
        st.divider()
        temperature = st.number_input("Temperature", 0.0, 1000.0, 0.7, 0.1, help="Higher values make output more random")
        max_tokens = st.number_input("Max Tokens (Verbosity)", 100, 10000, 2000, 100)
        streaming_on = st.toggle("Streaming Generation", value=True)

    with st.expander("📚 Zotero DB & RAG Parameters", expanded=True):
        zotero_collections = get_zotero_collections()
        target_collection = st.selectbox("Embed Collection/Subfolder", options=["All Items"] + list(zotero_collections.keys()), help="Select a specific Zotero Collection to embed, or All Items.")
        active_collection_key = zotero_collections.get(target_collection) if target_collection != "All Items" else None
        
        zotero_limit = st.number_input("Max Items to scan", 1, 5000, 500, 50)
        force_rebuild = st.toggle("Force Rebuild Database", value=False, help="Delete existing embeddings and start fresh.")
        custom_separators = st.text_input("Custom Separators", value="\\n\\n, \\n, ., , \u200b")
        st.divider()
        chunk_size = st.number_input("Chunk Size", 100, 2000, 1000, 100)
        chunk_overlap = st.number_input("Chunk Overlap", 0, 500, 200, 50)
        k_chunks = st.number_input("Top-K Retrieval", 1, 20, 5)
        display_k = st.number_input("Display Top-K Sources", 1, 20, 3)
    
    with st.expander("🔬 Search Engine Toggles", expanded=False):
        use_decomposition = st.toggle("Query Decomposition", value=True)
        use_multiquery = st.toggle("Multi-Query Expansion", value=False)
        n_queries = st.slider("Variations (n)", 1, 5, 3) if use_multiquery else 1
        use_reranking = st.toggle("LLM Zero-Shot Re-ranking", value=True)
        alpha = st.slider("Hybrid Search Alpha (0=Keyword, 1=Semantic)", 0.0, 1.0, 0.5, 0.1)

    st.divider()
    if st.button("🗑️ Wipe Session Memory", use_container_width=True, type="secondary"):
        st.session_state.clear()
        st.rerun()

# --- Main UI Pipeline ---
st.title("🎙️ Local Research Assistant (Zotero Edition)")
st.caption("A completely unified step-by-step pipeline ensuring maximum control, exclusively powered by Zotero.")

st.header("Step 1: Talk & Edit Base Text")
with st.container(border=True):
    col_audio, col_text = st.columns([1, 2])
    with col_audio:
        st.subheader("Audio Ingestion")
        st.info("Record your thoughts to begin the process, or manually type in the editor on the right.", icon="ℹ️")
        if st.button("🔴 Stop Recording" if st.session_state.is_recording else "🎙️ Start Recording", use_container_width=True):
            st.session_state.is_recording = not st.session_state.is_recording
            if st.session_state.is_recording:
                st.session_state.is_editing_base = False
                transcriber.start_recording()
            else:
                transcriber.stop_recording()
                final_text, _ = transcriber.transcribe_chunk()
                if final_text: st.session_state.raw_transcript += f" {final_text}"
            st.rerun()

    with col_text:
        st.subheader("Base Transcript")
        if st.session_state.is_recording:
            new_text, _ = transcriber.transcribe_chunk()
            if new_text: st.session_state.raw_transcript += f" {new_text}"
            st.markdown(f"> *Listening...*\n\n{st.session_state.raw_transcript}")
            time.sleep(0.5)
            st.rerun()
        else:
            mode_base = st.toggle("Edit Base Text Mode", value=st.session_state.is_editing_base, key="toggle_edit_base")
            if mode_base:
                st.session_state.is_editing_base = True
                edited = st.text_area("Edit Transcript", st.session_state.raw_transcript, height=200, label_visibility="collapsed")
                if edited != st.session_state.raw_transcript:
                    st.session_state.raw_transcript = edited
            else:
                st.session_state.is_editing_base = False
                if st.session_state.raw_transcript:
                    st.markdown(st.session_state.raw_transcript)
                else:
                    st.caption("*No text provided yet. Please record audio or toggle Edit mode to type.*")


st.header("Step 2: Enhance via LLM")
with st.container(border=True):
    col_prompt, col_output = st.columns([1, 2])
    with col_prompt:
        st.subheader("System Prompts")
        prompts_dict = load_prompts()
        selected_prompt_name = st.selectbox("Active Prompt", list(prompts_dict.keys()), help="Select a system prompt to guide the LLM's enhancement style.")
        system_prompt = st.text_area("Edit Current Prompt", prompts_dict[selected_prompt_name], height=150)
        
        with st.expander("Save / Modify Prompt"):
            new_prompt_name = st.text_input("Prompt Name", value=selected_prompt_name)
            if st.button("Save Prompt"):
                if new_prompt_name and system_prompt:
                    save_prompt(new_prompt_name, system_prompt)
                    st.success("Saved!")
                    st.rerun()
                    
        if st.button("✨ Enhance Text", use_container_width=True, type="primary"):
            if not st.session_state.raw_transcript.strip():
                st.warning("Please provide base text in Step 1 first.")
            else:
                st.session_state.enhanced_text = ""
                text_placeholder = st.empty()
                if streaming_on:
                    full_text = ""
                    buffer = ""
                    with st.status(f"Generative pass using {text_model}...", expanded=True) as status:
                        for chunk in manager.generate_stream(st.session_state.raw_transcript, system_prompt, text_model, temperature, max_tokens):
                            full_text += chunk
                            buffer += chunk
                            if len(buffer) > 30 or "\n" in buffer:
                                text_placeholder.markdown(full_text + "▌") 
                                buffer = ""
                        status.update(label="Text Enhancement Complete!", state="complete", expanded=False)
                    text_placeholder.empty() 
                    st.session_state.enhanced_text = full_text
                else:
                    with st.status(f"Generating Output with {text_model}...", expanded=True) as status:
                        full_text = manager.generate_sync(st.session_state.raw_transcript, system_prompt, text_model, temperature, max_tokens)
                        status.update(label="Text Enhancement Complete!", state="complete", expanded=False)
                        st.session_state.enhanced_text = full_text
    
    with col_output:
        st.subheader("Enhanced Output (.md)")
        if st.session_state.enhanced_text:
            mode_enh = st.toggle("Edit Enhanced Text Mode", value=st.session_state.is_editing_enhanced, key="toggle_edit_enh")
            if mode_enh:
                st.session_state.is_editing_enhanced = True
                edited = st.text_area("Edit Enhanced Text", st.session_state.enhanced_text, height=300, label_visibility="collapsed")
                if edited != st.session_state.enhanced_text: st.session_state.enhanced_text = edited
            else:
                st.session_state.is_editing_enhanced = False
                st.markdown(st.session_state.enhanced_text)
        else:
            st.caption("*No enhanced text generated yet.*")


st.header("Step 3: Embed Zotero Knowledge Base")
with st.container(border=True):
    subfolder_str = f"'{target_collection}'" if target_collection != "All Items" else "All Items"
    st.info(f"Targeting: **{subfolder_str}**. Incremental saving is active; existing documents are skipped.", icon="📚")
    
    if st.button("🔄 Sync & Embed Zotero Library to VectorDB", use_container_width=True):
        separators = [s.strip().replace('\\n', '\n') for s in custom_separators.split(',') if s.strip()]
        if not separators: separators = ["\n\n", "\n", ".", " ", ""]
        
        with st.status("Embedding Zotero Library", expanded=True) as status:
            progress_bar = st.progress(0, text="Preparing to embed documents...")
            def update_progress(current, total, msg):
                progress = min(max(current / total if total > 0 else 1.0, 0.0), 1.0)
                progress_bar.progress(progress, text=msg)
                
            count = rag.embed_zotero_library(
                zotero_client=zclient, active_model=embed_model, collection_id=active_collection_key,
                limit=zotero_limit, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                force_rebuild=force_rebuild, separators=separators, progress_callback=update_progress
            )
            progress_bar.progress(1.0, text="Complete!")
            status.update(label=f"Successfully indexed {count} new chunks into '{embed_model}' DB.", state="complete", expanded=False)


# Prep TrustPipeline class globally since 4/5 depend on it
tp = TrustPipeline(text_model, zotero_client=zclient)

st.header("Step 4: Execute Advanced RAG Search & Extract Citations")
with st.container(border=True):
    target_text = st.session_state.enhanced_text if st.session_state.enhanced_text else st.session_state.raw_transcript
    col_btn, col_info = st.columns([1, 2])
    with col_info:
        if target_text:
            text_type = "Enhanced Text" if st.session_state.enhanced_text else "Base Transcript"
            st.info(f"The search will query the database using the **{text_type}**, then immediately fetch corresponding APA Zotero Citations.", icon="ℹ️")
        else:
            st.warning("Please provide/generate text in Step 1 or 2 to query the database.")

    with col_btn:
        if st.button("🔍 Execute Advanced Search & Citations", use_container_width=True, type="primary"):
            if not target_text:
                st.error("No query text available.")
            else:
                with st.status("Executing RAG Pipeline & Citation Extraction Phase...", expanded=True) as status:
                    def rag_status(msg): status.write(f"⚙️ {msg}")    
                    st.session_state.retrieved_chunks = rag.advanced_search(
                        target_text, top_k=k_chunks, use_multiquery=use_multiquery, 
                        n_queries=n_queries, use_decomposition=use_decomposition, use_reranking=use_reranking,
                        alpha=alpha, text_model=text_model, embed_model=embed_model, status_callback=rag_status
                    )
                    
                    # Extract Citations automatically for the gathered chunks
                    status.write("⚙️ Connecting to Zotero to generate APA Citations...")
                    all_chunks = st.session_state.retrieved_chunks.get("hybrid", []) + st.session_state.retrieved_chunks.get("pure_keyword", [])
                    st.session_state.tp_citations_list = tp.step_b_retrieve_citations(all_chunks)
                    
                    status.update(label="Search & Extraction Complete!", state="complete", expanded=False)
                st.session_state.trust_step_1_done = True

    if st.session_state.retrieved_chunks and isinstance(st.session_state.retrieved_chunks, dict):
        st.markdown("### Search Results & Generated Citations")
        st.caption("Review search results below and select the sources you wish to include in your final synthesis (Step 5).")
        
        for section_title, section_key, chunks_list in [
            ("🧠 Hybrid Search (Context + Keyword)", "hybrid", st.session_state.retrieved_chunks.get("hybrid", [])[:display_k]),
            ("🎯 Pure Keyword Search", "pure_keyword", st.session_state.retrieved_chunks.get("pure_keyword", [])[:display_k])
        ]:
            if not chunks_list: continue
            st.markdown(f"#### {section_title}")
            for idx, chunk in enumerate(chunks_list):
                unique_key = f"{section_key}_{idx}"
                score = chunk.get('final_score', chunk.get('semantic_score', chunk.get('keyword_score', 0)))
                
                with st.expander(f"[{idx+1}] {chunk['filename']} (Score: {score:.3f})"):
                    st.checkbox("✅ Include this source & its citation in Final Synthesis", key=f"source_sel_{unique_key}")
                    st.markdown(chunk['text'])
                    
                    # Pre-calculation for PDF rendering
                    all_chunks_in_dict = st.session_state.retrieved_chunks.get("hybrid", []) + st.session_state.retrieved_chunks.get("pure_keyword", [])
                    file_chunks = [c['text'] for c in all_chunks_in_dict if c['filepath'] == chunk['filepath']]
                    
                    try:
                        pages_to_keep = get_pages_to_highlight(chunk['filepath'], tuple(file_chunks))
                        num_pages = len(pages_to_keep)
                        st.write(f"**Pages to be rendered:** {num_pages}")
                        
                        if num_pages > 30:
                            st.warning(f"⚠️ Warning: Rendering {num_pages} pages may slow down your system. 30 pages or fewer is recommended.")
                            
                        if num_pages > 0 and st.button("📄 View Highlighted local PDF", key=f"view_pdf_{unique_key}"):
                            with st.status(f"Loading local {chunk['filename']}...", expanded=True) as status:
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
                                    st.error("Could not locally load or highlight PDF. PyMuPDF requires the attachment to be openable.")
                    except Exception as e:
                        st.error(f"Cannot render PDF: {e}")

        # Display global Citations recovered
        if st.session_state.trust_step_1_done and st.session_state.tp_citations_list:
            st.divider()
            with st.expander("📚 View Auto-Generated Extracted Citations", expanded=False):
                for c in st.session_state.tp_citations_list:
                    st.markdown(f"- {c}")

st.header("Step 5: Trust Pipeline - Final Academic Synthesis")
with st.container(border=True):
    if not st.session_state.trust_step_1_done:
        st.warning("Please Execute Advanced Search (Step 4) to prepare citations before generating final synthesis.")
    else:
        pipeline_chunks = []
        if isinstance(st.session_state.retrieved_chunks, dict):
            for sect_key in ["hybrid", "pure_keyword"]:
                for i, chunk in enumerate(st.session_state.retrieved_chunks.get(sect_key, [])):
                    if st.session_state.get(f"source_sel_{sect_key}_{i}", False):
                        pipeline_chunks.append(chunk)

        prompts_dict = load_prompts()
        selected_style = st.selectbox("Pipeline Synthesis Prompt:", list(prompts_dict.keys()), help="Select the instruction to guide final integration.")
        prompt_instruction = st.text_area("Custom Instructions (Optional):", value=prompts_dict[selected_style], height=100)

        if st.button("🚀 Generate Final Academic Synthesis", type="primary", use_container_width=True):
            if not pipeline_chunks:
                st.warning("No sources selected! Please check the boxes in Step 4.")
            else:
                with st.status("Executing Trust Pipeline Final Phase...", expanded=True) as pt_status:
                    base_text = st.session_state.enhanced_text if st.session_state.enhanced_text else st.session_state.raw_transcript
                    pt_status.write("Enhancing base text strictly with the requested sources...")
                    sources_text = tp.format_sources(pipeline_chunks)
                    st.session_state.tp_enhanced_text = tp.step_a_enhance_text(base_text, sources_text, prompt_instruction)
                    
                    pt_status.write("Generating final output based on extracted citations and enhanced text...")
                    
                    # Ensure only the citations for the selected chunks are fed to the LLM (or just all recovered)
                    selected_filenames = [c.get('filename') for c in pipeline_chunks]
                    filtered_citations = []
                    for cit in st.session_state.tp_citations_list:
                        for fn in selected_filenames:
                            if fn in cit and cit not in filtered_citations: filtered_citations.append(cit)
                    
                    citations_text = "\n".join(filtered_citations) if filtered_citations else "No citations provided."
                    
                    final_academic_text = tp.step_c_create_academic_paragraph(
                        base_text, st.session_state.tp_enhanced_text, citations_text, prompt_instruction
                    )
                    pt_status.update(label="Trust Pipeline Completed Successfully!", state="complete", expanded=False)
                st.session_state.trust_pipeline_output = final_academic_text

        if st.session_state.trust_pipeline_output:
            st.markdown("### Final Output (.md)")
            mode_final = st.toggle("Edit Final Output Mode", value=st.session_state.is_editing_final, key="toggle_edit_final")
            if mode_final:
                st.session_state.is_editing_final = True
                edited = st.text_area("Edit Final Text", st.session_state.trust_pipeline_output, height=400, label_visibility="collapsed")
                if edited != st.session_state.trust_pipeline_output: st.session_state.trust_pipeline_output = edited
            else:
                st.session_state.is_editing_final = False
                st.markdown(st.session_state.trust_pipeline_output)
                
            st.download_button(
                label="💾 Download Final Academic Synthesis (.md)",
                data=st.session_state.trust_pipeline_output, file_name="final_academic_synthesis.md", mime="text/markdown", use_container_width=True
            )